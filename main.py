import os
os.environ["FLAGS_use_onednn"] = "0"
import cv2
import numpy as np
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR

class SmartScanner:
    """
    智能文档扫描器核心类
    功能：文档检测、透视矫正、图像增强、OCR识别
    """
    def __init__(self):
        # 初始化 OCR 模型 (use_gpu=False 表示用 CPU，有 NVIDIA 显卡可改为 True)
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ch')
        print("✅ 智能扫描器已初始化，OCR 模型加载完毕。")

    def order_points(self, pts):
        """安全的点排序函数"""
        # 确保 pts 是 4x2 的形状
        pts = np.array(pts, dtype="float32")
        if pts.shape[0] != 4:
            raise ValueError(f"期望4个点，但得到了 {pts.shape[0]} 个点")
        
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # 左上
        rect[2] = pts[np.argmax(s)]  # 右下
        
        diff = np.diff(pts, axis=1)
        # 确保 diff 是一维的
        if diff.ndim > 1:
            diff = diff.flatten()
        rect[1] = pts[np.argmin(diff)]  # 右上
        rect[3] = pts[np.argmax(diff)]  # 左下
        
        return rect

    def preprocess(self, image_path):
        """步骤 1: 读取与边缘检测"""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return img, edges

    def find_document_contour(self, edges):
        """步骤 2: 寻找最大四边形轮廓"""
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for c in contours[:10]:  # 检查前10个最大的轮廓
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                # 确保轮廓足够大
                if cv2.contourArea(approx) > 1000:  # 最小面积阈值
                    return approx.reshape(4, 2)
        return None

    def warp_image(self, img, pts):
        """步骤 3: 透视矫正"""
        pts = self.order_points(pts)
        width = max(np.linalg.norm(pts[0]-pts[1]), np.linalg.norm(pts[2]-pts[3]))
        height = max(np.linalg.norm(pts[0]-pts[3]), np.linalg.norm(pts[1]-pts[2]))
        
        dst_pts = np.array([
            [0, 0], [width - 1, 0],
            [width - 1, height - 1], [0, height - 1]
        ], dtype="float32")
        
        M = cv2.getPerspectiveTransform(pts, dst_pts)
        warped = cv2.warpPerspective(img, M, (int(width), int(height)))
        return warped

    def enhance_image(self, warped):
        """步骤 4: 图像增强 (背景除法 + 二值化)"""
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        # 使用大核高斯模糊获取背景
        background = cv2.GaussianBlur(warped_gray, (51, 51), 0)
        # 背景除法
        enhanced = (warped_gray.astype("float32") / (background.astype("float32") + 1e-5)) * 255
        enhanced = np.clip(enhanced, 0, 255).astype("uint8")
        # 自适应二值化 (C=8 可以过滤掉浅色横线)
        final_img = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 8)
        return final_img

    def scan(self, image_path):
        """
        主流程：一键扫描
        返回：矫正后的图，增强后的图，识别出的文字
        """
        print(f" 正在处理图像: {image_path} ...")
        
        # 1. 预处理
        img, edges = self.preprocess(image_path)
        
        # 2. 找轮廓
        cnt = self.find_document_contour(edges)
        if cnt is None:
            raise ValueError("未找到文档轮廓，请确保图片中有清晰的矩形物体。")
        
        # 3. 矫正
        warped = self.warp_image(img, cnt)
        
        # 4. 增强
        enhanced = self.enhance_image(warped)
        
       # 5. OCR 识别（PaddleOCR 3.x 版本）
        try:
            result = self.ocr.predict(warped)
            
            text_content = ""
            if result:
                for res in result:
                    if 'rec_texts' in res:
                        texts = res['rec_texts']
                        for text in texts:
                            text_content += text + "\n"
        except Exception as e:
            print(f"⚠️  OCR 识别出错: {e}")
            text_content = "OCR 识别失败"
                
            print("✅ 处理完成！")
            return warped, enhanced, text_content

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 1. 实例化扫描器
    scanner = SmartScanner()
    
    # 2. 运行扫描
    # 请确保 test_doc.jpg 存在
    try:
        warped_img, enhanced_img, text = scanner.scan("test_doc.jpg")
        
        # 3. 展示结果
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.title("Warped (Corrected)")
        plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title("Enhanced (Scanner Style)")
        plt.imshow(enhanced_img, cmap='gray')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # 4. 打印识别结果
        print("\n" + "="*30)
        print("📝 识别出的文字内容:")
        print("="*30)
        print(text)
        print("="*30)
        
    except Exception as e:
        print(f"❌ 发生错误: {e}")