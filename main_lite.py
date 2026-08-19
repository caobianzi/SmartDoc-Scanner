import cv2
import numpy as np

class SmartScanner:
    """智能文档扫描器 - 轻量版（无 OCR，支持参数调节）"""
    
    def __init__(self):
        print("✅ 智能扫描器已初始化（轻量版）")

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        if diff.ndim > 1:
            diff = diff.flatten()
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def preprocess(self, image_path, canny_threshold1=50, canny_threshold2=150):
        """预处理：边缘检测，支持参数调节"""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)
        return img, edges

    def find_document_contour(self, edges):
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for c in contours[:10]:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                if cv2.contourArea(approx) > 1000:
                    return approx.reshape(4, 2)
        return None

    def warp_image(self, img, pts):
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

    def enhance_image(self, warped, blur_kernel_size=51, adaptive_block_size=11, adaptive_c=8):
        """图像增强，支持参数调节"""
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        
        # 确保核大小是奇数
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1
        
        background = cv2.GaussianBlur(warped_gray, (blur_kernel_size, blur_kernel_size), 0)
        enhanced = (warped_gray.astype("float32") / (background.astype("float32") + 1e-5)) * 255
        enhanced = np.clip(enhanced, 0, 255).astype("uint8")
        
        # 确保块大小是奇数
        if adaptive_block_size % 2 == 0:
            adaptive_block_size += 1
        
        final_img = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, adaptive_block_size, adaptive_c)
        return final_img

    def scan(self, image_path, canny_threshold1=50, canny_threshold2=150, 
             blur_kernel_size=51, adaptive_block_size=11, adaptive_c=8):
        """主流程，支持所有参数调节"""
        print(f" 正在处理图像: {image_path} ...")
        print(f"  参数: Canny({canny_threshold1}, {canny_threshold2}), "
              f"Blur({blur_kernel_size}), Adaptive({adaptive_block_size}, {adaptive_c})")
        
        # 1. 预处理（带参数）
        img, edges = self.preprocess(image_path, canny_threshold1, canny_threshold2)
        
        # 2. 找轮廓
        cnt = self.find_document_contour(edges)
        if cnt is None:
            raise ValueError("未找到文档轮廓，请确保图片中有清晰的矩形物体。")
        
        # 3. 矫正
        warped = self.warp_image(img, cnt)
        
        # 4. 增强（带参数）
        enhanced = self.enhance_image(warped, blur_kernel_size, adaptive_block_size, adaptive_c)
        
        text_content = "OCR 功能在云端已禁用"
        
        print("✅ 处理完成！")
        return warped, enhanced, text_content