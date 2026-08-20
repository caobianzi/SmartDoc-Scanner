import cv2
import numpy as np
from doc_detector import DocumentDetector

class SmartScanner:
    """智能文档扫描器 - YOLO 版"""
    
    def __init__(self):
        self.detector = DocumentDetector()
        print("✅ 智能扫描器已初始化（YOLO 版）")

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
        
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1
            
        background = cv2.GaussianBlur(warped_gray, (blur_kernel_size, blur_kernel_size), 0)
        enhanced = (warped_gray.astype("float32") / (background.astype("float32") + 1e-5)) * 255
        enhanced = np.clip(enhanced, 0, 255).astype("uint8")
        
        if adaptive_block_size % 2 == 0:
            adaptive_block_size += 1
            
        final_img = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, adaptive_block_size, adaptive_c)
        return final_img

    def scan(self, image_path, blur_kernel_size=51, adaptive_block_size=11, adaptive_c=8,epsilon_factor=0.02):
        """主流程：YOLO 检测 -> 透视变换 -> 图像增强"""
        print(f" 正在处理图像: {image_path} ...")
        
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")

        from doc_detector import DocumentDetector
        detector = DocumentDetector()
        # 1. YOLO 检测文档区域
        pts = detector.detect(image_path, epsilon_factor)
        if pts is None:
            raise ValueError("未检测到文档，请确保图片中有清晰的文档/纸张。")
        
        print(f"✅ 检测到文档区域: {pts}")
        
        # 2. 透视变换
        warped = self.warp_image(img, pts)
        
        # 3. 图像增强
        enhanced = self.enhance_image(warped, blur_kernel_size, adaptive_block_size, adaptive_c)
        
        print("✅ 处理完成！")
        return warped, enhanced, "OCR 功能在云端已禁用"