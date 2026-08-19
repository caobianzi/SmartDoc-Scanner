import cv2
import numpy as np

class SmartScanner:
    """智能文档扫描器 - 轻量版（无 OCR）"""
    
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

    def preprocess(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
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

    def enhance_image(self, warped):
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        background = cv2.GaussianBlur(warped_gray, (51, 51), 0)
        enhanced = (warped_gray.astype("float32") / (background.astype("float32") + 1e-5)) * 255
        enhanced = np.clip(enhanced, 0, 255).astype("uint8")
        final_img = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 8)
        return final_img

    def scan(self, image_path):
        print(f" 正在处理图像: {image_path} ...")
        
        img, edges = self.preprocess(image_path)
        cnt = self.find_document_contour(edges)
        if cnt is None:
            raise ValueError("未找到文档轮廓")
        
        warped = self.warp_image(img, cnt)
        enhanced = self.enhance_image(warped)
        
        text_content = "OCR 功能在云端已禁用"
        
        print("✅ 处理完成！")
        return warped, enhanced, text_content