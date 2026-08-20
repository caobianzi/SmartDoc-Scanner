import cv2
import numpy as np

class DocumentDetector:
    """
    终极文档检测器 (边缘 + 形态学)
    专门对付复杂背景（地板、桌布）
    """
    
    def __init__(self):
        print("✅ 终极文档检测器已初始化")
        
    def detect(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # 1. 转灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. 高斯模糊 (去噪)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. 自适应阈值 (比固定阈值更稳健)
        # 这一步能把图片变成黑白分明的二值图，突出文字和边缘
        thresh = cv2.adaptiveThreshold(blurred, 255, 1, 1, 11, 2)
        
        # 4. 形态学操作：闭运算
        # 作用：把断裂的线条连起来，把纸张变成一个整体
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 5. Canny 边缘检测
        edges = cv2.Canny(closed, 50, 150)
        
        # 6. 再次形态学操作：膨胀
        # 作用：把边缘加粗，方便找轮廓
        dilated_edges = cv2.dilate(edges, kernel, iterations=2)
        
        # 7. 找轮廓
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # 8. 智能筛选：找到“最像文档”的轮廓
        # 规则：面积大 + 近似四边形
        doc_contour = None
        max_area = 0
        
        for c in contours:
            area = cv2.contourArea(c)
            # 过滤掉太小的噪点 (假设图片至少是 640x480，面积 > 50000 才算大)
            if area < 50000:
                continue
                
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            # 如果找到了四边形，且面积最大，那就是它了！
            if len(approx) == 4 and area > max_area:
                max_area = area
                doc_contour = approx
        
        # 9. 兜底：如果没找到四边形，就取面积最大的那个轮廓的外接矩形
        if doc_contour is None:
            print("⚠️ 未找到四边形，尝试使用最大轮廓的外接矩形")
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) < 50000:
                return None # 实在找不到
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            return np.array(box, dtype="float32")
        
        print("✅ 成功检测到文档轮廓")
        return doc_contour.reshape(4, 2)