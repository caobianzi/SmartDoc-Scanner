import cv2
import numpy as np

class DocumentDetector:
    """
    文档检测器 - 稳健版
    降低阈值，增加容错
    """
    
    def __init__(self):
        print("✅ 文档检测器已初始化")
        
    def detect(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            print("❌ 无法读取图片")
            return None
        
        h, w = img.shape[:2]
        print(f"📐 图片尺寸: {w}x{h}")
            
        # 1. 转灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. 自适应阈值
        thresh = cv2.adaptiveThreshold(blurred, 255, 1, 1, 11, 2)
        
        # 4. 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 5. Canny 边缘检测（降低阈值）
        edges = cv2.Canny(closed, 30, 100)
        
        # 6. 膨胀
        dilated_edges = cv2.dilate(edges, kernel, iterations=2)
        
        # 7. 找轮廓
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("❌ 未找到任何轮廓")
            return None
        
        print(f"🔍 找到 {len(contours)} 个轮廓")
            
        # 8. 智能筛选
        doc_contour = None
        max_area = 0
        
        # 降低面积阈值：图片面积的 10%
        min_area = (w * h) * 0.1
        
        for c in contours:
            area = cv2.contourArea(c)
            print(f"  轮廓面积: {area}")
            
            if area < min_area:
                continue
                
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            if len(approx) == 4 and area > max_area:
                max_area = area
                doc_contour = approx
        
        # 9. 兜底策略
        if doc_contour is None:
            print("⚠️ 未找到四边形，使用外接矩形兜底")
            # 找面积最大的轮廓
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) < min_area:
                print("❌ 最大轮廓面积太小，返回图片四角")
                return np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype="float32")
            
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            print("✅ 使用外接矩形")
            return np.array(box, dtype="float32")
        
        print(f"✅ 成功检测到文档轮廓，面积: {max_area}")
        return doc_contour.reshape(4, 2)