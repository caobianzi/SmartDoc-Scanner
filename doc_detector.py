import cv2
import numpy as np

class DocumentDetector:
    """
    智能文档检测器 (工业级增强版)
    策略：利用"白纸高亮"特性 + 形态学操作，无视复杂背景纹理
    """
    
    def __init__(self):
        print("✅ 智能文档检测器已初始化 (基于亮度分割算法)")
        
    def detect(self, image_path):
        """
        检测图片中的文档区域
        """
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # 1. 转灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. 高斯模糊去噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. 核心策略：阈值分割
        # 原理：白纸通常非常亮 (像素值 > 200)，而地板/桌子通常较暗
        # 这一步能直接过滤掉地板的条纹（因为条纹通常有阴影，比较暗）
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        
        # 4. 形态学操作：闭运算
        # 作用：填补纸张内部的文字空洞，把纸张变成一个实心的白色块
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 5. 找轮廓
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # 6. 找到面积最大的轮廓（假设就是那张纸）
        c = max(contours, key=cv2.contourArea)
        
        # 过滤掉太小的噪点
        if cv2.contourArea(c) < 5000:
            return None
            
        # 7. 近似多边形（找四边形）
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # 如果找到了四边形，直接返回
        if len(approx) == 4:
            print("✅ 策略1：成功通过亮度分割找到四边形文档")
            return approx.reshape(4, 2)
        
        # 8. 兜底策略：如果没找到四边形，返回最大轮廓的最小外接矩形
        print("⚠️ 策略1未找到四边形，使用最小外接矩形兜底")
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="float32")
        return box