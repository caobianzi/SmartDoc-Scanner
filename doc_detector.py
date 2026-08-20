from ultralytics import YOLO
import cv2
import numpy as np

class DocumentDetector:
    """基于 YOLOv8 文档专用模型的检测器"""
    
    def __init__(self):
        # 加载专门针对文档微调过的 YOLOv8 模型
        # 这个模型认识“纸”、“文档”、“表格”，无视复杂背景
        print(" 正在加载文档专用 YOLO 模型 (首次运行需下载，请耐心等待)...")
        self.model = YOLO('keremberke/yolov8n-document')
        print("✅ 文档专用 YOLO 检测器已初始化")
        
    def detect(self, image_path):
        """
        检测图片中的文档区域
        """
        # 运行推理 (conf=0.25 降低阈值，让模型更敏感)
        results = self.model(image_path, conf=0.25, verbose=False)
        
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        
        if len(boxes) == 0:
            return None
            
        # 找到面积最大的那个框
        largest_box = None
        max_area = 0
        
        for box in boxes:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                largest_box = box
                
        if largest_box is None:
            return None
            
        # 将 xyxy 转换为四个角点
        x1, y1, x2, y2 = largest_box
        pts = np.array([
            [x1, y1],  # 左上
            [x2, y1],  # 右上
            [x2, y2],  # 右下
            [x1, y2]   # 左下
        ], dtype="float32")
        
        return pts