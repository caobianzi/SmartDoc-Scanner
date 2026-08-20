from ultralytics import YOLO
import cv2
import numpy as np

class DocumentDetector:
    """基于 YOLOv8 的文档检测器"""
    
    def __init__(self):
        # 加载预训练的 YOLOv8 模型 (nano 版本，速度最快)
        # 这里我们先用通用的 COCO 模型，后面可以换成专门检测文档的模型
        self.model = YOLO('yolov8n.pt')
        print("✅ YOLOv8 文档检测器已初始化")
        
    def detect(self, image_path):
        """
        检测图片中的文档区域
        返回：文档的四个角点坐标 (numpy array) 或 None
        """
        # 运行推理
        results = self.model(image_path, verbose=False)
        
        # 获取结果
        result = results[0]
        
        # 获取边界框 (xyxy 格式: xmin, ymin, xmax, ymax)
        boxes = result.boxes.xyxy.cpu().numpy()
        
        if len(boxes) == 0:
            return None
            
        # 找到面积最大的那个框（假设最大的就是文档）
        # 实际应用中，这里应该用专门检测 "book" 或 "paper" 的类别
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
            
        # 将 xyxy 转换为四个角点 (左上, 右上, 右下, 左下)
        x1, y1, x2, y2 = largest_box
        pts = np.array([
            [x1, y1],  # 左上
            [x2, y1],  # 右上
            [x2, y2],  # 右下
            [x1, y2]   # 左下
        ], dtype="float32")
        
        return pts