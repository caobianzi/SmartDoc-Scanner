import cv2
import numpy as np
import matplotlib.pyplot as plt

def order_points(pts):
    """
    将四个点排序为：左上、右上、右下、左下
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # 1. 按 x 坐标排序：前两个是左边的点，后两个是右边的点
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上（x+y 最小）
    rect[2] = pts[np.argmax(s)]  # 右下（x+y 最大）
    
    # 2. 计算差值：y - x
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # 右上（y-x 最小）
    rect[3] = pts[np.argmax(diff)] # 左下（y-x 最大）
    
    return rect

def find_and_transform(image_path: str) -> None:
    """
    寻找图像中最大的四边形轮廓，并进行透视矫正。
    """
    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    
    # 保持一份原图用于最后展示
    img_original = img.copy()
    
    # 2. 预处理（复用阶段一的知识）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # 3. 寻找轮廓 (这是关键！)
    # cv2.RETR_LIST: 检索所有轮廓，不建立层级关系
    # cv2.CHAIN_APPROX_SIMPLE: 压缩水平、垂直和对角线方向的元素，只保留端点（省内存）
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # 4. 筛选最大的四边形轮廓
    # 按面积从大到小排序
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5] # 取前5个最大的
    
    screen_cnt = None
    for c in contours:
        # 计算轮廓的周长
        peri = cv2.arcLength(c, True)
        # 近似多边形，epsilon 是精度（周长的2%）
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # 如果近似后的多边形有4个顶点，我们就认为找到了文档/物体！
        if len(approx) == 4:
            screen_cnt = approx
            break
            
    # 5. 透视变换（拉平图像）
    if screen_cnt is not None:
        # 获取四个顶点的坐标
        pts = order_points(screen_cnt.reshape(4, 2))
        
        # 计算变换后的宽高（取最大宽高，保证不裁剪）
        width = max(np.linalg.norm(pts[0]-pts[1]), np.linalg.norm(pts[2]-pts[3]))
        height = max(np.linalg.norm(pts[0]-pts[3]), np.linalg.norm(pts[1]-pts[2]))
        
        # 定义目标矩形的四个角点（左上，右上，右下，左下）
        dst_pts = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")
        
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(pts.astype("float32"), dst_pts)
        
        # 应用变换
        warped = cv2.warpPerspective(img, M, (int(width), int(height)))
        
        # 6. 展示结果
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.title("Original")
        plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title("Edges & Contour")
        # 在原图上画出找到的轮廓
        img_with_contour = img_original.copy()
        cv2.drawContours(img_with_contour, [screen_cnt], -1, (0, 255, 0), 3) # 绿色粗线
        plt.imshow(cv2.cvtColor(img_with_contour, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title("Warped (Top-down View)")
        plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # 保存结果
        cv2.imwrite("output_warped.jpg", warped)
        print(f"透视矫正完成！已保存为 output_warped.jpg，尺寸: {int(width)}x{int(height)}")
    else:
        print("未找到明显的四边形轮廓，请尝试更换图片（找一张有明显边界的文档或包装盒）。")

if __name__ == "__main__":
    # 建议换一张有明显矩形边界的图，比如一张A4纸、一本书、或者一个快递盒
    # 如果还是用花椒锅巴，可能因为形状不规则找不到4个顶点
    TARGET_IMAGE = "test_doc.jpg" 
    find_and_transform(TARGET_IMAGE)