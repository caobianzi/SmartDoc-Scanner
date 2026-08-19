import cv2
import numpy as np
import matplotlib.pyplot as plt

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def process_document(image_path: str):
    # 1. 读取与预处理
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # 2. 找轮廓与透视矫正 (复用之前的逻辑)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    screen_cnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screen_cnt = approx
            break
            
    if screen_cnt is None:
        print("未找到文档轮廓")
        return

    pts = order_points(screen_cnt.reshape(4, 2))
    width = max(np.linalg.norm(pts[0]-pts[1]), np.linalg.norm(pts[2]-pts[3]))
    height = max(np.linalg.norm(pts[0]-pts[3]), np.linalg.norm(pts[1]-pts[2]))
    dst_pts = np.array([[0,0], [width-1,0], [width-1,height-1], [0,height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (int(width), int(height)))
    
    # ==========================================
    # 3. 核心：图像增强 (背景除法)
    # ==========================================
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # 第一步：生成“背景图”。用超大核的高斯模糊，把字模糊掉，只保留光照/阴影。
    # 核大小必须是奇数，且要足够大（比如 21x21 或 31x31）
    background = cv2.GaussianBlur(warped_gray, (51, 51), 0)
    
    # 第二步：背景除法。公式：result = original / background * 255
    # 注意：防止除以0，加一个极小值 1e-5
    enhanced = (warped_gray.astype("float32") / (background.astype("float32") + 1e-5)) * 255
    
    # 第三步：裁剪数值到 0-255，并转回整数
    enhanced = np.clip(enhanced, 0, 255).astype("uint8")
    
    # 第四步：二值化（可选，让字更黑，纸更白）。这里用自适应阈值
    # 如果不需要纯黑白，只想要清晰灰度图，注释掉下面这行即可
    # final_img = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # final_img = enhanced # 暂时先看增强后的灰度图
    
    # 使用自适应阈值进行二值化，让文字变纯黑，纸张变纯白
    # 11 是邻域大小，2 是常数 C（微调用）
    final_img = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 8)
#代码中 adaptiveThreshold 的最后一个参数 C，C 是一个常数，从计算出的局部阈值中减去。C 越大，阈值越高，只有非常黑的像素（比如手写字）才会被保留，浅色的横线就会被过滤成白色。
    # 4. 展示结果
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Warped (Raw)")
    plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Enhanced (Scanner Style)")
    plt.imshow(final_img, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    process_document("test_doc.jpg")