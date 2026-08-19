import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_image(image_path:str) -> None:
    """
    读取图像，进行基础处理并展示。
    
    :param image_path: 输入图像的路径
    """
    #1.读取图像
    #OpenCV默认图像后本质是一个Numpy的多维数组（ndarray）。
    #OpenCV默认读取的彩色图像通道顺序是BGR，而不是常见的RGB！
    img_bgr = cv2.imread(image_path)

    if image_path is None:
        raise FileNotFoundError(f"无法读取图像，请检查路径:{image_path}")


    #2.色彩空间转换
    #将BGR转换为 灰度图（Grayscale).灰度图是后续边缘检测的基础。
    img_gray = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
    # 将 BGR 转换为 RGB，仅为了后续用 Matplotlib 正确显示彩色图
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    #3.基础滤波（去噪）
    #使用高斯模糊。核大小（5，5）必须是奇数。
    #原理：用周围像素的甲醛平均来平滑图像，消除高频噪音。
    img_blurred= cv2.GaussianBlur(img_gray,(5,5),0)


    #4.边缘检测（Canny）
    #阈值50和150 是经验值，用于控制边缘的敏感度。
    edges= cv2.Canny(img_blurred,threshold1=50,threshold2=150)


    #5.结果展示与保存
    #使用Matplotlib展示，因为它支持中文和更灵活的排版
    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.title("Original(RGB)")
    plt.imshow(img_rgb)
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.title("Grayscale & Blurred")
    plt.imshow(img_blurred,cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Canny Edges")
    plt.imshow(edges, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    #使用OpenCV保存边缘检测结果到本地
    cv2.imwrite("output_edges.jpg",edges)
    print("处理完成，边缘图已保存为 output_edges.jpg")

if __name__ == "__main__":
    #请准备一张包含清晰边缘的图片（如文档、书本、包装盒），放在同级目录下
    #修改这里的文件名为你自己的图片名
    TARGET_IMAGE = "test_doc.jpg"
    process_image(TARGET_IMAGE)
