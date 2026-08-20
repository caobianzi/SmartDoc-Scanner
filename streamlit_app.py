import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import tempfile

# 设置页面配置
st.set_page_config(
    page_title="SmartDoc-Scanner",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS，让界面更漂亮
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 3rem;
    }
    .stImage > div {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# 标题
st.markdown('<p class="main-header">📄 SmartDoc-Scanner</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">智能文档扫描系统 | 基于传统CV与深度学习</p>', unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.header("️ 参数设置")
    st.markdown("---")
    debug_mode = st.checkbox("🔧 调试模式", value=False)
    if debug_mode:
        st.info("""
        **调试信息**：
        - 策略1：寻找标准四边形（最准确）
        - 策略2：使用最小外接矩形（容错）
        - 策略3：返回原图（兜底）
        """)
    
    st.subheader("透视矫正")
    canny_threshold1 = st.slider("Canny 阈值1", 0, 200, 50)
    canny_threshold2 = st.slider("Canny 阈值2", 0, 250, 150)
    
    st.subheader("图像增强")
    blur_kernel_size = st.slider("高斯模糊核大小", 21, 101, 51, step=2)
    adaptive_block_size = st.slider("自适应二值化块大小", 5, 21, 11, step=2)
    adaptive_c = st.slider("自适应二值化常数 C", 1, 15, 8)
    
    st.markdown("---")
    st.info("💡 **提示**：调整参数可以优化不同光照条件下的处理效果")
    
    # 显示项目信息
    st.markdown("---")
    st.markdown("**技术栈**")
    st.markdown("- OpenCV 4.8+")
    st.markdown("- Python 3.10+")
    st.markdown("- NumPy")
    
    # GitHub 链接
    st.markdown("---")
    st.markdown("[🔗 查看 GitHub 仓库](https://github.com/caobianzi/SmartDoc-Scanner)")

# 主界面
st.markdown("### 📤 上传文档图片")
st.markdown("支持拖拽上传或直接选择文件。系统将自动进行透视矫正和图像增强。")

uploaded_file = st.file_uploader(
    "选择图片...",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
    key="file_uploader"  # 添加 key 以便重置
)

# 如果使用了示例图片，覆盖 uploaded_file
if 'example_file' in st.session_state:
    uploaded_file = st.session_state['example_file']
# 示例图片按钮
# 示例图片按钮
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("📷 使用示例图片", use_container_width=True):
        if os.path.exists("test_doc.jpg"):
            st.session_state['use_example'] = True
            st.rerun()
        else:
            st.warning("未找到示例图片 test_doc.jpg")

# 处理示例图片逻辑
if 'use_example' not in st.session_state:
    st.session_state['use_example'] = False

if st.session_state['use_example'] and os.path.exists("test_doc.jpg"):
    uploaded_file = open("test_doc.jpg", "rb")
    st.session_state['use_example'] = False  # 重置状态

if uploaded_file is not None:
    # 显示原始图片
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📷 原始图片")
        input_image = Image.open(uploaded_file)
        st.image(input_image, use_container_width=True)
    
    # 处理按钮
    with st.spinner('🔍 正在进行边缘检测与轮廓提取...'):
        with st.spinner('📐 正在计算透视变换矩阵...'):
            with st.spinner('✨ 正在进行图像增强...'):
                # 转换为 OpenCV 格式
                img_array = np.array(input_image)
                if len(img_array.shape) == 2:
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                else:
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # 临时保存
                temp_path = "temp_streamlit.jpg"
                cv2.imwrite(temp_path, img_bgr)
                
                # 调用核心算法
                try:
                    
                    # 直接处理 numpy 数组而不是文件路径
                    from main_lite import SmartScanner
                    scanner = SmartScanner()
                    
                    # 运行扫描（传递参数！）
                    warped, enhanced, _ = scanner.scan(      
                        temp_path,
                        canny_threshold1=canny_threshold1,
                        canny_threshold2=canny_threshold2,
                        blur_kernel_size=blur_kernel_size,
                        adaptive_block_size=adaptive_block_size,
                        adaptive_c=adaptive_c
                    )
                    # 显示调试信息
                    if debug_mode:
                        st.success("✅ 文档检测成功！")
                    
                    # 转换为 RGB 供 Streamlit 显示
                    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
                    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
                    
                    # 显示结果
                    with col2:
                        st.markdown("#### ✅ 透视矫正 (Warped)")
                        st.image(warped_rgb, use_container_width=True)
                    
                    st.markdown("### 🎨 增强结果对比")
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        st.markdown("##### 📄 矫正后原图")
                        st.image(warped_rgb, use_container_width=True)
                    
                    with col4:
                        st.markdown("##### ✨ 增强后 (Scanner Style)")
                        st.image(enhanced_rgb, use_container_width=True)
                    
                    # 下载按钮
                    st.markdown("---")
                    st.markdown("### 💾 下载结果")
                    
                    col_d1, col_d2 = st.columns(2)
                    
                    with col_d1:
                        # 保存为 bytes
                        _, warped_encoded = cv2.imencode('.png', cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2BGR))
                        st.download_button(
                            label="📥 下载矫正图片",
                            data=warped_encoded.tobytes(),
                            file_name="warped_result.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    
                    with col_d2:
                        _, enhanced_encoded = cv2.imencode('.png', cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR))
                        st.download_button(
                            label="📥 下载增强图片",
                            data=enhanced_encoded.tobytes(),
                            file_name="enhanced_result.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    
                    st.success("✅ 处理完成！")
                    
                except Exception as e:
                    st.error(f"❌ 处理出错：{str(e)}")
                    st.exception(e)

# 底部说明
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9rem;'>
    <p> <b>使用建议</b>：拍摄时尽量让文档边缘清晰可见，效果更佳</p>
    <p>🔧 <b>参数调整</b>：如果自动检测失败，可在左侧调整 Canny 阈值和模糊核大小</p>
</div>
""", unsafe_allow_html=True)

# 隐藏 Streamlit 默认菜单
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)