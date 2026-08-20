# SmartDoc-Scanner 智能文档扫描系统

🌐 **在线演示：[https://smartdoc-scanner.streamlit.app](https://smartdoc-scanner.streamlit.app)**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red.svg)](https://smartdoc-scanner.streamlit.app)

## 📖 项目简介

这是一个基于传统计算机视觉与深度学习的智能文档扫描系统，模拟"扫描全能王"的核心功能。
项目展示了从图像预处理、几何变换到图像增强的完整 CV 流程，适合计算机视觉初学者学习。

## 🎯 核心功能与效果展示

### 1. 自动文档检测与透视矫正
- 使用 Canny 边缘检测提取文档轮廓
- 通过单应性矩阵（Homography Matrix）实现透视变换
- 将倾斜拍摄的文档自动拉正
- 本系统适用于俯拍角度（< 45度），极大角度可能导致矫正失败。

### 2. 智能图像增强
- **背景除法**（Background Division）：去除光照不均和阴影
- **自适应二值化**：生成清晰的扫描件效果
- 实现从"照片"到"扫描件"的转换

### 📸 效果演示 (Demo Result)

下图展示了从“手机拍摄倾斜文档”到“生成专业扫描件”的全过程：
*   **左图 (Warped)**：透视矫正后的原始文档。
*   **右图 (Enhanced)**：经过背景除法与自适应二值化后的最终扫描件效果。阴影完全消除，字迹清晰锐利。

<img width="1282" height="500" alt="demo_result" src="https://github.com/user-attachments/assets/1653e0e8-1f52-4a24-8f22-a263f50ac71f" />


## 🛠️ 技术栈

- **图像处理**：OpenCV 4.8+
- **深度学习**：PaddlePaddle + PaddleOCR
- **开发语言**：Python 3.10+
- **工程实践**：面向对象设计、虚拟环境管理

## 📦 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/caobianzi/SmartDoc-Scanner.git
cd SmartDoc-Scanner
##  在线演示（Streamlit Web 应用）

本项目提供了交互式的 Web 界面，无需配置环境即可体验！

### 本地运行
```bash
# 安装依赖
pip install -r requirements.txt

# 启动 Streamlit 应用
streamlit run streamlit_app.py


