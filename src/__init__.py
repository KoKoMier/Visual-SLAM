# -*- coding: utf-8 -*-
"""
ORB-SLAM3 + HF-Net 深度学习特征架构对齐版视觉/视觉惯性 SLAM 验证框架
"""
import os
import sys

# 自动将当前 Python 环境 (如 miniconda3/envs/hfnet) 的 Library/bin 加入 PATH，确保 OpenCV 3.4.2 DLL 正常加载
_conda_lib_bin = os.path.join(sys.prefix, "Library", "bin")
if os.path.isdir(_conda_lib_bin) and _conda_lib_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _conda_lib_bin + os.pathsep + os.environ.get("PATH", "")
