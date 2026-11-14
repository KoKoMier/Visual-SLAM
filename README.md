# Visual-SLAM (Stereo / Stereo-Inertial / Monocular SLAM for RK3566 & Desktop)

面向无人机 6DoF 位姿估计与 **Rockchip RK3566 (4× ARM Cortex-A55 @ 1.8GHz)** 嵌入式部署的模块化视觉/视觉惯性 SLAM 系统。底层数学算子、前端特征跟踪与后端稀疏图优化严格 1:1 对齐 C++ `ORB-SLAM3`（`Eigen3` + `Sophus` + `g2o`）。

---

## 核心特性

1. **李群李代数与流形预积分 (`src/core/lie_algebra.py`, `src/frontend/imu_preintegration.py`)**
   - 显式实现 $\text{SO}(3)/\text{SE}(3)$ 指数/对数映射（往返精度 `5.22e-16`）与右雅可比矩阵 $\mathbf{J}_r(\boldsymbol{\phi})$。
   - 完整实现 Forster 流形预积分（$\Delta \mathbf{R}_{ij}, \Delta \mathbf{v}_{ij}, \Delta \mathbf{p}_{ij}$）、$15 \times 15$ 协方差递推与零偏一阶雅可比在线修正。
2. **RK3566 混合光流 + 四叉树 ORB 前端 (`src/frontend/`)**
   - **普通帧快速路径**：基于 200Hz IMU 陀螺仪先验的 4 层双向金字塔 LK 光流跟踪 + 二维结构张量极线立体匹配（普通帧中位耗时 `24.75 ms`）。
   - **关键帧路径**：8 层图像金字塔 + 四叉树均匀化 FAST/ORB 特征提取 + 局部共视地图投影跟踪 (`TrackLocalMap`)。
   - **多传感器模式**：支持 **双目 + IMU (`track_stereo_inertial`)**、**单目 + IMU (`track_monocular_inertial`)**、**纯双目 (`track_stereo`)** 与 **纯单目 (`track_monocular`)**。
3. **显式解析雅可比与稀疏舒尔补后端 (`src/backend/`)**
   - 4 轮 Huber 鲁棒核 + 卡方检验的 6DoF 视觉-惯性紧耦合位姿优化 (`pose_optimization`) 与重力方向 Roll/Pitch 闭环可观性约束。
   - 基于显式稀疏舒尔补消元 (`Sparse Schur Complement`) 的滑动窗口局部捆集调整 (`local_bundle_adjustment`) 与本质图回环优化 (`optimize_essential_graph`)。
4. **全量运行黑盒记录仪 (`src/blackbox_logger.py`)**
   - 每次运行自动生成逐帧 6DoF 位姿、IMU 原始/调平欧拉角、光流残差、旋转门控与真值误差黑盒日志 (`frames.csv`, `events.jsonl`, `summary.json`)，支持跨数据集与实机运行对比诊断。

---

## EuRoC `MH_01_easy` 真实无人机数据集评测结果

| 评测指标 | 实测数值 (Stereo + 200Hz IMU) |
| :--- | :--- |
| **绝对轨迹误差 ATE ($\text{SE}(3)$ RMSE, 真实尺度 $s=1.0$)** | **`3.45 cm` (`0.0345 m`)** |
| **绝对轨迹误差平均值 (Mean) / 中位数 (Median)** | **`3.09 cm` / `2.73 cm`** |
| **活跃关键帧数 / 3D 地图点数 (150 帧)** | **`21` KeyFrames / `845` MapPoints** |
| **普通帧前端跟踪中位耗时** | **`24.75 ms`** (`< 33.3 ms` 30 FPS 实时红线) |

![EuRoC SLAM Evaluation](outputs/euroc_slam_evaluation.png)

![Camera 6DoF Motion Storyboard](outputs/euroc_camera_motion_storyboard.png)

---

## 快速运行指南

```bash
# 1. 运行合成 3D 闭环数据验证 (零外部数据集依赖)
python run_phase1_demo.py

# 2. 运行 EuRoC MAV 真实数据集双目+IMU评测
python run_phase1_demo.py --euroc data/EuRoC_MH_01_easy/mav0 --max_frames 150

# 3. 运行实机 USB 单目摄像头 (640x480) + 串口 IMU (COM4 @ 9600) 实时单目惯性 SLAM (按 'q' 退出)
python run_phase1_demo.py --live --mono_imu --cam_id 1 --imu_port COM4

# 4. 运行实机 USB 同步拼接双目摄像头 (1280x480) + 串口 IMU 实时双目惯性 SLAM
python run_phase1_demo.py --live --cam_id 0 --imu_port COM4

# 5. 单目 / 双目摄像头内外参 + 相机与 IMU 手眼外参 (T_b_c1) 标定工具
python calibrate_mono_camera.py --cam_id 1
python calibrate_stereo_camera.py --generate_board
python calibrate_stereo_camera.py --capture --concat_width 1280 --height 480 --imu_port COM4 --imu_baud 9600
python calibrate_stereo_camera.py --calibrate --square_mm 25.0
```
