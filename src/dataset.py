# -*- coding: utf-8 -*-
"""
dataset.py
==========
数据集加载器与高保真双目+IMU 闭环仿真生成器
1. EuRoCDataset: 标准 EuRoC MAV 数据集加载器 (支持 mav0/cam0, cam1, imu0, state_groundtruth_estimate0)。
2. SyntheticStereoInertialDataset: 生成带丰富多尺度纹理的 3D 室内外场景双目图像流、200Hz IMU 测量与精确真值轨迹，
   用于零外部依赖下一键验证四叉树 ORB、极线立体匹配、光流跟踪、舒尔补 Local BA 与回环闭合全流程。
"""

from __future__ import annotations
import os
import csv
import numpy as np
import cv2

from .core.lie_algebra import SE3, exp_so3, log_so3
from .core.camera import StereoCamera
from .frontend.imu_preintegration import ImuMeasurement


class EuRoCDataset:
    """
    标准 EuRoC MAV 数据集读取器 (如 MH_01_easy/mav0)
    """
    def __init__(self, mav0_dir: str):
        self.mav0_dir = mav0_dir
        self.cam0_csv = os.path.join(mav0_dir, "cam0", "data.csv")
        self.cam0_img_dir = os.path.join(mav0_dir, "cam0", "data")
        self.cam1_img_dir = os.path.join(mav0_dir, "cam1", "data")
        self.imu_csv = os.path.join(mav0_dir, "imu0", "data.csv")
        self.gt_csv = os.path.join(mav0_dir, "state_groundtruth_estimate0", "data.csv")

        self.frames: list[tuple[float, str, str]] = []
        self.imu_data: list[ImuMeasurement] = []
        self.gt_timestamps: np.ndarray = np.zeros(0, dtype=np.float64)
        self.gt_positions: np.ndarray = np.zeros((0, 3), dtype=np.float64)

        self._load_metadata()

    def _load_metadata(self) -> None:
        if not os.path.exists(self.cam0_csv):
            raise FileNotFoundError(f"未找到 EuRoC cam0 数据文件: {self.cam0_csv}")

        with open(self.cam0_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2 or row[0].startswith("#"):
                    continue
                ts_ns = int(row[0].strip())
                fname = row[1].strip()
                path_l = os.path.join(self.cam0_img_dir, fname)
                path_r = os.path.join(self.cam1_img_dir, fname)
                if os.path.exists(path_l) and os.path.exists(path_r):
                    self.frames.append((ts_ns * 1e-9, path_l, path_r))

        if os.path.exists(self.imu_csv):
            with open(self.imu_csv, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 7 or row[0].startswith("#"):
                        continue
                    ts = int(row[0].strip()) * 1e-9
                    gyro = np.array([float(row[1]), float(row[2]), float(row[3])], dtype=np.float64)
                    acc = np.array([float(row[4]), float(row[5]), float(row[6])], dtype=np.float64)
                    self.imu_data.append(ImuMeasurement(timestamp=ts, acc=acc, gyro=gyro))

        if os.path.exists(self.gt_csv):
            gt_ts = []
            gt_pos = []
            t_b_c0 = np.array([-0.0216401454975, -0.064676986768, 0.00981073058949], dtype=np.float64)
            with open(self.gt_csv, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 8 or row[0].startswith("#"):
                        continue
                    gt_ts.append(int(row[0].strip()) * 1e-9)
                    p_wb = np.array([float(row[1]), float(row[2]), float(row[3])], dtype=np.float64)
                    qw, qx, qy, qz = float(row[4]), float(row[5]), float(row[6]), float(row[7])
                    # 四元数转旋转矩阵 R_wb，计算左目相机光心真实世界坐标 p_wc = p_wb + R_wb * t_b_c0
                    R_wb = np.array([
                        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
                        [2*(qx*qy + qw*qz),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
                        [2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx*qx + qy*qy)]
                    ], dtype=np.float64)
                    p_wc = p_wb + (R_wb @ t_b_c0)
                    gt_pos.append(p_wc)
            self.gt_timestamps = np.asarray(gt_ts, dtype=np.float64)
            self.gt_positions = np.asarray(gt_pos, dtype=np.float64)

    def __len__(self) -> int:
        return len(self.frames)

    def get_item(self, idx: int) -> tuple[np.ndarray, np.ndarray, float, list[ImuMeasurement], np.ndarray | None]:
        ts, path_l, path_r = self.frames[idx]
        img_l = cv2.imread(path_l, cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(path_r, cv2.IMREAD_GRAYSCALE)

        prev_ts = self.frames[idx - 1][0] if idx > 0 else ts - 0.05
        imus = [m for m in self.imu_data if prev_ts < m.timestamp <= ts]

        gt_pos = None
        if len(self.gt_timestamps) > 0:
            nearest_i = int(np.argmin(np.abs(self.gt_timestamps - ts)))
            if abs(self.gt_timestamps[nearest_i] - ts) < 0.05:
                gt_pos = self.gt_positions[nearest_i].copy()

        return img_l, img_r, ts, imus, gt_pos


class SyntheticStereoInertialDataset:
    """
    高真实感 3D 纹理场景双目 + IMU 仿真序列生成器
    生成具有丰富 FAST/ORB 纹理斑块的 3D 空间地标，渲染无畸变极线对齐的左右目灰度图像与 200Hz IMU 数据。
    """
    def __init__(
        self,
        camera: StereoCamera,
        num_frames: int = 80,
        num_landmarks: int = 450,
        seed: int = 2026
    ):
        self.camera = camera
        self.num_frames = int(num_frames)
        self.rng = np.random.default_rng(seed)

        # 1. 生成围绕相机轨迹分布的 3D 纹理特征簇与随机结构斑块
        self.landmarks_w, self.landmark_patterns = self._generate_3d_textured_world(num_landmarks)

        # 2. 生成平滑的 3D 闭环/曲线相机运动真值轨迹 T_wc(t)
        self.timestamps, self.gt_poses_wc, self.imu_batches = self._generate_trajectory_and_imu()

    def _generate_3d_textured_world(self, n_landmarks: int) -> tuple[np.ndarray, np.ndarray]:
        """在环形走廊空间生成 3D 结构点及每个点的唯一局部像素微纹理模板 (9x9)"""
        pts = []
        # 分布在相机前方及环线四周 (深度 1.5m ~ 7.5m)
        for _ in range(n_landmarks):
            angle = self.rng.uniform(-np.pi, np.pi)
            radius = self.rng.uniform(2.2, 6.2)
            x = radius * np.sin(angle)
            z = 3.0 + radius * np.cos(angle)
            y = self.rng.uniform(-1.6, 1.6)
            pts.append([x, y, z])
        landmarks_w = np.asarray(pts, dtype=np.float64)

        # 为每个 3D 点生成独特的 9x9 高对比度棋盘/角点微纹理，使 FAST 与 ORB 描述子具有稳定可区分性
        patterns = self.rng.integers(40, 235, size=(n_landmarks, 9, 9), dtype=np.uint8)
        for i in range(n_landmarks):
            # 增强中心角点梯度以触发 FAST 检测器
            if i % 2 == 0:
                patterns[i, 3:6, 3:6] = 250
                patterns[i, 0:2, :] = 30
                patterns[i, :, 0:2] = 30
            else:
                patterns[i, 3:6, 3:6] = 15
                patterns[i, 0:2, :] = 225
                patterns[i, :, 0:2] = 225
        return landmarks_w, patterns

    def _generate_trajectory_and_imu(self) -> tuple[np.ndarray, list[SE3], list[list[ImuMeasurement]]]:
        """生成 20 FPS 相机平滑 6DoF 轨迹与帧间 200Hz IMU 测量"""
        dt_cam = 1.0 / self.camera.fps
        timestamps = np.arange(self.num_frames, dtype=np.float64) * dt_cam
        gt_poses_wc: list[SE3] = []

        # 设计平滑的椭圆 + 波浪 3D 轨迹，首尾相接形成闭环
        for k, t in enumerate(timestamps):
            phase = 2.0 * np.pi * (float(k) / float(self.num_frames))
            tx = 1.2 * np.sin(phase)
            ty = 0.25 * np.sin(2.0 * phase)
            tz = 1.0 * (1.0 - np.cos(phase))
            pos = np.array([tx, ty, tz], dtype=np.float64)

            yaw = -0.35 * np.sin(phase)
            pitch = 0.08 * np.cos(phase)
            roll = 0.05 * np.sin(2.0 * phase)
            R_wc = exp_so3(np.array([pitch, yaw, roll], dtype=np.float64))
            gt_poses_wc.append(SE3(R_wc, pos))

        # 根据解析位姿差分生成帧间 200Hz IMU 测量 (每个相机间隔 10 条 IMU)
        g_w = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        imu_batches: list[list[ImuMeasurement]] = [[]]
        n_sub = 10

        for k in range(1, self.num_frames):
            T_prev = gt_poses_wc[k - 1]
            T_curr = gt_poses_wc[k]
            t0 = timestamps[k - 1]
            dt_sub = dt_cam / n_sub

            # 帧间平均角速度与加速度
            omega_body = log_so3(T_prev.R.T @ T_curr.R) / dt_cam
            v_prev = (T_curr.t - T_prev.t) / dt_cam if k == 1 else (gt_poses_wc[k - 1].t - gt_poses_wc[k - 2].t) / dt_cam
            v_curr = (T_curr.t - T_prev.t) / dt_cam
            a_world = (v_curr - v_prev) / dt_cam - g_w

            batch: list[ImuMeasurement] = []
            for s in range(1, n_sub + 1):
                ts_imu = t0 + s * dt_sub
                alpha = float(s) / float(n_sub)
                R_interp = T_prev.R @ exp_so3(omega_body * (alpha * dt_cam))
                acc_body = R_interp.T @ a_world
                batch.append(ImuMeasurement(timestamp=ts_imu, acc=acc_body, gyro=omega_body))
            imu_batches.append(batch)

        return timestamps, gt_poses_wc, imu_batches

    def render_stereo_pair(self, T_wc: SE3) -> tuple[np.ndarray, np.ndarray]:
        """根据给定相机位姿 T_wc 渲染左目与右目灰度图像"""
        h, w = self.camera.height, self.camera.width
        img_l = np.full((h, w), 115, dtype=np.uint8)
        img_r = np.full((h, w), 115, dtype=np.uint8)

        T_cw = T_wc.inverse()
        pts_c = T_cw.transform(self.landmarks_w)

        # 按深度从远到近排序绘制 (画家算法处理遮挡)
        valid_mask = (pts_c[:, 2] > 0.6) & (pts_c[:, 2] < 12.0)
        valid_indices = np.nonzero(valid_mask)[0]
        order = valid_indices[np.argsort(-pts_c[valid_indices, 2])]

        fx, fy, cx, cy, bf = (
            self.camera.fx, self.camera.fy,
            self.camera.cx, self.camera.cy, self.camera.bf
        )

        for idx in order:
            X, Y, Z = pts_c[idx]
            inv_z = 1.0 / Z
            ul = int(np.round(fx * X * inv_z + cx))
            vl = int(np.round(fy * Y * inv_z + cy))
            ur = int(np.round(ul - bf * inv_z))

            if 6 <= vl < h - 6:
                pat = self.landmark_patterns[idx]
                if 6 <= ul < w - 6:
                    img_l[vl - 4 : vl + 5, ul - 4 : ul + 5] = pat
                if 6 <= ur < w - 6:
                    img_r[vl - 4 : vl + 5, ur - 4 : ur + 5] = pat

        return img_l, img_r
