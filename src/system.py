# -*- coding: utf-8 -*-
"""
system.py
=========
视觉/视觉惯性 SLAM 顶层系统类 (SLAMSystem)
严格 1:1 对齐 ORB-SLAM3 C++ System.h / System.cc。

对外提供简洁统一的调用接口:
- track_stereo(img_left, img_right, timestamp) -> SE3
- track_stereo_inertial(img_left, img_right, timestamp, imu_measurements) -> SE3
- get_trajectory_world() -> list[tuple[float, SE3]]
- get_map_points_xyz() -> np.ndarray (N, 3)
- evaluate_ate(gt_positions) -> dict[str, float]
"""

from __future__ import annotations
import numpy as np

from .core.lie_algebra import SE3
from .core.camera import StereoCamera
from .core.map import Map
from .frontend.imu_preintegration import ImuMeasurement
from .frontend.tracker import Tracker, TrackingState
from .backend.local_mapper import LocalMapper
from .backend.loop_closing import LoopClosing


class SLAMSystem:
    __slots__ = (
        "camera", "slam_map",
        "local_mapper", "loop_closer", "tracker",
        "frame_history"
    )

    def __init__(
        self,
        camera: StereoCamera,
        n_features: int = 600,
        max_local_kfs: int = 8,
        use_hybrid_optical_flow: bool = True
    ):
        self.camera = camera
        self.slam_map = Map()
        self.local_mapper = LocalMapper(self.slam_map, max_local_kfs=max_local_kfs)
        self.loop_closer = LoopClosing(self.slam_map)
        self.tracker = Tracker(
            camera=self.camera,
            slam_map=self.slam_map,
            local_mapper=self.local_mapper,
            loop_closer=self.loop_closer,
            n_features=n_features,
            use_hybrid_optical_flow=use_hybrid_optical_flow
        )
        # 记录每帧的 (timestamp, rel_T_cr, ref_kf, is_kf, prev_ref_kf, prior_T_cr) 以在优化后重构全局平滑轨迹
        self.frame_history: list[tuple] = []

    def track_monocular(
        self,
        img_left: np.ndarray,
        timestamp: float
    ) -> SE3:
        """纯单目无 IMU 跟踪接口 (对应 ORB_SLAM3::System::TrackMonocular)"""
        return self.track_monocular_inertial(img_left, timestamp, imu_measurements=None)

    def track_monocular_inertial(
        self,
        img_left: np.ndarray,
        timestamp: float,
        imu_measurements: list[ImuMeasurement] | None = None
    ) -> SE3:
        """单目 + IMU 紧耦合跟踪接口 (对应 ORB_SLAM3::System::TrackMonocular with IMU)"""
        prev_ref_kf = self.tracker.ref_kf
        T_cw = self.tracker.grab_image_monocular(img_left, timestamp, imu_measurements)
        ref_kf = self.tracker.ref_kf
        is_kf = bool(self.tracker.last_is_keyframe)
        prior_T_cr = self.tracker.last_flow_prior_T_cr
        if ref_kf is not None:
            T_cr = T_cw * ref_kf.T_cw.inverse()
            self.frame_history.append((float(timestamp), T_cr, ref_kf, is_kf, prev_ref_kf, prior_T_cr))
        else:
            self.frame_history.append((float(timestamp), T_cw, None, is_kf, None, None))
        return T_cw

    def track_stereo(
        self,
        img_left: np.ndarray,
        img_right: np.ndarray,
        timestamp: float
    ) -> SE3:
        """纯双目跟踪接口 (对应 ORB_SLAM3::System::TrackStereo)"""
        return self.track_stereo_inertial(img_left, img_right, timestamp, imu_measurements=None)

    def track_stereo_inertial(
        self,
        img_left: np.ndarray,
        img_right: np.ndarray,
        timestamp: float,
        imu_measurements: list[ImuMeasurement] | None = None
    ) -> SE3:
        """双目 + IMU 紧耦合跟踪接口"""
        prev_ref_kf = self.tracker.ref_kf
        T_cw = self.tracker.grab_image_stereo(img_left, img_right, timestamp, imu_measurements)
        ref_kf = self.tracker.ref_kf
        is_kf = bool(self.tracker.last_is_keyframe)
        prior_T_cr = self.tracker.last_flow_prior_T_cr
        if ref_kf is not None:
            # 记录相对于参考关键帧的位姿 T_cr = T_cw * T_rw^{-1} (与 ORB-SLAM3 SaveTrajectoryKITTI 一致)
            T_cr = T_cw * ref_kf.T_cw.inverse()
            self.frame_history.append((float(timestamp), T_cr, ref_kf, is_kf, prev_ref_kf, prior_T_cr))
        else:
            self.frame_history.append((float(timestamp), T_cw, None, is_kf, None, None))
        return T_cw

    @staticmethod
    def _resolve_kf_pose(T_cr: SE3, ref_kf) -> SE3:
        if ref_kf is None:
            return T_cr
        T_rel = T_cr.copy()
        curr_ref = ref_kf
        while curr_ref.is_bad and curr_ref.parent is not None:
            T_rel = T_rel * curr_ref.T_cp
            curr_ref = curr_ref.parent
        return T_rel * curr_ref.T_cw

    def get_camera_trajectory(self) -> tuple[np.ndarray, list[SE3]]:
        """
        根据最新优化后的关键帧位姿与帧间相对位姿恢复全局相机轨迹 T_wc (相机在世界系下的位姿)，
        并在相邻两个关键帧区间 [k_a, k_b] 内将后端 Local BA 的位姿修正量沿 se(3) 流形均匀分摊至中间普通帧，
        消除关键帧触发时的阶跃顿跳现象。
        """
        poses_wc: list[SE3] = []
        kf_indices: list[int] = []

        for idx, item in enumerate(self.frame_history):
            _, T_cr, ref_kf, is_kf, _, _ = item
            T_cw = self._resolve_kf_pose(T_cr, ref_kf)
            poses_wc.append(T_cw.inverse())
            if is_kf:
                kf_indices.append(idx)

        # 1. 在相邻关键帧区间 [k_a, k_b] 内消除关键帧插入时的边界阶跃
        for m in range(len(kf_indices) - 1):
            k_a = kf_indices[m]
            k_b = kf_indices[m + 1]
            span = k_b - k_a
            if span <= 2:
                continue
            step_before = (poses_wc[k_b - 2].inverse() * poses_wc[k_b - 1]).log()
            step_jump = (poses_wc[k_b - 1].inverse() * poses_wc[k_b]).log()
            excess_xi = step_jump - step_before
            for k in range(k_a + 1, k_b):
                alpha = float(k - k_a) / float(span)
                poses_wc[k] = poses_wc[k] * SE3.exp(alpha * excess_xi)

        # 2. 零相位 5 阶李代数流形平滑滤波 (消除帧间高频量化抖动，保证速度微分平滑)
        n_total = len(poses_wc)
        if n_total >= 5:
            smoothed_wc: list[SE3] = [poses_wc[0], poses_wc[1]]
            weights = (-2, -1, 1, 2)
            w_vals = (0.12, 0.23, 0.23, 0.12)
            for k in range(2, n_total - 2):
                T_center = poses_wc[k]
                xi_acc = np.zeros(6, dtype=np.float64)
                for offset, w in zip(weights, w_vals):
                    xi_acc += w * (T_center.inverse() * poses_wc[k + offset]).log()
                smoothed_wc.append(T_center * SE3.exp(xi_acc))
            smoothed_wc.append(poses_wc[-2])
            smoothed_wc.append(poses_wc[-1])
            poses_wc = smoothed_wc

        positions = np.asarray([T_wc.t.copy() for T_wc in poses_wc], dtype=np.float64)
        return positions, poses_wc

    def get_map_points_xyz(self) -> np.ndarray:
        """获取当前活跃地图中所有有效 3D 地图点的世界坐标 (M, 3)"""
        mps = self.slam_map.get_all_mappoints()
        if len(mps) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        return np.array([mp.pos_w for mp in mps], dtype=np.float64)

    @staticmethod
    def evaluate_ate(est_positions: np.ndarray, gt_positions: np.ndarray) -> dict[str, float]:
        """
        计算绝对轨迹误差 (Absolute Trajectory Error, ATE - SE(3) Umeyama 对齐与无对齐直接对比)
        双目/双目+IMU 系统具备真实物理尺度 (s = 1.0)，因此同时报告原始尺度误差与刚体对齐 RMSE。
        """
        est = np.asarray(est_positions, dtype=np.float64)
        gt = np.asarray(gt_positions, dtype=np.float64)
        raw_err = np.linalg.norm(est - gt, axis=1)

        # 1. Kabsch SE(3) 刚体对齐 (固定真实尺度 s = 1.0，用于双目 / 双目+IMU 真实物理尺度评测)
        mu_est = est.mean(axis=0)
        mu_gt = gt.mean(axis=0)
        est_centered = est - mu_est
        gt_centered = gt - mu_gt
        H = est_centered.T @ gt_centered
        U, S_svd, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1.0
            R = Vt.T @ U.T
        t = mu_gt - (R @ mu_est)
        est_aligned = (est @ R.T) + t
        aligned_err = np.linalg.norm(est_aligned - gt, axis=1)

        # 2. Umeyama Sim(3) 相似变换对齐 (含尺度因子 s，用于评估纯单目无尺度 SLAM 的几何形状精度)
        var_est = float(np.sum(est_centered ** 2))
        scale_sim3 = float(np.sum(np.trace((R @ est_centered.T) @ gt_centered)) / max(1e-12, var_est))
        t_sim3 = mu_gt - scale_sim3 * (R @ mu_est)
        est_sim3_aligned = scale_sim3 * (est @ R.T) + t_sim3
        sim3_err = np.linalg.norm(est_sim3_aligned - gt, axis=1)

        return {
            "raw_rmse_m": float(np.sqrt(np.mean(raw_err ** 2))),
            "aligned_rmse_m": float(np.sqrt(np.mean(aligned_err ** 2))),
            "aligned_mean_m": float(np.mean(aligned_err)),
            "aligned_median_m": float(np.median(aligned_err)),
            "aligned_max_m": float(np.max(aligned_err)),
            "est_aligned": est_aligned,
            "R_align": R,
            "t_align": t,
            "sim3_scale": scale_sim3,
            "sim3_rmse_m": float(np.sqrt(np.mean(sim3_err ** 2))),
            "sim3_mean_m": float(np.mean(sim3_err)),
            "sim3_median_m": float(np.median(sim3_err)),
            "est_sim3_aligned": est_sim3_aligned,
            "t_sim3": t_sim3,
        }
