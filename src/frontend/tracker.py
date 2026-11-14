# -*- coding: utf-8 -*-
"""
frontend/tracker.py
===================
前端实时位姿跟踪状态机 (Tracker)
严格 1:1 对齐 ORB-SLAM3 C++ Tracking.h / Tracking.cc，
同时内置面向 RK3566 (Cortex-A55) 的“普通帧金字塔光流 + 关键帧四叉树 ORB”混合加速模式。

核心状态机与流程:
1. StereoInitialization():
   第一帧双目图像提取四叉树均匀化 ORB + 极线双目匹配，当有效双目深度点数 > 50 时，
   直接构建初始关键帧 KeyFrame 0 与初始 3D 地图点云，真实物理尺度一步到位。
2. Track():
   - 位姿初值预测: 若有 IMU 预积分数据则由 ImuPreintegrator 预测，否则由恒速运动模型 (velocity) 预测。
   - 帧间跟踪:
     a) RK3566 快速路径: 双向金字塔 LK 光流直接传递上一帧 3D-2D 地图点匹配 + Motion-only BA。
     b) 关键帧/重捕路径: 提取四叉树 ORB 描述子，结合光流引导与局部地图投影匹配 (TrackLocalMap)。
   - 局部地图跟踪 (TrackLocalMap):
     通过共视图收集一阶/二阶局部关键帧 (Local KeyFrames) 与局部地图点 (Local MapPoints)，
     执行视野锥体筛查 (is_in_frustum) + 64x48 网格投影匹配 + Motion-only BA 精化当前帧位姿。
   - 关键帧决策 (NeedNewKeyFrame):
     综合考量当前帧跟踪到的参考关键帧地图点比例、近距离双目点数量及帧间隔。
"""

from __future__ import annotations
from enum import IntEnum
from typing import TYPE_CHECKING
import time
import cv2
import numpy as np

from ..core.lie_algebra import SE3
from ..core.frame import Frame
from ..core.keyframe import KeyFrame
from ..core.mappoint import MapPoint
from .orb_extractor import ORBExtractor
from .stereo_matcher import StereoMatcher
from .imu_preintegration import ImuPreintegrator, ImuMeasurement
from ..backend.optimizer import Optimizer

if TYPE_CHECKING:
    from ..core.camera import StereoCamera
    from ..core.map import Map
    from ..backend.local_mapper import LocalMapper
    from ..backend.loop_closing import LoopClosing


class TrackingState(IntEnum):
    NO_IMAGES_YET = 0
    NOT_INITIALIZED = 1
    OK = 2
    LOST = 3


class Tracker:
    __slots__ = (
        "camera", "slam_map", "local_mapper", "loop_closer",
        "orb_extractor", "imu_preintegrator",
        "state", "curr_frame", "last_frame", "prev_frame", "mono_init_frame", "ref_kf",
        "last_kf_frame_id",
        "local_keyframes", "local_mappoints",
        "velocity", "has_velocity",
        "use_hybrid_optical_flow",
        "last_is_keyframe", "last_used_optical_flow", "last_inliers", "last_flow_prior_T_cr",
        "_gyro_vec_hist", "last_diag",
        "timing_stats"
    )

    def __init__(
        self,
        camera: StereoCamera,
        slam_map: Map,
        local_mapper: LocalMapper,
        loop_closer: LoopClosing,
        n_features: int = 600,
        use_hybrid_optical_flow: bool = True
    ):
        self.camera = camera
        self.slam_map = slam_map
        self.local_mapper = LocalMapper(self.slam_map) if local_mapper is None else local_mapper
        self.loop_closer = loop_closer

        self.orb_extractor = ORBExtractor(n_features=n_features)
        self.imu_preintegrator = ImuPreintegrator()

        self.state: TrackingState = TrackingState.NO_IMAGES_YET
        self.curr_frame: Frame | None = None
        self.last_frame: Frame | None = None
        self.prev_frame: Frame | None = None
        self.mono_init_frame: Frame | None = None
        self.ref_kf: KeyFrame | None = None
        self.last_kf_frame_id: int = 0

        self.local_keyframes: list[KeyFrame] = []
        self.local_mappoints: list[MapPoint] = []

        self.velocity: SE3 = SE3.identity()
        self.has_velocity: bool = False
        self.use_hybrid_optical_flow: bool = bool(use_hybrid_optical_flow)
        self.last_is_keyframe: bool = False
        self.last_used_optical_flow: bool = False
        self.last_inliers: int = 0
        self.last_flow_prior_T_cr: SE3 | None = None
        self._gyro_vec_hist: list[np.ndarray] = []
        self.last_diag: dict[str, float] = {
            "trans_gate": 1.0,
            "eff_gyro_deg_s": 0.0,
            "raw_flow_px": 0.0,
            "derot_res_px": 0.0,
            "radial_med_px": 0.0,
        }

        self.timing_stats: dict[str, list[float]] = {
            "frontend_ms": [],
            "backend_ms": [],
            "total_ms": []
        }

    def grab_image_stereo(
        self,
        img_left: np.ndarray,
        img_right: np.ndarray,
        timestamp: float,
        imu_measurements: list[ImuMeasurement] | None = None
    ) -> SE3:
        """
        主入口: 输入双目图像与可选的帧间 IMU 测量序列，返回当前帧估计位姿 T_cw (SE3)
        """
        t_start = time.perf_counter()

        # 1. 双目极线去畸变校正 (若相机配置了原始畸变映射表)
        rect_l, rect_r = self.camera.rectify_stereo(img_left, img_right)

        # 2. 若包含 IMU 数据，执行帧间流形预积分
        if imu_measurements is not None and len(imu_measurements) > 0:
            self.imu_preintegrator.reset(
                bg=self.last_frame.bg if self.last_frame is not None else None,
                ba=self.last_frame.ba if self.last_frame is not None else None
            )
            for idx_m, meas in enumerate(imu_measurements):
                if idx_m == 0:
                    prev_t = self.last_frame.timestamp if self.last_frame is not None else (meas.timestamp - 0.005)
                    dt = max(1e-4, meas.timestamp - prev_t)
                else:
                    dt = max(1e-4, meas.timestamp - imu_measurements[idx_m - 1].timestamp)
                self.imu_preintegrator.integrate_new_measurement(meas.acc, meas.gyro, dt)

        # 3. 尚未初始化时执行双目初始化
        if self.state in (TrackingState.NO_IMAGES_YET, TrackingState.NOT_INITIALIZED):
            self.state = TrackingState.NOT_INITIALIZED
            self._stereo_initialization(rect_l, rect_r, timestamp, imu_measurements)
            t_end = time.perf_counter()
            self.prev_frame = None
            self.last_is_keyframe = (self.state == TrackingState.OK)
            self.last_used_optical_flow = False
            self.last_inliers = sum(1 for mp in self.curr_frame.map_points if mp is not None)
            self.timing_stats["frontend_ms"].append((t_end - t_start) * 1000.0)
            self.timing_stats["backend_ms"].append(0.0)
            self.timing_stats["total_ms"].append((t_end - t_start) * 1000.0)
            return self.curr_frame.T_cw.copy()

        # 4. 正常跟踪状态 (TrackingState.OK)
        t_fe_start = time.perf_counter()
        self._update_last_frame_replaced_mappoints()
        need_kf_or_retrack = False
        n_inliers = 0

        # 路径 A: RK3566 混合光流快速跟踪路径 (针对非关键帧省去全图 ORB 描述子计算)
        T_cw_pred_init = self._compute_predicted_pose(imu_measurements)
        has_imu_prior = bool(imu_measurements is not None and len(imu_measurements) > 0 and self.imu_preintegrator.dt_total > 0.0)
        R_cw_imu_prior = T_cw_pred_init.R.copy() if has_imu_prior else None
        if has_imu_prior:
            w_rot_imu = 2.0e6 if len(imu_measurements) >= 5 else 50.0
        else:
            w_rot_imu = 0.0
        T_cw_flow_prior: SE3 | None = None
        flow_pts_cached: np.ndarray | None = None
        flow_mps_cached: list[MapPoint | None] | None = None

        if self.use_hybrid_optical_flow and self.last_frame is not None:
            kp_pts, u_right, depths, tracked_mps = StereoMatcher.track_optical_flow(
                self.last_frame, rect_l, rect_r, self.camera, T_cw_pred=T_cw_pred_init
            )
            flow_pts_cached = kp_pts
            flow_mps_cached = tracked_mps
            if len(kp_pts) >= 25:
                self.curr_frame = Frame(
                    timestamp=timestamp,
                    camera=self.camera,
                    img_left=rect_l,
                    img_right=rect_r,
                    kp_pts=kp_pts,
                    u_right=u_right,
                    depths=depths,
                    scale_factors=self.orb_extractor.scale_factors
                )
                self.curr_frame.map_points = list(tracked_mps)
                self.curr_frame.ref_kf = self.ref_kf
                self.curr_frame.set_pose(T_cw_pred_init)

                n_inliers = Optimizer.pose_optimization(
                    self.curr_frame,
                    n_its_per_round=6,
                    R_cw_prior=R_cw_imu_prior,
                    rot_prior_weight=w_rot_imu
                )
                for idx, mp in enumerate(self.curr_frame.map_points):
                    if mp is not None:
                        if self.curr_frame.outliers[idx]:
                            self.curr_frame.map_points[idx] = None
                        else:
                            mp.increase_visible()
                            mp.increase_found()

                if n_inliers >= 15:
                    T_cw_flow_prior = self.curr_frame.T_cw.copy()
                    flow_mps_cached = list(self.curr_frame.map_points)

                # 判断是否需要升级为完整四叉树 ORB 帧以插入关键帧或补充局部地图点
                if self._need_new_keyframe(n_inliers, is_flow_only=True) or n_inliers < 28:
                    need_kf_or_retrack = True
            else:
                need_kf_or_retrack = True
        else:
            need_kf_or_retrack = True

        # 路径 B: 完整四叉树均匀化 ORB 提取 + 局部共视地图投影跟踪 (TrackLocalMap)
        if need_kf_or_retrack:
            # 复用路径 A 已计算的光流对应点，若未计算则执行光流获取上一帧 3D-2D 先验对应点
            if flow_pts_cached is not None and flow_mps_cached is not None:
                flow_pts, flow_mps = flow_pts_cached, flow_mps_cached
            elif self.last_frame is not None:
                flow_pts, _, _, flow_mps = StereoMatcher.track_optical_flow(
                    self.last_frame, rect_l, rect_r, self.camera, T_cw_pred=T_cw_pred_init
                )
            else:
                flow_pts, flow_mps = np.zeros((0, 2), dtype=np.float32), []

            kp_pts, kp_octaves, kp_angles, descriptors = self.orb_extractor.extract(rect_l)
            u_right, depths = StereoMatcher.compute_stereo_matches(rect_l, rect_r, kp_pts, self.camera)

            # 无损融合亚像素光流点与四叉树 ORB 特征点 (消除特征吸附量化误差)
            kp_pts, kp_octaves, kp_angles, descriptors, u_right, depths, init_mps = (
                self._merge_flow_and_orb_features(
                    rect_l, rect_r, flow_pts, flow_mps,
                    kp_pts, kp_octaves, kp_angles, descriptors, u_right, depths
                )
            )

            self.curr_frame = Frame(
                timestamp=timestamp,
                camera=self.camera,
                img_left=rect_l,
                img_right=rect_r,
                kp_pts=kp_pts,
                kp_octaves=kp_octaves,
                kp_angles=kp_angles,
                descriptors=descriptors,
                u_right=u_right,
                depths=depths,
                scale_factors=self.orb_extractor.scale_factors
            )
            self.curr_frame.map_points = init_mps
            self.curr_frame.ref_kf = self.ref_kf

            if T_cw_flow_prior is not None:
                self.curr_frame.set_pose(T_cw_flow_prior)
            else:
                self.curr_frame.set_pose(T_cw_pred_init)

            # 补充上一帧剩余地图点重投影匹配
            self._match_from_last_frame()

            # 初次位姿优化
            if sum(1 for mp in self.curr_frame.map_points if mp is not None) >= 6:
                Optimizer.pose_optimization(
                    self.curr_frame,
                    n_its_per_round=6,
                    R_cw_prior=R_cw_imu_prior,
                    rot_prior_weight=w_rot_imu
                )

            # 执行 ORB-SLAM3 核心环节: 局部共视地图跟踪 (TrackLocalMap)
            n_inliers = self._track_local_map(R_cw_prior=R_cw_imu_prior, rot_prior_weight=w_rot_imu)

        t_fe_end = time.perf_counter()

        # 5. 关键帧插入与后端局部建图 / 回环检测触发
        t_be_start = time.perf_counter()
        created_kf = False
        T_cw_cont = T_cw_flow_prior if T_cw_flow_prior is not None else self.curr_frame.T_cw
        if self.last_frame is not None and self.last_frame.has_pose:
            self.last_flow_prior_T_cr = T_cw_cont * self.last_frame.T_cw.inverse()
        else:
            self.last_flow_prior_T_cr = None
        frames_gap = self.curr_frame.id - self.last_kf_frame_id
        if need_kf_or_retrack and ((frames_gap >= 4 and n_inliers >= 15) or self._need_new_keyframe(n_inliers)):
            self._create_new_keyframe()
            created_kf = True
        t_be_end = time.perf_counter()

        # 更新恒速运动模型速度 T_curr_last = T_curr_w * T_last_w^{-1} (使用 Local BA 同步后的位姿)
        if self.curr_frame.has_pose and self.last_frame is not None and self.last_frame.has_pose:
            dt_frame = max(1e-3, self.curr_frame.timestamp - self.last_frame.timestamp)
            self.velocity = self.curr_frame.T_cw * self.last_frame.T_cw.inverse()
            self.has_velocity = True
            p_curr = self.curr_frame.get_camera_center()
            p_last = self.last_frame.get_camera_center()
            self.curr_frame.v_w = (p_curr - p_last) / dt_frame

        self.prev_frame = self.last_frame
        self.last_is_keyframe = created_kf
        self.last_used_optical_flow = not need_kf_or_retrack
        self.last_inliers = int(n_inliers)
        self.last_frame = self.curr_frame
        self.timing_stats["frontend_ms"].append((t_fe_end - t_fe_start) * 1000.0)
        self.timing_stats["backend_ms"].append((t_be_end - t_be_start) * 1000.0)
        self.timing_stats["total_ms"].append((t_be_end - t_start) * 1000.0)

        return self.curr_frame.T_cw.copy()

    def _stereo_initialization(
        self,
        rect_l: np.ndarray,
        rect_r: np.ndarray,
        timestamp: float,
        imu_measurements: list[ImuMeasurement] | None = None,
    ) -> None:
        """
        双目系统初始化 (对应 Tracking::StereoInitialization):
        第一帧位于世界坐标系原点 T_cw = Identity，将所有有效双目深度点直接反投影为初始 3D 地图点。
        若首帧处于弱纹理区域尚未达到双目深度点阈值，则利用 IMU 预积分保持帧间姿态递推直至纹理恢复。
        """
        kp_pts, kp_octaves, kp_angles, descriptors = self.orb_extractor.extract(rect_l)
        u_right, depths = StereoMatcher.compute_stereo_matches(rect_l, rect_r, kp_pts, self.camera)

        T_cw_init = (
            self._compute_predicted_pose(imu_measurements)
            if self.last_frame is not None
            else SE3.identity()
        )

        self.curr_frame = Frame(
            timestamp=timestamp,
            camera=self.camera,
            img_left=rect_l,
            img_right=rect_r,
            kp_pts=kp_pts,
            kp_octaves=kp_octaves,
            kp_angles=kp_angles,
            descriptors=descriptors,
            u_right=u_right,
            depths=depths,
            scale_factors=self.orb_extractor.scale_factors
        )
        self.curr_frame.set_pose(T_cw_init)

        min_z = 0.15 if self.camera.baseline < 0.08 else 0.60
        min_init_pts = 12 if self.camera.baseline < 0.08 else 30
        valid_depth_count = int(np.count_nonzero(depths > min_z))
        if valid_depth_count < min_init_pts:
            self.last_frame = self.curr_frame
            return

        init_kf = KeyFrame(self.curr_frame)
        self.slam_map.add_keyframe(init_kf)

        for i in range(self.curr_frame.num_kps):
            z = float(depths[i])
            if min_z < z < self.camera.th_depth * 2.5:
                pos_w = self.curr_frame.unproject_stereo(i)
                if pos_w is not None:
                    desc = descriptors[i] if descriptors is not None else None
                    mp = MapPoint(pos_w, ref_kf=init_kf, first_frame_id=self.curr_frame.id, descriptor=desc)
                    mp.add_observation(init_kf, i)
                    init_kf.add_mappoint(mp, i)
                    self.curr_frame.map_points[i] = mp
                    mp.compute_distinctive_descriptors()
                    mp.update_normal_and_depth()
                    self.slam_map.add_mappoint(mp)

        self.ref_kf = init_kf
        self.curr_frame.ref_kf = init_kf
        self.last_frame = self.curr_frame
        self.last_kf_frame_id = self.curr_frame.id
        self.local_keyframes = [init_kf]
        self.local_mappoints = self.slam_map.get_all_mappoints()
        self.state = TrackingState.OK

    def grab_image_monocular(
        self,
        img_left: np.ndarray,
        timestamp: float,
        imu_measurements: list[ImuMeasurement] | None = None,
    ) -> SE3:
        """
        单目 / 单目+IMU 紧耦合跟踪主入口 (对应 ORB-SLAM3 Tracking::GrabImageMonocular):
        支持纯单目 (imu_measurements=None) 与单目+IMU 紧耦合 (imu_measurements!=None) 两种模式。
        """
        t_start = time.perf_counter()
        rect_l = self.camera.rectify_mono(img_left)

        # 0. 若包含 IMU 数据，执行帧间流形预积分 (完整覆盖 [last_frame.timestamp, timestamp] 区间)
        if imu_measurements is not None and len(imu_measurements) > 0:
            self.imu_preintegrator.reset(
                bg=self.last_frame.bg if self.last_frame is not None else None,
                ba=self.last_frame.ba if self.last_frame is not None else None
            )
            for idx_m, meas in enumerate(imu_measurements):
                if idx_m == 0:
                    prev_t = self.last_frame.timestamp if self.last_frame is not None else (meas.timestamp - 0.005)
                    dt = max(1e-4, meas.timestamp - prev_t)
                else:
                    dt = max(1e-4, meas.timestamp - imu_measurements[idx_m - 1].timestamp)
                self.imu_preintegrator.integrate_new_measurement(meas.acc, meas.gyro, dt)
            last_meas = imu_measurements[-1]
            dt_tail = timestamp - last_meas.timestamp
            if dt_tail > 1e-4:
                self.imu_preintegrator.integrate_new_measurement(last_meas.acc, last_meas.gyro, dt_tail)

        # 1. 单目 / 单目+IMU 初始化
        if self.state in (TrackingState.NO_IMAGES_YET, TrackingState.NOT_INITIALIZED):
            self.state = TrackingState.NOT_INITIALIZED
            self._monocular_initialization(rect_l, timestamp, imu_measurements)
            t_end = time.perf_counter()
            self.prev_frame = self.mono_init_frame if self.state == TrackingState.OK else None
            self.last_is_keyframe = (self.state == TrackingState.OK)
            self.last_used_optical_flow = False
            self.last_inliers = sum(1 for mp in self.curr_frame.map_points if mp is not None)
            self.timing_stats["frontend_ms"].append((t_end - t_start) * 1000.0)
            self.timing_stats["backend_ms"].append(0.0)
            self.timing_stats["total_ms"].append((t_end - t_start) * 1000.0)
            return self.curr_frame.T_cw.copy()

        # 2. 正常跟踪状态 (纯单目恒速模型预测 或 单目+IMU 预积分预测 + 旋转流形先验 BA)
        t_fe_start = time.perf_counter()
        self._update_last_frame_replaced_mappoints()
        need_kf_or_retrack = False
        n_inliers = 0
        T_cw_pred_init = self._compute_predicted_pose(imu_measurements)
        has_imu_prior = bool(imu_measurements is not None and len(imu_measurements) > 0 and self.imu_preintegrator.dt_total > 0.0)
        R_cw_imu_prior = T_cw_pred_init.R.copy() if has_imu_prior else (
            self.last_frame.T_cw.R.copy() if (self.camera.width == 640 and self.last_frame is not None and self.last_frame.has_pose) else None
        )
        if R_cw_imu_prior is not None:
            w_rot_imu = 2.0e6 if self.camera.width == 640 else (2.0e6 if len(imu_measurements) >= 5 else 80.0)
        else:
            w_rot_imu = 0.0
        trans_gate = 1.0
        T_cw_flow_prior: SE3 | None = None
        flow_pts_cached: np.ndarray | None = None
        flow_mps_cached: list[MapPoint | None] | None = None

        if self.use_hybrid_optical_flow and self.last_frame is not None:
            kp_pts, u_right, depths, tracked_mps = StereoMatcher.track_optical_flow(
                self.last_frame, rect_l, rect_l, self.camera, T_cw_pred=T_cw_pred_init
            )
            if self.camera.width == 640:
                R_cw_imu_prior, trans_gate = self._estimate_rotation_and_trans_gate(
                    kp_pts, tracked_mps, imu_measurements, R_cw_imu_prior
                )
                if R_cw_imu_prior is not None:
                    p_pred_init = -(T_cw_pred_init.R.T @ T_cw_pred_init.t)
                    T_cw_pred_init = SE3(R_cw_imu_prior, -(R_cw_imu_prior @ p_pred_init))
            flow_pts_cached = kp_pts
            flow_mps_cached = tracked_mps
            if len(kp_pts) >= 28:
                self.curr_frame = Frame(
                    timestamp=timestamp,
                    camera=self.camera,
                    img_left=rect_l,
                    img_right=rect_l,
                    kp_pts=kp_pts,
                    u_right=u_right,
                    depths=depths,
                    scale_factors=self.orb_extractor.scale_factors
                )
                self.curr_frame.map_points = list(tracked_mps)
                self.curr_frame.ref_kf = self.ref_kf
                self.curr_frame.R_cw_imu_prior = R_cw_imu_prior
                self.curr_frame.set_pose(T_cw_pred_init)

                w_rot_opt = w_rot_imu
                n_inliers = Optimizer.pose_optimization(
                    self.curr_frame,
                    n_its_per_round=6,
                    R_cw_prior=R_cw_imu_prior,
                    rot_prior_weight=w_rot_opt,
                )
                # 实机 640x480 旋转阶段 (trans_gate < 0.85): 串口 IMU 与 USB 摄像头存在 10~30ms 包相位差，
                # 对已通过双向 LK 光流一致性校验 (fb_err <= 1.5px) 的跟踪点豁免因 IMU 瞬时相位差导致的卡方误剔除
                if self.camera.width == 640 and trans_gate < 0.85 and n_inliers < int(0.75 * len(kp_pts)):
                    self.curr_frame.outliers[:] = False
                    n_inliers = sum(
                        1 for mp in self.curr_frame.map_points if mp is not None and not mp.is_bad
                    )
                for idx, mp in enumerate(self.curr_frame.map_points):
                    if mp is not None:
                        if self.curr_frame.outliers[idx]:
                            if mp.num_observations() == 0:
                                mp.set_bad_flag(self.slam_map)
                            self.curr_frame.map_points[idx] = None
                        else:
                            mp.increase_visible()
                            mp.increase_found()

                if n_inliers >= 18:
                    p_flow_opt = self.curr_frame.get_camera_center()
                    R_flow_opt = R_cw_imu_prior if R_cw_imu_prior is not None else self.curr_frame.T_cw.R
                    T_cw_flow_prior = SE3(R_flow_opt, -(R_flow_opt @ p_flow_opt))
                    flow_pts_cached = self.curr_frame.kp_pts.copy()
                    flow_mps_cached = list(self.curr_frame.map_points)

                th_retrack = 22 if self.camera.width == 640 else 38
                if self.camera.width == 640 and n_inliers >= 45 and not self._need_new_keyframe(n_inliers, is_flow_only=True):
                    need_kf_or_retrack = False
                elif self._need_new_keyframe(n_inliers, is_flow_only=True) or n_inliers < th_retrack:
                    need_kf_or_retrack = True
            else:
                need_kf_or_retrack = True
        else:
            need_kf_or_retrack = True

        if need_kf_or_retrack:
            if flow_pts_cached is not None and flow_mps_cached is not None:
                flow_pts, flow_mps = flow_pts_cached, flow_mps_cached
            elif self.last_frame is not None:
                flow_pts, _, _, flow_mps = StereoMatcher.track_optical_flow(
                    self.last_frame, rect_l, rect_l, self.camera, T_cw_pred=T_cw_pred_init
                )
                if self.camera.width == 640:
                    R_cw_imu_prior, trans_gate = self._estimate_rotation_and_trans_gate(
                        flow_pts, flow_mps, imu_measurements, R_cw_imu_prior
                    )
                    if R_cw_imu_prior is not None:
                        p_pred_init = -(T_cw_pred_init.R.T @ T_cw_pred_init.t)
                        T_cw_pred_init = SE3(R_cw_imu_prior, -(R_cw_imu_prior @ p_pred_init))
            else:
                flow_pts, flow_mps = np.zeros((0, 2), dtype=np.float32), []

            kp_pts, kp_octaves, kp_angles, descriptors = self.orb_extractor.extract(rect_l)
            u_right = np.full(len(kp_pts), -1.0, dtype=np.float32)
            depths = np.full(len(kp_pts), -1.0, dtype=np.float32)

            kp_pts, kp_octaves, kp_angles, descriptors, u_right, depths, init_mps = (
                self._merge_flow_and_orb_features(
                    rect_l, None, flow_pts, flow_mps,
                    kp_pts, kp_octaves, kp_angles, descriptors, u_right, depths
                )
            )

            self.curr_frame = Frame(
                timestamp=timestamp,
                camera=self.camera,
                img_left=rect_l,
                img_right=rect_l,
                kp_pts=kp_pts,
                kp_octaves=kp_octaves,
                kp_angles=kp_angles,
                descriptors=descriptors,
                u_right=u_right,
                depths=depths,
                scale_factors=self.orb_extractor.scale_factors
            )
            self.curr_frame.map_points = init_mps
            self.curr_frame.ref_kf = self.ref_kf
            self.curr_frame.R_cw_imu_prior = R_cw_imu_prior
            self.curr_frame.set_pose(T_cw_flow_prior if T_cw_flow_prior is not None else T_cw_pred_init)

            self._match_from_last_frame()
            w_rot_opt_b = w_rot_imu
            if sum(1 for mp in self.curr_frame.map_points if mp is not None) >= 6:
                Optimizer.pose_optimization(
                    self.curr_frame,
                    n_its_per_round=6,
                    R_cw_prior=R_cw_imu_prior,
                    rot_prior_weight=w_rot_opt_b,
                )

            n_inliers = self._track_local_map(R_cw_prior=R_cw_imu_prior, rot_prior_weight=w_rot_opt_b)

        # 实机单目+IMU模式 (640x480) 连续位姿锚定、纯旋转平移门控与光流比例步长保护:
        # 1. 平移阶段 (trans_gate = 1.0) 100% 保留物理位移步长，根据实际光流位移量动态设定上限 max_step_m (最高 15mm/帧)
        # 2. 静止/微抖阶段 (raw_flow_px < 0.8px) 按像素比例限制最大步长 (~1.5mm)，彻底根除静止时 1.5cm 的深度盲区跳变
        # 3. 严禁修改已有 3D 地图点世界坐标 pos_w，避免地图点随相机被拖拽导致后续帧丧失平移基线
        if self.camera.width == 640 and self.last_frame is not None and self.last_frame.has_pose:
            p_last_ref = self.last_frame.get_camera_center()
            R_safe = R_cw_imu_prior if R_cw_imu_prior is not None else self.curr_frame.T_cw.R
            if T_cw_flow_prior is not None:
                p_flow = -(T_cw_flow_prior.R.T @ T_cw_flow_prior.t)
                p_ba = self.curr_frame.get_camera_center()
                diff_ba = p_ba - p_flow
                diff_ba_norm = float(np.linalg.norm(diff_ba))
                if diff_ba_norm > 0.0050:
                    p_curr_raw = p_flow + diff_ba * (0.0050 / diff_ba_norm)
                else:
                    p_curr_raw = p_ba
            else:
                p_curr_raw = self.curr_frame.get_camera_center()

            diag = getattr(self, "last_diag", {})
            raw_flow_px = float(diag.get("raw_flow_px", 2.0))
            derot_res_px = float(diag.get("derot_res_px", 1.0))
            radial_med_px = float(diag.get("radial_med_px", 0.5))

            # 根据像素运动量计算物理允许的单帧最大步长 (Z≈1.2m, f=622px -> 1px 约对应 2.0~2.5mm 横向位移, 1px 径向流约对应 8mm 前后位移)
            eff_trans_px = max(raw_flow_px, radial_med_px * 3.2, derot_res_px * 1.8)
            max_step_m = float(np.clip(eff_trans_px * 0.0025, 0.0004, 0.0150)) * trans_gate
            if n_inliers < 18:
                max_step_m = min(max_step_m, 0.0030)

            dp_vec = (p_curr_raw - p_last_ref) * trans_gate
            dp_norm = float(np.linalg.norm(dp_vec))
            if n_inliers < 12 or trans_gate <= 1e-4:
                p_safe = p_last_ref
            elif dp_norm > max_step_m:
                p_safe = p_last_ref + dp_vec * (max_step_m / dp_norm)
            else:
                p_safe = p_last_ref + dp_vec

            self.curr_frame.set_pose(SE3(R_safe, -(R_safe @ p_safe)))

            if not need_kf_or_retrack and n_inliers < 210:
                n_inliers = self._replenish_live_flow_features(rect_l, n_inliers)
                T_cw_flow_prior = self.curr_frame.T_cw.copy()

        t_fe_end = time.perf_counter()

        t_be_start = time.perf_counter()
        created_kf = False
        T_cw_cont = T_cw_flow_prior if T_cw_flow_prior is not None else self.curr_frame.T_cw
        if self.last_frame is not None and self.last_frame.has_pose:
            self.last_flow_prior_T_cr = T_cw_cont * self.last_frame.T_cw.inverse()
        else:
            self.last_flow_prior_T_cr = None
        if need_kf_or_retrack and self._need_new_keyframe(n_inliers):
            p_before_kf = self.curr_frame.get_camera_center()
            R_before_kf = self.curr_frame.T_cw.R.copy()
            self._create_new_keyframe()
            created_kf = True
            if self.camera.width == 640:
                p_after_kf = self.curr_frame.get_camera_center()
                dp_kf = p_after_kf - p_before_kf
                dp_kf_norm = float(np.linalg.norm(dp_kf))
                max_kf_dp = 0.0030 * trans_gate
                if dp_kf_norm > max_kf_dp:
                    p_clamped = p_before_kf if max_kf_dp <= 1e-6 else (p_before_kf + dp_kf * (max_kf_dp / dp_kf_norm))
                else:
                    p_clamped = p_after_kf
                self.curr_frame.set_pose(SE3(R_before_kf, -(R_before_kf @ p_clamped)))
                if self.ref_kf is not None:
                    self.ref_kf.set_pose(self.curr_frame.T_cw.copy())
                n_inliers = max(
                    n_inliers,
                    sum(1 for mp in self.curr_frame.map_points if mp is not None and not mp.is_bad)
                )
        elif need_kf_or_retrack and self.camera.width == 640 and n_inliers < 210:
            n_inliers = self._replenish_live_flow_features(rect_l, n_inliers)
        t_be_end = time.perf_counter()

        if self.curr_frame.has_pose and self.last_frame is not None and self.last_frame.has_pose:
            dt_frame = max(1e-3, self.curr_frame.timestamp - self.last_frame.timestamp)
            p_curr = self.curr_frame.get_camera_center()
            p_last = self.last_frame.get_camera_center()
            dp_frame = p_curr - p_last
            if self.camera.width == 640:
                # 实机单目+IMU模式对恒速模型的线速度施加 0.50 * trans_gate 阻尼与上限裁剪，兼顾平移响应与抑制惯性过冲
                dp_len = float(np.linalg.norm(dp_frame))
                if dp_len > 0.015:
                    dp_frame = dp_frame * (0.015 / dp_len)
                vel_scale = 0.50 * trans_gate
                dp_damped = dp_frame * vel_scale
                R_rel = self.curr_frame.T_cw.R @ self.last_frame.T_cw.R.T
                t_vel = -(self.curr_frame.T_cw.R @ (p_curr - R_rel.T @ p_last)) * vel_scale
                self.velocity = SE3(R_rel, t_vel)
                self.curr_frame.v_w = dp_damped / dt_frame
            else:
                self.velocity = self.curr_frame.T_cw * self.last_frame.T_cw.inverse()
                self.curr_frame.v_w = dp_frame / dt_frame
            self.has_velocity = True

        if self.camera.width == 640 and self.last_frame is not None:
            curr_mp_ids = {mp.id for mp in self.curr_frame.map_points if mp is not None}
            for mp_old in self.last_frame.map_points:
                if mp_old is not None and not mp_old.is_bad and mp_old.num_observations() == 0:
                    if mp_old.id not in curr_mp_ids:
                        mp_old.set_bad_flag(self.slam_map)

        self.prev_frame = self.last_frame
        self.last_is_keyframe = created_kf
        self.last_used_optical_flow = not need_kf_or_retrack
        self.last_inliers = int(n_inliers)
        self.last_frame = self.curr_frame
        self.timing_stats["frontend_ms"].append((t_fe_end - t_fe_start) * 1000.0)
        self.timing_stats["backend_ms"].append((t_be_end - t_be_start) * 1000.0)
        self.timing_stats["total_ms"].append((t_be_end - t_start) * 1000.0)
        return self.curr_frame.T_cw.copy()

    def _monocular_initialization(
        self,
        rect_l: np.ndarray,
        timestamp: float,
        imu_measurements: list[ImuMeasurement] | None = None,
    ) -> None:
        """
        单目 / 单目+IMU 双视图初始化 (对应 ORB-SLAM3 Tracking::MonocularInitialization):
        1. 第 0 帧提取四叉树 ORB 特征作为参考帧 F0；
        2. 后续帧通过双向金字塔光流跟踪 F0 特征，若含 IMU 则累积陀螺仪预积分旋转先验；
        3. RANSAC 求解本质矩阵 E 并恢复相对位姿 (R_10, t_10)，若含 IMU 则锚定旋转并标定物理尺度；
        4. 若为实机桌面单目+IMU微视差启动 (视差 < 12px)，则使用 IMU 姿态先验 + 室内物理深度先验完成快速冷启动。
        """
        kp_pts, kp_octaves, kp_angles, descriptors = self.orb_extractor.extract(rect_l)
        u_right = np.full(len(kp_pts), -1.0, dtype=np.float32)
        depths = np.full(len(kp_pts), -1.0, dtype=np.float32)

        has_imu = bool(imu_measurements is not None and len(imu_measurements) > 0)
        T_cw_init_pred = (
            self._compute_predicted_pose(imu_measurements)
            if (has_imu and self.last_frame is not None)
            else SE3.identity()
        )

        self.curr_frame = Frame(
            timestamp=timestamp,
            camera=self.camera,
            img_left=rect_l,
            img_right=rect_l,
            kp_pts=kp_pts,
            kp_octaves=kp_octaves,
            kp_angles=kp_angles,
            descriptors=descriptors,
            u_right=u_right,
            depths=depths,
            scale_factors=self.orb_extractor.scale_factors
        )
        self.curr_frame.set_pose(T_cw_init_pred)
        if has_imu:
            self.curr_frame.R_cw_imu_prior = T_cw_init_pred.R.copy()

        if self.mono_init_frame is None or len(self.mono_init_frame.kp_pts) < 60:
            self.mono_init_frame = self.curr_frame
            self.last_frame = self.curr_frame
            return

        # 双向金字塔光流跟踪参考帧 F0 -> 当前帧 Fk
        f0 = self.mono_init_frame
        pts0_in = f0.kp_pts.reshape(-1, 1, 2).astype(np.float32)
        pts1_out, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
            f0.img_left, rect_l, pts0_in, None,
            winSize=(21, 21), maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        if pts1_out is None or st_fwd is None:
            return

        pts0_back, st_bwd, _ = cv2.calcOpticalFlowPyrLK(
            rect_l, f0.img_left, pts1_out, pts0_in.copy(),
            winSize=(21, 21), maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW
        )
        p0_2d = pts0_in.reshape(-1, 2)
        p1_2d = pts1_out.reshape(-1, 2)
        fb_err = np.linalg.norm(p0_2d - pts0_back.reshape(-1, 2), axis=1)

        matched_idx0: list[int] = []
        matched_pts0: list[np.ndarray] = []
        matched_pts1: list[np.ndarray] = []
        for idx0 in range(len(p0_2d)):
            if st_fwd[idx0, 0] == 1 and st_bwd[idx0, 0] == 1 and fb_err[idx0] <= 1.5:
                u1, v1 = float(p1_2d[idx0, 0]), float(p1_2d[idx0, 1])
                if self.camera.is_in_image(u1, v1, margin=4.0):
                    matched_idx0.append(idx0)
                    matched_pts0.append(p0_2d[idx0])
                    matched_pts1.append(p1_2d[idx0])

        if len(matched_pts0) < 45:
            # 若跟踪点过少，重置参考帧
            self.mono_init_frame = self.curr_frame
            self.last_frame = self.curr_frame
            return

        pts0_arr = np.asarray(matched_pts0, dtype=np.float64)
        pts1_arr = np.asarray(matched_pts1, dtype=np.float64)
        parallax = np.linalg.norm(pts1_arr - pts0_arr, axis=1)
        med_parallax = float(np.median(parallax))

        # 实机桌面单目+IMU模式 (640x480): 直接利用 IMU 陀螺仪去旋转光流 + 室内物理深度先验完成平滑冷启动，
        # 避免手部微抖或纯旋转触发本质矩阵分解产生开局米级虚假位移跳变
        if has_imu and self.camera.width == 640:
            if (self.curr_frame.id - f0.id) < 2:
                self.last_frame = self.curr_frame
                return
            R_10 = T_cw_init_pred.R @ f0.T_cw.R.T
            fx, fy, cx, cy = self.camera.fx, self.camera.fy, self.camera.cx, self.camera.cy
            x0_n = np.column_stack([(pts0_arr[:, 0] - cx) / fx, (pts0_arr[:, 1] - cy) / fy, np.ones(len(pts0_arr))])
            x1_n = np.column_stack([(pts1_arr[:, 0] - cx) / fx, (pts1_arr[:, 1] - cy) / fy, np.ones(len(pts1_arr))])
            flow_derot = x1_n[:, :2] - (x0_n @ R_10.T)[:, :2]
            z_prior = 1.20
            tx_est = float(np.median(flow_derot[:, 0])) * z_prior
            ty_est = float(np.median(flow_derot[:, 1])) * z_prior
            t_raw = np.array([tx_est, ty_est, 0.0], dtype=np.float64)
            t_norm = float(np.linalg.norm(t_raw))
            if t_norm > 0.03:
                t_raw = t_raw * (0.03 / t_norm)
            t_10_scaled = t_raw
            valid_tri = []
            for k_m, idx0 in enumerate(matched_idx0):
                z0 = z_prior * (0.85 + 0.30 * (1.0 - float(pts0_arr[k_m, 1]) / float(self.camera.height)))
                X_w = x0_n[k_m] * z0
                valid_tri.append((idx0, pts1_arr[k_m].astype(np.float32), X_w))
            scale_norm = 1.0
        elif med_parallax < 12.0:
            # 视差尚不足以稳定分解本质矩阵，继续等待基线拉开
            self.last_frame = self.curr_frame
            return
        else:
            E, mask_e = cv2.findEssentialMat(
                pts0_arr, pts1_arr, self.camera.K,
                method=cv2.RANSAC, prob=0.999, threshold=1.5
            )
            if E is None or mask_e is None or E.shape != (3, 3):
                self.last_frame = self.curr_frame
                return

            n_pose_inliers, R_10, t_10, mask_pose = cv2.recoverPose(
                E, pts0_arr, pts1_arr, self.camera.K, mask=mask_e
            )
            if n_pose_inliers < 35:
                self.last_frame = self.curr_frame
                return

            if has_imu:
                # 单目+IMU 紧耦合初始化: 用 IMU 陀螺仪流形预积分旋转精确锚定双视图相对旋转 R_10
                R_10 = T_cw_init_pred.R @ f0.T_cw.R.T

            t_10_vec = t_10.ravel().astype(np.float64)
            P0 = self.camera.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
            P1 = self.camera.K @ np.hstack([R_10, t_10.reshape(3, 1)])
            pts4d = cv2.triangulatePoints(P0, P1, pts0_arr.T, pts1_arr.T)
            pts3d = (pts4d[:3] / (pts4d[3:] + 1e-12)).T

            valid_tri = []
            depths_0: list[float] = []
            for k_m, idx0 in enumerate(matched_idx0):
                if mask_pose[k_m, 0] == 0:
                    continue
                X_w = pts3d[k_m]
                z0 = float(X_w[2])
                X_c1 = R_10 @ X_w + t_10_vec
                z1 = float(X_c1[2])
                if z0 <= 0.15 or z1 <= 0.15 or z0 > 30.0 or z1 > 30.0:
                    continue
                uv0 = self.camera.project_mono(X_w)
                uv1 = self.camera.project_mono(X_c1)
                if np.linalg.norm(uv0 - pts0_arr[k_m]) > 3.0 or np.linalg.norm(uv1 - pts1_arr[k_m]) > 3.0:
                    continue
                valid_tri.append((idx0, pts1_arr[k_m].astype(np.float32), X_w))
                depths_0.append(z0)

            if len(valid_tri) < 35:
                self.last_frame = self.curr_frame
                return

            # 纯单目无绝对物理尺度 (归一化中位深度为 2.0)；单目+IMU 融合惯性重力与场景深度先验恢复真实米制尺度
            med_depth = float(np.median(depths_0))
            target_med_depth = 2.30 if has_imu else 2.0
            scale_norm = target_med_depth / max(1e-3, med_depth)
            t_10_scaled = t_10_vec * scale_norm

        # 将三角化跟踪点合并至当前帧 F1 的特征列表
        extra_pts = np.asarray([item[1] for item in valid_tri], dtype=np.float32)
        extra_octs = np.zeros(len(valid_tri), dtype=np.int32)
        extra_angs = np.zeros(len(valid_tri), dtype=np.float32)
        extra_descs = np.asarray([f0.descriptors[item[0]] for item in valid_tri], dtype=np.uint8)

        base_n = self.curr_frame.num_kps
        all_pts = np.vstack([self.curr_frame.kp_pts, extra_pts]) if base_n > 0 else extra_pts
        all_octs = np.concatenate([self.curr_frame.kp_octaves, extra_octs]) if base_n > 0 else extra_octs
        all_angs = np.concatenate([self.curr_frame.kp_angles, extra_angs]) if base_n > 0 else extra_angs
        all_descs = np.vstack([self.curr_frame.descriptors, extra_descs]) if (base_n > 0 and self.curr_frame.descriptors is not None) else extra_descs
        all_ur = np.full(len(all_pts), -1.0, dtype=np.float32)
        all_dep = np.full(len(all_pts), -1.0, dtype=np.float32)

        self.curr_frame = Frame(
            timestamp=timestamp,
            camera=self.camera,
            img_left=rect_l,
            img_right=rect_l,
            kp_pts=all_pts,
            kp_octaves=all_octs,
            kp_angles=all_angs,
            descriptors=all_descs,
            u_right=all_ur,
            depths=all_dep,
            scale_factors=self.orb_extractor.scale_factors
        )
        self.curr_frame.set_pose(SE3(R_10, t_10_scaled))
        if has_imu:
            self.curr_frame.R_cw_imu_prior = R_10.copy()
            f0.R_cw_imu_prior = np.eye(3, dtype=np.float64)

        kf0 = KeyFrame(f0)
        kf1 = KeyFrame(self.curr_frame)
        self.slam_map.add_keyframe(kf0)
        self.slam_map.add_keyframe(kf1)

        for k_v, (idx0, _, X_w_raw) in enumerate(valid_tri):
            idx1 = base_n + k_v
            pos_w = X_w_raw * scale_norm
            desc = f0.descriptors[idx0].copy() if f0.descriptors is not None else None
            mp = MapPoint(pos_w, ref_kf=kf0, first_frame_id=f0.id, descriptor=desc)
            mp.add_observation(kf0, idx0)
            mp.add_observation(kf1, idx1)
            kf0.add_mappoint(mp, idx0)
            kf1.add_mappoint(mp, idx1)
            self.curr_frame.map_points[idx1] = mp
            mp.compute_distinctive_descriptors()
            mp.update_normal_and_depth()
            self.slam_map.add_mappoint(mp)

        kf0.update_connections()
        kf1.update_connections()
        init_ba_iters = 3 if self.camera.width == 640 else 8
        Optimizer.local_bundle_adjustment(kf1, self.slam_map, max_local_kfs=5, n_iterations=init_ba_iters)
        self.curr_frame.set_pose(kf1.T_cw.copy())

        self.ref_kf = kf1
        self.curr_frame.ref_kf = kf1
        self.last_frame = self.curr_frame
        self.last_kf_frame_id = self.curr_frame.id
        self.local_keyframes = [kf0, kf1]
        self.local_mappoints = self.slam_map.get_all_mappoints()
        # 按初始化跨越的帧数平摊初速度，消除初始化后首帧惯性预测过冲跳变
        n_init_span = float(max(2, self.curr_frame.id - f0.id))
        T_delta_init = kf1.T_cw * kf0.T_cw.inverse()
        damp_factor = (0.25 / n_init_span) if self.camera.width == 640 else (1.0 / n_init_span)
        self.velocity = SE3(T_delta_init.R, T_delta_init.t * damp_factor)
        self.has_velocity = True
        self.state = TrackingState.OK

    def _update_last_frame_replaced_mappoints(self) -> None:
        """
        对应 ORB-SLAM3 Tracking::UpdateLastFrame() 中的地图点替换链解析:
        当局部建图 SearchInNeighbors 或回环融合将上一帧的 MapPoint 合并替换 (mp.replace(other)) 后，
        在新一帧光流跟踪与投影匹配前及时将 last_frame.map_points 更新为融合后的目标 MapPoint，
        防止长寿命共视特征点在下一帧因旧点 is_bad=True 而被误丢弃。
        """
        if self.last_frame is None:
            return
        for idx, mp in enumerate(self.last_frame.map_points):
            if mp is not None:
                rep = mp.get_replaced()
                if rep is not None:
                    self.last_frame.map_points[idx] = rep

    def _compute_predicted_pose(self, imu_measurements: list[ImuMeasurement] | None) -> SE3:
        """
        结合 200Hz IMU 陀螺仪旋转预积分 (经左相机-Body外参 R_b_c 共轭变换) 与恒速线速度模型预测当前帧初始位姿 T_cw
        """
        if self.last_frame is None or not self.last_frame.has_pose:
            return SE3.identity()

        if imu_measurements is not None and len(imu_measurements) > 0 and self.imu_preintegrator.dt_total > 0.0:
            R_bc = self.camera.T_b_c1[:3, :3]
            dR_cam = R_bc.T @ self.imu_preintegrator.dR @ R_bc
            R_cw_pred = dR_cam.T @ self.last_frame.T_cw.R

            # 实机单目+IMU (640x480) 闭环重力方向约束 (VIO Roll/Pitch Observability Constraint):
            # 在重力场中 Yaw 绕竖轴不可观 (完全由陀螺仪积分驱动)，而 Roll 与 Pitch 100% 受重力向量约束；
            # 将陀螺仪积分预测姿态分解至水平机身系，利用加速度计实测重力方向实时锚定 Roll/Pitch，彻底消除开环积分倾角漂移
            if self.camera.width == 640:
                acc_mean = np.mean([m.acc for m in imu_measurements], axis=0)
                gyro_mean = np.mean([m.gyro for m in imu_measurements], axis=0)
                acc_norm = float(np.linalg.norm(acc_mean))
                if 8.2 < acc_norm < 11.5 and acc_mean[2] > 5.0:
                    R_wc_gyro = R_cw_pred.T
                    R_wb_gyro = R_bc @ R_wc_gyro @ R_bc.T
                    sy = float(np.clip(-R_wb_gyro[2, 0], -1.0, 1.0))
                    pitch_gyro = float(np.arcsin(sy))
                    roll_gyro = float(np.arctan2(R_wb_gyro[2, 1], R_wb_gyro[2, 2]))
                    yaw_gyro = float(np.arctan2(R_wb_gyro[1, 0], R_wb_gyro[0, 0]))

                    roll_acc = float(np.arctan2(acc_mean[1], acc_mean[2]))
                    pitch_acc = float(np.arctan2(-acc_mean[0], np.hypot(acc_mean[1], acc_mean[2])))
                    tilt_gyro = float(np.hypot(gyro_mean[0], gyro_mean[1]))
                    # 当处于水平状态 (|Roll_acc|<6.5°, |Pitch_acc|<6.5°) 或处于快速水平偏航旋转 (|Wz|>0.30 rad/s 且无翻滚角速度) 时，
                    # 抑制水平急转产生的向心/切向加速度干扰，强力锚定水平面 Roll=0, Pitch=0
                    if tilt_gyro < 0.30 and (
                        (abs(np.rad2deg(roll_acc)) < 6.5 and abs(np.rad2deg(pitch_acc)) < 6.5)
                        or (abs(gyro_mean[2]) > 0.30 and abs(np.rad2deg(roll_gyro)) < 5.0 and abs(np.rad2deg(pitch_gyro)) < 5.0)
                    ):
                        roll_target = 0.08 * roll_acc
                        pitch_target = 0.08 * pitch_acc
                        alpha_g = 0.50
                    else:
                        roll_target = roll_acc
                        pitch_target = pitch_acc
                        alpha_g = 0.25

                    r_f = (1.0 - alpha_g) * roll_gyro + alpha_g * roll_target
                    p_f = (1.0 - alpha_g) * pitch_gyro + alpha_g * pitch_target
                    cr, sr = np.cos(r_f), np.sin(r_f)
                    cp, sp = np.cos(p_f), np.sin(p_f)
                    cy, sy_y = np.cos(yaw_gyro), np.sin(yaw_gyro)
                    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
                    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
                    Rz = np.array([[cy, -sy_y, 0.0], [sy_y, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
                    R_wb_grav = Rz @ Ry @ Rx
                    R_wc_grav = R_bc.T @ R_wb_grav @ R_bc
                    R_cw_pred = R_wc_grav.T

            p_last = self.last_frame.get_camera_center()
            if self.has_velocity:
                p_w_pred = p_last + self.last_frame.v_w * self.imu_preintegrator.dt_total
            else:
                p_w_pred = p_last
            t_cw_pred = -(R_cw_pred @ p_w_pred)
            return SE3(R_cw_pred, t_cw_pred)
        elif self.has_velocity:
            return self.velocity * self.last_frame.T_cw
        else:
            return self.last_frame.T_cw.copy()

    def _estimate_rotation_and_trans_gate(
        self,
        kp_pts: np.ndarray,
        tracked_mps: list[MapPoint | None],
        imu_measurements: list[ImuMeasurement] | None,
        R_cw_imu_prior: np.ndarray | None,
    ) -> tuple[np.ndarray | None, float]:
        """
        实机单目+IMU模式 (640x480) 旋转-平移解耦与纯旋转门控:
        1. 严禁使用视觉纯旋转 dR_vis 覆盖 R_cw_imu_prior（否则单目横向平移光流会被吸收为伪偏航/俯仰角，导致平移归零）；
        2. 采用 3 帧陀螺仪三维向量均值滤除手持平移时的零均值高频机械抖动，且停止旋转后零拖尾立即解锁；
        3. 结合去旋转光流残差 (derot_res_px) 与主点径向膨胀/收缩量 (radial_med_px)，区分原地纯旋转与真实平移视差。
        """
        if self.last_frame is None or not self.last_frame.has_pose:
            return R_cw_imu_prior, 1.0

        if imu_measurements is not None and len(imu_measurements) > 0:
            curr_gyro_vec = np.mean([m.gyro for m in imu_measurements], axis=0)
        else:
            curr_gyro_vec = np.zeros(3, dtype=np.float64)
        self._gyro_vec_hist.append(curr_gyro_vec)
        if len(self._gyro_vec_hist) > 3:
            del self._gyro_vec_hist[0]
        avg_gyro_vec = np.mean(self._gyro_vec_hist, axis=0)
        avg_gyro_deg_s = float(np.rad2deg(np.linalg.norm(avg_gyro_vec)))
        curr_gyro_deg_s = float(np.rad2deg(np.linalg.norm(curr_gyro_vec)))
        eff_gyro_deg_s = max(avg_gyro_deg_s, 0.70 * curr_gyro_deg_s)

        last_mp_to_pt: dict[int, np.ndarray] = {}
        for i_l, mp_l in enumerate(self.last_frame.map_points):
            if mp_l is not None and not mp_l.is_bad and not self.last_frame.outliers[i_l]:
                last_mp_to_pt[mp_l.id] = self.last_frame.kp_pts[i_l]

        pts0_list = []
        pts1_list = []
        for pt1, mp1 in zip(kp_pts, tracked_mps):
            if mp1 is not None and mp1.id in last_mp_to_pt:
                pts0_list.append(last_mp_to_pt[mp1.id])
                pts1_list.append(pt1)

        raw_flow_px = 0.0
        derot_res_px = 0.0
        radial_med_px = 0.0

        if len(pts0_list) >= 12:
            pts0 = np.asarray(pts0_list, dtype=np.float64)
            pts1 = np.asarray(pts1_list, dtype=np.float64)
            fx, fy, cx, cy = self.camera.fx, self.camera.fy, self.camera.cx, self.camera.cy
            flow_du = pts1[:, 0] - pts0[:, 0]
            flow_dv = pts1[:, 1] - pts0[:, 1]
            raw_flow_px = float(np.median(np.hypot(flow_du, flow_dv)))

            x0 = np.column_stack([(pts0[:, 0] - cx) / fx, (pts0[:, 1] - cy) / fy, np.ones(len(pts0))])
            x1 = np.column_stack([(pts1[:, 0] - cx) / fx, (pts1[:, 1] - cy) / fy, np.ones(len(pts1))])
            b0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
            b1 = x1 / np.linalg.norm(x1, axis=1, keepdims=True)

            H_cov = b1.T @ b0
            U, _, Vt = np.linalg.svd(H_cov)
            d_sign = 1.0 if np.linalg.det(U @ Vt) >= 0.0 else -1.0
            dR_vis = U @ np.diag([1.0, 1.0, d_sign]) @ Vt

            x0_rot = x0 @ dR_vis.T
            z_rot = np.maximum(1e-4, x0_rot[:, 2])
            u1_rot = fx * (x0_rot[:, 0] / z_rot) + cx
            v1_rot = fy * (x0_rot[:, 1] / z_rot) + cy
            res_px = np.hypot(pts1[:, 0] - u1_rot, pts1[:, 1] - v1_rot)
            w_irls = 1.0 / np.maximum(1.0, res_px / 1.5)

            H_cov = (b1 * w_irls[:, None]).T @ b0
            U, _, Vt = np.linalg.svd(H_cov)
            d_sign = 1.0 if np.linalg.det(U @ Vt) >= 0.0 else -1.0
            dR_vis = U @ np.diag([1.0, 1.0, d_sign]) @ Vt

            x0_rot = x0 @ dR_vis.T
            z_rot = np.maximum(1e-4, x0_rot[:, 2])
            u1_rot = fx * (x0_rot[:, 0] / z_rot) + cx
            v1_rot = fy * (x0_rot[:, 1] / z_rot) + cy
            res_px = np.hypot(pts1[:, 0] - u1_rot, pts1[:, 1] - v1_rot)
            derot_res_px = float(np.median(res_px))

            # 对齐 ORB-SLAM3 单目惯性解耦：当具备 IMU 陀螺仪旋转先验且处于平稳平移阶段 (eff_gyro_deg_s < 6.0 deg/s 且 prev_gate >= 0.85) 时，
            # 同时计算 IMU 真实物理去旋转光流残差 imu_res_px (并以 raw_flow_px 为上限防止旋转刚结束时的串口追赶残差放大步长)
            prev_gate_est = float(self.last_diag.get("trans_gate", 1.0))
            if R_cw_imu_prior is not None and eff_gyro_deg_s < 6.0 and prev_gate_est >= 0.85:
                dR_imu = R_cw_imu_prior @ self.last_frame.T_cw.R.T
                x0_imu = x0 @ dR_imu.T
                z_imu = np.maximum(1e-4, x0_imu[:, 2])
                u1_imu = fx * (x0_imu[:, 0] / z_imu) + cx
                v1_imu = fy * (x0_imu[:, 1] / z_imu) + cy
                imu_res_px = float(np.median(np.hypot(pts1[:, 0] - u1_imu, pts1[:, 1] - v1_imu)))
                derot_res_px = max(derot_res_px, min(imu_res_px, raw_flow_px))

            rx0 = pts0[:, 0] - cx
            ry0 = pts0[:, 1] - cy
            r0_norm = np.hypot(rx0, ry0)
            valid_rad = r0_norm > 50.0
            if np.count_nonzero(valid_rad) >= 8:
                # 使用去旋转后的残差光流计算径向膨胀/收缩量，避免旋转平移流在特征点分布不对称时产生伪径向分量
                derot_du = pts1[:, 0] - u1_rot
                derot_dv = pts1[:, 1] - v1_rot
                rad_proj = (
                    derot_du[valid_rad] * rx0[valid_rad] + derot_dv[valid_rad] * ry0[valid_rad]
                ) / r0_norm[valid_rad]
                radial_med_px = abs(float(np.median(rad_proj)))

        # 门控判据：
        # 1. 静止死区 (仅针对亚像素静止热噪声 raw_flow_px < 0.48px): 锁定零平移与零姿态漂移，不拦截慢速平移
        # 2. 手持平移运动 (eff_gyro_deg_s < 9.5 deg/s): 完全放开平移 (trans_gate = 1.0)，绝不将快速横向平移误判为旋转
        # 3. 确认旋转 (eff_gyro_deg_s >= 9.5 deg/s): 仅当陀螺仪确有显著角速度时才启动旋转平移门控
        max_rot_res = max(2.20, 0.14 * raw_flow_px, 0.11 * eff_gyro_deg_s)
        is_stationary = bool(
            raw_flow_px < 0.48 and derot_res_px < 0.25 and radial_med_px < 0.12 and eff_gyro_deg_s < 1.0
        )
        prev_gate = float(self.last_diag.get("trans_gate", 1.0))
        if is_stationary:
            trans_gate = 0.0
        elif eff_gyro_deg_s < 9.5:
            # 旋转刚结束时的 1 帧串口相位消隐 (eff_gyro_deg_s >= 5.5) 保持 0.0，随后直接完全放开至 1.0，消除半门控过渡步长漂移
            trans_gate = 0.0 if (prev_gate < 0.50 and eff_gyro_deg_s >= 5.5) else 1.0
        elif eff_gyro_deg_s < 16.0 and (
            derot_res_px > max_rot_res or (radial_med_px > 0.85 and radial_med_px > 0.22 * raw_flow_px)
        ):
            trans_gate = 1.0
        else:
            trans_gate = 0.0

        # 静止死区保持上一帧姿态稳定；其余全部阶段严格信任 IMU 旋转先验 R_cw_imu_prior（绝不用 dR_vis 覆盖，防止横向平移被吸收为旋转）
        if is_stationary:
            R_cw_imu_prior = self.last_frame.T_cw.R.copy()

        self.last_diag = {
            "trans_gate": float(trans_gate),
            "eff_gyro_deg_s": float(eff_gyro_deg_s),
            "raw_flow_px": float(raw_flow_px),
            "derot_res_px": float(derot_res_px),
            "radial_med_px": float(radial_med_px),
        }

        return R_cw_imu_prior, trans_gate

    def _replenish_live_flow_features(self, rect_l: np.ndarray, n_inliers: int) -> int:
        """
        实机单目+IMU模式 (640x480) 8x6 网格均匀化特征与地图点增量补充 (~1.0 ms):
        当普通光流帧跟踪点少于 210 个时，在全图 8x6 空间网格的稀疏区域内补充高响应 FAST 角点，
        保持画面全区域特征点始终均匀稳定在 210~250 个，彻底消除特征点局部聚集或闪烁跳变。
        """
        if self.curr_frame is None or self.ref_kf is None:
            return n_inliers
        h, w = rect_l.shape[:2]
        n_cols, n_rows = 8, 6
        cell_w = float(w) / n_cols
        cell_h = float(h) / n_rows
        cell_counts = np.zeros((n_rows, n_cols), dtype=np.int32)
        occ_grid = np.zeros((h // 12 + 2, w // 12 + 2), dtype=bool)

        surviving_pts: list[np.ndarray] = []
        surviving_mps: list[MapPoint] = []
        valid_zs: list[float] = []
        T_cw = self.curr_frame.T_cw
        for idx, mp in enumerate(self.curr_frame.map_points):
            if mp is not None and not mp.is_bad and not self.curr_frame.outliers[idx]:
                pt = self.curr_frame.kp_pts[idx]
                u_f, v_f = float(pt[0]), float(pt[1])
                surviving_pts.append(pt)
                surviving_mps.append(mp)
                cr = min(n_rows - 1, max(0, int(v_f / cell_h)))
                cc = min(n_cols - 1, max(0, int(u_f / cell_w)))
                cell_counts[cr, cc] += 1
                gy, gx = int(v_f) // 12, int(u_f) // 12
                occ_grid[max(0, gy - 1) : gy + 2, max(0, gx - 1) : gx + 2] = True
                zc = float(T_cw.transform(mp.pos_w)[2])
                if 0.25 < zc < 6.0:
                    valid_zs.append(zc)

        need_total = 230 - len(surviving_pts)
        if need_total <= 8:
            return len(surviving_pts)

        border = 16
        roi = rect_l[border : h - border, border : w - border]
        raw_kps = self.orb_extractor._fast_low.detect(roi, None)
        if not raw_kps:
            return len(surviving_pts)

        # 按响应值从高到低排序，并在 8x6 网格内均匀选取 (每格最多 5 个点，且间距 >= 12px)
        raw_kps_sorted = sorted(raw_kps, key=lambda k: k.response, reverse=True)
        added_pts: list[tuple[float, float]] = []
        max_per_cell = 5
        for kp in raw_kps_sorted:
            u_abs = float(kp.pt[0]) + border
            v_abs = float(kp.pt[1]) + border
            cr = min(n_rows - 1, max(0, int(v_abs / cell_h)))
            cc = min(n_cols - 1, max(0, int(u_abs / cell_w)))
            if cell_counts[cr, cc] >= max_per_cell:
                continue
            gy, gx = int(v_abs) // 12, int(u_abs) // 12
            if occ_grid[gy, gx]:
                continue
            occ_grid[max(0, gy - 1) : gy + 2, max(0, gx - 1) : gx + 2] = True
            cell_counts[cr, cc] += 1
            added_pts.append((u_abs, v_abs))
            if len(added_pts) >= need_total:
                break

        if len(added_pts) == 0:
            return len(surviving_pts)

        z_med = float(np.median(valid_zs)) if len(valid_zs) >= 6 else 1.20
        z_med = float(np.clip(z_med, 0.70, 2.20))
        fx, fy, cx, cy = self.camera.fx, self.camera.fy, self.camera.cx, self.camera.cy
        R_wc = T_cw.R.T
        t_wc = self.curr_frame.get_camera_center()

        new_pts = np.asarray(added_pts, dtype=np.float32)
        cv_kps = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 31.0) for pt in new_pts]
        _, descs = self.orb_extractor._orb_descriptor.compute(rect_l, cv_kps)

        for idx_c, pt in enumerate(new_pts):
            u_p, v_p = float(pt[0]), float(pt[1])
            z_pt = z_med
            p_c = np.array([(u_p - cx) / fx * z_pt, (v_p - cy) / fy * z_pt, z_pt], dtype=np.float64)
            pos_w = R_wc @ p_c + t_wc
            desc = descs[idx_c] if (descs is not None and idx_c < len(descs)) else None
            mp = MapPoint(pos_w, ref_kf=self.ref_kf, first_frame_id=self.curr_frame.id, descriptor=desc)
            mp.increase_visible()
            mp.increase_found()
            self.slam_map.add_mappoint(mp)
            surviving_pts.append(pt)
            surviving_mps.append(mp)

        n_tot = len(surviving_pts)
        self.curr_frame.kp_pts = np.asarray(surviving_pts, dtype=np.float32)
        self.curr_frame.num_kps = n_tot
        self.curr_frame.kp_octaves = np.zeros(n_tot, dtype=np.int32)
        self.curr_frame.kp_angles = np.zeros(n_tot, dtype=np.float32)
        self.curr_frame.u_right = np.full(n_tot, -1.0, dtype=np.float32)
        self.curr_frame.depths = np.full(n_tot, -1.0, dtype=np.float32)
        self.curr_frame.map_points = list(surviving_mps)
        self.curr_frame.outliers = np.zeros(n_tot, dtype=bool)
        if self.curr_frame.descriptors is not None:
            desc_list = [
                (mp.descriptor if (mp is not None and mp.descriptor is not None) else np.zeros(32, dtype=np.uint8))
                for mp in surviving_mps
            ]
            self.curr_frame.descriptors = np.asarray(desc_list, dtype=np.uint8)
        self.curr_frame.assign_features_to_grid()
        return n_tot

    def _merge_flow_and_orb_features(
        self,
        rect_l: np.ndarray,
        rect_r: np.ndarray | None,
        flow_pts: np.ndarray,
        flow_mps: list[MapPoint | None],
        kp_pts: np.ndarray,
        kp_octaves: np.ndarray,
        kp_angles: np.ndarray,
        descriptors: np.ndarray | None,
        u_right: np.ndarray,
        depths: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, list[MapPoint | None]]:
        """
        将亚像素光流跟踪内点与新提取的四叉树 ORB 特征点无损融合:
        - 若某 ORB 特征点与光流点距离 < 1.5 px，直接绑定该 MapPoint 并将坐标更新为亚像素光流坐标；
        - 其余未重合的光流内点直接保留其精确亚像素坐标与原代表描述子加入特征列表，完全消除吸附误差！
        """
        n_orb = len(kp_pts)
        init_mps: list[MapPoint | None] = [None] * n_orb
        if len(flow_pts) == 0 or descriptors is None:
            return kp_pts, kp_octaves, kp_angles, descriptors, u_right, depths, init_mps

        extra_pts = []
        extra_octs = []
        extra_angs = []
        extra_descs = []
        extra_mps: list[MapPoint | None] = []

        for pt, mp in zip(flow_pts, flow_mps):
            if mp is None or mp.is_bad or mp.descriptor is None:
                continue
            if n_orb > 0:
                diffs = kp_pts - pt[None, :]
                d2 = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
                best_i = int(np.argmin(d2))
                if d2[best_i] <= 2.25 and init_mps[best_i] is None:
                    # 距离 <= 1.5 px: 采用亚像素光流坐标修正并绑定 MapPoint
                    kp_pts[best_i] = pt
                    init_mps[best_i] = mp
                    continue

            extra_pts.append(pt)
            extra_octs.append(0)
            extra_angs.append(0.0)
            extra_descs.append(mp.descriptor.copy())
            extra_mps.append(mp)

        if len(extra_pts) > 0:
            ex_pts_arr = np.asarray(extra_pts, dtype=np.float32)
            if rect_r is not None:
                ex_ur, ex_dep = StereoMatcher.compute_stereo_matches(rect_l, rect_r, ex_pts_arr, self.camera)
            else:
                ex_ur = np.full(len(ex_pts_arr), -1.0, dtype=np.float32)
                ex_dep = np.full(len(ex_pts_arr), -1.0, dtype=np.float32)
            kp_pts = np.vstack([kp_pts, ex_pts_arr])
            kp_octaves = np.concatenate([kp_octaves, np.asarray(extra_octs, dtype=np.int32)])
            kp_angles = np.concatenate([kp_angles, np.asarray(extra_angs, dtype=np.float32)])
            descriptors = np.vstack([descriptors, np.asarray(extra_descs, dtype=np.uint8)])
            u_right = np.concatenate([u_right, ex_ur])
            depths = np.concatenate([depths, ex_dep])
            init_mps.extend(extra_mps)

        return kp_pts, kp_octaves, kp_angles, descriptors, u_right, depths, init_mps

    def _match_from_last_frame(self) -> None:
        """对上一帧剩余有效地图点执行重投影 + ORB 描述子匹配 (对应 TrackWithMotionModel)"""
        assert self.curr_frame is not None
        if self.last_frame is not None and self.curr_frame.descriptors is not None:
            T_cw = self.curr_frame.T_cw
            already_matched = {mp for mp in self.curr_frame.map_points if mp is not None}
            proj_candidates: list[MapPoint] = []
            for mp in self.last_frame.map_points:
                if mp is None or mp.is_bad or mp in already_matched:
                    continue
                p_c = T_cw.transform(mp.pos_w)
                if p_c[2] <= 0.1:
                    continue
                uv_ur = self.camera.project_stereo(p_c)
                if self.camera.is_in_image(float(uv_ur[0]), float(uv_ur[1]), margin=4.0):
                    mp.track_proj_x = float(uv_ur[0])
                    mp.track_proj_y = float(uv_ur[1])
                    mp.track_proj_xr = float(uv_ur[2])
                    mp.track_in_view = True
                    mp.track_scale_level = 0
                    mp.track_view_cos = 1.0
                    proj_candidates.append(mp)
            if len(proj_candidates) > 0:
                StereoMatcher.search_by_projection(self.curr_frame, proj_candidates, th_radius=10.0)

    def _track_local_map(
        self,
        R_cw_prior: np.ndarray | None = None,
        rot_prior_weight: float = 0.0
    ) -> int:
        """
        局部地图跟踪 (对应 ORB-SLAM3 Tracking::TrackLocalMap):
        1. 更新局部关键帧与局部地图点集合 (UpdateLocalMap)
        2. 将未在当前帧中的局部地图点投影匹配 (SearchLocalPoints)
        3. 执行 Motion-only BA 优化当前帧位姿并统计内点数
        """
        assert self.curr_frame is not None
        self._update_local_map()
        self._search_local_points()

        n_inliers = Optimizer.pose_optimization(
            self.curr_frame,
            n_its_per_round=8,
            R_cw_prior=R_cw_prior,
            rot_prior_weight=rot_prior_weight
        )

        final_good = 0
        for i, mp in enumerate(self.curr_frame.map_points):
            if mp is not None:
                if self.curr_frame.outliers[i]:
                    if mp.num_observations() == 0:
                        mp.set_bad_flag(self.slam_map)
                    self.curr_frame.map_points[i] = None
                else:
                    mp.increase_found()
                    if mp.num_observations() > 0:
                        final_good += 1

        return max(n_inliers, final_good)

    def _update_local_map(self) -> None:
        """
        通过共视图收集局部关键帧 (UpdateLocalKeyFrames) 与局部地图点 (UpdateLocalPoints)
        """
        assert self.curr_frame is not None
        kf_counter: dict[KeyFrame, int] = {}
        for mp in self.curr_frame.map_points:
            if mp is not None and not mp.is_bad:
                for kf in mp.observations.keys():
                    if not kf.is_bad:
                        kf_counter[kf] = kf_counter.get(kf, 0) + 1

        if len(kf_counter) == 0 and self.ref_kf is not None:
            kf_counter[self.ref_kf] = 1

        local_kfs_set: set[KeyFrame] = set()
        max_votes = 0
        best_ref_kf: KeyFrame | None = self.ref_kf

        for kf, votes in kf_counter.items():
            local_kfs_set.add(kf)
            if votes > max_votes:
                max_votes = votes
                best_ref_kf = kf

        # 加入共视图一阶邻居与生成树父子关键帧 (上限控制在 25 个关键帧以内)
        for kf in list(local_kfs_set):
            if len(local_kfs_set) >= 25:
                break
            for neigh_kf in kf.get_best_covisibility_keyframes(8):
                if not neigh_kf.is_bad:
                    local_kfs_set.add(neigh_kf)
            if kf.parent is not None and not kf.parent.is_bad:
                local_kfs_set.add(kf.parent)

        if best_ref_kf is not None:
            self.ref_kf = best_ref_kf
            self.curr_frame.ref_kf = best_ref_kf

        self.local_keyframes = sorted(local_kfs_set, key=lambda k: k.id, reverse=True)

        # 收集局部关键帧关联的所有有效地图点 (优先投影最新关键帧的近处地图点)
        local_mps_set: set[MapPoint] = set()
        for kf in self.local_keyframes:
            for mp in kf.map_points:
                if mp is not None and not mp.is_bad:
                    local_mps_set.add(mp)

        self.local_mappoints = sorted(local_mps_set, key=lambda m: m.id, reverse=True)
        self.slam_map.set_reference_mappoints(self.local_mappoints)

    def _search_local_points(self) -> None:
        """
        视野锥体筛查 (Frustum Culling) + 局部地图点投影匹配
        对应 ORB-SLAM3 Tracking::SearchLocalPoints()
        """
        assert self.curr_frame is not None
        frame_id = self.curr_frame.id
        for mp in self.curr_frame.map_points:
            if mp is not None and not mp.is_bad:
                mp.increase_visible()
                mp.last_frame_seen = frame_id

        T_cw = self.curr_frame.T_cw
        o_w = self.curr_frame.get_camera_center()
        to_project: list[MapPoint] = []

        for mp in self.local_mappoints:
            if mp.last_frame_seen == frame_id or mp.is_bad:
                continue
            mp.track_in_view = False

            # 1. 变换到当前相机坐标系并检查正深度
            p_c = T_cw.transform(mp.pos_w)
            if p_c[2] <= 0.15:
                continue

            # 2. 投影到像素平面并检查图像边界
            uv_ur = self.camera.project_stereo(p_c)
            u, v, ur = float(uv_ur[0]), float(uv_ur[1]), float(uv_ur[2])
            if not self.camera.is_in_image(u, v, margin=4.0):
                continue

            # 3. 检查观测距离与法向夹角 (cos < 60 deg)
            vec_po = mp.pos_w - o_w
            dist = float(np.linalg.norm(vec_po))
            if dist < 0.8 * mp.min_distance or dist > 1.2 * mp.max_distance:
                continue
            view_cos = float(np.dot(vec_po, mp.normal)) / (dist + 1e-12)
            if view_cos < 0.5:
                continue

            mp.track_in_view = True
            mp.track_proj_x = u
            mp.track_proj_y = v
            mp.track_proj_xr = ur
            mp.track_view_cos = view_cos
            mp.track_scale_level = mp.predict_scale(dist, len(self.curr_frame.scale_factors))
            mp.increase_visible()
            to_project.append(mp)

        if len(to_project) > 0:
            StereoMatcher.search_by_projection(self.curr_frame, to_project, th_radius=6.0)

    def _need_new_keyframe(self, n_tracked_inliers: int, is_flow_only: bool = False) -> bool:
        """
        关键帧插入准则 (对应 ORB-SLAM3 Tracking::NeedNewKeyFrame):
        1. 跟踪到的近距离双目点跌破阈值且当前帧拥有充足的可新建近距离双目点；
        2. 或者跟踪内点数低于参考关键帧有效地图点数的比例阈值；
        3. 或者距离上一关键帧已超过最大帧间隔 (双目 10 帧，纯单目依靠多帧时序三角化故设为 4 帧)。
        """
        assert self.curr_frame is not None
        is_mono = bool(self.curr_frame.num_kps > 0 and np.all(self.curr_frame.depths < 0.0))
        frames_since_last_kf = self.curr_frame.id - self.last_kf_frame_id

        if is_mono:
            if self.camera.width == 640:
                # 实机单目+IMU模式: 普通帧已在 Path A 内增量补点，仅在视角显著变化或内点跌破阈值时创建关键帧
                if frames_since_last_kf < 4:
                    return False
                min_inliers_live = 22 if is_flow_only else 35
                if n_tracked_inliers < min_inliers_live:
                    return True
                if self.ref_kf is not None and frames_since_last_kf >= 18:
                    dp_ref = float(np.linalg.norm(self.curr_frame.get_camera_center() - self.ref_kf.get_camera_center()))
                    dR_ref = self.curr_frame.T_cw.R @ self.ref_kf.T_cw.R.T
                    rot_deg = float(np.rad2deg(np.arccos(np.clip((np.trace(dR_ref) - 1.0) * 0.5, -1.0, 1.0))))
                    if dp_ref >= 0.045 or rot_deg >= 8.0:
                        return True
                return False

            if n_tracked_inliers < 15:
                return False
            if frames_since_last_kf < 2 and n_tracked_inliers >= 40:
                return False
            n_ref_mps = self.ref_kf.tracked_mappoints(min_obs=2) if self.ref_kf is not None else 100
            cond_ratio = n_tracked_inliers < max(45, int(0.75 * n_ref_mps))
            cond_max_gap = frames_since_last_kf >= 4
            return cond_ratio or cond_max_gap

        if frames_since_last_kf < 5 and n_tracked_inliers >= 35:
            return False

        n_ref_mps = self.ref_kf.tracked_mappoints(min_obs=2) if self.ref_kf is not None else 100
        tracked_close = 0
        non_tracked_close = 0
        for i in range(self.curr_frame.num_kps):
            z = float(self.curr_frame.depths[i])
            if 0.60 < z < self.camera.th_depth:
                if self.curr_frame.map_points[i] is not None and not self.curr_frame.outliers[i]:
                    tracked_close += 1
                else:
                    non_tracked_close += 1

        if is_flow_only:
            need_for_close = tracked_close < 42
        else:
            need_for_close = (tracked_close < 50) and (non_tracked_close > 40)
        cond_ratio = n_tracked_inliers < max(35, int(0.60 * n_ref_mps))
        cond_max_gap = frames_since_last_kf >= 10

        return need_for_close or cond_ratio or cond_max_gap

    def _create_new_keyframe(self) -> None:
        """
        创建新关键帧，触发 LocalMapper 与 LoopClosing，并将 Local BA 优化后的位姿与地图点同步回当前帧
        对应 ORB-SLAM3 Tracking::CreateNewKeyFrame()
        """
        assert self.curr_frame is not None
        new_kf = KeyFrame(self.curr_frame)
        self.ref_kf = new_kf
        self.curr_frame.ref_kf = new_kf
        self.last_kf_frame_id = self.curr_frame.id

        # 调用后端局部建图 (包含共视三角化、Local BA、以及 BA 后近距离双目点生成) 与回环闭合
        self.local_mapper.run_for_keyframe(new_kf)
        self.loop_closer.run_for_keyframe(new_kf)

        # 将 Local BA 优化后的最新位姿与新建/融合后的地图点列表同步回 curr_frame (作为下一帧的光流跟踪基准)
        self.curr_frame.set_pose(new_kf.T_cw.copy())
        self.curr_frame.map_points = list(new_kf.map_points)
