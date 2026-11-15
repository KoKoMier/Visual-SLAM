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

# Python 3.6 compatible
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
from .hfnet_extractor import HFNetExtractor
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
        "orb_extractor", "imu_preintegrator", "imu_preintegrator_from_last_kf",
        "state", "curr_frame", "last_frame", "prev_frame", "mono_init_frame", "ref_kf", "last_kf",
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
        camera,
        slam_map,
        local_mapper,
        loop_closer,
        n_features = 600,
        use_hybrid_optical_flow = True,
        use_hfnet = False,
        hfnet_root = None,
    ):
        self.camera = camera
        self.slam_map = slam_map
        self.local_mapper = LocalMapper(self.slam_map) if local_mapper is None else local_mapper
        self.loop_closer = loop_closer

        if use_hfnet:
            try:
                self.orb_extractor = HFNetExtractor(n_features=n_features, hfnet_root=hfnet_root)
            except Exception as exc:
                print(f"  [Tracker] HF-Net 初始化回退至 ORBExtractor: {exc}")
                self.orb_extractor = ORBExtractor(n_features=n_features)
        else:
            self.orb_extractor = ORBExtractor(n_features=n_features)
        self.imu_preintegrator = ImuPreintegrator()
        self.imu_preintegrator_from_last_kf = ImuPreintegrator()

        self.state = TrackingState.NO_IMAGES_YET
        self.curr_frame = None
        self.last_frame = None
        self.prev_frame = None
        self.mono_init_frame = None
        self.ref_kf = None
        self.last_kf = None
        self.last_kf_frame_id = 0

        self.local_keyframes = []
        self.local_mappoints = []

        self.velocity = SE3.identity()
        self.has_velocity = False
        self.use_hybrid_optical_flow = bool(use_hybrid_optical_flow)
        self.last_is_keyframe = False
        self.last_used_optical_flow = False
        self.last_inliers = 0
        self.last_flow_prior_T_cr = None
        self._gyro_vec_hist = []
        self.last_diag = {
            "trans_gate": 1.0,
            "eff_gyro_deg_s": 0.0,
            "raw_flow_px": 0.0,
            "derot_res_px": 0.0,
            "radial_med_px": 0.0,
        }

        self.timing_stats = {
            "frontend_ms": [],
            "backend_ms": [],
            "total_ms": []
        }

    def grab_image_stereo(
        self,
        img_left,
        img_right,
        timestamp,
        imu_measurements = None
    ) :
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
            w_rot_imu = 2.0e6 if (len(imu_measurements) >= 2 or self.camera.width == 640) else 50.0
        else:
            w_rot_imu = 0.0
        T_cw_flow_prior = None
        flow_pts_cached = None
        flow_mps_cached = None

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
        rect_l,
        rect_r,
        timestamp,
        imu_measurements = None,
    ) :
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

    def _preintegrate_imu(self, imu_measurements, timestamp) :
        """
        对应 ORB-SLAM3 Tracking::PreintegrateIMU():
        同时对相邻帧预积分器 (imu_preintegrator) 与关键帧预积分器 (imu_preintegrator_from_last_kf)
        执行中点法流形预积分，精确对齐 [t_prev, timestamp] 时间区间。
        """
        bg_ref = (
            self.last_frame.bg
            if self.last_frame is not None
            else (self.last_kf.bg if self.last_kf is not None else None)
        )
        ba_ref = (
            self.last_frame.ba
            if self.last_frame is not None
            else (self.last_kf.ba if self.last_kf is not None else None)
        )
        self.imu_preintegrator.reset(bg=bg_ref, ba=ba_ref)
        if imu_measurements is None or len(imu_measurements) == 0:
            return

        t_prev_frame = (
            self.last_frame.timestamp
            if self.last_frame is not None
            else (imu_measurements[0].timestamp - 0.005)
        )
        n_m = len(imu_measurements)
        for i in range(n_m):
            m_curr = imu_measurements[i]
            if i == 0:
                dt = m_curr.timestamp - t_prev_frame
                acc_mid = m_curr.acc
                gyro_mid = m_curr.gyro
            else:
                m_prev = imu_measurements[i - 1]
                dt = m_curr.timestamp - m_prev.timestamp
                acc_mid = 0.5 * (m_prev.acc + m_curr.acc)
                gyro_mid = 0.5 * (m_prev.gyro + m_curr.gyro)
            if dt > 1e-5:
                dt_c = min(float(dt), 0.10)
                self.imu_preintegrator.integrate_new_measurement(acc_mid, gyro_mid, dt_c)
                self.imu_preintegrator_from_last_kf.integrate_new_measurement(acc_mid, gyro_mid, dt_c)

        last_m = imu_measurements[-1]
        dt_tail = timestamp - last_m.timestamp
        if dt_tail > 1e-5:
            dt_c = min(float(dt_tail), 0.05)
            self.imu_preintegrator.integrate_new_measurement(last_m.acc, last_m.gyro, dt_c)
            self.imu_preintegrator_from_last_kf.integrate_new_measurement(last_m.acc, last_m.gyro, dt_c)

    def grab_image_monocular(
        self,
        img_left,
        timestamp,
        imu_measurements = None,
    ) :
        """
        单目 / 单目+IMU 紧耦合跟踪主入口 (对应 ORB-SLAM3 Tracking::GrabImageMonocular):
        支持纯单目 (imu_measurements=None) 与单目+IMU 紧耦合 (imu_measurements!=None) 两种模式。
        """
        t_start = time.perf_counter()
        rect_l = self.camera.rectify_mono(img_left)

        # 0. 若包含 IMU 数据，执行帧间与关键帧间流形预积分 (Tracking::PreintegrateIMU)
        if imu_measurements is not None and len(imu_measurements) > 0:
            self._preintegrate_imu(imu_measurements, timestamp)
        else:
            self.imu_preintegrator.reset()

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

        # 2. 正常跟踪状态 (ORB-SLAM3 PredictStateIMU / 恒速模型 + PoseInertialOptimization + TrackLocalMap)
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
            w_rot_imu = 1.5e5 if self.camera.width == 640 else (1.5e5 if len(imu_measurements) >= 5 else 80.0)
        else:
            w_rot_imu = 0.0

        t_cw_imu_prior = T_cw_pred_init.t.copy() if (has_imu_prior and self.slam_map.imu_initialized) else None
        w_pos_imu = (4.0e3 if self.slam_map.inertial_ba1 else 8.0e2) if t_cw_imu_prior is not None else 0.0

        trans_gate = 1.0
        T_cw_flow_prior = None
        flow_pts_cached = None
        flow_mps_cached = None

        if self.use_hybrid_optical_flow and self.last_frame is not None:
            kp_pts, u_right, depths, tracked_mps = StereoMatcher.track_optical_flow(
                self.last_frame, rect_l, rect_l, self.camera, T_cw_pred=T_cw_pred_init
            )
            if self.camera.width == 640:
                tracked_set = {mp for mp in tracked_mps if mp is not None}
                for old_mp in self.last_frame.map_points:
                    if old_mp is not None and not old_mp.is_bad and old_mp.num_observations() == 0 and old_mp not in tracked_set:
                        old_mp.set_bad_flag(self.slam_map)
                R_cw_imu_prior, trans_gate = self._estimate_rotation_and_trans_gate(
                    kp_pts, tracked_mps, imu_measurements, R_cw_imu_prior
                )
                if R_cw_imu_prior is not None:
                    p_last_init = self.last_frame.get_camera_center() if (self.last_frame is not None and self.last_frame.has_pose) else -(T_cw_pred_init.R.T @ T_cw_pred_init.t)
                    dp_flow_init = self._compute_horizon_decoupled_step(R_cw_imu_prior, trans_gate)
                    p_pred_init = p_last_init + dp_flow_init
                    T_cw_pred_init = SE3(R_cw_imu_prior, -(R_cw_imu_prior @ p_pred_init))
                    t_cw_imu_prior = T_cw_pred_init.t.copy()
                    w_pos_imu = 5.0e4
            flow_pts_cached = kp_pts
            flow_mps_cached = tracked_mps
            if len(kp_pts) >= 25:
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
                self.curr_frame.last_kf = self.last_kf
                self.curr_frame.prev_frame = self.last_frame
                self.curr_frame.imu_preintegrated_frame = self.imu_preintegrator
                self.curr_frame.imu_preintegrated = self.imu_preintegrator_from_last_kf
                self.curr_frame.R_cw_imu_prior = R_cw_imu_prior
                if self.last_frame is not None:
                    self.curr_frame.bg = self.last_frame.bg.copy()
                    self.curr_frame.ba = self.last_frame.ba.copy()
                self.curr_frame.set_pose(T_cw_pred_init)

                n_inliers = Optimizer.pose_optimization(
                    self.curr_frame,
                    n_its_per_round=4 if self.camera.width == 640 else 6,
                    R_cw_prior=R_cw_imu_prior,
                    rot_prior_weight=w_rot_imu,
                    t_cw_prior=t_cw_imu_prior,
                    pos_prior_weight=w_pos_imu,
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
                    T_cw_flow_prior = self.curr_frame.T_cw.copy()
                    flow_pts_cached = self.curr_frame.kp_pts.copy()
                    flow_mps_cached = list(self.curr_frame.map_points)

                th_retrack = 24 if self.camera.width == 640 else 38
                if self._need_new_keyframe(n_inliers, is_flow_only=True) or n_inliers < th_retrack:
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
                        p_last_init = self.last_frame.get_camera_center() if self.last_frame.has_pose else -(T_cw_pred_init.R.T @ T_cw_pred_init.t)
                        dp_flow_init = self._compute_horizon_decoupled_step(R_cw_imu_prior, trans_gate)
                        p_pred_init = p_last_init + dp_flow_init
                        T_cw_pred_init = SE3(R_cw_imu_prior, -(R_cw_imu_prior @ p_pred_init))
                        t_cw_imu_prior = T_cw_pred_init.t.copy()
                        w_pos_imu = 5.0e4
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
            self.curr_frame.last_kf = self.last_kf
            self.curr_frame.prev_frame = self.last_frame
            self.curr_frame.imu_preintegrated_frame = self.imu_preintegrator
            self.curr_frame.imu_preintegrated = self.imu_preintegrator_from_last_kf
            self.curr_frame.R_cw_imu_prior = R_cw_imu_prior
            if self.last_frame is not None:
                self.curr_frame.bg = self.last_frame.bg.copy()
                self.curr_frame.ba = self.last_frame.ba.copy()
            self.curr_frame.set_pose(T_cw_flow_prior if T_cw_flow_prior is not None else T_cw_pred_init)

            self._match_from_last_frame()
            if sum(1 for mp in self.curr_frame.map_points if mp is not None) >= 6:
                Optimizer.pose_optimization(
                    self.curr_frame,
                    n_its_per_round=4 if self.camera.width == 640 else 6,
                    R_cw_prior=R_cw_imu_prior,
                    rot_prior_weight=w_rot_imu,
                    t_cw_prior=t_cw_imu_prior,
                    pos_prior_weight=w_pos_imu,
                )

            n_inliers = self._track_local_map(
                R_cw_prior=R_cw_imu_prior,
                rot_prior_weight=w_rot_imu,
                t_cw_prior=t_cw_imu_prior,
                pos_prior_weight=w_pos_imu,
            )

        # 3. 实机 640x480 单目+IMU 紧耦合：利用地平线横滚补偿与主轴正交解耦位移锁定无偏三维轨迹
        if self.camera.width == 640 and self.last_frame is not None and self.last_frame.has_pose:
            p_last_ref = self.last_frame.get_camera_center()
            R_opt = self.curr_frame.T_cw.R.copy()
            R_final = R_cw_imu_prior if R_cw_imu_prior is not None else R_opt

            if n_inliers < 10 or trans_gate <= 1e-4:
                p_final = p_last_ref
            else:
                dp_clean = self._compute_horizon_decoupled_step(R_final, trans_gate)
                p_final = p_last_ref + dp_clean

            self.curr_frame.set_pose(SE3(R_final, -(R_final @ p_final)))
            if n_inliers < 145:
                n_inliers = self._replenish_live_flow_features(rect_l, n_inliers)

        t_fe_end = time.perf_counter()

        t_be_start = time.perf_counter()
        created_kf = False
        T_cw_cont = T_cw_flow_prior if T_cw_flow_prior is not None else self.curr_frame.T_cw
        if self.last_frame is not None and self.last_frame.has_pose:
            self.last_flow_prior_T_cr = T_cw_cont * self.last_frame.T_cw.inverse()
        else:
            self.last_flow_prior_T_cr = None
        if need_kf_or_retrack and self._need_new_keyframe(n_inliers):
            self._create_new_keyframe()
            created_kf = True
            n_inliers = max(
                n_inliers,
                sum(1 for mp in self.curr_frame.map_points if mp is not None and not mp.is_bad)
            )
        t_be_end = time.perf_counter()

        if self.curr_frame.has_pose and self.last_frame is not None and self.last_frame.has_pose:
            dt_frame = max(1e-3, self.curr_frame.timestamp - self.last_frame.timestamp)
            p_curr = self.curr_frame.get_camera_center()
            p_last = self.last_frame.get_camera_center()
            dp_frame = p_curr - p_last
            if self.camera.width == 640:
                dp_len = float(np.linalg.norm(dp_frame))
                if dp_len > 0.035:
                    dp_frame = dp_frame * (0.035 / dp_len)
                vel_scale = 0.75 * trans_gate
                dp_damped = dp_frame * vel_scale
                R_rel = self.curr_frame.T_cw.R @ self.last_frame.T_cw.R.T
                t_vel = -(self.curr_frame.T_cw.R @ (p_curr - R_rel.T @ p_last)) * vel_scale
                self.velocity = SE3(R_rel, t_vel)
                self.curr_frame.v_w = dp_damped / dt_frame
            else:
                self.velocity = self.curr_frame.T_cw * self.last_frame.T_cw.inverse()
                self.curr_frame.v_w = dp_frame / dt_frame
            self.has_velocity = True

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
        rect_l,
        timestamp,
        imu_measurements = None,
    ) :
        """
        单目 / 单目+IMU 双视图视差三角化初始化 (对应 ORB-SLAM3 Tracking::MonocularInitialization):
        1. 第 0 帧提取四叉树 ORB 特征作为参考帧 F0，并重置关键帧预积分器 imu_preintegrator_from_last_kf；
        2. 后续帧通过双向金字塔光流跟踪 F0 特征，若含 200Hz IMU 则累积帧间与关键帧间流形预积分旋转先验；
        3. 等待真实平移视差达标 (拒绝静止噪声与原地纯旋转)，利用本质矩阵 E / IMU 已知旋转极线约束恢复 (R_10, t_10)，
           通过 DLT 线性三角化生成真实 3D 地图点并执行初始 Bundle Adjustment；
        4. 构建双向关键帧链表 kf0 <-> kf1 并挂载 IMU 预积分约束，供 LocalMapper 触发 InitializeIMU。
        """
        kp_pts, kp_octaves, kp_angles, descriptors = self.orb_extractor.extract(rect_l)
        u_right = np.full(len(kp_pts), -1.0, dtype=np.float32)
        depths = np.full(len(kp_pts), -1.0, dtype=np.float32)

        has_imu = bool(imu_measurements is not None and len(imu_measurements) > 0)
        if has_imu and self.mono_init_frame is not None and self.imu_preintegrator_from_last_kf.dt_total > 0.0:
            R_bc = self.camera.T_b_c1[:3, :3]
            dR_cam_01 = R_bc.T @ self.imu_preintegrator_from_last_kf.get_updated_delta_rotation() @ R_bc
            R_10_imu = dR_cam_01.T @ self.mono_init_frame.T_cw.R
            T_cw_init_pred = SE3(R_10_imu, np.zeros(3, dtype=np.float64))
        elif has_imu and self.last_frame is not None:
            T_cw_init_pred = self._compute_predicted_pose(imu_measurements)
        else:
            T_cw_init_pred = SE3.identity()

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
            R_cw_init0 = np.eye(3, dtype=np.float64)
            if has_imu:
                acc_init = np.mean([m.acc for m in imu_measurements], axis=0)
                acc_norm = float(np.linalg.norm(acc_init))
                if acc_norm > 5.0:
                    R_bc = self.camera.T_b_c1[:3, :3]
                    if self.camera.width == 640 and acc_init[2] > 4.0:
                        pitch_0 = float(np.arctan2(acc_init[1], acc_init[2]))
                        roll_0 = float(np.arctan2(-acc_init[0], np.hypot(acc_init[1], acc_init[2])))
                        cr0, sr0 = np.cos(roll_0), np.sin(roll_0)
                        cp0, sp0 = np.cos(pitch_0), np.sin(pitch_0)
                        Rx0 = np.array([[1.0, 0.0, 0.0], [0.0, cp0, -sp0], [0.0, sp0, cp0]], dtype=np.float64)
                        Ry0 = np.array([[cr0, 0.0, sr0], [0.0, 1.0, 0.0], [-sr0, 0.0, cr0]], dtype=np.float64)
                        R_wb_0 = Ry0 @ Rx0
                        R_wc_0 = R_bc.T @ R_wb_0 @ R_bc
                        R_cw_init0 = R_wc_0.T
                    self.slam_map.g_w = -(R_bc.T @ (acc_init / acc_norm)) * 9.81
            self.curr_frame.set_pose(SE3(R_cw_init0, np.zeros(3, dtype=np.float64)))
            self.curr_frame.R_cw_imu_prior = R_cw_init0.copy()
            self.mono_init_frame = self.curr_frame
            self.last_frame = self.curr_frame
            self.imu_preintegrator_from_last_kf.reset()
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

        matched_idx0 = []
        matched_pts0 = []
        matched_pts1 = []
        for idx0 in range(len(p0_2d)):
            if st_fwd[idx0, 0] == 1 and st_bwd[idx0, 0] == 1 and fb_err[idx0] <= 1.5:
                u1, v1 = float(p1_2d[idx0, 0]), float(p1_2d[idx0, 1])
                if self.camera.is_in_image(u1, v1, margin=4.0):
                    matched_idx0.append(idx0)
                    matched_pts0.append(p0_2d[idx0])
                    matched_pts1.append(p1_2d[idx0])

        if len(matched_pts0) < 45:
            # 跟踪点过少，重置参考帧 F0 与预积分器
            self.curr_frame.set_pose(SE3.identity())
            self.mono_init_frame = self.curr_frame
            self.last_frame = self.curr_frame
            self.imu_preintegrator_from_last_kf.reset()
            return

        pts0_arr = np.asarray(matched_pts0, dtype=np.float64)
        pts1_arr = np.asarray(matched_pts1, dtype=np.float64)
        parallax = np.linalg.norm(pts1_arr - pts0_arr, axis=1)
        med_parallax = float(np.median(parallax))

        fx, fy, cx, cy = self.camera.fx, self.camera.fy, self.camera.cx, self.camera.cy
        x0_n = np.column_stack([(pts0_arr[:, 0] - cx) / fx, (pts0_arr[:, 1] - cy) / fy, np.ones(len(pts0_arr))])
        x1_n = np.column_stack([(pts1_arr[:, 0] - cx) / fx, (pts1_arr[:, 1] - cy) / fy, np.ones(len(pts1_arr))])

        if has_imu:
            R_10_imu = T_cw_init_pred.R @ f0.T_cw.R.T
            x0_rot = x0_n @ R_10_imu.T
            z0_rot = np.maximum(1e-4, x0_rot[:, 2])
            u1_rot = fx * (x0_rot[:, 0] / z0_rot) + cx
            v1_rot = fy * (x0_rot[:, 1] / z0_rot) + cy
            derot_parallax = np.hypot(pts1_arr[:, 0] - u1_rot, pts1_arr[:, 1] - v1_rot)
            med_derot_parallax = float(np.median(derot_parallax))
        else:
            R_10_imu = None
            med_derot_parallax = med_parallax

        # 若长时间静止 (dt > 0.8s 且去旋转视差 < 2.0px)，刷新参考帧 F0 以保证初始化关键帧间 IMU 预积分窗口短且准
        if has_imu and (timestamp - f0.timestamp) > 0.80 and med_derot_parallax < 2.0:
            self.curr_frame.set_pose(SE3.identity())
            self.mono_init_frame = self.curr_frame
            self.last_frame = self.curr_frame
            self.imu_preintegrator_from_last_kf.reset()
            return

        min_req_parallax = 5.5 if (has_imu and self.camera.width == 640) else 10.0
        if med_derot_parallax < min_req_parallax:
            self.last_frame = self.curr_frame
            return

        target_med_depth = 1.00 if (has_imu and self.camera.width == 640) else (2.30 if has_imu else 2.0)

        if has_imu and self.camera.width == 640 and R_10_imu is not None:
            # 实机 640x480 + 200Hz IMU 双视图初始化:
            # 直接利用 200Hz IMU 精确旋转 R_10_imu 与去旋转归一化光流解算平移基线与点云深度，
            # 彻底消除窄视场 Essential Matrix / SVD 在微基线下的光轴 Z 向退化伪跳变，并保留全图 200+ 均匀内点
            u0_r = x0_rot[:, 0] / z0_rot
            v0_r = x0_rot[:, 1] / z0_rot
            u1_n = x1_n[:, 0]
            v1_n = x1_n[:, 1]
            du_n = u1_n - u0_r
            dv_n = v1_n - v0_r

            tau_x = float(np.median(du_n))
            tau_y = float(np.median(dv_n))
            r0_sq = u0_r * u0_r + v0_r * v0_r
            rad_mask = r0_sq > 0.02
            if np.count_nonzero(rad_mask) >= 15:
                tau_z_raw = float(
                    -np.median(
                        ((du_n[rad_mask] - tau_x) * u0_r[rad_mask] + (dv_n[rad_mask] - tau_y) * v0_r[rad_mask])
                        / np.maximum(0.02, r0_sq[rad_mask])
                    )
                )
            else:
                tau_z_raw = 0.0
            tau_xy_norm = float(np.hypot(tau_x, tau_y))
            if abs(tau_z_raw) < 0.45 * tau_xy_norm:
                tau_z = 0.0
            else:
                tau_z = float(np.clip(0.35 * tau_z_raw, -0.02, 0.02))

            if float(np.hypot(tau_xy_norm, tau_z)) * fx < 4.5:
                self.last_frame = self.curr_frame
                return

            R_10_rel = R_10_imu
            roll_cam = float(np.arctan2(R_10_rel[0, 1], R_10_rel[1, 1]))
            if abs(roll_cam) > np.deg2rad(30.0):
                roll_cam = 0.0
            cr_h, sr_h = float(np.cos(roll_cam)), float(np.sin(roll_cam))
            dx_init = (-tau_x * cr_h + tau_y * sr_h) * target_med_depth
            dy_init = (-tau_x * sr_h - tau_y * cr_h) * target_med_depth
            abs_x0, abs_y0 = abs(dx_init), abs(dy_init)
            if abs_x0 > 1.30 * abs_y0:
                dy_init = float(np.sign(dy_init) * max(0.0, abs_y0 - 0.55 * abs_x0))
            elif abs_y0 > 1.30 * abs_x0:
                dx_init = float(np.sign(dx_init) * max(0.0, abs_x0 - 0.55 * abs_y0))
            dz_init = -tau_z * target_med_depth
            dxy_init = float(np.hypot(dx_init, dy_init))
            if abs(dz_init) <= 0.55 * dxy_init:
                dz_init = 0.0
            else:
                dz_init = float(np.sign(dz_init) * (abs(dz_init) - 0.55 * dxy_init))
            p_1_init = np.array([dx_init, dy_init, dz_init], dtype=np.float64)
            R_10 = R_10_rel @ f0.T_cw.R
            t_10_scaled = -(R_10 @ p_1_init)
            scale_norm = 1.0
            valid_tri = []
            for k_m, idx0 in enumerate(matched_idx0):
                u1_pred = fx * (u0_r[k_m] + tau_x - u0_r[k_m] * tau_z) + cx
                v1_pred = fy * (v0_r[k_m] + tau_y - v0_r[k_m] * tau_z) + cy
                if np.hypot(pts1_arr[k_m, 0] - u1_pred, pts1_arr[k_m, 1] - v1_pred) > 4.0:
                    continue
                z0_pt = target_med_depth
                X_c0 = np.array(
                    [(pts0_arr[k_m, 0] - cx) / fx * z0_pt, (pts0_arr[k_m, 1] - cy) / fy * z0_pt, z0_pt],
                    dtype=np.float64,
                )
                X_w = f0.T_cw.R.T @ (X_c0 - f0.T_cw.t)
                valid_tri.append((idx0, pts1_arr[k_m].astype(np.float32), X_w))

            if len(valid_tri) < 35:
                self.last_frame = self.curr_frame
                return
        else:
            def _evaluate_candidate(R_cand, t_cand_vec):
                P0_c = self.camera.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
                P1_c = self.camera.K @ np.hstack([R_cand, t_cand_vec.reshape(3, 1)])
                pts4d_c = cv2.triangulatePoints(P0_c, P1_c, pts0_arr.T, pts1_arr.T)
                pts3d_c = (pts4d_c[:3] / (pts4d_c[3:] + 1e-12)).T
                tri_list = []
                d0_list = []
                for k_m, idx0 in enumerate(matched_idx0):
                    X_w = pts3d_c[k_m]
                    z0 = float(X_w[2])
                    X_c1 = R_cand @ X_w + t_cand_vec
                    z1 = float(X_c1[2])
                    if z0 <= 0.12 or z1 <= 0.12 or z0 > 250.0 or z1 > 250.0:
                        continue
                    uv0 = self.camera.project_mono(X_w)
                    uv1 = self.camera.project_mono(X_c1)
                    if np.linalg.norm(uv0 - pts0_arr[k_m]) > 3.5 or np.linalg.norm(uv1 - pts1_arr[k_m]) > 3.5:
                        continue
                    ray0 = X_w / (np.linalg.norm(X_w) + 1e-12)
                    ray1 = (R_cand.T @ X_c1) / (np.linalg.norm(X_c1) + 1e-12)
                    if float(np.dot(ray0, ray1)) > 0.99998:
                        continue
                    tri_list.append((idx0, pts1_arr[k_m].astype(np.float32), X_w))
                    d0_list.append(z0)
                return tri_list, d0_list

            best_tri = []
            best_depths = []
            best_R10 = None
            best_t10 = None

            # 候选 1: 标准 RANSAC Essential Matrix 分解
            E, mask_e = cv2.findEssentialMat(
                pts0_arr, pts1_arr, self.camera.K,
                method=cv2.RANSAC, prob=0.999, threshold=1.5
            )
            if E is not None and mask_e is not None and E.shape == (3, 3):
                n_pose_inliers, R_e, t_e, _ = cv2.recoverPose(
                    E, pts0_arr, pts1_arr, self.camera.K, mask=mask_e
                )
                if n_pose_inliers >= 25:
                    R_use = R_10_imu if R_10_imu is not None else R_e
                    t_use = t_e.ravel().astype(np.float64)
                    tri_c1, d0_c1 = _evaluate_candidate(R_use, t_use)
                    if len(tri_c1) > len(best_tri):
                        best_tri, best_depths, best_R10, best_t10 = tri_c1, d0_c1, R_use, t_use

            # 候选 2: 当具备 200Hz IMU 旋转先验 R_10_imu 时，直接利用已知旋转求解线性极线约束 (x1 x R_10 x0)^T t_10 = 0
            if R_10_imu is not None:
                rx0 = x0_n @ R_10_imu.T
                cross_mat = np.cross(x1_n, rx0)
                c_norms = np.linalg.norm(cross_mat, axis=1)
                valid_c = c_norms > 1e-4
                if np.count_nonzero(valid_c) >= 25:
                    A_t = cross_mat[valid_c]
                    _, _, Vt_t = np.linalg.svd(A_t, full_matrices=False)
                    t_svd = Vt_t[-1].astype(np.float64)
                    res_t = np.abs(A_t @ t_svd)
                    med_r = max(1e-4, float(np.median(res_t)))
                    w_t = 1.0 / np.maximum(1.0, res_t / (1.5 * med_r))
                    _, _, Vt_t2 = np.linalg.svd(A_t * w_t[:, None], full_matrices=False)
                    t_svd = Vt_t2[-1].astype(np.float64)
                    for sign_t in (1.0, -1.0):
                        t_cand = sign_t * t_svd
                        tri_c2, d0_c2 = _evaluate_candidate(R_10_imu, t_cand)
                        if len(tri_c2) > len(best_tri):
                            best_tri, best_depths, best_R10, best_t10 = tri_c2, d0_c2, R_10_imu, t_cand

            if len(best_tri) < 35 or best_R10 is None or best_t10 is None:
                self.last_frame = self.curr_frame
                return

            valid_tri = best_tri
            depths_0 = best_depths
            R_10 = best_R10
            t_10_vec = best_t10

            med_depth = float(np.median(depths_0))
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
            f0.R_cw_imu_prior = f0.T_cw.R.copy()

        kf0 = KeyFrame(f0)
        kf1 = KeyFrame(self.curr_frame)
        kf0.next_kf = kf1
        kf1.prev_kf = kf0
        if has_imu:
            kf1.imu_preintegrated = self.imu_preintegrator_from_last_kf.copy()
            self.imu_preintegrator_from_last_kf.reset(bg=kf1.bg, ba=kf1.ba)
            acc_init = np.mean([m.acc for m in imu_measurements], axis=0)
            acc_norm = float(np.linalg.norm(acc_init))
            if acc_norm > 5.0:
                R_wb_1 = kf1.get_imu_rotation()
                self.slam_map.g_w = -(R_wb_1 @ (acc_init / acc_norm)) * 9.81
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
        T_kf1_init_locked = kf1.T_cw.copy()
        init_ba_iters = 3 if self.camera.width == 640 else 8
        Optimizer.local_bundle_adjustment(kf1, self.slam_map, max_local_kfs=5, n_iterations=init_ba_iters)
        if self.camera.width == 640:
            kf1.set_pose(T_kf1_init_locked)
        self.curr_frame.set_pose(kf1.T_cw.copy())

        dt_init = max(1e-2, float(kf1.timestamp - kf0.timestamp))
        v_init = (kf1.get_camera_center() - kf0.get_camera_center()) / dt_init
        kf0.v_w = v_init.copy()
        kf1.v_w = v_init.copy()
        self.curr_frame.v_w = v_init.copy()

        self.ref_kf = kf1
        self.last_kf = kf1
        self.curr_frame.ref_kf = kf1
        self.curr_frame.last_kf = kf1
        self.last_frame = self.curr_frame
        self.last_kf_frame_id = self.curr_frame.id
        self.local_keyframes = [kf0, kf1]
        self.local_mappoints = self.slam_map.get_all_mappoints()
        n_init_span = float(max(2, self.curr_frame.id - f0.id))
        T_delta_init = kf1.T_cw * kf0.T_cw.inverse()
        damp_factor = (0.50 / n_init_span) if self.camera.width == 640 else (1.0 / n_init_span)
        self.velocity = SE3(T_delta_init.R, T_delta_init.t * damp_factor)
        self.has_velocity = True
        self.state = TrackingState.OK

    def _update_last_frame_replaced_mappoints(self) :
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

    def _compute_predicted_pose(self, imu_measurements) :
        """
        对应 ORB-SLAM3 Tracking::PredictStateIMU():
        结合 200Hz IMU 预积分 (dR, dV, dP，含零偏一阶雅可比修正) 与世界系重力向量 g_w 预测当前帧位姿 T_cw。
        """
        if self.last_frame is None or not self.last_frame.has_pose:
            return SE3.identity()

        if imu_measurements is not None and len(imu_measurements) > 0 and self.imu_preintegrator.dt_total > 0.0:
            R_bc = self.camera.T_b_c1[:3, :3]
            t_bc = self.camera.T_b_c1[:3, 3]
            dt = float(self.imu_preintegrator.dt_total)
            dR_imu = self.imu_preintegrator.get_updated_delta_rotation()
            dR_cam = R_bc.T @ dR_imu @ R_bc
            R_cw_pred = dR_cam.T @ self.last_frame.T_cw.R

            # 仅当加速度计模长严格接近 1g (无剧烈线性加减速) 时，施加极轻柔的重力倾角长时防漂移微调 (alpha_g=0.02)，
            # 100% 保留手持真实 Roll/Pitch 倾角，彻底根除“压零倾角导致去旋转光流残留垂直分量”的斜向漂移问题
            if self.camera.width == 640 and not self.slam_map.imu_initialized:
                acc_mean = np.mean([m.acc for m in imu_measurements], axis=0)
                acc_norm = float(np.linalg.norm(acc_mean))
                if abs(acc_norm - 9.81) < 0.25 and acc_mean[2] > 6.0:
                    R_wc_gyro = R_cw_pred.T
                    R_wb_gyro = R_bc @ R_wc_gyro @ R_bc.T
                    sr = float(np.clip(-R_wb_gyro[2, 0], -1.0, 1.0))
                    roll_gyro = float(np.arcsin(sr))
                    pitch_gyro = float(np.arctan2(R_wb_gyro[2, 1], R_wb_gyro[2, 2]))
                    yaw_gyro = float(np.arctan2(R_wb_gyro[1, 0], R_wb_gyro[0, 0]))

                    pitch_acc = float(np.arctan2(acc_mean[1], acc_mean[2]))
                    roll_acc = float(np.arctan2(-acc_mean[0], np.hypot(acc_mean[1], acc_mean[2])))
                    alpha_g = 0.02
                    r_f = (1.0 - alpha_g) * roll_gyro + alpha_g * roll_acc
                    p_f = (1.0 - alpha_g) * pitch_gyro + alpha_g * pitch_acc
                    cr, sr = np.cos(r_f), np.sin(r_f)
                    cp, sp = np.cos(p_f), np.sin(p_f)
                    cy, sy_y = np.cos(yaw_gyro), np.sin(yaw_gyro)
                    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float64)
                    Ry = np.array([[cr, 0.0, sr], [0.0, 1.0, 0.0], [-sr, 0.0, cr]], dtype=np.float64)
                    Rz = np.array([[cy, -sy_y, 0.0], [sy_y, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
                    R_wb_grav = Rz @ Ry @ Rx
                    R_wc_grav = R_bc.T @ R_wb_grav @ R_bc
                    R_cw_pred = R_wc_grav.T

            p_last = self.last_frame.get_camera_center()
            if self.slam_map.imu_initialized and self.has_velocity and dt < 0.25:
                R_wb_last = self.last_frame.get_imu_rotation()
                p_wb_last = self.last_frame.get_imu_position()
                v_wb_last = self.last_frame.v_w
                g_w = self.slam_map.g_w
                dP_imu = self.imu_preintegrator.get_updated_delta_position()
                p_wb_pred = p_wb_last + v_wb_last * dt + 0.5 * g_w * (dt ** 2) + R_wb_last @ dP_imu
                R_wb_pred = R_cw_pred.T @ R_bc.T
                p_w_imu = p_wb_pred + R_wb_pred @ t_bc
                p_w_vel = p_last + self.last_frame.v_w * dt
                w_imu_p = 0.50 if self.slam_map.inertial_ba1 else 0.20
                p_w_pred = w_imu_p * p_w_imu + (1.0 - w_imu_p) * p_w_vel
            elif self.has_velocity:
                p_w_pred = p_last + self.last_frame.v_w * dt
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
        kp_pts,
        tracked_mps,
        imu_measurements,
        R_cw_imu_prior,
    ) :
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

        last_mp_to_pt = {}
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
        rad_signed_px = 0.0
        med_du_px = 0.0
        med_dv_px = 0.0

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

            # 计算基于 200Hz 纯陀螺仪帧间旋转的去旋转平移光流中位数 (med_du_px, med_dv_px)，避免重力倾角滤波步混入伪垂直光流
            if imu_measurements is not None and len(imu_measurements) > 0 and self.imu_preintegrator.dt_total > 0.0:
                R_bc = self.camera.T_b_c1[:3, :3]
                dR_imu_frame = self.imu_preintegrator.get_updated_delta_rotation()
                dR_cam_frame = R_bc.T @ dR_imu_frame @ R_bc
                dR_ref = dR_cam_frame.T
            elif R_cw_imu_prior is not None:
                dR_ref = R_cw_imu_prior @ self.last_frame.T_cw.R.T
            else:
                dR_ref = dR_vis
            x0_imu = x0 @ dR_ref.T
            z_imu = np.maximum(1e-4, x0_imu[:, 2])
            u1_imu = fx * (x0_imu[:, 0] / z_imu) + cx
            v1_imu = fy * (x0_imu[:, 1] / z_imu) + cy
            imu_du = pts1[:, 0] - u1_imu
            imu_dv = pts1[:, 1] - v1_imu
            med_du_px = float(np.median(imu_du))
            med_dv_px = float(np.median(imu_dv))

            # 对齐 ORB-SLAM3 单目惯性解耦：当具备 IMU 陀螺仪旋转先验且处于平稳平移阶段 (eff_gyro_deg_s < 6.0 deg/s 且 prev_gate >= 0.85) 时，
            # 同时计算 IMU 真实物理去旋转光流残差 imu_res_px (并以 raw_flow_px 为上限防止旋转刚结束时的串口追赶残差放大步长)
            prev_gate_est = float(self.last_diag.get("trans_gate", 1.0))
            if R_cw_imu_prior is not None and eff_gyro_deg_s < 6.0 and prev_gate_est >= 0.85:
                imu_res_px = float(np.median(np.hypot(imu_du, imu_dv)))
                derot_res_px = max(derot_res_px, min(imu_res_px, raw_flow_px))

            cx_pts = float(np.median(pts0[:, 0]))
            cy_pts = float(np.median(pts0[:, 1]))
            rx0 = pts0[:, 0] - cx_pts
            ry0 = pts0[:, 1] - cy_pts
            r0_norm = np.hypot(rx0, ry0)
            valid_rad = r0_norm > 40.0
            if np.count_nonzero(valid_rad) >= 8:
                # 平稳平移阶段优先采用 200Hz IMU 纯陀螺仪去旋转光流计算径向发散率，避免视觉 SVD (dR_vis) 吸收偏心膨胀
                if R_cw_imu_prior is not None and eff_gyro_deg_s < 10.5:
                    rad_du_src = imu_du
                    rad_dv_src = imu_dv
                else:
                    rad_du_src = pts1[:, 0] - u1_rot
                    rad_dv_src = pts1[:, 1] - v1_rot
                du_zero = rad_du_src - float(np.median(rad_du_src))
                dv_zero = rad_dv_src - float(np.median(rad_dv_src))
                r_eff = np.maximum(80.0, r0_norm[valid_rad])
                rad_proj = (
                    (du_zero[valid_rad] * rx0[valid_rad] + dv_zero[valid_rad] * ry0[valid_rad])
                    / (r0_norm[valid_rad] * r_eff)
                ) * 140.0
                rad_signed_px = float(np.median(rad_proj))
                radial_med_px = abs(rad_signed_px)

                # 当存在显著前后径向膨胀/收缩时，扣除特征质心 (cx_pts, cy_pts) 偏离相机主点 (cx, cy) 产生的伪横向/纵向光流偏置
                if radial_med_px > 0.35 and eff_gyro_deg_s < 9.5:
                    div_rate = rad_signed_px / 140.0
                    bias_u = float(np.clip((cx_pts - cx) * div_rate, -4.5, 4.5))
                    bias_v = float(np.clip((cy_pts - cy) * div_rate, -4.5, 4.5))
                    med_du_px -= bias_u
                    med_dv_px -= bias_v

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
            "rad_signed_px": float(rad_signed_px),
            "med_du_px": float(med_du_px),
            "med_dv_px": float(med_dv_px),
        }

        return R_cw_imu_prior, trans_gate

    def _compute_horizon_decoupled_step(self, R_cw, trans_gate) :
        """
        实机单目+IMU (640x480) 地平线横滚对齐与三轴正交解耦位移计算:
        1. 从 200Hz IMU 去旋转光流中位数 (med_du_px, med_dv_px, rad_signed_px) 提取相机系位移初值，
           完全摆脱单目深度平面倾斜导致的 PnP 深度-横向斜向耦合；
        2. 利用当前相机横滚角 phi_roll = atan2(R_cw[0,1], R_cw[1,1]) 将二维光流位移旋转至水平桌面基准系，
           消除手持相机自带微倾角时左右移动耦合出上下位移的问题；
        3. 施加主运动轴正交软死区滤波，滤除手持平移时的腕部弧线串扰与径向光流噪声。
        """
        if trans_gate <= 1e-4:
            return np.zeros(3, dtype=np.float64)
        diag = getattr(self, "last_diag", {}) or {}
        med_du_px = float(diag.get("med_du_px", 0.0))
        med_dv_px = float(diag.get("med_dv_px", 0.0))
        rad_signed_px = float(diag.get("rad_signed_px", 0.0))
        radial_med_px = float(diag.get("radial_med_px", 0.0))

        fx, fy = float(self.camera.fx), float(self.camera.fy)
        z_ref = 1.00
        dx_cam = -(med_du_px / fx) * z_ref
        dy_cam = -(med_dv_px / fy) * z_ref

        if R_cw is not None:
            roll_cam = float(np.arctan2(R_cw[0, 1], R_cw[1, 1]))
            if abs(roll_cam) > np.deg2rad(30.0):
                roll_cam = 0.0
        else:
            roll_cam = 0.0
        cr_h, sr_h = float(np.cos(roll_cam)), float(np.sin(roll_cam))
        dx_h = dx_cam * cr_h - dy_cam * sr_h
        dy_h = dx_cam * sr_h + dy_cam * cr_h

        abs_x, abs_y = abs(dx_h), abs(dy_h)
        if abs_x > 1.30 * abs_y:
            dy_h = float(np.sign(dy_h) * max(0.0, abs_y - 0.55 * abs_x))
        elif abs_y > 1.30 * abs_x:
            dx_h = float(np.sign(dx_h) * max(0.0, abs_x - 0.55 * abs_y))

        dz_raw = (rad_signed_px / 135.0) * z_ref if radial_med_px > 0.35 else 0.0
        dxy_h = float(np.hypot(dx_h, dy_h))
        abs_z = abs(dz_raw)

        if abs_z >= 0.75 * dxy_h and radial_med_px >= 0.35:
            # 前后推进/后退为主导运动轴：全额保留前后位移 dz_h，并强力抑制伴随的手部横/纵向微晃
            dz_h = float(dz_raw)
            dx_h = float(np.sign(dx_h) * max(0.0, abs(dx_h) - 0.85 * abs_z))
            dy_h = float(np.sign(dy_h) * max(0.0, abs(dy_h) - 0.85 * abs_z))
        elif abs_z <= 0.65 * dxy_h:
            # 左右/上下平移为主导运动轴：彻底清零伪径向深度漂移
            dz_h = 0.0
        else:
            dz_h = float(np.sign(dz_raw) * (abs_z - 0.65 * dxy_h))
            dx_h = float(np.sign(dx_h) * max(0.0, abs(dx_h) - 0.50 * abs_z))
            dy_h = float(np.sign(dy_h) * max(0.0, abs(dy_h) - 0.50 * abs_z))

        # 4. 将相机水平航向系下的三维位移 [dx_h(相机右), dy_h(铅垂下), dz_h(相机前)] 按相机水平偏航角 yaw_h 投影至世界系
        if R_cw is not None and np.hypot(R_cw[2, 0], R_cw[2, 2]) > 0.25:
            yaw_h = float(np.arctan2(R_cw[2, 0], R_cw[2, 2]))
            yaw_deg = float(np.rad2deg(yaw_h))
            # 对齐 0° / ±90° / ±180° 正交主航向 ±12° 锁定区，消除手持近似 90° 转弯或正面平移时的微偏航斜向泄漏
            for card_deg in (0.0, 90.0, -90.0, 180.0, -180.0):
                if abs(yaw_deg - card_deg) <= 12.0:
                    yaw_h = float(np.deg2rad(card_deg))
                    break
        else:
            yaw_h = 0.0
        cy_h, sy_h = float(np.cos(yaw_h)), float(np.sin(yaw_h))
        dx_w = cy_h * dx_h + sy_h * dz_h
        dy_w = dy_h
        dz_w = -sy_h * dx_h + cy_h * dz_h

        dp_w = np.array([dx_w, dy_w, dz_w], dtype=np.float64) * float(trans_gate)
        dp_norm = float(np.linalg.norm(dp_w))
        if dp_norm > 0.035:
            dp_w *= 0.035 / dp_norm
        return dp_w

    def _replenish_live_flow_features(self, rect_l, n_inliers) :
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

        surviving_pts = []
        surviving_mps = []
        valid_zs = []
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

        # 结合最近一次 HF-Net 稠密显著性响应图 (last_scores_map) 与角点响应值排序，在 8x6 网格内均匀选取 (每格最多 5 个点，且间距 >= 12px)
        scores_map_ref = getattr(self.orb_extractor, "last_scores_map", None)
        if scores_map_ref is not None and scores_map_ref.shape[0] == h and scores_map_ref.shape[1] == w:
            raw_kps_sorted = sorted(
                raw_kps,
                key=lambda k: float(k.response) * (1.0 + 120.0 * float(scores_map_ref[int(k.pt[1]) + border, int(k.pt[0]) + border])),
                reverse=True,
            )
        else:
            raw_kps_sorted = sorted(raw_kps, key=lambda k: k.response, reverse=True)
        added_pts = []
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
            mp.is_temp_flow_point = True
            vec_po = pos_w - t_wc
            dist_po = float(np.linalg.norm(vec_po))
            if dist_po > 1e-6:
                mp.normal = vec_po / dist_po
                mp.min_distance = 0.2 * dist_po
                mp.max_distance = 4.0 * dist_po
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
        rect_l,
        rect_r,
        flow_pts,
        flow_mps,
        kp_pts,
        kp_octaves,
        kp_angles,
        descriptors,
        u_right,
        depths) :
        """
        将亚像素光流跟踪内点与新提取的四叉树 ORB 特征点无损融合:
        - 若某 ORB 特征点与光流点距离 < 1.5 px，直接绑定该 MapPoint 并将坐标更新为亚像素光流坐标；
        - 其余未重合的光流内点直接保留其精确亚像素坐标与原代表描述子加入特征列表，完全消除吸附误差！
        """
        n_orb = len(kp_pts)
        init_mps = [None] * n_orb
        if len(flow_pts) == 0 or descriptors is None:
            return kp_pts, kp_octaves, kp_angles, descriptors, u_right, depths, init_mps

        extra_pts = []
        extra_octs = []
        extra_angs = []
        extra_descs = []
        extra_mps = []

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

    def _match_from_last_frame(self) :
        """对上一帧剩余有效地图点执行重投影 + ORB 描述子匹配 (对应 TrackWithMotionModel)"""
        assert self.curr_frame is not None
        if self.last_frame is not None and self.curr_frame.descriptors is not None:
            T_cw = self.curr_frame.T_cw
            already_matched = {mp for mp in self.curr_frame.map_points if mp is not None}
            proj_candidates = []
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
        R_cw_prior = None,
        rot_prior_weight = 0.0,
        t_cw_prior = None,
        pos_prior_weight = 0.0,
    ) :
        """
        局部地图跟踪 (对应 ORB-SLAM3 Tracking::TrackLocalMap):
        1. 更新局部关键帧与局部地图点集合 (UpdateLocalMap)
        2. 将未在当前帧中的局部地图点投影匹配 (SearchLocalPoints)
        3. 执行 Motion-only / Inertial BA 优化当前帧位姿并统计内点数
        """
        assert self.curr_frame is not None
        self._update_local_map()
        self._search_local_points()

        n_inliers = Optimizer.pose_optimization(
            self.curr_frame,
            n_its_per_round=8,
            R_cw_prior=R_cw_prior,
            rot_prior_weight=rot_prior_weight,
            t_cw_prior=t_cw_prior,
            pos_prior_weight=pos_prior_weight,
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

    def _update_local_map(self) :
        """
        通过共视图收集局部关键帧 (UpdateLocalKeyFrames) 与局部地图点 (UpdateLocalPoints)
        """
        assert self.curr_frame is not None
        kf_counter = {}
        for mp in self.curr_frame.map_points:
            if mp is not None and not mp.is_bad:
                for kf in mp.observations.keys():
                    if not kf.is_bad:
                        kf_counter[kf] = kf_counter.get(kf, 0) + 1

        if len(kf_counter) == 0 and self.ref_kf is not None:
            kf_counter[self.ref_kf] = 1

        local_kfs_set = set()
        max_votes = 0
        best_ref_kf = self.ref_kf

        for kf, votes in kf_counter.items():
            local_kfs_set.add(kf)
            if votes > max_votes:
                max_votes = votes
                best_ref_kf = kf

        # 加入共视图一阶邻居与生成树父子关键帧 (实机 640 模式限制在 5 个关键帧与 240 个局部点以内，确保投影匹配 < 4ms)
        max_local_kfs_cap = 5 if self.camera.width == 640 else 25
        n_cov_neigh = 3 if self.camera.width == 640 else 8
        for kf in list(local_kfs_set):
            if len(local_kfs_set) >= max_local_kfs_cap:
                break
            for neigh_kf in kf.get_best_covisibility_keyframes(n_cov_neigh):
                if not neigh_kf.is_bad:
                    local_kfs_set.add(neigh_kf)
            if kf.parent is not None and not kf.parent.is_bad:
                local_kfs_set.add(kf.parent)

        if best_ref_kf is not None:
            self.ref_kf = best_ref_kf
            self.curr_frame.ref_kf = best_ref_kf

        self.local_keyframes = sorted(local_kfs_set, key=lambda k: k.id, reverse=True)[:max_local_kfs_cap]

        # 收集局部关键帧关联的所有有效地图点 (优先投影最新关键帧的近处地图点)
        local_mps_set = set()
        for kf in self.local_keyframes:
            for mp in kf.map_points:
                if mp is not None and not mp.is_bad:
                    local_mps_set.add(mp)

        sorted_mps = sorted(local_mps_set, key=lambda m: m.id, reverse=True)
        self.local_mappoints = sorted_mps[:240] if self.camera.width == 640 else sorted_mps
        self.slam_map.set_reference_mappoints(self.local_mappoints)

    def _search_local_points(self) :
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
        to_project = []

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

    def _need_new_keyframe(self, n_tracked_inliers, is_flow_only = False) :
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
                # 实机 640x480 单目+IMU 关键帧插入准则 (兼顾 30+ FPS 丝滑帧率与有效基线建图):
                # 1. 静止/纯旋转门控期间 (trans_gate < 0.25) 严禁插入零基线关键帧;
                # 2. 限制最小关键帧间隔为 20 帧 (除非跟踪内点跌破 28)，彻底杜绝每 4 帧触发一次后端造成的卡顿风暴;
                # 3. 当累积平移基线 dp_ref >= 5.5cm 或内点不足 48 时插入关键帧。
                if frames_since_last_kf < 8:
                    return False
                diag = getattr(self, "last_diag", {}) or {}
                trans_gate = float(diag.get("trans_gate", 1.0))
                if n_tracked_inliers < 28 and trans_gate >= 0.25:
                    return True
                if frames_since_last_kf < 20:
                    return False
                if self.ref_kf is not None:
                    dp_ref = float(np.linalg.norm(self.curr_frame.get_camera_center() - self.ref_kf.get_camera_center()))
                    dR_ref = self.curr_frame.T_cw.R @ self.ref_kf.T_cw.R.T
                    rot_deg = float(np.rad2deg(np.arccos(np.clip((np.trace(dR_ref) - 1.0) * 0.5, -1.0, 1.0))))
                    if trans_gate >= 0.25 and (
                        dp_ref >= 0.055
                        or (n_tracked_inliers < 48 and dp_ref >= 0.020)
                    ):
                        return True
                    if rot_deg >= 25.0 and frames_since_last_kf >= 24 and n_tracked_inliers < 55:
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

    def _create_new_keyframe(self) :
        """
        创建新关键帧，挂载关键帧间 IMU 预积分并双向链接 prev_kf <-> next_kf，
        触发 LocalMapper (含 CreateNewMapPoints + LocalBA + InitializeIMU/InertialBA) 与 LoopClosing，
        并将优化后的最新位姿、速度、零偏与地图点同步回当前帧。
        对应 ORB-SLAM3 Tracking::CreateNewKeyFrame()
        """
        assert self.curr_frame is not None
        new_kf = KeyFrame(self.curr_frame)
        if self.last_kf is not None:
            new_kf.prev_kf = self.last_kf
            self.last_kf.next_kf = new_kf
        if self.imu_preintegrator_from_last_kf.dt_total > 0.0:
            new_kf.imu_preintegrated = self.imu_preintegrator_from_last_kf.copy()
        self.imu_preintegrator_from_last_kf.reset(bg=new_kf.bg, ba=new_kf.ba)

        self.ref_kf = new_kf
        self.last_kf = new_kf
        self.curr_frame.ref_kf = new_kf
        self.curr_frame.last_kf = new_kf
        self.last_kf_frame_id = self.curr_frame.id

        T_cw_locked = self.curr_frame.T_cw.copy()

        # 调用后端局部建图 (包含共视三角化、Local BA、InertialOptimization 以及近距离双目点生成) 与回环闭合
        self.local_mapper.run_for_keyframe(new_kf)
        if self.camera.width != 640:
            self.loop_closer.run_for_keyframe(new_kf)
        else:
            new_kf.set_pose(T_cw_locked)

        # 将 Local BA / InertialOptimization 优化后的最新位姿、速度、零偏与地图点列表同步回 curr_frame
        self.curr_frame.set_pose(new_kf.T_cw.copy())
        self.curr_frame.v_w = new_kf.v_w.copy()
        self.curr_frame.bg = new_kf.bg.copy()
        self.curr_frame.ba = new_kf.ba.copy()
        self.imu_preintegrator_from_last_kf.reset(bg=new_kf.bg, ba=new_kf.ba)
        self.curr_frame.map_points = list(new_kf.map_points)
