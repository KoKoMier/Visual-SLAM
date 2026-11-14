# -*- coding: utf-8 -*-
"""
core/frame.py
=============
普通图像帧类 (Frame)
严格 1:1 对齐 ORB-SLAM3 C++ Frame.h / Frame.cc。

核心设计 (针对 RK3566 与 ORB-SLAM3 对齐):
1. 维护 64x48 空间网格索引 (FRAME_GRID_COLS=64, FRAME_GRID_ROWS=48)，
   使局部地图投影匹配 get_features_in_area(x, y, r) 时间复杂度降至 O(1)。
2. 支持混合前端特征模式:
   - 普通帧快速路径: 仅维护 2D 关键点坐标 kp_pts (N, 2) 与双目深度 depths (N,) / 右目横坐标 u_right (N,)。
   - 关键帧路径: 额外维护四叉树 ORB 描述子 descriptors (N, 32)、金字塔层级 kp_octaves (N,) 与方向角。
3. 维护相机位姿 T_cw (世界到相机 SE3) 及 IMU 状态 (速度 v_w, 陀螺仪零偏 bg, 加速度计零偏 ba)。
"""

# Python 3.6 compatible
from typing import TYPE_CHECKING
import numpy as np

from .lie_algebra import SE3

if TYPE_CHECKING:
    from .camera import StereoCamera
    from .mappoint import MapPoint
    from .keyframe import KeyFrame


class Frame:
    _next_id = 0
    FRAME_GRID_COLS = 64
    FRAME_GRID_ROWS = 48

    __slots__ = (
        "id", "timestamp", "camera",
        "img_left", "img_right",
        "kp_pts", "kp_octaves", "kp_angles", "descriptors",
        "u_right", "depths",
        "num_kps",
        "map_points", "outliers",
        "T_cw", "has_pose",
        "v_w", "bg", "ba", "R_cw_imu_prior",
        "ref_kf",
        "scale_factors", "inv_scale_factors",
        "level_sigma2", "inv_level_sigma2",
        "grid_element_width_inv", "grid_element_height_inv",
        "grid"
    )

    def __init__(
        self,
        timestamp,
        camera,
        img_left = None,
        img_right = None,
        kp_pts = None,
        kp_octaves = None,
        kp_angles = None,
        descriptors = None,
        u_right = None,
        depths = None,
        scale_factors = None
    ):
        self.id = Frame._next_id
        Frame._next_id += 1

        self.timestamp = float(timestamp)
        self.camera = camera
        self.img_left = img_left
        self.img_right = img_right

        if kp_pts is None or len(kp_pts) == 0:
            self.kp_pts = np.zeros((0, 2), dtype=np.float32)
            self.num_kps = 0
        else:
            self.kp_pts = np.asarray(kp_pts, dtype=np.float32).reshape(-1, 2)
            self.num_kps = len(self.kp_pts)

        if kp_octaves is None:
            self.kp_octaves = np.zeros(self.num_kps, dtype=np.int32)
        else:
            self.kp_octaves = np.asarray(kp_octaves, dtype=np.int32).reshape(self.num_kps)

        if kp_angles is None:
            self.kp_angles = np.zeros(self.num_kps, dtype=np.float32)
        else:
            self.kp_angles = np.asarray(kp_angles, dtype=np.float32).reshape(self.num_kps)

        self.descriptors = descriptors

        if u_right is None:
            self.u_right = np.full(self.num_kps, -1.0, dtype=np.float32)
        else:
            self.u_right = np.asarray(u_right, dtype=np.float32).reshape(self.num_kps)

        if depths is None:
            self.depths = np.full(self.num_kps, -1.0, dtype=np.float32)
        else:
            self.depths = np.asarray(depths, dtype=np.float32).reshape(self.num_kps)

        self.map_points = [None] * self.num_kps
        self.outliers = np.zeros(self.num_kps, dtype=bool)

        # 位姿与 IMU 运动状态
        self.T_cw = SE3.identity()
        self.has_pose = False
        self.v_w = np.zeros(3, dtype=np.float64)
        self.bg = np.zeros(3, dtype=np.float64)
        self.ba = np.zeros(3, dtype=np.float64)
        self.R_cw_imu_prior = None
        self.ref_kf = None

        # 金字塔尺度因子与不确定度权重 (ORB-SLAM3 标准 8 层金字塔, 缩放比 1.2)
        if scale_factors is None:
            self.scale_factors = np.array([1.2 ** i for i in range(8)], dtype=np.float64)
        else:
            self.scale_factors = np.asarray(scale_factors, dtype=np.float64)
        self.inv_scale_factors = 1.0 / self.scale_factors
        self.level_sigma2 = self.scale_factors ** 2
        self.inv_level_sigma2 = 1.0 / self.level_sigma2

        # 64 x 48 空间加速网格 (对应 Frame::AssignFeaturesToGrid)
        self.grid_element_width_inv = float(self.FRAME_GRID_COLS) / float(self.camera.width)
        self.grid_element_height_inv = float(self.FRAME_GRID_ROWS) / float(self.camera.height)
        self.grid = [
            [[] for _ in range(self.FRAME_GRID_ROWS)]
            for _ in range(self.FRAME_GRID_COLS)
        ]
        self.assign_features_to_grid()

    def assign_features_to_grid(self) :
        """将当前帧所有特征点分配至 64x48 空间网格桶中"""
        for c in range(self.FRAME_GRID_COLS):
            for r in range(self.FRAME_GRID_ROWS):
                self.grid[c][r].clear()

        for i in range(self.num_kps):
            u, v = float(self.kp_pts[i, 0]), float(self.kp_pts[i, 1])
            gx = int(np.floor(u * self.grid_element_width_inv))
            gy = int(np.floor(v * self.grid_element_height_inv))
            if 0 <= gx < self.FRAME_GRID_COLS and 0 <= gy < self.FRAME_GRID_ROWS:
                self.grid[gx][gy].append(i)

    def set_pose(self, T_cw) :
        self.T_cw = T_cw.copy()
        self.has_pose = True

    def get_camera_center(self) :
        """返回相机光心在世界坐标系下的位置 O_w = -R_cw^T * t_cw"""
        return -(self.T_cw.R.T @ self.T_cw.t)

    def get_pose_inverse(self) :
        """返回 T_wc (相机到世界变换)"""
        return self.T_cw.inverse()

    def unproject_stereo(self, i) :
        """
        将第 i 个具有有效双目深度的特征点反投影到世界坐标系 P_w (3,)
        对应 ORB-SLAM3 Frame::UnprojectStereo(i)
        """
        z = float(self.depths[i])
        if z <= 0.0:
            return None
        u, v = float(self.kp_pts[i, 0]), float(self.kp_pts[i, 1])
        p_c = self.camera.unproject_depth(u, v, z)
        T_wc = self.T_cw.inverse()
        return T_wc.transform(p_c)

    def get_features_in_area(
        self,
        x,
        y,
        r,
        min_level = -1,
        max_level = -1
    ) :
        """
        在以 (x, y) 为中心、半径为 r 的圆形区域内快速检索候选特征点索引
        对应 ORB-SLAM3 Frame::GetFeaturesInArea()
        """
        if r <= 0.0 or self.num_kps == 0:
            return []

        min_cell_x = max(0, int(np.floor((x - r) * self.grid_element_width_inv)))
        if min_cell_x >= self.FRAME_GRID_COLS:
            return []
        max_cell_x = min(
            self.FRAME_GRID_COLS - 1,
            int(np.ceil((x + r) * self.grid_element_width_inv))
        )
        if max_cell_x < 0:
            return []

        min_cell_y = max(0, int(np.floor((y - r) * self.grid_element_height_inv)))
        if min_cell_y >= self.FRAME_GRID_ROWS:
            return []
        max_cell_y = min(
            self.FRAME_GRID_ROWS - 1,
            int(np.ceil((y + r) * self.grid_element_height_inv))
        )
        if max_cell_y < 0:
            return []

        check_levels = (min_level >= 0) or (max_level >= 0)
        r_sq = r * r
        indices = []

        for ix in range(min_cell_x, max_cell_x + 1):
            for iy in range(min_cell_y, max_cell_y + 1):
                cell = self.grid[ix][iy]
                for idx in cell:
                    if check_levels:
                        lvl = int(self.kp_octaves[idx])
                        if min_level >= 0 and lvl < min_level:
                            continue
                        if max_level >= 0 and lvl > max_level:
                            continue
                    dx = float(self.kp_pts[idx, 0]) - x
                    dy = float(self.kp_pts[idx, 1]) - y
                    if dx * dx + dy * dy <= r_sq:
                        indices.append(idx)
        return indices
