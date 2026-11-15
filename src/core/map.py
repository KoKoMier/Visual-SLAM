# -*- coding: utf-8 -*-
"""
core/map.py
===========
全局/活跃子地图容器类 (Map)
严格 1:1 对齐 ORB-SLAM3 C++ Map.h / Map.cc。
维护所有活跃 KeyFrame、MapPoint 以及当前局部参考地图点 (reference_mappoints)。
"""

# Python 3.6 compatible
from typing import TYPE_CHECKING
import numpy as np

from .lie_algebra import SE3

if TYPE_CHECKING:
    from .keyframe import KeyFrame
    from .mappoint import MapPoint


class Map:
    _next_id = 0

    __slots__ = (
        "id", "keyframes", "mappoints",
        "reference_mappoints", "max_kf_id",
        "big_change_idx",
        "imu_initialized", "inertial_ba1", "inertial_ba2", "g_w"
    )

    def __init__(self):
        self.id = Map._next_id
        Map._next_id += 1

        self.keyframes = set()
        self.mappoints = set()
        self.reference_mappoints = []
        self.max_kf_id = 0
        self.big_change_idx = 0
        self.imu_initialized = False
        self.inertial_ba1 = False
        self.inertial_ba2 = False
        self.g_w = np.array([0.0, 0.0, -9.81], dtype=np.float64)

    def add_keyframe(self, kf) :
        self.keyframes.add(kf)
        if kf.id > self.max_kf_id:
            self.max_kf_id = kf.id

    def add_mappoint(self, mp) :
        self.mappoints.add(mp)

    def erase_mappoint(self, mp) :
        self.mappoints.discard(mp)

    def erase_keyframe(self, kf) :
        self.keyframes.discard(kf)

    def set_reference_mappoints(self, mps) :
        self.reference_mappoints = list(mps)

    def inform_new_big_change(self) :
        self.big_change_idx += 1

    def is_imu_initialized(self) :
        return bool(self.imu_initialized)

    def set_imu_initialized(self, val = True) :
        self.imu_initialized = bool(val)

    def is_inertial_ba1(self) :
        return bool(self.inertial_ba1)

    def set_inertial_ba1(self, val = True) :
        self.inertial_ba1 = bool(val)

    def is_inertial_ba2(self) :
        return bool(self.inertial_ba2)

    def set_inertial_ba2(self, val = True) :
        self.inertial_ba2 = bool(val)

    def get_all_keyframes(self) :
        return sorted(
            [kf for kf in self.keyframes if not kf.is_bad],
            key=lambda k: k.id
        )

    def get_all_mappoints(self) :
        return sorted(
            [mp for mp in self.mappoints if not mp.is_bad],
            key=lambda m: m.id
        )

    def num_keyframes(self) :
        return len(self.keyframes)

    def num_mappoints(self) :
        return len(self.mappoints)

    def clear(self) :
        self.keyframes.clear()
        self.mappoints.clear()
        self.reference_mappoints.clear()
        self.max_kf_id = 0
        self.big_change_idx = 0
        self.imu_initialized = False
        self.inertial_ba1 = False
        self.inertial_ba2 = False
        self.g_w = np.array([0.0, 0.0, -9.81], dtype=np.float64)

    def apply_scaled_rotation(self, R_yw, scale, b_scaled_vel = True) :
        """
        对整个地图的所有 KeyFrame 位姿、速度与 MapPoint 坐标执行重力对齐旋转与尺度缩放
        严格对应 ORB-SLAM3 Map::ApplyScaledRotation(const Sophus::SE3f &T, const float s, const bool bScaledVel)
        """
        s = float(scale)
        R_yw = np.asarray(R_yw, dtype=np.float64).reshape(3, 3)

        for kf in self.keyframes:
            if kf.is_bad:
                continue
            T_wc = kf.get_pose_inverse()
            R_yc = R_yw @ T_wc.R
            t_yc = s * (R_yw @ T_wc.t)
            T_cy = SE3(R_yc, t_yc).inverse()
            kf.set_pose(T_cy)
            if kf.R_cw_imu_prior is not None:
                kf.R_cw_imu_prior = kf.R_cw_imu_prior @ R_yw.T
            if b_scaled_vel:
                kf.v_w = s * (R_yw @ kf.v_w)
            else:
                kf.v_w = R_yw @ kf.v_w

        for mp in self.mappoints:
            if mp.is_bad:
                continue
            mp.pos_w = s * (R_yw @ mp.pos_w)
            mp.update_normal_and_depth()

        self.big_change_idx += 1
