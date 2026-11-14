# -*- coding: utf-8 -*-
"""
core/mappoint.py
================
3D 地图点类 (MapPoint)
严格 1:1 对齐 ORB-SLAM3 C++ MapPoint.h / MapPoint.cc。

核心职责:
1. 维护地图点在世界坐标系下的 3D 坐标 pos_w (3,)。
2. 维护观测到该点的所有关键帧及对应特征点索引 observations: dict[KeyFrame, int]。
3. 计算最具代表性的 ORB 描述子 (ComputeDistinctiveDescriptors: 最小中值汉明距离)。
4. 更新平均观测方向向量 normal 与有效尺度观测距离范围 [min_distance, max_distance]。
5. 跟踪可见性统计 (visible_count / found_count) 用于局部建图剔除劣质点。
"""

# Python 3.6 compatible
from typing import TYPE_CHECKING
import numpy as np
import cv2

if TYPE_CHECKING:
    from .keyframe import KeyFrame
    from .map import Map

# 模块级预计算 0~255 字节 popcount 查表，避免每次调用 compute_distinctive_descriptors 时重复构建
_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)


class MapPoint:
    _next_id = 0

    __slots__ = (
        "id", "pos_w", "normal", "descriptor",
        "ref_kf", "first_kf_id", "first_frame_id",
        "observations", "visible_count", "found_count",
        "min_distance", "max_distance",
        "is_bad", "is_temp_flow_point", "replaced_by", "track_proj_x", "track_proj_y",
        "track_proj_xr", "track_in_view", "track_scale_level",
        "track_view_cos", "last_frame_seen"
    )

    def __init__(
        self,
        pos_w,
        ref_kf = None,
        first_frame_id = -1,
        descriptor = None
    ):
        self.id = MapPoint._next_id
        MapPoint._next_id += 1

        self.pos_w = np.asarray(pos_w, dtype=np.float64).reshape(3).copy()
        self.normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.descriptor = descriptor.copy() if descriptor is not None else None

        self.ref_kf = ref_kf
        self.first_kf_id = ref_kf.id if ref_kf is not None else -1
        self.first_frame_id = int(first_frame_id)

        # 观测字典: KeyFrame -> 特征点索引 idx
        self.observations = {}
        self.visible_count = 1
        self.found_count = 1

        self.min_distance = 0.0
        self.max_distance = float("inf")

        self.is_bad = False
        self.is_temp_flow_point = False
        self.replaced_by = None

        # 前端局部地图投影缓存变量 (对应 ORB-SLAM3 mTrackProjX 等)
        self.track_proj_x = 0.0
        self.track_proj_y = 0.0
        self.track_proj_xr = 0.0
        self.track_in_view = False
        self.track_scale_level = 0
        self.track_view_cos = 1.0
        self.last_frame_seen = -1

    def set_world_pos(self, pos_w) :
        self.pos_w = np.asarray(pos_w, dtype=np.float64).reshape(3).copy()

    def get_world_pos(self) :
        return self.pos_w

    def num_observations(self) :
        return len(self.observations)

    def add_observation(self, kf, idx) :
        """记录关键帧 kf 的第 idx 个特征点观测到了本 MapPoint"""
        if self.is_bad:
            return
        if kf not in self.observations:
            self.observations[kf] = int(idx)
            if self.ref_kf is None:
                self.ref_kf = kf

    def erase_observation(self, kf, slam_map = None) :
        """移除关键帧观测；若有效观测少于 2 个关键帧且非刚生成的双目点，标记为 bad"""
        if kf in self.observations:
            del self.observations[kf]
            if self.ref_kf == kf and len(self.observations) > 0:
                self.ref_kf = next(iter(self.observations.keys()))
            if len(self.observations) <= 1:
                self.set_bad_flag(slam_map)

    def set_bad_flag(self, slam_map = None) :
        """销毁劣质地图点并通知所有观测关键帧清除引用"""
        self.is_bad = True
        obs_copy = list(self.observations.items())
        self.observations.clear()
        for kf, idx in obs_copy:
            kf.erase_mappoint_match(idx)
        if slam_map is not None:
            slam_map.erase_mappoint(self)

    def replace(self, other, slam_map = None) :
        """在局部建图或回环融合时，将当前点合并到 other 地图点"""
        if other.id == self.id:
            return
        obs_copy = list(self.observations.items())
        self.observations.clear()
        self.is_bad = True
        self.replaced_by = other

        for kf, idx in obs_copy:
            if not other.is_in_keyframe(kf):
                kf.replace_mappoint_match(idx, other)
                other.add_observation(kf, idx)
            else:
                kf.erase_mappoint_match(idx)

        other.increase_found(self.found_count)
        other.increase_visible(self.visible_count)
        other.compute_distinctive_descriptors()
        if slam_map is not None:
            slam_map.erase_mappoint(self)

    def get_replaced(self) :
        """
        对应 ORB-SLAM3 MapPoint::GetReplaced():
        返回当前地图点被融合替换后的最终目标 MapPoint（支持传递替换链解析）。
        """
        mp = self.replaced_by
        while mp is not None and mp.replaced_by is not None and mp.replaced_by.id != mp.id:
            mp = mp.replaced_by
        return mp

    def is_in_keyframe(self, kf) :
        return kf in self.observations

    def get_index_in_keyframe(self, kf) :
        return self.observations.get(kf, -1)

    def increase_visible(self, n = 1) :
        self.visible_count += int(n)

    def increase_found(self, n = 1) :
        self.found_count += int(n)

    def get_found_ratio(self) :
        if self.visible_count == 0:
            return 0.0
        return float(self.found_count) / float(self.visible_count)

    def compute_distinctive_descriptors(self) :
        """
        计算最具代表性的 ORB 描述子 (对应 ORB-SLAM3 MapPoint::ComputeDistinctiveDescriptors):
        收集所有观测关键帧中对应的 256-bit ORB 描述子，计算两两汉明距离矩阵，
        选择与其他描述子中值距离最小的那一个作为该地图点的代表描述子。
        """
        if self.is_bad or len(self.observations) == 0:
            return
        descs = []
        for kf, idx in self.observations.items():
            if not kf.is_bad and kf.descriptors is not None and idx < len(kf.descriptors):
                descs.append(kf.descriptors[idx])
        if len(descs) == 0:
            return
        if len(descs) <= 2:
            # 1 或 2 个观测时两两距离对称，直接取首个描述子即可，省去 unpackbits 开销
            self.descriptor = descs[0].copy()
            return

        desc_mat = np.asarray(descs, dtype=np.uint8)
        n = len(desc_mat)
        # 利用模块级 popcount 查表快速计算 N x N 汉明距离矩阵
        xor_mat = desc_mat[:, None, :] ^ desc_mat[None, :, :]  # (N, N, 32)
        dist_mat = _POPCOUNT_LUT[xor_mat].sum(axis=-1)  # (N, N)
        dist_sorted = np.sort(dist_mat, axis=1)
        median_dists = dist_sorted[:, n // 2]
        best_idx = int(np.argmin(median_dists))
        self.descriptor = desc_mat[best_idx].copy()

    def update_normal_and_depth(self) :
        """
        更新地图点平均观测方向向量 normal 与金字塔尺度距离范围
        对应 ORB-SLAM3 MapPoint::UpdateNormalAndDepth()
        """
        if self.is_bad or len(self.observations) == 0 or self.ref_kf is None:
            return
        normals = []
        for kf, _ in self.observations.items():
            if not kf.is_bad:
                o_w = kf.get_camera_center()
                v = self.pos_w - o_w
                norm_v = float(np.linalg.norm(v))
                if norm_v > 1e-8:
                    normals.append(v / norm_v)
        if len(normals) > 0:
            mean_n = np.mean(normals, axis=0)
            norm_n = float(np.linalg.norm(mean_n))
            if norm_n > 1e-8:
                self.normal = mean_n / norm_n

        # 计算参考关键帧下的观测距离与金字塔尺度上下界
        ref_center = self.ref_kf.get_camera_center()
        dist = float(np.linalg.norm(self.pos_w - ref_center))
        ref_idx = self.observations.get(self.ref_kf, 0)
        level = int(self.ref_kf.kp_octaves[ref_idx]) if ref_idx < len(self.ref_kf.kp_octaves) else 0
        scale_factor = self.ref_kf.scale_factors[level]
        n_levels = len(self.ref_kf.scale_factors)
        max_scale = self.ref_kf.scale_factors[-1]

        self.max_distance = dist * scale_factor
        self.min_distance = self.max_distance / max_scale

    def predict_scale(self, current_dist, n_levels = 8, log_scale_factor = np.log(1.2)) :
        """根据当前距离预测特征点在图像金字塔中的最佳层级 (0 ~ n_levels-1)"""
        if current_dist <= 1e-6 or self.max_distance <= 1e-6:
            return 0
        ratio = self.max_distance / current_dist
        level = int(np.ceil(np.log(ratio) / log_scale_factor))
        return int(np.clip(level, 0, n_levels - 1))
