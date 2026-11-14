# -*- coding: utf-8 -*-
"""
core/keyframe.py
================
关键帧与共视图节点类 (KeyFrame)
严格 1:1 对齐 ORB-SLAM3 C++ KeyFrame.h / KeyFrame.cc。

核心机制:
1. 共视图 (Covisibility Graph):
   - connected_kf_weights: dict[KeyFrame, int] 记录与其它关键帧共享的 MapPoint 数量。
   - ordered_connected_kfs: 按共视权重从大到小排序的邻居关键帧列表 (权重 >= 15)。
2. 生成树 (Spanning Tree):
   - 维护 parent (共视权重最大的历史关键帧) 与 children 集合，构成最小生成树主干。
3. 回环边 (Loop Edges):
   - 维护回环闭合时新增的强约束边 loop_edges，与生成树及高共视边(>=100)共同构成本质图 (Essential Graph)。
"""

# Python 3.6 compatible
from typing import TYPE_CHECKING
import numpy as np

from .lie_algebra import SE3

if TYPE_CHECKING:
    from .frame import Frame
    from .mappoint import MapPoint
    from .map import Map


class KeyFrame:
    _next_id = 0

    __slots__ = (
        "id", "frame_id", "timestamp", "camera",
        "img_left",
        "kp_pts", "kp_octaves", "kp_angles", "descriptors",
        "u_right", "depths", "num_kps",
        "map_points",
        "T_cw", "v_w", "bg", "ba", "R_cw_imu_prior",
        "scale_factors", "inv_scale_factors",
        "level_sigma2", "inv_level_sigma2",
        "grid_element_width_inv", "grid_element_height_inv", "grid",
        "connected_kf_weights", "ordered_connected_kfs", "ordered_weights",
        "first_connection", "parent", "children", "loop_edges",
        "not_erase", "to_be_erased", "is_bad", "T_cp",
        "bow_vec", "feat_vec"
    )

    def __init__(self, frame):
        self.id = KeyFrame._next_id
        KeyFrame._next_id += 1

        self.frame_id = frame.id
        self.timestamp = frame.timestamp
        self.camera = frame.camera
        self.img_left = frame.img_left

        self.kp_pts = frame.kp_pts.copy()
        self.kp_octaves = frame.kp_octaves.copy()
        self.kp_angles = frame.kp_angles.copy()
        self.descriptors = frame.descriptors.copy() if frame.descriptors is not None else None
        self.u_right = frame.u_right.copy()
        self.depths = frame.depths.copy()
        self.num_kps = frame.num_kps

        self.map_points = list(frame.map_points)

        self.T_cw = frame.T_cw.copy()
        self.v_w = frame.v_w.copy()
        self.bg = frame.bg.copy()
        self.ba = frame.ba.copy()
        self.R_cw_imu_prior = (
            frame.R_cw_imu_prior.copy() if frame.R_cw_imu_prior is not None else None
        )

        self.scale_factors = frame.scale_factors.copy()
        self.inv_scale_factors = frame.inv_scale_factors.copy()
        self.level_sigma2 = frame.level_sigma2.copy()
        self.inv_level_sigma2 = frame.inv_level_sigma2.copy()

        self.grid_element_width_inv = frame.grid_element_width_inv
        self.grid_element_height_inv = frame.grid_element_height_inv
        self.grid = [[ list(cell) for cell in col ] for col in frame.grid]

        # 共视图、生成树与本质图拓扑连接
        self.connected_kf_weights = {}
        self.ordered_connected_kfs = []
        self.ordered_weights = []

        self.first_connection = True
        self.parent = None
        self.children = set()
        self.loop_edges = set()

        self.not_erase = False
        self.to_be_erased = False
        self.is_bad = False
        self.T_cp = SE3.identity()

        # 词袋/全局描述子缓存 (用于快速回环检索)
        self.bow_vec = None
        self.feat_vec = None

    def set_pose(self, T_cw) :
        self.T_cw = T_cw.copy()

    def get_pose(self) :
        return self.T_cw

    def get_pose_inverse(self) :
        return self.T_cw.inverse()

    def get_camera_center(self) :
        return -(self.T_cw.R.T @ self.T_cw.t)

    def add_mappoint(self, mp, idx) :
        self.map_points[idx] = mp

    def erase_mappoint_match(self, idx) :
        if 0 <= idx < self.num_kps:
            self.map_points[idx] = None

    def replace_mappoint_match(self, idx, mp) :
        if 0 <= idx < self.num_kps:
            self.map_points[idx] = mp

    def get_mappoint_matches(self) :
        return self.map_points

    def tracked_mappoints(self, min_obs = 1) :
        """统计当前关键帧中观测次数 >= min_obs 的有效地图点数量"""
        count = 0
        for mp in self.map_points:
            if mp is not None and not mp.is_bad:
                if min_obs <= 1 or mp.num_observations() >= min_obs:
                    count += 1
        return count

    def get_features_in_area(
        self,
        x,
        y,
        r,
        min_level = -1,
        max_level = -1
    ) :
        """与 Frame.get_features_in_area 完全一致的 64x48 网格空间查找"""
        if r <= 0.0 or self.num_kps == 0:
            return []
        cols = len(self.grid)
        rows = len(self.grid[0])
        min_cell_x = max(0, int(np.floor((x - r) * self.grid_element_width_inv)))
        if min_cell_x >= cols:
            return []
        max_cell_x = min(cols - 1, int(np.ceil((x + r) * self.grid_element_width_inv)))
        if max_cell_x < 0:
            return []

        min_cell_y = max(0, int(np.floor((y - r) * self.grid_element_height_inv)))
        if min_cell_y >= rows:
            return []
        max_cell_y = min(rows - 1, int(np.ceil((y + r) * self.grid_element_height_inv)))
        if max_cell_y < 0:
            return []

        check_levels = (min_level >= 0) or (max_level >= 0)
        r_sq = r * r
        indices = []
        for ix in range(min_cell_x, max_cell_x + 1):
            for iy in range(min_cell_y, max_cell_y + 1):
                for idx in self.grid[ix][iy]:
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

    def unproject_stereo(self, i) :
        z = float(self.depths[i])
        if z <= 0.0:
            return None
        u, v = float(self.kp_pts[i, 0]), float(self.kp_pts[i, 1])
        p_c = self.camera.unproject_depth(u, v, z)
        return self.T_cw.inverse().transform(p_c)

    def compute_scene_median_depth(self, q = 2) :
        """
        对应 ORB-SLAM3 KeyFrame::ComputeSceneMedianDepth(q):
        计算该关键帧所有有效地图点在相机坐标系下 Z 轴深度的中位数 (q=2) 或均值 (q=1)。
        """
        zs = []
        R_cw = self.T_cw.R
        t_cw = self.T_cw.t
        for mp in self.map_points:
            if mp is not None and not mp.is_bad:
                z = float(R_cw[2] @ mp.pos_w + t_cw[2])
                if z > 0.0:
                    zs.append(z)
        if len(zs) == 0:
            return 1.0
        zs.sort()
        if q == 2:
            return float(zs[len(zs) // 2])
        else:
            return float(np.mean(zs))

    # =========================================================================
    # 共视图 (Covisibility Graph) 与生成树 (Spanning Tree) 核心维护方法
    # =========================================================================
    def add_connection(self, kf, weight) :
        self.connected_kf_weights[kf] = int(weight)
        self.update_best_covisibles()

    def erase_connection(self, kf) :
        if kf in self.connected_kf_weights:
            del self.connected_kf_weights[kf]
            self.update_best_covisibles()

    def update_best_covisibles(self) :
        """按共视地图点权重降序排列相连关键帧 (权重相同时按 id 降序保证严格确定性)"""
        valid_items = [(kf, w) for kf, w in self.connected_kf_weights.items() if not kf.is_bad]
        valid_items.sort(key=lambda item: (item[1], item[0].id), reverse=True)
        self.ordered_connected_kfs = [item[0] for item in valid_items]
        self.ordered_weights = [item[1] for item in valid_items]

    def update_connections(self) :
        """
        更新当前关键帧在共视图中的边与生成树父节点
        对应 ORB-SLAM3 KeyFrame::UpdateConnections()
        阈值 th = 15 (共视点 >= 15 建立共视边；若均 < 15 则连接共视最多的那一帧)
        """
        kf_counter = {}
        for mp in self.map_points:
            if mp is None or mp.is_bad:
                continue
            for other_kf in mp.observations.keys():
                if other_kf.id == self.id or other_kf.is_bad:
                    continue
                kf_counter[other_kf] = kf_counter.get(other_kf, 0) + 1

        if len(kf_counter) == 0:
            return

        th = 15
        max_weight = 0
        best_kf = None

        self.connected_kf_weights.clear()
        for kf, weight in kf_counter.items():
            if weight > max_weight:
                max_weight = weight
                best_kf = kf
            if weight >= th:
                self.connected_kf_weights[kf] = weight
                kf.add_connection(self, weight)

        if len(self.connected_kf_weights) == 0 and best_kf is not None:
            self.connected_kf_weights[best_kf] = max_weight
            best_kf.add_connection(self, max_weight)

        self.update_best_covisibles()

        if self.first_connection and self.id != 0 and len(self.ordered_connected_kfs) > 0:
            self.parent = self.ordered_connected_kfs[0]
            self.parent.add_child(self)
            self.first_connection = False

    def get_connected_keyframes(self) :
        return self.ordered_connected_kfs

    def get_best_covisibility_keyframes(self, n) :
        return self.ordered_connected_kfs[:n]

    def get_covisibles_by_weight(self, w) :
        return [kf for kf, weight in zip(self.ordered_connected_kfs, self.ordered_weights) if weight >= w]

    def get_weight(self, kf) :
        return self.connected_kf_weights.get(kf, 0)

    def add_child(self, kf) :
        self.children.add(kf)

    def erase_child(self, kf) :
        self.children.discard(kf)

    def change_parent(self, new_parent) :
        if new_parent.id == self.id:
            return
        if self.parent is not None:
            self.parent.erase_child(self)
        self.parent = new_parent
        new_parent.add_child(self)

    def add_loop_edge(self, kf) :
        self.not_erase = True
        self.loop_edges.add(kf)

    def set_bad_flag(self, slam_map = None) :
        """
        冗余关键帧安全剔除 (对应 ORB-SLAM3 KeyFrame::SetBadFlag):
        1. 若是第 0 帧或正在参与回环边约束，禁止剔除。
        2. 断开所有共视边与地图点观测。
        3. 为生成树中的子节点重新分配最优父节点，保证生成树不断裂。
        """
        if self.id == 0:
            return
        if self.not_erase:
            self.to_be_erased = True
            return

        for kf in list(self.connected_kf_weights.keys()):
            kf.erase_connection(self)

        for i, mp in enumerate(self.map_points):
            if mp is not None:
                mp.erase_observation(self, slam_map)
                self.map_points[i] = None

        self.connected_kf_weights.clear()
        self.ordered_connected_kfs.clear()

        # 更新生成树: 为当前节点的子节点重新在祖先/共视帧中寻找父节点，并记录相对于父节点的位姿 T_cp
        if self.parent is not None:
            self.T_cp = self.T_cw * self.parent.T_cw.inverse()
            self.parent.erase_child(self)
            for child in list(self.children):
                if child.is_bad:
                    continue
                child.change_parent(self.parent)
        self.children.clear()
        self.is_bad = True

        if slam_map is not None:
            slam_map.erase_keyframe(self)
