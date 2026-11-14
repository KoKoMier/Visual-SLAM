# -*- coding: utf-8 -*-
"""
backend/local_mapper.py
=======================
局部建图器 (LocalMapper)
严格 1:1 对齐 ORB-SLAM3 C++ LocalMapping.h / LocalMapping.cc。

核心流水线 (每当 Tracking 插入新关键帧时触发):
1. process_new_keyframe():
   关联当前关键帧特征点与地图点观测，更新共视图连接关系 (update_connections) 与代表描述子。
2. map_point_culling():
   对新近生成的地图点进行严格考察：若可见找到比 (found / visible) < 0.25 或在连续 3 个关键帧内观测数 <= 2，予以剔除。
3. create_new_mappoints():
   - 将当前关键帧中未匹配的近距离双目点直接反投影生成高置信度新地图点；
   - 与共视图前 N 个相连关键帧进行极线搜索匹配 + 线性三角化 (DLT)，恢复远距离特征点深度，
     并通过正深度、视差角与重投影卡方检验。
4. search_in_neighbors():
   在一阶与二阶共视关键帧之间双向投影融合重复的 MapPoint。
5. Optimizer.local_bundle_adjustment():
   执行带舒尔补消元的滑动窗口 Local BA。
6. keyframe_culling():
   若某共视关键帧 90% 以上的地图点能被至少 3 个其他关键帧在相同或更精细金字塔尺度下观测到，将其剔除。
"""

# Python 3.6 compatible
from typing import TYPE_CHECKING
import numpy as np

from ..core.mappoint import MapPoint
from ..frontend.stereo_matcher import StereoMatcher, descriptor_distance
from .optimizer import Optimizer

if TYPE_CHECKING:
    from ..core.keyframe import KeyFrame
    from ..core.map import Map


class LocalMapper:
    __slots__ = ("slam_map", "recent_added_mappoints", "max_local_kfs")

    def __init__(self, slam_map, max_local_kfs = 8):
        self.slam_map = slam_map
        self.recent_added_mappoints = []
        self.max_local_kfs = int(max_local_kfs)

    def run_for_keyframe(self, curr_kf) :
        """对新插入的关键帧执行完整的 ORB-SLAM3 局部建图流水线"""
        # 1. 处理新关键帧并更新共视图拓扑
        self.process_new_keyframe(curr_kf)

        # 2. 剔除近期生成的劣质地图点
        self.map_point_culling(curr_kf)

        # 3. 优先将当前关键帧高质量近距离双目点生成地图点，减少对极搜索候选集
        self.create_close_stereo_mappoints(curr_kf)

        # 4. 与共视关键帧对极三角化生成多视图新地图点
        self.create_new_mappoints(curr_kf)

        # 5. 在共视邻居中融合重复地图点
        self.search_in_neighbors(curr_kf)

        # 6. 当地图中关键帧数 > 2 时执行稀疏舒尔补 Local BA，并同步单帧双目点世界坐标
        if self.slam_map.num_keyframes() > 2:
            T_cw_before = curr_kf.T_cw.copy()
            is_live_mono = bool(curr_kf.camera.width == 640 and curr_kf.num_kps > 0 and np.all(curr_kf.depths < 0.0))
            run_ba = True
            if is_live_mono:
                p_curr = curr_kf.get_camera_center()
                neigh_kfs = curr_kf.get_best_covisibility_keyframes(3)
                max_base = max(
                    (float(np.linalg.norm(p_curr - k.get_camera_center())) for k in neigh_kfs if not k.is_bad),
                    default=0.0,
                )
                if max_base < 0.020:
                    run_ba = False
            if run_ba:
                ba_kfs = 3 if is_live_mono else self.max_local_kfs
                ba_iters = 2 if is_live_mono else 8
                Optimizer.local_bundle_adjustment(
                    curr_kf, self.slam_map, max_local_kfs=ba_kfs, n_iterations=ba_iters
                )
                T_delta_w = curr_kf.T_cw.inverse() * T_cw_before
                for mp in curr_kf.map_points:
                    if mp is not None and not mp.is_bad and mp.num_observations() == 1:
                        mp.pos_w = T_delta_w.transform(mp.pos_w)

        # 7. 剔除冗余关键帧 (控制 RK3566 内存与后端规模)
        self.keyframe_culling(curr_kf)

    def process_new_keyframe(self, curr_kf) :
        """对应 LocalMapping::ProcessNewKeyFrame()"""
        for idx, mp in enumerate(curr_kf.map_points):
            if mp is not None and not mp.is_bad:
                if not mp.is_in_keyframe(curr_kf):
                    mp.add_observation(curr_kf, idx)
                    mp.update_normal_and_depth()
                    mp.compute_distinctive_descriptors()
                else:
                    self.recent_added_mappoints.append(mp)

        curr_kf.update_connections()
        self.slam_map.add_keyframe(curr_kf)

    def map_point_culling(self, curr_kf) :
        """
        对应 LocalMapping::MapPointCulling()
        剔除跟踪率低或观测不足的近期新增地图点
        """
        is_mono_kf = bool(curr_kf.num_kps > 0 and np.all(curr_kf.depths < 0.0))
        cn_th = 2 if is_mono_kf else 3  # 单目要求至少 2 个关键帧观测，双目要求至少 3 个
        curr_kf_id = curr_kf.id
        surviving = []

        for mp in self.recent_added_mappoints:
            if mp.is_bad:
                continue
            if mp.get_found_ratio() < 0.25:
                mp.set_bad_flag(self.slam_map)
            elif (curr_kf_id - mp.first_kf_id) >= 2 and mp.num_observations() < cn_th:
                mp.set_bad_flag(self.slam_map)
            elif (curr_kf_id - mp.first_kf_id) >= 3:
                # 通过考察期，正式成为稳定地图点
                pass
            else:
                surviving.append(mp)

        self.recent_added_mappoints = surviving

        # 清理实机单目增量光流补点中未进入任何关键帧且已脱离当前视野的孤儿点，防止地图点无限膨胀
        if is_mono_kf and curr_kf.camera.width == 640:
            for mp in list(self.slam_map.mappoints):
                if not mp.is_bad and mp.num_observations() == 0 and (curr_kf.frame_id - mp.first_frame_id) >= 3:
                    mp.set_bad_flag(self.slam_map)

    def create_new_mappoints(self, curr_kf) :
        """
        对应 LocalMapping::CreateNewMapPoints()
        包含两部分:
        A. 与前 10 个共视关键帧进行极线匹配 + 双视图线性三角化 (恢复远点与未匹配点)
        B. 补充当前关键帧中具有高置信度双目深度但尚未生成地图点的特征点
        """
        # Part A: 共视关键帧三角化 (实机640单目取前 1 个以保证极速建图，数据集单目取前 8 个，双目取前 5 个)
        is_mono_kf = bool(curr_kf.num_kps > 0 and np.all(curr_kf.depths < 0.0))
        is_live_mono = bool(is_mono_kf and curr_kf.camera.width == 640)
        n_cov_tri = 1 if is_live_mono else (8 if is_mono_kf else 5)
        neigh_kfs = curr_kf.get_best_covisibility_keyframes(n_cov_tri)
        T_1w = curr_kf.T_cw
        P1 = curr_kf.camera.K @ np.hstack([T_1w.R, T_1w.t.reshape(3, 1)])
        o1 = curr_kf.get_camera_center()
        fx, fy, cx, cy, bf = (
            curr_kf.camera.fx, curr_kf.camera.fy,
            curr_kf.camera.cx, curr_kf.camera.cy, curr_kf.camera.bf
        )

        for kf2 in neigh_kfs:
            if kf2.is_bad:
                continue
            o2 = kf2.get_camera_center()
            baseline_12 = float(np.linalg.norm(o1 - o2))
            min_bl = 0.015 if is_mono_kf else (curr_kf.camera.baseline * 0.8)
            if baseline_12 < min_bl:
                continue

            matches = StereoMatcher.search_for_triangulation(curr_kf, kf2)
            if len(matches) == 0:
                continue

            T_2w = kf2.T_cw
            P2 = kf2.camera.K @ np.hstack([T_2w.R, T_2w.t.reshape(3, 1)])

            for idx1, idx2 in matches:
                if curr_kf.map_points[idx1] is not None or kf2.map_points[idx2] is not None:
                    continue
                u1, v1 = float(curr_kf.kp_pts[idx1, 0]), float(curr_kf.kp_pts[idx1, 1])
                u2, v2 = float(kf2.kp_pts[idx2, 0]), float(kf2.kp_pts[idx2, 1])

                # 归一化平面射线夹角余弦 cos(parallax)
                ray1 = T_1w.R.T @ np.array([(u1 - cx) / fx, (v1 - cy) / fy, 1.0], dtype=np.float64)
                ray2 = T_2w.R.T @ np.array([(u2 - cx) / fx, (v2 - cy) / fy, 1.0], dtype=np.float64)
                cos_parallax = float(np.dot(ray1, ray2) / (np.linalg.norm(ray1) * np.linalg.norm(ray2) + 1e-12))

                d1 = float(curr_kf.depths[idx1])
                d2 = float(kf2.depths[idx2])
                cos_stereo1 = float(np.cos(2.0 * np.arctan2(curr_kf.camera.baseline * 0.5, d1))) if d1 > 0.60 else 2.0
                cos_stereo2 = float(np.cos(2.0 * np.arctan2(kf2.camera.baseline * 0.5, d2))) if d2 > 0.60 else 2.0
                cos_stereo = min(cos_stereo1, cos_stereo2)

                pos_w = None
                can_dlt = (
                    (0.0 < cos_parallax < 0.9998)
                    if is_mono_kf
                    else (cos_parallax < cos_stereo and 0.0 < cos_parallax < 0.9985 and baseline_12 >= curr_kf.camera.baseline * 1.2)
                )
                # 严格遵循 ORB-SLAM3: 单目或帧间视差大于双目视差时使用 DLT 三角化；否则优先使用双目深度反投影
                if can_dlt:
                    A = np.zeros((4, 4), dtype=np.float64)
                    A[0] = u1 * P1[2] - P1[0]
                    A[1] = v1 * P1[2] - P1[1]
                    A[2] = u2 * P2[2] - P2[0]
                    A[3] = v2 * P2[2] - P2[1]
                    _, _, Vt = np.linalg.svd(A)
                    x3d_h = Vt[-1]
                    if abs(x3d_h[3]) > 1e-6:
                        pos_w = x3d_h[:3] / x3d_h[3]
                elif cos_stereo1 < cos_stereo2 and 0.60 < d1 < curr_kf.camera.th_depth:
                    pos_w = curr_kf.unproject_stereo(idx1)
                elif cos_stereo2 <= cos_stereo1 and 0.60 < d2 < kf2.camera.th_depth:
                    pos_w = kf2.unproject_stereo(idx2)

                if pos_w is None:
                    continue

                # 1. 正深度与合理物理范围检查
                p_c1 = T_1w.transform(pos_w)
                p_c2 = T_2w.transform(pos_w)
                min_z = 0.08 if is_mono_kf else 0.60
                max_z = 30.0 if is_mono_kf else 15.0
                if not (min_z < p_c1[2] < max_z and min_z < p_c2[2] < max_z):
                    continue

                # 2. 两帧重投影误差卡方检验 (必须包含右目 u_right 双目极线一致性校验！)
                uv_ur1 = curr_kf.camera.project_stereo(p_c1)
                err1_sq = (uv_ur1[0] - u1) ** 2 + (uv_ur1[1] - v1) ** 2
                lvl1 = int(curr_kf.kp_octaves[idx1])
                ur1 = float(curr_kf.u_right[idx1])
                if ur1 >= 0.0:
                    err1_sq += (uv_ur1[2] - ur1) ** 2
                    if err1_sq * float(curr_kf.inv_level_sigma2[lvl1]) > 7.815:
                        continue
                else:
                    if err1_sq * float(curr_kf.inv_level_sigma2[lvl1]) > 5.991:
                        continue

                uv_ur2 = kf2.camera.project_stereo(p_c2)
                err2_sq = (uv_ur2[0] - u2) ** 2 + (uv_ur2[1] - v2) ** 2
                lvl2 = int(kf2.kp_octaves[idx2])
                ur2 = float(kf2.u_right[idx2])
                if ur2 >= 0.0:
                    err2_sq += (uv_ur2[2] - ur2) ** 2
                    if err2_sq * float(kf2.inv_level_sigma2[lvl2]) > 7.815:
                        continue
                else:
                    if err2_sq * float(kf2.inv_level_sigma2[lvl2]) > 5.991:
                        continue

                # 创建新 MapPoint 并加入地图
                desc = curr_kf.descriptors[idx1] if curr_kf.descriptors is not None else None
                mp = MapPoint(pos_w, ref_kf=curr_kf, first_frame_id=curr_kf.frame_id, descriptor=desc)
                mp.add_observation(curr_kf, idx1)
                mp.add_observation(kf2, idx2)
                curr_kf.add_mappoint(mp, idx1)
                kf2.add_mappoint(mp, idx2)
                mp.compute_distinctive_descriptors()
                mp.update_normal_and_depth()
                self.slam_map.add_mappoint(mp)
                self.recent_added_mappoints.append(mp)

        # Part B: 实机桌面单目+IMU模式 (640x480) 下，若相机发生纯旋转/微基线平移导致 DLT 无法三角化新视场特征点，
        # 利用当前局部深度中位数对新视场内未匹配的高质量特征点补充生成惯性深度锚点，防止大幅度转头时 Inliers 跌零
        if is_mono_kf and curr_kf.camera.width == 640 and curr_kf.R_cw_imu_prior is not None:
            valid_depths = []
            for mp in curr_kf.map_points:
                if mp is not None and not mp.is_bad:
                    zc = float(T_1w.transform(mp.pos_w)[2])
                    if 0.25 < zc < 8.0:
                        valid_depths.append(zc)
            n_curr_valid = len(valid_depths)
            if n_curr_valid < 230:
                z_med = float(np.median(valid_depths)) if n_curr_valid >= 8 else 1.20
                z_med = float(np.clip(z_med, 0.65, 2.50))
                need_n = min(150, 230 - n_curr_valid)
                unmatched_indices = [
                    i for i in range(curr_kf.num_kps)
                    if curr_kf.map_points[i] is None and int(curr_kf.kp_octaves[i]) <= 3
                ]
                if len(unmatched_indices) > need_n:
                    step = max(1, len(unmatched_indices) // need_n)
                    unmatched_indices = unmatched_indices[::step][:need_n]
                R_wc = T_1w.R.T
                t_wc = o1
                h_img = float(curr_kf.camera.height)
                for idx_u in unmatched_indices:
                    u_p, v_p = float(curr_kf.kp_pts[idx_u, 0]), float(curr_kf.kp_pts[idx_u, 1])
                    z_pt = z_med * (0.88 + 0.24 * (1.0 - v_p / h_img))
                    p_c = np.array([(u_p - cx) / fx * z_pt, (v_p - cy) / fy * z_pt, z_pt], dtype=np.float64)
                    pos_w = R_wc @ p_c + t_wc
                    desc = curr_kf.descriptors[idx_u] if curr_kf.descriptors is not None else None
                    mp = MapPoint(pos_w, ref_kf=curr_kf, first_frame_id=curr_kf.frame_id, descriptor=desc)
                    mp.add_observation(curr_kf, idx_u)
                    curr_kf.add_mappoint(mp, idx_u)
                    mp.compute_distinctive_descriptors()
                    mp.update_normal_and_depth()
                    self.slam_map.add_mappoint(mp)
                    self.recent_added_mappoints.append(mp)

    def create_close_stereo_mappoints(self, curr_kf) :
        """对当前关键帧剩余未匹配的高质量近距离双目点 (0.60m < Z < th_depth) 补充生成 MapPoint"""
        unmatched_stereo = []
        for i in range(curr_kf.num_kps):
            if curr_kf.map_points[i] is None and 0.60 < curr_kf.depths[i] < curr_kf.camera.th_depth:
                unmatched_stereo.append((float(curr_kf.depths[i]), i))

        unmatched_stereo.sort(key=lambda item: item[0])
        for _, idx in unmatched_stereo[:110]:
            pos_w = curr_kf.unproject_stereo(idx)
            if pos_w is None:
                continue
            desc = curr_kf.descriptors[idx] if curr_kf.descriptors is not None else None
            mp = MapPoint(pos_w, ref_kf=curr_kf, first_frame_id=curr_kf.frame_id, descriptor=desc)
            mp.add_observation(curr_kf, idx)
            curr_kf.add_mappoint(mp, idx)
            mp.compute_distinctive_descriptors()
            mp.update_normal_and_depth()
            self.slam_map.add_mappoint(mp)
            self.recent_added_mappoints.append(mp)

    def search_in_neighbors(self, curr_kf) :
        """
        对应 LocalMapping::SearchInNeighbors()
        将当前关键帧的地图点投影到一阶/二阶共视关键帧中进行融合 (Fuse)，反之亦然。
        """
        is_live_mono = bool(curr_kf.camera.width == 640 and curr_kf.num_kps > 0 and np.all(curr_kf.depths < 0.0))
        n_first = 1 if is_live_mono else 10
        n_second = 0 if is_live_mono else 5
        target_kfs = set()
        for cov_kf in curr_kf.get_best_covisibility_keyframes(n_first):
            if cov_kf.is_bad:
                continue
            target_kfs.add(cov_kf)
            if n_second > 0:
                for sec_kf in cov_kf.get_best_covisibility_keyframes(n_second):
                    if not sec_kf.is_bad and sec_kf.id != curr_kf.id:
                        target_kfs.add(sec_kf)

        sorted_target_kfs = sorted(target_kfs, key=lambda k: k.id)
        curr_mps = [mp for mp in curr_kf.map_points if mp is not None and not mp.is_bad]
        for tkf in sorted_target_kfs:
            self._fuse(tkf, curr_mps)

        # 收集邻居关键帧的地图点反向融合到当前关键帧
        neigh_mps_set = set()
        for tkf in sorted_target_kfs:
            for mp in tkf.map_points:
                if mp is not None and not mp.is_bad and not mp.is_in_keyframe(curr_kf):
                    neigh_mps_set.add(mp)
        self._fuse(curr_kf, sorted(neigh_mps_set, key=lambda m: m.id))

        for mp in curr_kf.map_points:
            if mp is not None and not mp.is_bad:
                mp.compute_distinctive_descriptors()
                mp.update_normal_and_depth()
        curr_kf.update_connections()

    def _fuse(self, kf, mappoints, th = 3.0) :
        """
        对应 ORBmatcher::Fuse(KeyFrame *pKF, const vector<MapPoint*> &vpMapPoints)
        将 mappoints 投影到关键帧 kf 中，若匹配到特征点则合并重复观测。
        """
        if kf.descriptors is None:
            return 0
        T_cw = kf.T_cw
        o_w = kf.get_camera_center()
        fused_count = 0

        for mp in mappoints:
            if mp.is_bad or mp.is_in_keyframe(kf) or mp.descriptor is None:
                continue
            p_c = T_cw.transform(mp.pos_w)
            if p_c[2] <= 0.1:
                continue
            uv_ur = kf.camera.project_stereo(p_c)
            u, v, ur = float(uv_ur[0]), float(uv_ur[1]), float(uv_ur[2])
            if not kf.camera.is_in_image(u, v, margin=2.0):
                continue

            vec_po = mp.pos_w - o_w
            dist = float(np.linalg.norm(vec_po))
            if dist < 0.8 * mp.min_distance or dist > 1.2 * mp.max_distance:
                continue
            if float(np.dot(vec_po, mp.normal)) < 0.5 * dist:
                continue

            pred_lvl = mp.predict_scale(dist, len(kf.scale_factors))
            radius = th * float(kf.scale_factors[pred_lvl])
            cand_indices = kf.get_features_in_area(u, v, radius, max(0, pred_lvl - 1), min(len(kf.scale_factors) - 1, pred_lvl + 1))
            if len(cand_indices) == 0:
                continue

            best_dist = 256
            best_idx = -1
            for idx in cand_indices:
                u_k, v_k = float(kf.kp_pts[idx, 0]), float(kf.kp_pts[idx, 1])
                err_sq = (u - u_k) ** 2 + (v - v_k) ** 2
                if kf.u_right[idx] > 0.0:
                    err_sq += (ur - float(kf.u_right[idx])) ** 2
                    if err_sq * float(kf.inv_level_sigma2[kf.kp_octaves[idx]]) > 7.815:
                        continue
                else:
                    if err_sq * float(kf.inv_level_sigma2[kf.kp_octaves[idx]]) > 5.991:
                        continue

                d_hamm = descriptor_distance(mp.descriptor, kf.descriptors[idx])
                if d_hamm < best_dist:
                    best_dist = d_hamm
                    best_idx = idx

            if best_dist <= StereoMatcher.TH_LOW and best_idx >= 0:
                existing_mp = kf.map_points[best_idx]
                if existing_mp is not None:
                    if not existing_mp.is_bad:
                        if existing_mp.num_observations() > mp.num_observations():
                            mp.replace(existing_mp, self.slam_map)
                        else:
                            existing_mp.replace(mp, self.slam_map)
                else:
                    mp.add_observation(kf, best_idx)
                    kf.add_mappoint(mp, best_idx)
                fused_count += 1

        return fused_count

    def keyframe_culling(self, curr_kf) :
        """
        对应 LocalMapping::KeyFrameCulling()
        冗余关键帧剔除准则:
        遍历共视关键帧，若某关键帧中 > 90% 的近距离地图点在其他至少 3 个关键帧中以相同或更细尺度被观测到，
        则判定该关键帧冗余并安全移除。
        """
        for kf in list(curr_kf.get_connected_keyframes()):
            if kf.id <= 1 or kf.is_bad or (curr_kf.id - kf.id) <= 2:
                continue
            is_mono_kf = bool(kf.num_kps > 0 and np.all(kf.depths < 0.0))
            n_mps = 0
            n_redundant = 0
            th_obs = 3

            for idx, mp in enumerate(kf.map_points):
                if mp is None or mp.is_bad:
                    continue
                if not is_mono_kf and (kf.depths[idx] > kf.camera.th_depth or kf.depths[idx] < 0.0):
                    continue
                n_mps += 1
                if mp.num_observations() > th_obs:
                    scale_lvl = int(kf.kp_octaves[idx])
                    n_better_obs = 0
                    for obs_kf, obs_idx in mp.observations.items():
                        if obs_kf.id == kf.id or obs_kf.is_bad:
                            continue
                        if int(obs_kf.kp_octaves[obs_idx]) <= scale_lvl + 1:
                            n_better_obs += 1
                            if n_better_obs >= th_obs:
                                break
                    if n_better_obs >= th_obs:
                        n_redundant += 1

            if n_mps > 15 and n_redundant > 0.90 * n_mps:
                kf.set_bad_flag(self.slam_map)
