# -*- coding: utf-8 -*-
"""
backend/loop_closing.py
=======================
回环检测与全局位姿图优化器 (LoopClosing)
严格 1:1 对齐 ORB-SLAM3 C++ LoopClosing.h / LoopClosing.cc。

核心流程:
1. 全局二值外观签名 / 词袋向量计算 (compute_global_signature):
   利用关键帧 256-bit ORB 描述子构建紧凑的二进制位频率直方图签名 (256 维)，
   在 C++ 端可直接映射为 DBoW2 词袋向量或紧凑二进制签名，实现微秒级候选帧初筛。
2. 历史候选帧检索 (detect_loop_candidates):
   排除与当前关键帧直接共视的局部邻居以及最近 20 个关键帧，检索外观相似度高且距离合理的历史帧。
3. 3D-3D RANSAC + SVD (Umeyama/Kabsch) 刚体变换验证 (compute_se3_ransac):
   通过 ORB 描述子匹配两关键帧关联的 3D 地图点，执行 RANSAC 求解相对位姿 T_curr_loop，
   并检查内点数量与重投影误差。
4. 回环闭合与本质图优化 (correct_loop):
   添加回环边 (add_loop_edge)，合并重复地图点，调用 Optimizer.optimize_essential_graph()
   消除全局累积漂移！
"""

# Python 3.6 compatible
from typing import TYPE_CHECKING
import numpy as np

from ..core.lie_algebra import SE3
from ..frontend.stereo_matcher import descriptor_distance_batch, StereoMatcher
from .optimizer import Optimizer

if TYPE_CHECKING:
    from ..core.keyframe import KeyFrame
    from ..core.map import Map


class LoopClosing:
    __slots__ = ("slam_map", "last_loop_kf_id", "min_kf_gap", "loop_count")

    def __init__(self, slam_map, min_kf_gap = 30):
        self.slam_map = slam_map
        self.last_loop_kf_id = -100
        self.min_kf_gap = int(min_kf_gap)
        self.loop_count = 0

    @staticmethod
    def compute_bow_signature(kf) :
        """
        计算关键帧的 256 维归一化二进制特征分布向量 (用于快速外观相似度检索)
        """
        if kf.bow_vec is not None:
            return kf.bow_vec
        if kf.descriptors is None or len(kf.descriptors) == 0:
            return None
        bits = np.unpackbits(kf.descriptors, axis=1).astype(np.float64)  # (N, 256)
        hist = bits.mean(axis=0)
        hist -= hist.mean()
        norm = float(np.linalg.norm(hist))
        if norm > 1e-8:
            hist /= norm
        kf.bow_vec = hist
        return kf.bow_vec

    def run_for_keyframe(self, curr_kf) :
        """
        对新关键帧执行回环检测与闭合校正
        返回是否成功触发并完成了回环闭合。
        """
        self.compute_bow_signature(curr_kf)
        if curr_kf.id < self.min_kf_gap or (curr_kf.id - self.last_loop_kf_id) < 10:
            return False

        candidates = self.detect_loop_candidates(curr_kf)
        if len(candidates) == 0:
            return False

        for cand_kf in candidates:
            success, T_corr_cw, matched_pairs = self.verify_loop_candidate(curr_kf, cand_kf)
            if success and T_corr_cw is not None:
                self.correct_loop(curr_kf, cand_kf, T_corr_cw, matched_pairs)
                self.last_loop_kf_id = curr_kf.id
                self.loop_count += 1
                return True

        return False

    def detect_loop_candidates(self, curr_kf) :
        """检索非局部共视的历史回环候选关键帧"""
        curr_sig = self.compute_bow_signature(curr_kf)
        if curr_sig is None:
            return []

        cov_set = set(curr_kf.get_connected_keyframes())
        cov_set.add(curr_kf)

        # 计算与局部共视邻居的最低外观得分作为基准阈值 (对齐 ORB-SLAM3 DBoW2 minScore)
        min_cov_score = 0.65
        for n_kf in curr_kf.get_best_covisibility_keyframes(5):
            n_sig = self.compute_bow_signature(n_kf)
            if n_sig is not None:
                score = float(np.dot(curr_sig, n_sig))
                if score < min_cov_score:
                    min_cov_score = max(0.45, score * 0.85)

        curr_pos = curr_kf.get_camera_center()
        scored_candidates = []

        for kf in self.slam_map.get_all_keyframes():
            if kf in cov_set or kf.is_bad:
                continue
            if (curr_kf.id - kf.id) < self.min_kf_gap:
                continue
            sig = self.compute_bow_signature(kf)
            if sig is None:
                continue
            score = float(np.dot(curr_sig, sig))
            dist = float(np.linalg.norm(curr_pos - kf.get_camera_center()))
            # 外观高度相似或空间重访候选
            if score >= min_cov_score or dist < 4.5:
                scored_candidates.append((score, kf))

        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored_candidates[:3]]

    def verify_loop_candidate(
        self,
        curr_kf,
        cand_kf) :
        """
        对候选回环帧执行 ORB 匹配 + 3D-3D RANSAC SE(3) 求解与内点校验
        """
        if curr_kf.descriptors is None or cand_kf.descriptors is None:
            return False, None, []

        valid_1 = [i for i, mp in enumerate(curr_kf.map_points) if mp is not None and not mp.is_bad]
        valid_2 = [j for j, mp in enumerate(cand_kf.map_points) if mp is not None and not mp.is_bad]
        if len(valid_1) < 15 or len(valid_2) < 15:
            return False, None, []

        desc2_mat = cand_kf.descriptors[valid_2]
        matches = []
        used_2 = set()

        for i1 in valid_1:
            d1 = curr_kf.descriptors[i1]
            dists = descriptor_distance_batch(d1, desc2_mat)
            order = np.argsort(dists)
            if len(order) >= 2:
                best_d = int(dists[order[0]])
                second_d = int(dists[order[1]])
                if best_d <= StereoMatcher.TH_HIGH and best_d < 0.8 * float(second_d):
                    i2 = valid_2[int(order[0])]
                    if i2 not in used_2:
                        matches.append((i1, i2))
                        used_2.add(i2)

        if len(matches) < 12:
            return False, None, []

        # 提取匹配点在各自相机坐标系下的 3D 坐标 P_c1 与在 cand_kf 下的世界坐标 P_w2
        pts_c1 = []
        pts_w2 = []
        for i1, i2 in matches:
            mp1 = curr_kf.map_points[i1]
            mp2 = cand_kf.map_points[i2]
            p_c1 = curr_kf.T_cw.transform(mp1.pos_w)
            pts_c1.append(p_c1)
            pts_w2.append(mp2.pos_w)

        pts_c1_arr = np.asarray(pts_c1, dtype=np.float64)
        pts_w2_arr = np.asarray(pts_w2, dtype=np.float64)

        # 3D-3D RANSAC + Kabsch/SVD 求解 T_corr_cw: P_c1 = R_cw * P_w2 + t_cw
        n_m = len(matches)
        best_inliers = np.zeros(n_m, dtype=bool)
        best_T = None
        rng = np.random.default_rng(42)

        for _ in range(60):
            sample_idx = rng.choice(n_m, size=3, replace=False)
            T_cand = self._solve_kabsch_se3(pts_w2_arr[sample_idx], pts_c1_arr[sample_idx])
            pred_c1 = T_cand.transform(pts_w2_arr)
            errs = np.linalg.norm(pred_c1 - pts_c1_arr, axis=1)
            inliers = errs < 0.12  # 3D 严格内点阈值 (12 cm)
            if np.count_nonzero(inliers) > np.count_nonzero(best_inliers):
                best_inliers = inliers
                best_T = T_cand

        n_inliers = int(np.count_nonzero(best_inliers))
        if n_inliers < 22 or best_T is None:
            return False, None, []

        # 使用所有内点重新精化 SE(3)
        refined_T = self._solve_kabsch_se3(pts_w2_arr[best_inliers], pts_c1_arr[best_inliers])
        inlier_pairs = [matches[k] for k in range(n_m) if best_inliers[k]]
        return True, refined_T, inlier_pairs

    @staticmethod
    def _solve_kabsch_se3(pts_src, pts_dst) :
        """
        使用 SVD (Kabsch / Umeyama 算法) 求解刚体变换 T (R, t) 使得 pts_dst approx R * pts_src + t
        """
        mu_src = pts_src.mean(axis=0)
        mu_dst = pts_dst.mean(axis=0)
        X = pts_src - mu_src
        Y = pts_dst - mu_dst
        H = X.T @ Y
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1.0
            R = Vt.T @ U.T
        t = mu_dst - (R @ mu_src)
        return SE3(R, t)

    def correct_loop(
        self,
        curr_kf,
        loop_kf,
        T_corr_cw,
        matched_pairs) :
        """
        执行回环闭合:
        1. 融合回环匹配点对的 MapPoint
        2. 注册回环边并更新共视图
        3. 调用本质图位姿图优化 (OptimizeEssentialGraph) 全局修正漂移
        """
        for idx_curr, idx_loop in matched_pairs:
            mp_curr = curr_kf.map_points[idx_curr]
            mp_loop = loop_kf.map_points[idx_loop]
            if mp_loop is not None and not mp_loop.is_bad:
                if mp_curr is not None and not mp_curr.is_bad:
                    mp_curr.replace(mp_loop, self.slam_map)
                else:
                    curr_kf.add_mappoint(mp_loop, idx_curr)
                    mp_loop.add_observation(curr_kf, idx_curr)

        curr_kf.add_loop_edge(loop_kf)
        loop_kf.add_loop_edge(curr_kf)
        curr_kf.update_connections()
        loop_kf.update_connections()

        Optimizer.optimize_essential_graph(self.slam_map, loop_kf, curr_kf, T_corr_cw)
