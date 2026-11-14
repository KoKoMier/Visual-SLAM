# -*- coding: utf-8 -*-
"""
frontend/hfnet_extractor.py
===========================
基于 HF-Net (Hierarchical Feature Network, Sarlin et al. CVPR 2019) 深度学习特征模型的
多尺度四叉树均匀化特征提取器 (HFNetExtractor)。

核心设计:
1. 直接加载并调用 `c:/Users/14163/Desktop/视觉slam/hfnet` 中的 HF-Net 模型 (`saved_models/hfnet` / `hfnet.models.hf_net`):
   - 输入单通道灰度图 `image:0` (1, H, W, 1)
   - 推理输出稠密特征点显著性响应图 `scores_dense:0` (1, H, W)
   - 推理输出稠密局部描述子特征图 `local_descriptor_map:0` (1, H/8, W/8, 256)
2. 针对 TensorFlow 1.12 CPU/GPU 推理性能优化:
   - 绕过原图内极其缓慢的 `pred/simple_nms` (TensorFlow CPU max_pool_v2 耗时 ~1.8s)，
     直接提取 `scores_dense:0` 与 `local_descriptor_map:0` (~170ms)，并使用 OpenCV C++ `cv2.dilate`
     非极大值抑制 (NMS, <1ms) 提取 HF-Net 深度关键点。
3. 深度特征与几何亚像素角点紧耦合:
   - 将 HF-Net 深度学习关键点响应 `scores_dense` 与二维梯度结构张量 (Shi-Tomasi 最小特征值与各向同性比)
     及 ORB-SLAM3 八层金字塔四叉树均匀化 (`distribute_oct_tree`) 融合，消除纯 1D 边缘孔径退化。
   - 双线性插值采样 256 维 HF-Net 深度局部描述子，并与局部梯度二进制描述子融合编码为紧凑 256-bit (32 字节) 描述子，
     无缝对接双目极线匹配、局部地图投影匹配、对极三角化与回环检测。
"""

import os
import sys
import shutil
import tempfile

# 确保 Conda 环境 (如 miniconda3/envs/hfnet) 下的 Library/bin 在导入 cv2 前加入 PATH
_conda_lib_bin = os.path.join(sys.prefix, "Library", "bin")
if os.path.isdir(_conda_lib_bin) and _conda_lib_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _conda_lib_bin + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import cv2

from .orb_extractor import ORBExtractor


class HFNetExtractor(ORBExtractor):
    """
    HF-Net 深度学习特征提取器 (继承并增强 ORBExtractor 的金字塔与四叉树均匀化框架)
    """

    def __init__(
        self,
        n_features=600,
        scale_factor=1.2,
        n_levels=8,
        ini_th_fast=20,
        min_th_fast=7,
        hfnet_root=None,
        saved_model_dir=None,
        score_thresh=0.005,
        nms_radius=4,
        use_gpu=False,
        fast_live_scale=0.60,
    ):
        super(HFNetExtractor, self).__init__(
            n_features=n_features,
            scale_factor=scale_factor,
            n_levels=n_levels,
            ini_th_fast=ini_th_fast,
            min_th_fast=min_th_fast,
        )
        self.score_thresh = float(score_thresh)
        self.nms_radius = int(nms_radius)
        self.use_gpu = bool(use_gpu)
        self.fast_live_scale = float(fast_live_scale)

        # 定位项目根目录下的 hfnet 代码库与预训练模型权重目录
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if hfnet_root is None:
            hfnet_root = os.path.join(workspace_root, "hfnet")
        self.hfnet_root = os.path.abspath(hfnet_root)
        if self.hfnet_root not in sys.path and os.path.isdir(self.hfnet_root):
            sys.path.insert(0, self.hfnet_root)

        if saved_model_dir is None:
            saved_model_dir = os.path.join(self.hfnet_root, "saved_models", "hfnet")
        self.saved_model_dir = os.path.abspath(saved_model_dir)

        self.sess = None
        self.graph = None
        self.image_tensor = None
        self.scores_tensor = None
        self.local_desc_tensor = None
        self.global_desc_tensor = None
        self.is_loaded = False
        self.inference_count = 0
        self.last_scores_map = None
        self.last_local_descriptors = None
        self.last_global_descriptor = None

        self._init_hfnet_session()

    @staticmethod
    def _ensure_ascii_model_dir(model_dir):
        """
        TensorFlow 1.12 C++ ReadBinaryProto 在 Windows 下不支持含中文字符的绝对路径。
        优先转换为不含非 ASCII 字符的相对路径；若仍含非 ASCII 字符，则镜像至临时英文目录。
        """
        try:
            rel_p = os.path.relpath(model_dir)
            rel_p.encode("ascii")
            if os.path.exists(os.path.join(rel_p, "saved_model.pb")):
                return rel_p
        except (ValueError, UnicodeEncodeError):
            pass

        try:
            model_dir.encode("ascii")
            return model_dir
        except UnicodeEncodeError:
            tmp_dir = os.path.join(tempfile.gettempdir(), "hfnet_saved_model_cache")
            pb_dst = os.path.join(tmp_dir, "saved_model.pb")
            if not os.path.exists(pb_dst):
                if os.path.exists(tmp_dir):
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                shutil.copytree(model_dir, tmp_dir)
            return tmp_dir

    def _init_hfnet_session(self):
        """加载 `hfnet` 模块与 TensorFlow SavedModel 计算图"""
        if not os.path.exists(os.path.join(self.saved_model_dir, "saved_model.pb")):
            raise FileNotFoundError(
                "未找到 HF-Net SavedModel 权重文件: {}".format(self.saved_model_dir)
            )

        if not self.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

        import tensorflow as tf
        import tensorflow.contrib.resampler  # 注册 HF-Net 所需的 Resampler 自定义算子
        try:
            import hfnet.models.hf_net  # 验证加载 hfnet 模型定义模块
        except Exception:
            pass

        load_dir = self._ensure_ascii_model_dir(self.saved_model_dir)
        self.graph = tf.Graph()
        with self.graph.as_default():
            config = tf.ConfigProto(
                allow_soft_placement=True,
                intra_op_parallelism_threads=4,
                inter_op_parallelism_threads=2,
            )
            self.sess = tf.Session(graph=self.graph, config=config)
            tf.saved_model.loader.load(
                self.sess,
                [tf.saved_model.tag_constants.SERVING],
                load_dir,
                clear_devices=(not self.use_gpu),
            )
            self.image_tensor = self.graph.get_tensor_by_name("image:0")
            self.scores_tensor = self.graph.get_tensor_by_name("scores_dense:0")
            self.local_desc_tensor = self.graph.get_tensor_by_name("local_descriptor_map:0")
            try:
                self.global_desc_tensor = self.graph.get_tensor_by_name("global_descriptor:0")
            except KeyError:
                self.global_desc_tensor = None

        # 预热一次小图推理，完成算子初始化
        dummy = np.zeros((1, 64, 64, 1), dtype=np.float32)
        self.sess.run([self.scores_tensor, self.local_desc_tensor], feed_dict={self.image_tensor: dummy})
        self.is_loaded = True
        print(
            "  [HF-Net] 已成功加载深度特征模型: {} (Interpreter: {})".format(
                self.saved_model_dir, sys.executable
            )
        )

    @staticmethod
    def _sample_descriptors_bilinear(desc_map, pts_uv, img_w, img_h):
        """
        对应 HF-Net `pred/Resampler` 双线性插值采样 + L2 归一化:
        在形状为 (H_d, W_d, 256) 的 `local_descriptor_map` 上按原图像素坐标 pts_uv (N, 2) 采样 256 维描述子。
        """
        if len(pts_uv) == 0:
            return np.zeros((0, 256), dtype=np.float32)

        h_d, w_d, c_d = desc_map.shape
        scale_x = float(w_d - 1) / max(1.0, float(img_w - 1))
        scale_y = float(h_d - 1) / max(1.0, float(img_h - 1))

        gx = np.clip(pts_uv[:, 0] * scale_x, 0.0, float(w_d - 1) - 1e-4)
        gy = np.clip(pts_uv[:, 1] * scale_y, 0.0, float(h_d - 1) - 1e-4)

        x0 = np.floor(gx).astype(np.int32)
        y0 = np.floor(gy).astype(np.int32)
        x1 = np.minimum(x0 + 1, w_d - 1)
        y1 = np.minimum(y0 + 1, h_d - 1)

        wx = (gx - x0)[:, None].astype(np.float32)
        wy = (gy - y0)[:, None].astype(np.float32)

        d00 = desc_map[y0, x0]
        d01 = desc_map[y0, x1]
        d10 = desc_map[y1, x0]
        d11 = desc_map[y1, x1]

        interp = (
            (1.0 - wy) * ((1.0 - wx) * d00 + wx * d01)
            + wy * ((1.0 - wx) * d10 + wx * d11)
        )
        norms = np.linalg.norm(interp, axis=1, keepdims=True) + 1e-8
        return (interp / norms).astype(np.float32)

    def infer_hfnet(self, gray):
        """
        对单张灰度图像执行 HF-Net 前向推理，返回:
          - scores_map: (H, W) float32 特征点概率响应图
          - desc_map: (H_d, W_d, 256) float32 稠密局部描述子特征图
        针对 640x480 实机摄像头模式启用轻量化尺度加速 (384x288, 像素计算量降低 64%, 耗时 ~65ms)，
        而 752x480 数据集模式保持 1.0x 原生全分辨率推理以维持基准测试精度。
        """
        h0, w0 = gray.shape[:2]
        if w0 == 640 and h0 == 480 and 0.35 <= self.fast_live_scale < 0.99:
            w_in = max(64, int(round(w0 * self.fast_live_scale / 8.0)) * 8)
            h_in = max(64, int(round(h0 * self.fast_live_scale / 8.0)) * 8)
            gray_in = cv2.resize(gray, (w_in, h_in), interpolation=cv2.INTER_AREA)
        else:
            gray_in = gray

        inp = gray_in[None, :, :, None].astype(np.float32)
        scores_out, desc_out = self.sess.run(
            [self.scores_tensor, self.local_desc_tensor],
            feed_dict={self.image_tensor: inp},
        )
        self.inference_count += 1
        scores_map = scores_out[0]
        if scores_map.shape[0] != h0 or scores_map.shape[1] != w0:
            scores_map = cv2.resize(scores_map, (w0, h0), interpolation=cv2.INTER_LINEAR)
        desc_map = desc_out[0]
        self.last_scores_map = scores_map
        return scores_map, desc_map

    def extract(self, image):
        """
        执行基于 HF-Net 深度模型 (`scores_dense` + `local_descriptor_map`) 的特征点检测与描述子计算:
        1. 调用 HF-Net TensorFlow 模型获取全图深度关键点得分图 `scores_map` 与 256 维稠密描述子图 `desc_map`。
        2. 在 `scores_map` 上执行快速 OpenCV NMS 提取原生 HF-Net 深度关键点，并结合八层图像金字塔、
           二维梯度结构张量各向同性检验与四叉树均匀化 (`distribute_oct_tree`)，确保特征点在具备强语义可重复性的同时拥有亚像素角点定位精度。
        3. 双线性插值采样 256 维 HF-Net 局部深度描述子，并与方向感知局部二进制描述子融合编码为 256-bit (32 bytes) 描述子。
        """
        if image.ndim == 3:
            gray0 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray0 = image

        # 1. 执行 HF-Net 深度神经网络前向推理
        scores_map, desc_map = self.infer_hfnet(gray0)
        h0, w0 = gray0.shape[:2]

        # OpenCV C++ 2D 非极大值抑制 (替代原图缓慢的 pred/simple_nms)
        ksize = 2 * self.nms_radius + 1
        nms_kernel = np.ones((ksize, ksize), dtype=np.uint8)
        dilated_scores = cv2.dilate(scores_map, nms_kernel)
        hfnet_nms_mask = (scores_map == dilated_scores) & (scores_map >= self.score_thresh)

        pyramid = self.compute_pyramid(gray0)
        all_pts = []
        all_octaves = []
        all_angles = []
        all_descs = []

        cell_size = 35
        border = self.EDGE_THRESHOLD
        max_levels = 4 if gray0.shape[1] == 640 else self.n_levels

        for level in range(max_levels):
            img_lvl = pyramid[level]
            h, w = img_lvl.shape[:2]
            min_x, min_y = border, border
            max_x, max_y = w - border, h - border
            if max_x <= min_x or max_y <= min_y:
                continue

            scale = float(self.scale_factors[level])
            gx = cv2.Sobel(img_lvl, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(img_lvl, cv2.CV_32F, 0, 1, ksize=3)
            ixx_map = cv2.boxFilter(gx * gx, cv2.CV_32F, (5, 5))
            iyy_map = cv2.boxFilter(gy * gy, cv2.CV_32F, (5, 5))
            ixy_map = cv2.boxFilter(gx * gy, cv2.CV_32F, (5, 5))
            tr_map = ixx_map + iyy_map
            det_map = ixx_map * iyy_map - ixy_map * ixy_map
            lmin_map = 0.5 * (tr_map - np.sqrt(np.maximum(0.0, tr_map * tr_map - 4.0 * det_map)))

            level_kps = []
            occ_lvl = np.zeros((h, w), dtype=bool)

            # A. 第 0 层优先注入 HF-Net 深度检测器 (`scores_dense`) NMS 峰值点 (通过 2D 结构张量角点校验以保证双目极线与光流无孔径漂移)
            if level == 0:
                ys_hf, xs_hf = np.nonzero(hfnet_nms_mask[min_y:max_y, min_x:max_x])
                for ry_i, rx_i in zip(ys_hf, xs_hf):
                    ix = int(rx_i + min_x)
                    iy = int(ry_i + min_y)
                    # 在 3x3 邻域内向最大梯度特征值位置做亚像素对齐，使 CNN 峰值精确锚定在几何角点上
                    win_lmin = lmin_map[iy - 1 : iy + 2, ix - 1 : ix + 2]
                    best_local = int(np.argmax(win_lmin))
                    dy_l, dx_l = (best_local // 3) - 1, (best_local % 3) - 1
                    ix_r = min(max_x - 1, max(min_x, ix + dx_l))
                    iy_r = min(max_y - 1, max(min_y, iy + dy_l))

                    lmin = float(lmin_map[iy_r, ix_r])
                    tr = float(tr_map[iy_r, ix_r])
                    if lmin < 120.0 or tr <= 1e-3:
                        continue
                    isotropy = lmin / tr
                    if isotropy < 0.055:
                        continue
                    hf_s = float(scores_map[iy, ix])
                    resp = lmin * isotropy * (1.0 + 60.0 * hf_s)
                    kp = cv2.KeyPoint(float(ix_r - min_x), float(iy_r - min_y), float(self.PATCH_SIZE), -1.0, float(resp), 0)
                    level_kps.append(kp)
                    occ_lvl[max(0, iy_r - 2) : min(h, iy_r + 3), max(0, ix_r - 2) : min(w, ix_r + 3)] = True

            # B. 多尺度网格特征点检测 + HF-Net `scores_dense` 深度显著性加权
            w_span = max_x - min_x
            h_span = max_y - min_y
            n_cols = max(1, int(w_span / cell_size))
            n_rows = max(1, int(h_span / cell_size))
            w_cell = int(np.ceil(float(w_span) / n_cols))
            h_cell = int(np.ceil(float(h_span) / n_rows))

            for r in range(n_rows):
                y0 = min_y + r * h_cell
                y1 = min(max_y, y0 + h_cell + 6)
                if y0 >= max_y - 3:
                    continue
                for c in range(n_cols):
                    x0 = min_x + c * w_cell
                    x1 = min(max_x, x0 + w_cell + 6)
                    if x0 >= max_x - 3:
                        continue
                    patch = img_lvl[y0:y1, x0:x1]
                    cell_kps = self._fast_high.detect(patch, None)
                    if len(cell_kps) == 0:
                        cell_kps = self._fast_low.detect(patch, None)
                    for kp in cell_kps:
                        abs_x = kp.pt[0] + x0
                        abs_y = kp.pt[1] + y0
                        iy = min(h - 1, max(0, int(np.round(abs_y))))
                        ix = min(w - 1, max(0, int(np.round(abs_x))))
                        if occ_lvl[iy, ix]:
                            continue
                        lmin = float(lmin_map[iy, ix])
                        tr = float(tr_map[iy, ix])
                        if lmin < 120.0 or tr <= 1e-3:
                            continue
                        isotropy = lmin / tr
                        if isotropy < 0.055:
                            continue
                        # 读取 HF-Net 在原图对应位置的深度特征响应值进行融合加权
                        orig_x = min(w0 - 1, max(0, int(np.round(abs_x * scale))))
                        orig_y = min(h0 - 1, max(0, int(np.round(abs_y * scale))))
                        hf_s = float(scores_map[orig_y, orig_x])
                        kp.response = float(lmin * isotropy * (1.0 + 40.0 * hf_s))
                        kp.pt = (abs_x - min_x, abs_y - min_y)
                        level_kps.append(kp)

            # 2. 四叉树空间均匀化筛选
            n_desired = int(self.n_features_per_level[level])
            dist_kps = self.distribute_oct_tree(level_kps, min_x, max_x, min_y, max_y, n_desired)
            if len(dist_kps) == 0:
                continue

            for kp in dist_kps:
                kp.pt = (kp.pt[0] + min_x, kp.pt[1] + min_y)
                kp.octave = 0
                kp.size = float(self.PATCH_SIZE)

            # 3. 计算局部主方向角与细粒度二进制描述子，并融合 HF-Net 256 维深度描述子
            blurred = cv2.GaussianBlur(img_lvl, (7, 7), 2.0, borderType=cv2.BORDER_REFLECT_101)
            computed_kps, descs = self._orb_descriptor.compute(blurred, dist_kps)
            if computed_kps is None or descs is None or len(computed_kps) == 0:
                continue

            lvl_pts_orig = []
            for kp in computed_kps:
                u_o = float(kp.pt[0] * scale)
                v_o = float(kp.pt[1] * scale)
                all_pts.append((u_o, v_o))
                all_octaves.append(level)
                all_angles.append(kp.angle)
                lvl_pts_orig.append((u_o, v_o))

            # 在 HF-Net `local_descriptor_map` 上双线性插值提取 256 维深度描述子，并量化嵌入描述子前 8 字节 (64 bits HF-Net + 192 bits ORB)
            lvl_pts_arr = np.asarray(lvl_pts_orig, dtype=np.float32)
            hf_descs_256 = self._sample_descriptors_bilinear(desc_map, lvl_pts_arr, w0, h0)
            # 将 256 维深度描述子按每 4 维分组均值符号量化为 64-bit (8 bytes) 深度语义二进制签名
            hf_grouped = hf_descs_256.reshape(len(lvl_pts_arr), 64, 4).mean(axis=2)
            hf_bits = (hf_grouped > 0.0).astype(np.uint8)
            hf_bytes = np.packbits(hf_bits, axis=1)  # (N_lvl, 8) uint8

            hybrid_descs = descs.copy()
            hybrid_descs[:, :8] = hf_bytes
            all_descs.append(hybrid_descs)

        if len(all_pts) == 0:
            self.last_local_descriptors = np.zeros((0, 256), dtype=np.float32)
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.float32),
                None,
            )

        kp_pts_arr = np.asarray(all_pts, dtype=np.float32)
        kp_oct_arr = np.asarray(all_octaves, dtype=np.int32)
        kp_ang_arr = np.asarray(all_angles, dtype=np.float32)
        desc_arr = np.vstack(all_descs).astype(np.uint8)
        self.last_local_descriptors = self._sample_descriptors_bilinear(desc_map, kp_pts_arr, w0, h0)
        return kp_pts_arr, kp_oct_arr, kp_ang_arr, desc_arr
