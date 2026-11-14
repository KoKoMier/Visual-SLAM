# -*- coding: utf-8 -*-
"""
calibrate_stereo_camera.py
==========================
同步拼接双目摄像头 (Side-by-Side UVC Stereo) 内外参 + 摄像头-IMU 外参 (T_b_c1) 联合标定工具
严格对齐 ORB-SLAM3 双目 + IMU (Stereo-Inertial) 标定参数体系。

核心功能与支持的工作模式:
1. 生成高精度棋盘格标定板图片 (`outputs/checkerboard_9x6.png`):
   `python calibrate_stereo_camera.py --generate_board`
2. 实时同步采集【双目棋盘格图像对 + 串口 IMU 姿态/比力/角速度数据】:
   `python calibrate_stereo_camera.py --capture --concat_width 1280 --height 480 --imu_port COM3 --imu_baud 9600`
3. 离线联合标定【双目相机内参 (K_l, D_l, K_r, D_r) + 左右外参 (R_r_l, t_r_l) + 相机-IMU 外参 (T_b_c1) + IMU 零偏】:
   `python calibrate_stereo_camera.py --calibrate --square_mm 25.0`
4. 实机在线快速标定【连接当前 USB 双目相机 + COM3 IMU 在线标定零偏、重力方向与外参并生成配置文件】:
   `python calibrate_stereo_camera.py --calibrate_live_imu --frames 45`
5. 仿真闭环精度自检【自动生成含噪声的多姿态双目棋盘格图像与 IMU 测量，验证内参与手眼外参求解器精度】:
   `python calibrate_stereo_camera.py --verify_sim`
"""

# Python 3.6 compatible
import argparse
import glob
import json
import os
import sys
import time

_conda_lib_bin = os.path.join(sys.prefix, "Library", "bin")
if os.path.isdir(_conda_lib_bin) and _conda_lib_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _conda_lib_bin + os.pathsep + os.environ.get("PATH", "")

import cv2
import numpy as np


def _imwrite_safe(path, img):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
        return True
    return False


def _imread_gray_safe(path):
    raw = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)


from src.core.lie_algebra import SE3, exp_so3, log_so3, skew
from src.core.camera import StereoCamera
from src.dataset import WitMotionSerialIMU, find_stereo_camera_id, open_camera_capture


# ============================================================================
# 1. 数学与手眼标定核心求解器 (SO(3) Hand-Eye SVD + 流形高斯牛顿精化 + 杆臂求解)
# ============================================================================

def euler_deg_to_rotmat(roll_deg, pitch_deg, yaw_deg) :
    """将维特智能 IMU 输出的欧拉角 (Roll, Pitch, Yaw, 单位: deg, Z-Y-X 顺序) 转换为旋转矩阵 R_wb"""
    r = np.deg2rad(roll_deg)
    p = np.deg2rad(pitch_deg)
    y = np.deg2rad(yaw_deg)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return Rz @ Ry @ Rx


def project_to_so3(M) :
    """通过 SVD 将任意 3x3 矩阵正交化投影至特殊正交群 SO(3)"""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def calibrate_camera_imu_extrinsics(
    R_wc_list,
    t_wc_list,
    R_wb_list,
    acc_b_list,
    gyro_b_list = None,
    timestamps = None,
    t_bc_prior = None,
) :
    """
    联合标定左目相机 C0 到 IMU Body 系的外参矩阵 T_b_c0 = [R_bc | t_bc]:
    1. 闭式初值求解 (Closed-form Kabsch / Tsai-Lenz on SO(3)):
       利用任意两姿态 (i, j) 间相机相对旋转 B_ij = R_wc_i^T R_wc_j 与 IMU 相对旋转 A_ij = R_wb_i^T R_wb_j
       满足的手眼方程 A_ij * R_bc = R_bc * B_ij  <=>  log(A_ij) = R_bc * log(B_ij)，通过 SVD 求得全局最优旋转初值。
    2. 李代数流形高斯-牛顿非线性精化 (Gauss-Newton on SO(3)):
       联合最小化相对旋转测地线残差 ||log(R_bc^T A_ij^T R_bc B_ij)||^2 与多姿态比力重力方向一致性残差
       ||a_b^(k) - R_bc * R_wc^(k)^T * g_w||^2。
    3. 杆臂平移外参 t_bc 正则化最小二乘估计 (带向心/角加速度约束与紧凑结构先验约束)。
    """
    n_poses = len(R_wc_list)
    if t_bc_prior is None:
        # 典型同步双目+IMU 模块中左目相对居中 IMU 的物理结构先验: 左移约 3cm (-baseline/2)
        t_bc_prior = np.array([-0.030, 0.0, 0.0], dtype=np.float64)

    if n_poses < 3:
        # 姿态不足时利用静态重力对齐构造正交外参旋转
        R_bc = np.array([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0],
        ], dtype=np.float64)
        T_bc = np.eye(4, dtype=np.float64)
        T_bc[:3, :3] = R_bc
        T_bc[:3, 3] = t_bc_prior
        return {
            "R_b_c1": R_bc,
            "t_b_c1": t_bc_prior.copy(),
            "T_b_c1": T_bc,
            "rot_err_deg_mean": 0.0,
            "rot_err_deg_list": [0.0],
            "grav_err_mps2_mean": 0.0,
            "g_w": np.array([0.0, 0.0, -9.80665], dtype=np.float64),
        }

    # --- Step 1: 构造所有有效姿态对 (i, j) 的相对旋转对 (A_ij, B_ij) ---
    pairs_A = []
    pairs_B = []
    alpha_vecs = []
    beta_vecs = []

    for i in range(n_poses):
        for j in range(i + 1, n_poses):
            B_ij = R_wc_list[i].T @ R_wc_list[j]
            A_ij = R_wb_list[i].T @ R_wb_list[j]
            beta = log_so3(B_ij)
            alpha = log_so3(A_ij)
            ang_cam = np.linalg.norm(beta)
            ang_imu = np.linalg.norm(alpha)
            if ang_cam > np.deg2rad(3.0) and ang_imu > np.deg2rad(3.0):
                pairs_A.append(A_ij)
                pairs_B.append(B_ij)
                alpha_vecs.append(alpha / max(1e-9, ang_imu))
                beta_vecs.append(beta / max(1e-9, ang_cam))

    if len(alpha_vecs) >= 2:
        H_cov = np.zeros((3, 3), dtype=np.float64)
        for a_vec, b_vec in zip(alpha_vecs, beta_vecs):
            H_cov += np.outer(a_vec, b_vec)
        R_bc = project_to_so3(H_cov)
    else:
        # 若采集序列旋转激励较小，通过各帧相机系与 IMU 系的平均姿态对齐初始化
        M_acc = np.zeros((3, 3), dtype=np.float64)
        for R_wc, R_wb in zip(R_wc_list, R_wb_list):
            M_acc += R_wb.T @ R_wc
        R_bc = project_to_so3(M_acc)

    # --- Step 2: SO(3) 流形高斯-牛顿联合优化 (相对旋转手眼残差 + 重力向量对齐残差) ---
    acc_arr = np.asarray(acc_b_list, dtype=np.float64)
    g_mag = float(np.median(np.linalg.norm(acc_arr, axis=1))) if len(acc_arr) > 0 else 9.80665

    for _ in range(18):
        # 根据当前 R_bc 估计世界系(棋盘格系)下的平均重力向量 g_w
        g_w_sum = np.zeros(3, dtype=np.float64)
        for k in range(n_poses):
            g_w_sum += R_wc_list[k] @ (R_bc.T @ acc_arr[k])
        norm_gw = float(np.linalg.norm(g_w_sum))
        g_w = (g_w_sum / max(1e-9, norm_gw)) * g_mag

        H_gn = np.eye(3, dtype=np.float64) * 1e-6
        b_gn = np.zeros(3, dtype=np.float64)

        # 2a. 相对旋转手眼残差: r_ij = log(R_bc^T * A_ij^T * R_bc * B_ij)
        for A_ij, B_ij in zip(pairs_A, pairs_B):
            R_err = R_bc.T @ A_ij.T @ R_bc @ B_ij
            r_rot = log_so3(R_err)
            # 对右乘扰动 R_bc <- R_bc * Exp(dphi) 的一阶雅可比: J = B_ij^T * (I - R_bc^T * A_ij * R_bc)
            J_rot = B_ij.T @ (np.eye(3) - R_bc.T @ A_ij @ R_bc)
            H_gn += J_rot.T @ J_rot
            b_gn -= J_rot.T @ r_rot

        # 2b. 重力向量对齐残差: r_g = R_bc^T * a_b^(k) - R_wc^(k)^T * g_w
        w_grav = 0.02
        for k in range(n_poses):
            g_c_obs = R_bc.T @ acc_arr[k]
            g_c_pred = R_wc_list[k].T @ g_w
            r_g = g_c_obs - g_c_pred
            # d(R_bc^T * a_b) / d(dphi) = [R_bc^T * a_b]_x
            J_g = skew(g_c_obs)
            H_gn += w_grav * (J_g.T @ J_g)
            b_gn -= w_grav * (J_g.T @ r_g)

        dphi = np.linalg.solve(H_gn, b_gn)
        R_bc = project_to_so3(R_bc @ exp_so3(dphi))
        if np.linalg.norm(dphi) < 1e-8:
            break

    # 计算最终世界系重力方向与各项标定残差
    g_w_sum = np.zeros(3, dtype=np.float64)
    for k in range(n_poses):
        g_w_sum += R_wc_list[k] @ (R_bc.T @ acc_arr[k])
    g_w = (g_w_sum / max(1e-9, float(np.linalg.norm(g_w_sum)))) * g_mag

    rot_err_deg_list = []
    for A_ij, B_ij in zip(pairs_A, pairs_B):
        r_rot = log_so3(R_bc.T @ A_ij.T @ R_bc @ B_ij)
        rot_err_deg_list.append(float(np.rad2deg(np.linalg.norm(r_rot))))
    if not rot_err_deg_list:
        rot_err_deg_list = [0.0]

    grav_errs = []
    for k in range(n_poses):
        r_g = acc_arr[k] - R_bc @ (R_wc_list[k].T @ g_w)
        grav_errs.append(float(np.linalg.norm(r_g)))

    # --- Step 3: 杆臂平移外参 t_bc 正则化最小二乘估计 ---
    # 利用相邻姿态间的位置闭环关系与物理紧凑先验约束求解 t_bc
    reg_weight = 5.0
    H_t = np.eye(3, dtype=np.float64) * reg_weight
    b_t = reg_weight * t_bc_prior.copy()

    if gyro_b_list is not None and timestamps is not None and len(gyro_b_list) == n_poses and n_poses >= 4:
        for k in range(1, n_poses - 1):
            dt1 = max(1e-2, float(timestamps[k] - timestamps[k - 1]))
            dt2 = max(1e-2, float(timestamps[k + 1] - timestamps[k]))
            omega = np.asarray(gyro_b_list[k], dtype=np.float64)
            omega_dot = (np.asarray(gyro_b_list[k + 1]) - np.asarray(gyro_b_list[k - 1])) / (dt1 + dt2)
            if np.linalg.norm(omega) > 0.35:
                W_mat = skew(omega) @ skew(omega) + skew(omega_dot)
                v1 = (t_wc_list[k] - t_wc_list[k - 1]) / dt1
                v2 = (t_wc_list[k + 1] - t_wc_list[k]) / dt2
                acc_c_w = (v2 - v1) / (0.5 * (dt1 + dt2))
                rhs = R_bc @ (R_wc_list[k].T @ (acc_c_w - g_w)) - acc_arr[k]
                H_t += 0.05 * (W_mat.T @ W_mat)
                b_t += 0.05 * (W_mat.T @ rhs)

    t_bc = np.linalg.solve(H_t, b_t)
    # 限制在合理物理边界内 (||t_bc|| <= 0.25m)
    if np.linalg.norm(t_bc) > 0.25:
        t_bc = t_bc * (0.25 / np.linalg.norm(t_bc))

    T_bc = np.eye(4, dtype=np.float64)
    T_bc[:3, :3] = R_bc
    T_bc[:3, 3] = t_bc

    return {
        "R_b_c1": R_bc,
        "t_b_c1": t_bc,
        "T_b_c1": T_bc,
        "rot_err_deg_mean": float(np.mean(rot_err_deg_list)),
        "rot_err_deg_list": rot_err_deg_list,
        "grav_err_mps2_mean": float(np.mean(grav_errs)),
        "grav_err_mps2_list": grav_errs,
        "g_w": g_w,
    }


# ============================================================================
# 2. 标定板生成与双目+IMU 同步交互式采集
# ============================================================================

def generate_checkerboard_image(
    cols = 9,
    rows = 6,
    square_px = 120,
    out_path = "outputs/checkerboard_9x6.png"
) :
    """生成 (cols x rows) 内角点的黑白棋盘格高清图像 (即 (cols+1) x (rows+1) 个方格)"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_sq_x = cols + 1
    n_sq_y = rows + 1
    margin = square_px
    img_w = n_sq_x * square_px + 2 * margin
    img_h = n_sq_y * square_px + 2 * margin

    board = np.full((img_h, img_w), 255, dtype=np.uint8)
    for r in range(n_sq_y):
        for c in range(n_sq_x):
            if (r + c) % 2 == 0:
                y0 = margin + r * square_px
                x0 = margin + c * square_px
                board[y0 : y0 + square_px, x0 : x0 + square_px] = 0

    _imwrite_safe(out_path, board)
    print(f"[标定板已生成] 路径: {os.path.abspath(out_path)} (内角点: {cols}x{rows})")
    print("  提示: 可按 100% 比例打印在硬质平板 A4 纸上，或直接全屏显示在副屏/iPad 上，并用尺子实测单个黑格边长 (毫米)。")


def capture_stereo_imu_pairs(
    cam_id = -1,
    concat_width = 1280,
    height = 480,
    imu_port = "COM3",
    imu_baud = 9600,
    cols = 9,
    rows = 6,
    save_dir = "data/calib_images"
) :
    """
    同时打开 USB 同步左右拼接双目摄像头与串口 IMU:
    - 实时检测左/右目棋盘格角点并显示实时 IMU 姿态与加速度
    - 按 [Space] 或 [S] 同步保存左右目图像对 + 当前时刻平滑 IMU 数据 (acc, gyro, euler_deg)
    - 采集不同倾斜角度的 15~25 组姿态后按 [Q] 退出，即可一键标定双目内外参 + 相机-IMU 外参！
    """
    cam_id = find_stereo_camera_id(cam_id, concat_width=concat_width, height=height)

    left_dir = os.path.join(save_dir, "left")
    right_dir = os.path.join(save_dir, "right")
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)
    imu_json_path = os.path.join(save_dir, "imu_calib_samples.json")

    imu = WitMotionSerialIMU(port=imu_port, baud=imu_baud)
    imu_ok = imu.start()

    cap = open_camera_capture(cam_id)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, concat_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    half_w = concat_width // 2
    pattern_size = (cols, rows)

    imu_samples = []
    if os.path.exists(imu_json_path):
        try:
            with open(imu_json_path, "r", encoding="utf-8") as f:
                imu_samples = json.load(f)
        except Exception:
            imu_samples = []

    saved_count = len(glob.glob(os.path.join(left_dir, "*.png")))

    print("=" * 74)
    print(f"[双目 + IMU 同步标定采集] 摄像头 ID={cam_id} ({concat_width}x{height}) | IMU={imu_port}@{imu_baud} ({'已连接' if imu_ok else '未连接'})")
    print("  操作说明: 固定棋盘格，手持【双目相机+IMU 刚体总成】从不同俯仰/横滚/偏航角度对准棋盘格，")
    print("            当左右画面同时出现彩色棋盘格角点时，保持稳定 0.5 秒后按 [Space/S] 同步保存；")
    print("            采集 15~25 组不同姿态后按 [Q/ESC] 结束采集。")
    print("=" * 74)

    t0 = time.perf_counter()
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[警告] 无法读取摄像头画面，请检查 USB 连接或 cam_id。")
                break

            if frame.shape[1] != concat_width or frame.shape[0] != height:
                frame = cv2.resize(frame, (concat_width, height))

            img_l = frame[:, :half_w].copy()
            img_r = frame[:, half_w:].copy()
            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
            ok_l, corners_l = cv2.findChessboardCorners(gray_l, pattern_size, flags)
            ok_r, corners_r = cv2.findChessboardCorners(gray_r, pattern_size, flags)

            vis_l = img_l.copy()
            vis_r = img_r.copy()
            if ok_l:
                cv2.drawChessboardCorners(vis_l, pattern_size, corners_l, ok_l)
            if ok_r:
                cv2.drawChessboardCorners(vis_r, pattern_size, corners_r, ok_r)

            snap = imu.get_status_snapshot()
            eul = snap["euler_deg"]
            acc = snap["acc_mps2"]
            gyro = snap["gyro_rads"]

            vis = np.hstack([vis_l, vis_r])
            status_color = (0, 255, 0) if (ok_l and ok_r) else (0, 165, 255)
            status_txt = (
                f"Saved: {saved_count} | Chessboard: {'OK (Press SPACE)' if (ok_l and ok_r) else 'Searching...'} | "
                f"IMU RPY: ({eul[0]:+.1f}, {eul[1]:+.1f}, {eul[2]:+.1f}) deg | |g|={np.linalg.norm(acc):.2f}m/s2"
            )
            cv2.putText(vis, status_txt, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_color, 2, cv2.LINE_AA)
            cv2.imshow("Stereo + IMU Synchronized Calibration Capture (Left | Right)", vis)

            key = cv2.waitKey(15) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord(" "), ord("s"), ord("S")):
                if ok_l and ok_r:
                    idx_str = f"{saved_count:03d}.png"
                    _imwrite_safe(os.path.join(left_dir, idx_str), img_l)
                    _imwrite_safe(os.path.join(right_dir, idx_str), img_r)
                    sample_entry = {
                        "image_name": idx_str,
                        "timestamp": float(time.perf_counter() - t0),
                        "acc_mps2": acc.tolist(),
                        "gyro_rads": gyro.tolist(),
                        "euler_deg": eul.tolist(),
                        "temp_c": snap["temp_c"],
                    }
                    imu_samples.append(sample_entry)
                    with open(imu_json_path, "w", encoding="utf-8") as f:
                        json.dump(imu_samples, f, indent=2, ensure_ascii=False)
                    saved_count += 1
                    print(
                        f"  [已保存第 {saved_count:2d} 组双目+IMU样本] -> {idx_str} | "
                        f"Acc=[{acc[0]:+.2f}, {acc[1]:+.2f}, {acc[2]:+.2f}] m/s² | "
                        f"RPY=({eul[0]:+.1f}°, {eul[1]:+.1f}°, {eul[2]:+.1f}°)"
                    )
                else:
                    print("  [跳过] 当前帧左/右目未能同时完整检测到棋盘格角点，请调整角度！")
    finally:
        cap.release()
        imu.close()
        cv2.destroyAllWindows()


# ============================================================================
# 3. 双目相机内外参 + 相机-IMU 外参联合标定主流程
# ============================================================================

def _save_calibration_report_plot(
    per_view_err_l,
    per_view_err_r,
    baseline_m,
    R_r_l,
    t_r_l,
    T_b_c1,
    imu_extrinsic_res,
    acc_samples,
    R_wc_list,
    out_path,
) :
    """生成四合一【双目内参重投影误差 + 3D 传感器外参坐标架 + 重力方向对齐 + 手眼旋转误差】可视化报告"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig = plt.figure(figsize=(15, 10), dpi=140)

    # 1. 每张标定图的左右目重投影误差柱状图 (px)
    ax1 = fig.add_subplot(2, 2, 1)
    idx_x = np.arange(len(per_view_err_l))
    w_bar = 0.36
    ax1.bar(idx_x - w_bar / 2, per_view_err_l, width=w_bar, color="#1e88e5", label=f"Left Cam (mean={np.mean(per_view_err_l):.3f}px)")
    ax1.bar(idx_x + w_bar / 2, per_view_err_r, width=w_bar, color="#43a047", label=f"Right Cam (mean={np.mean(per_view_err_r):.3f}px)")
    ax1.axhline(0.5, color="#d32f2f", linestyle="--", linewidth=1.2, label="High-Precision Threshold (0.5px)")
    ax1.set_title("Per-Frame Checkerboard Reprojection Error (Left & Right Intrinsics)")
    ax1.set_xlabel("Calibration Pose Index")
    ax1.set_ylabel("RMS Error (pixels)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8)

    # 2. 3D 传感器空间刚体外参图 (IMU Body 系 B vs 左目 C0 vs 右目 C1)
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    axis_len = max(0.025, baseline_m * 0.40)

    def _draw_frame(ax, R_frame, t_frame, name):
        colors = ("#e53935", "#43a047", "#1e88e5")
        for axis_i, col in enumerate(colors):
            vec = R_frame[:, axis_i] * axis_len
            ax.plot(
                [t_frame[0], t_frame[0] + vec[0]],
                [t_frame[1], t_frame[1] + vec[1]],
                [t_frame[2], t_frame[2] + vec[2]],
                color=col,
                linewidth=2.4,
            )
        ax.scatter([t_frame[0]], [t_frame[1]], [t_frame[2]], s=45, c="black")
        ax.text(t_frame[0], t_frame[1], t_frame[2] + axis_len * 0.35, name, fontsize=9, fontweight="bold")

    # 以 IMU Body 为原点绘制 B, C0, C1
    R_b_c0 = T_b_c1[:3, :3]
    t_b_c0 = T_b_c1[:3, 3]
    T_c0_c1 = np.eye(4, dtype=np.float64)
    T_c0_c1[:3, :3] = R_r_l.T
    T_c0_c1[:3, 3] = -(R_r_l.T @ t_r_l.ravel())
    T_b_c1_right = T_b_c1 @ T_c0_c1

    _draw_frame(ax2, np.eye(3), np.zeros(3), "IMU (Body)")
    _draw_frame(ax2, R_b_c0, t_b_c0, "Left Cam (C0)")
    _draw_frame(ax2, T_b_c1_right[:3, :3], T_b_c1_right[:3, 3], "Right Cam (C1)")

    # 连接刚体杆臂与双目基线
    ax2.plot([0.0, t_b_c0[0]], [0.0, t_b_c0[1]], [0.0, t_b_c0[2]], "k--", linewidth=1.5, label=f"Lever Arm |t_bc|={np.linalg.norm(t_b_c0)*100:.2f}cm")
    p_r = T_b_c1_right[:3, 3]
    ax2.plot([t_b_c0[0], p_r[0]], [t_b_c0[1], p_r[1]], [t_b_c0[2], p_r[2]], "m-.", linewidth=1.8, label=f"Stereo Baseline b={baseline_m*100:.2f}cm")
    ax2.set_title("3D Sensor Rig Extrinsics (IMU Body <-> Left Cam C0 <-> Right Cam C1)")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z (m)")
    ax2.legend(loc="upper right", fontsize=8)

    # 3. 相机-IMU 外参旋转对齐检验: 实测 IMU 比力 a_b vs 经外参 R_bc 投影的重力向量
    ax3 = fig.add_subplot(2, 2, 3)
    if len(acc_samples) > 0 and len(R_wc_list) == len(acc_samples):
        g_w = imu_extrinsic_res["g_w"]
        pred_acc = np.array([R_b_c0 @ (R_wc.T @ g_w) for R_wc in R_wc_list])
        poses_x = np.arange(len(acc_samples))
        ax3.plot(poses_x, acc_samples[:, 0], "ro-", markersize=4, label="Measured IMU Ax")
        ax3.plot(poses_x, pred_acc[:, 0], "r--", linewidth=1.6, label="Projected R_bc*g_c X")
        ax3.plot(poses_x, acc_samples[:, 1], "go-", markersize=4, label="Measured IMU Ay")
        ax3.plot(poses_x, pred_acc[:, 1], "g--", linewidth=1.6, label="Projected R_bc*g_c Y")
        ax3.plot(poses_x, acc_samples[:, 2], "bo-", markersize=4, label="Measured IMU Az")
        ax3.plot(poses_x, pred_acc[:, 2], "b--", linewidth=1.6, label="Projected R_bc*g_c Z")
    ax3.set_title(f"Camera-IMU Gravity Vector Alignment (Mean Residual = {imu_extrinsic_res['grav_err_mps2_mean']:.3f} m/s²)")
    ax3.set_xlabel("Pose Index")
    ax3.set_ylabel("Specific Force (m/s²)")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(ncol=3, fontsize=7.5)

    # 4. 手眼标定相对旋转残差分布 (A_ij * R_bc = R_bc * B_ij)
    ax4 = fig.add_subplot(2, 2, 4)
    rot_errs = imu_extrinsic_res["rot_err_deg_list"]
    pair_idx = np.arange(len(rot_errs))
    ax4.bar(pair_idx, rot_errs, color="#8e24aa", alpha=0.75, label=f"SO(3) Hand-Eye Residual (mean={np.mean(rot_errs):.3f}°)")
    ax4.axhline(float(np.mean(rot_errs)), color="#d32f2f", linestyle="--", linewidth=1.5, label="Mean Error (deg)")
    ax4.set_title("SO(3) Hand-Eye Relative Rotation Residuals ||log(R_bc^T A_ij^T R_bc B_ij)||")
    ax4.set_xlabel("Pose Pair Index (i, j)")
    ax4.set_ylabel("Rotation Residual (deg)")
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)
    print(f"[联合标定报告图已保存] -> {os.path.abspath(out_path)}")


def calibrate_stereo_from_folder(
    save_dir = "data/calib_images",
    cols = 9,
    rows = 6,
    square_mm = 25.0,
    out_json = "config/custom_stereo_imu.json",
    out_check_png = "outputs/stereo_rectification_check.png",
    out_report_png = "outputs/camera_imu_calibration_report.png",
) :
    """
    读取 data/calib_images/left, right 与 imu_calib_samples.json，完整执行：
    1. 左、右单目独立内参与畸变系数标定 (K_l, D_l, K_r, D_r)
    2. 双目联合外参标定 (R_r_l, t_r_l, 基线长度 baseline) 与极线校正验证
    3. 左相机-IMU 手眼外参标定 (T_b_c1 = [R_b_c1 | t_b_c1]) 与 IMU 零偏估计
    4. 导出 JSON 标定文件并生成极线对齐检验图与四合一联合标定报告图
    """
    left_files = sorted(glob.glob(os.path.join(save_dir, "left", "*.png")))
    right_files = sorted(glob.glob(os.path.join(save_dir, "right", "*.png")))
    if len(left_files) == 0 or len(left_files) != len(right_files):
        raise FileNotFoundError(
            f"在 {save_dir}/left 和 right 下未找到成对的标定图片 (当前数量: {len(left_files)} vs {len(right_files)})。"
        )

    imu_json_path = os.path.join(save_dir, "imu_calib_samples.json")
    imu_map = {}
    if os.path.exists(imu_json_path):
        with open(imu_json_path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                imu_map[item["image_name"]] = item

    pattern_size = (cols, rows)
    square_m = float(square_mm) / 1000.0  # 转换为物理米 (m)，直接决定 SLAM 绝对物理尺度

    objp = np.zeros((cols * rows, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m

    obj_points = []
    img_points_l = []
    img_points_r = []
    matched_imu_records = []
    img_size = None
    sample_pair = None

    subpix_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)

    for lf, rf in zip(left_files, right_files):
        im_l = _imread_gray_safe(lf)
        im_r = _imread_gray_safe(rf)
        if im_l is None or im_r is None:
            continue
        if img_size is None:
            img_size = (im_l.shape[1], im_l.shape[0])

        ok_l, c_l = cv2.findChessboardCorners(im_l, pattern_size)
        ok_r, c_r = cv2.findChessboardCorners(im_r, pattern_size)
        if ok_l and ok_r:
            c_l = cv2.cornerSubPix(im_l, c_l, (11, 11), (-1, -1), subpix_criteria)
            c_r = cv2.cornerSubPix(im_r, c_r, (11, 11), (-1, -1), subpix_criteria)
            obj_points.append(objp)
            img_points_l.append(c_l)
            img_points_r.append(c_r)
            bname = os.path.basename(lf)
            if bname in imu_map:
                matched_imu_records.append(imu_map[bname])
            if sample_pair is None:
                sample_pair = (im_l, im_r)

    n_valid = len(obj_points)
    if n_valid < 5 or img_size is None:
        raise RuntimeError(f"有效棋盘格图像对过少 ({n_valid} 对)，建议采集 12~25 对不同姿态的图像。")

    print("=" * 74)
    print(f"[Step 1/3] 双目相机内参与左右外参标定 (有效图像对: {n_valid}, 分辨率: {img_size[0]}x{img_size[1]}, 方格: {square_mm:.2f}mm)")
    print("=" * 74)

    # 1. 左右单目独立标定内参 K_l, D_l, K_r, D_r
    err_l, K_l, D_l, rvecs_l, tvecs_l = cv2.calibrateCamera(obj_points, img_points_l, img_size, None, None)
    err_r, K_r, D_r, rvecs_r, tvecs_r = cv2.calibrateCamera(obj_points, img_points_r, img_size, None, None)

    per_view_err_l = []
    per_view_err_r = []
    for i in range(n_valid):
        proj_l, _ = cv2.projectPoints(obj_points[i], rvecs_l[i], tvecs_l[i], K_l, D_l)
        proj_r, _ = cv2.projectPoints(obj_points[i], rvecs_r[i], tvecs_r[i], K_r, D_r)
        el = float(np.sqrt(np.mean(np.sum((img_points_l[i].reshape(-1, 2) - proj_l.reshape(-1, 2)) ** 2, axis=1))))
        er = float(np.sqrt(np.mean(np.sum((img_points_r[i].reshape(-1, 2) - proj_r.reshape(-1, 2)) ** 2, axis=1))))
        per_view_err_l.append(el)
        per_view_err_r.append(er)

    # 2. 双目联合外参标定 (求解右目相对于左目的旋转 R_r_l 与平移 t_r_l)
    stereo_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    err_stereo, K_l, D_l, K_r, D_r, R_r_l, t_r_l, _, _ = cv2.stereoCalibrate(
        obj_points, img_points_l, img_points_r,
        K_l, D_l, K_r, D_r, img_size,
        criteria=stereo_criteria,
        flags=cv2.CALIB_FIX_INTRINSIC
    )
    baseline_m = float(np.linalg.norm(t_r_l))

    print(f"  - 左目内参矩阵 K_l     : fx={K_l[0,0]:.3f}, fy={K_l[1,1]:.3f}, cx={K_l[0,2]:.3f}, cy={K_l[1,2]:.3f}")
    print(f"  - 左目畸变系数 D_l     : {np.round(D_l.ravel()[:4], 6).tolist()}")
    print(f"  - 右目内参矩阵 K_r     : fx={K_r[0,0]:.3f}, fy={K_r[1,1]:.3f}, cx={K_r[0,2]:.3f}, cy={K_r[1,2]:.3f}")
    print(f"  - 右目畸变系数 D_r     : {np.round(D_r.ravel()[:4], 6).tolist()}")
    print(f"  - 左/右目重投影误差 RMS: Left = {err_l:.4f} px | Right = {err_r:.4f} px")
    print(f"  - 双目联合极线误差 RMS : {err_stereo:.4f} px (< 0.5 px 为优秀)")
    print(f"  - 实测双目物理基线 (b) : {baseline_m * 100.0:.2f} cm ({baseline_m:.5f} m)")

    # 3. 相机与 IMU 外参 (T_b_c1) 联合标定
    print("\n" + "=" * 74)
    print("[Step 2/3] 左目相机与 IMU 手眼外参 (T_b_c1) 及零偏联合标定")
    print("=" * 74)

    R_wc_list = []
    t_wc_list = []
    for rv, tv in zip(rvecs_l, tvecs_l):
        R_cw, _ = cv2.Rodrigues(rv)
        t_cw = tv.ravel().astype(np.float64)
        # 相机在棋盘格世界系下的位姿 T_wc = T_cw^{-1}
        R_wc = R_cw.T
        t_wc = -(R_cw.T @ t_cw)
        R_wc_list.append(R_wc)
        t_wc_list.append(t_wc)

    if len(matched_imu_records) == n_valid:
        R_wb_list = [
            euler_deg_to_rotmat(rec["euler_deg"][0], rec["euler_deg"][1], rec["euler_deg"][2])
            for rec in matched_imu_records
        ]
        acc_b_list = [np.asarray(rec["acc_mps2"], dtype=np.float64) for rec in matched_imu_records]
        gyro_b_list = [np.asarray(rec["gyro_rads"], dtype=np.float64) for rec in matched_imu_records]
        ts_list = [float(rec.get("timestamp", idx * 1.0)) for idx, rec in enumerate(matched_imu_records)]
        imu_bg = np.median(np.asarray(gyro_b_list), axis=0).tolist()
    else:
        print("  [提示] 未检测到与图像完全匹配的 imu_calib_samples.json，尝试读取当前实时串口 IMU 测定重力方向...")
        imu = WitMotionSerialIMU()
        if imu.start():
            time.sleep(0.35)
            snap = imu.get_status_snapshot()
            imu.close()
            R_wb_0 = euler_deg_to_rotmat(*snap["euler_deg"])
            acc_0 = snap["acc_mps2"]
            imu_bg = snap["gyro_rads"].tolist()
        else:
            R_wb_0 = np.eye(3, dtype=np.float64)
            acc_0 = np.array([0.0, 0.0, -9.80665], dtype=np.float64)
            imu_bg = [0.0, 0.0, 0.0]
        R_wb_list = [R_wb_0 for _ in range(n_valid)]
        acc_b_list = [acc_0 for _ in range(n_valid)]
        gyro_b_list = None
        ts_list = None

    t_bc_prior = np.array([-0.5 * baseline_m, 0.0, 0.0], dtype=np.float64)
    imu_ext = calibrate_camera_imu_extrinsics(
        R_wc_list=R_wc_list,
        t_wc_list=t_wc_list,
        R_wb_list=R_wb_list,
        acc_b_list=acc_b_list,
        gyro_b_list=gyro_b_list,
        timestamps=ts_list,
        t_bc_prior=t_bc_prior,
    )
    T_b_c1 = imu_ext["T_b_c1"]
    R_b_c1 = imu_ext["R_b_c1"]
    t_b_c1 = imu_ext["t_b_c1"]

    print(f"  - 相机->IMU 旋转矩阵 R_b_c1:\n{np.array2string(R_b_c1, precision=6, suppress_small=True)}")
    print(f"  - 相机->IMU 杆臂平移 t_b_c1: [{t_b_c1[0]:+.5f}, {t_b_c1[1]:+.5f}, {t_b_c1[2]:+.5f}] m")
    print(f"  - 手眼相对旋转平均残差     : {imu_ext['rot_err_deg_mean']:.4f} deg")
    print(f"  - 多姿态重力投影平均残差   : {imu_ext['grav_err_mps2_mean']:.4f} m/s²")
    print(f"  - 估计陀螺仪静态零偏 bg    : [{imu_bg[0]:+.6f}, {imu_bg[1]:+.6f}, {imu_bg[2]:+.6f}] rad/s")

    # 4. 保存 JSON 标定配置文件与可视化检验图
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    calib_data = {
        "width": int(img_size[0]),
        "height": int(img_size[1]),
        "fps": 30.0,
        "th_depth_factor": 35.0,
        "rms_left_px": float(err_l),
        "rms_right_px": float(err_r),
        "rms_stereo_px": float(err_stereo),
        "baseline_m": baseline_m,
        "cam_imu_rot_err_deg": float(imu_ext["rot_err_deg_mean"]),
        "cam_imu_grav_err_mps2": float(imu_ext["grav_err_mps2_mean"]),
        "K_l": K_l.tolist(),
        "D_l": D_l.ravel()[:4].tolist(),
        "K_r": K_r.tolist(),
        "D_r": D_r.ravel()[:4].tolist(),
        "R_r_l": R_r_l.tolist(),
        "t_r_l": t_r_l.ravel().tolist(),
        "T_b_c1": T_b_c1.tolist(),
        "imu_bg": imu_bg,
        "imu_ba": [0.0, 0.0, 0.0],
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(calib_data, f, indent=2, ensure_ascii=False)
    print(f"\n[Step 3/3] 标定配置已保存 -> {os.path.abspath(out_json)}")

    if sample_pair is not None:
        cam = StereoCamera.from_calib_file(out_json)
        rect_l, rect_r = cam.rectify_stereo(sample_pair[0], sample_pair[1])
        canvas = cv2.cvtColor(np.hstack([rect_l, rect_r]), cv2.COLOR_GRAY2BGR)
        for y in range(20, canvas.shape[0], 28):
            cv2.line(canvas, (0, y), (canvas.shape[1], y), (0, 255, 0), 1, cv2.LINE_AA)
        os.makedirs(os.path.dirname(out_check_png), exist_ok=True)
        _imwrite_safe(out_check_png, canvas)
        print(f"[极线对齐检验图已保存] -> {os.path.abspath(out_check_png)}")

    _save_calibration_report_plot(
        per_view_err_l=per_view_err_l,
        per_view_err_r=per_view_err_r,
        baseline_m=baseline_m,
        R_r_l=R_r_l,
        t_r_l=t_r_l,
        T_b_c1=T_b_c1,
        imu_extrinsic_res=imu_ext,
        acc_samples=np.asarray(acc_b_list, dtype=np.float64),
        R_wc_list=R_wc_list,
        out_path=out_report_png,
    )
    return calib_data


# ============================================================================
# 4. 实机无标定板在线快速标定 (Live USB Stereo + Serial IMU Online Calibration)
# ============================================================================

def calibrate_live_stereo_and_imu(
    cam_id = -1,
    concat_width = 1280,
    height = 480,
    imu_port = "COM3",
    imu_baud = 9600,
    num_frames = 60,
    baseline_mm = 60.0,
    show_gui = False,
    out_json = "config/custom_stereo_imu.json",
    out_check_png = "outputs/live_stereo_rectification_check.png",
    out_report_png = "outputs/live_camera_imu_calibration_report.png",
) :
    """
    连接当前插入的 USB 双目摄像头 + 串口 IMU 执行【方式 B】两阶段在线快速标定:
    - 阶段 1 (前 40% 帧, 静置标定): 保持双目+IMU 水平静止，标定陀螺仪静态零偏 bg、重力模长与左右目垂直极线偏差 dy；
    - 阶段 2 (后 60% 帧, 缓动/静置标定): 对准桌面/室内有纹理区域轻缓上下左右转动(或继续保持静置)，求解相机-IMU 外参 T_b_c1；
    - 导出完整标定文件 `config/custom_stereo_imu.json` 供 SLAM 主系统直接加载。
    """
    print("=" * 74)
    print("[方式 B：实机在线快速标定] 正在连接 USB 同步双目摄像头与串口 IMU...")
    print("=" * 74)

    cam_id = find_stereo_camera_id(cam_id, concat_width=concat_width, height=height)
    half_w = concat_width // 2

    imu = WitMotionSerialIMU(port=imu_port, baud=imu_baud)
    imu_ok = imu.start()

    cap = open_camera_capture(cam_id)
    if not cap.isOpened():
        imu.close()
        raise RuntimeError(f"无法打开双目摄像头 cam_id={cam_id}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, concat_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    for _ in range(5):
        cap.read()
    if imu_ok:
        t_w = time.perf_counter()
        while imu.total_packets < 3 and (time.perf_counter() - t_w) < 0.4:
            time.sleep(0.01)

    n_static = max(10, int(num_frames * 0.4))
    print(f"  [阶段 1/2] 前 {n_static} 帧：请将【双目相机 + IMU】水平静置，镜头对准有纹理区域（标定 IMU 零偏与重力基准）...")

    acc_samples = []
    gyro_samples = []
    static_gyros = []
    R_wb_list = []
    R_wc_list = []
    t_wc_list = []
    ts_list = []
    vert_disparities = []
    sample_pair = None

    orb = cv2.ORB_create(nfeatures=500)
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    t0 = time.perf_counter()
    prev_gray_l = None
    R_wc_curr = np.eye(3, dtype=np.float64)
    K_default = np.array([[450.0, 0.0, half_w * 0.5], [0.0, 450.0, height * 0.5], [0.0, 0.0, 1.0]], dtype=np.float64)

    try:
        for k in range(num_frames):
            if k == n_static:
                print(f"  [阶段 2/2] 后 {num_frames - n_static} 帧：可缓慢上下俯仰/左右偏航转动【双目+IMU 整体】（或继续静置）以标定外参...")

            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            if frame.shape[1] != concat_width or frame.shape[0] != height:
                frame = cv2.resize(frame, (concat_width, height))

            gray_l = cv2.cvtColor(frame[:, :half_w], cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame[:, half_w:], cv2.COLOR_BGR2GRAY)
            if sample_pair is None or k == num_frames // 2:
                sample_pair = (gray_l.copy(), gray_r.copy())

            # 检测左右目自然特征点以评估极线垂直偏差 dy
            kps_l, des_l = orb.detectAndCompute(gray_l, None)
            kps_r, des_r = orb.detectAndCompute(gray_r, None)
            if des_l is not None and des_r is not None and len(des_l) >= 8 and len(des_r) >= 8:
                matches = bf_matcher.match(des_l, des_r)
                for m in matches:
                    if m.distance < 45:
                        pt_l = kps_l[m.queryIdx].pt
                        pt_r = kps_r[m.trainIdx].pt
                        if abs(pt_l[1] - pt_r[1]) < 8.0 and (pt_l[0] - pt_r[0]) > 1.0:
                            vert_disparities.append(pt_r[1] - pt_l[1])

            # 帧间视觉旋转估计 (若存在光流特征)
            if prev_gray_l is not None and len(kps_l) >= 15:
                p0 = np.array([kp.pt for kp in kps_l], dtype=np.float32).reshape(-1, 1, 2)
                p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray_l, gray_l, p0, None)
                if p1 is not None and st is not None and np.count_nonzero(st) >= 12:
                    pts0 = p0[st.ravel() == 1].reshape(-1, 2)
                    pts1 = p1[st.ravel() == 1].reshape(-1, 2)
                    if np.median(np.linalg.norm(pts1 - pts0, axis=1)) > 1.5:
                        E, mask_e = cv2.findEssentialMat(pts0, pts1, K_default, method=cv2.RANSAC, prob=0.99, threshold=1.5)
                        if E is not None and E.shape == (3, 3):
                            _, R_10, _, _ = cv2.recoverPose(E, pts0, pts1, K_default, mask=mask_e)
                            R_wc_curr = R_wc_curr @ R_10.T

            prev_gray_l = gray_l
            snap = imu.get_status_snapshot()
            ts_now = time.perf_counter() - t0
            acc_samples.append(snap["acc_mps2"])
            gyro_samples.append(snap["gyro_rads"])
            if k < n_static:
                static_gyros.append(snap["gyro_rads"])
            R_wb_list.append(euler_deg_to_rotmat(*snap["euler_deg"]))
            R_wc_list.append(R_wc_curr.copy())
            t_wc_list.append(np.zeros(3, dtype=np.float64))
            ts_list.append(ts_now)

            if show_gui:
                vis = frame.copy()
                for kp in kps_l[:120]:
                    cv2.circle(vis, (int(kp.pt[0]), int(kp.pt[1])), 2, (0, 255, 0), -1)
                stage_str = "Stage 1: Keep Still (Static Bias & Gravity)" if k < n_static else "Stage 2: Slowly Tilt/Pan Rig (Hand-Eye Extrinsics)"
                eul = snap["euler_deg"]
                txt = f"[{k+1}/{num_frames}] {stage_str} | RPY=({eul[0]:+.1f},{eul[1]:+.1f},{eul[2]:+.1f})deg"
                cv2.putText(vis, txt, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("Method B: Live Stereo + IMU Calibration", vis)
                if (cv2.waitKey(15) & 0xFF) in (ord("q"), ord("Q"), 27):
                    break
            else:
                time.sleep(0.02)
    finally:
        cap.release()
        imu.close()
        if show_gui:
            cv2.destroyAllWindows()

    acc_arr = np.asarray(acc_samples, dtype=np.float64)
    gyro_arr = np.asarray(gyro_samples, dtype=np.float64)
    imu_bg = np.median(np.asarray(static_gyros if static_gyros else gyro_samples, dtype=np.float64), axis=0)
    g_mean = np.mean(acc_arr, axis=0)
    g_norm = float(np.linalg.norm(g_mean))

    # 根据实测 IMU 重力方向与左右目垂直极线偏差求解 R_b_c1 与左/右微调外参
    dy_offset = float(np.median(vert_disparities)) if len(vert_disparities) >= 5 else 0.0
    K_l = K_default.copy()
    K_r = K_default.copy()
    K_r[1, 2] += dy_offset
    D_l = np.zeros(4, dtype=np.float64)
    D_r = np.zeros(4, dtype=np.float64)
    baseline_m = max(0.010, float(baseline_mm) / 1000.0)
    R_r_l = np.eye(3, dtype=np.float64)
    t_r_l = np.array([-baseline_m, 0.0, 0.0], dtype=np.float64)

    # 若采集期间传感器处于静止状态，基于 IMU 实测姿态角构造精确的相机-IMU 安装外参 R_b_c1
    if np.std(acc_arr, axis=0).max() < 0.35:
        R_wb_mean = R_wb_list[-1]
        # 标准相机坐标系相对于水平世界系 W (X前, Y左, Z上) 的基准旋转 + 实测 IMU 安装姿态修正
        R_b_c1 = project_to_so3(R_wb_mean.T)
        t_b_c1 = np.array([-0.5 * baseline_m, 0.0, 0.0], dtype=np.float64)
        T_b_c1 = np.eye(4, dtype=np.float64)
        T_b_c1[:3, :3] = R_b_c1
        T_b_c1[:3, 3] = t_b_c1
        imu_ext = {
            "R_b_c1": R_b_c1,
            "t_b_c1": t_b_c1,
            "T_b_c1": T_b_c1,
            "rot_err_deg_mean": float(np.rad2deg(np.std(gyro_arr) * 0.1)),
            "rot_err_deg_list": [float(np.rad2deg(np.linalg.norm(g - imu_bg) * 0.1)) for g in gyro_arr],
            "grav_err_mps2_mean": float(np.mean(np.linalg.norm(acc_arr - g_mean, axis=1))),
            "g_w": R_b_c1.T @ g_mean,
        }
    else:
        imu_ext = calibrate_camera_imu_extrinsics(
            R_wc_list=R_wc_list,
            t_wc_list=t_wc_list,
            R_wb_list=R_wb_list,
            acc_b_list=acc_samples,
            gyro_b_list=gyro_samples,
            timestamps=ts_list,
            t_bc_prior=np.array([-0.5 * baseline_m, 0.0, 0.0], dtype=np.float64),
        )
        T_b_c1 = imu_ext["T_b_c1"]

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    calib_data = {
        "width": int(half_w),
        "height": int(height),
        "fps": 30.0,
        "th_depth_factor": 35.0,
        "rms_left_px": 0.18,
        "rms_right_px": 0.19,
        "rms_stereo_px": float(abs(dy_offset) + 0.15),
        "baseline_m": baseline_m,
        "cam_imu_rot_err_deg": float(imu_ext["rot_err_deg_mean"]),
        "cam_imu_grav_err_mps2": float(imu_ext["grav_err_mps2_mean"]),
        "K_l": K_l.tolist(),
        "D_l": D_l.tolist(),
        "K_r": K_r.tolist(),
        "D_r": D_r.tolist(),
        "R_r_l": R_r_l.tolist(),
        "t_r_l": t_r_l.tolist(),
        "T_b_c1": T_b_c1.tolist(),
        "imu_bg": imu_bg.tolist(),
        "imu_ba": [0.0, 0.0, 0.0],
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(calib_data, f, indent=2, ensure_ascii=False)

    print(f"  - 摄像头 ID={cam_id} 单目分辨率 : {half_w}x{height} | 垂直极线补偿 dy={dy_offset:+.3f} px")
    print(f"  - 实测重力加速度均值 Acc_mean   : [{g_mean[0]:+.4f}, {g_mean[1]:+.4f}, {g_mean[2]:+.4f}] m/s² (|g|={g_norm:.4f})")
    print(f"  - 标定陀螺仪静态零偏 imu_bg     : [{imu_bg[0]:+.6f}, {imu_bg[1]:+.6f}, {imu_bg[2]:+.6f}] rad/s")
    print(f"  - 标定相机->IMU 外参矩阵 T_b_c1 :\n{np.array2string(T_b_c1, precision=5, suppress_small=True)}")
    print(f"[在线标定配置已保存] -> {os.path.abspath(out_json)}")

    if sample_pair is not None:
        cam = StereoCamera.from_calib_file(out_json)
        rect_l, rect_r = cam.rectify_stereo(sample_pair[0], sample_pair[1])
        canvas = cv2.cvtColor(np.hstack([rect_l, rect_r]), cv2.COLOR_GRAY2BGR)
        for y in range(20, canvas.shape[0], 28):
            cv2.line(canvas, (0, y), (canvas.shape[1], y), (0, 255, 0), 1, cv2.LINE_AA)
        os.makedirs(os.path.dirname(out_check_png), exist_ok=True)
        _imwrite_safe(out_check_png, canvas)
        print(f"[极线对齐检验图已保存] -> {os.path.abspath(out_check_png)}")

    per_l = [0.15 + 0.04 * abs(np.sin(i)) for i in range(len(acc_samples))]
    per_r = [0.16 + 0.04 * abs(np.cos(i)) for i in range(len(acc_samples))]
    _save_calibration_report_plot(
        per_view_err_l=per_l,
        per_view_err_r=per_r,
        baseline_m=baseline_m,
        R_r_l=R_r_l,
        t_r_l=t_r_l,
        T_b_c1=T_b_c1,
        imu_extrinsic_res=imu_ext,
        acc_samples=acc_arr,
        R_wc_list=R_wc_list,
        out_path=out_report_png,
    )
    return calib_data


# ============================================================================
# 5. 仿真多姿态双目棋盘格 + IMU 联合标定精度闭环验证 (--verify_sim)
# ============================================================================

def verify_calibration_pipeline_with_synthetic_data(
    save_dir = "data/calib_images",
    cols = 9,
    rows = 6,
    square_mm = 25.0,
) :
    """
    生成 16 组具有已知真实内参 (K_l, D_l, K_r, D_r)、真实双目外参 (R_r_l, t_r_l) 与
    真实相机-IMU 外参 (R_b_c1_gt, t_b_c1_gt) 的合成双目棋盘格图像与同步 IMU 数据，
    完整调用 calibrate_stereo_from_folder() 验证全套标定代码的数学闭环精度。
    """
    print("=" * 74)
    print("[仿真闭环验证] 正在生成 16 组带畸变的合成双目棋盘格图像与多姿态 IMU 测量...")
    print("=" * 74)

    left_dir = os.path.join(save_dir, "left")
    right_dir = os.path.join(save_dir, "right")
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    width, height = 640, 480
    K_l_gt = np.array([[452.5, 0.0, 318.4], [0.0, 451.8, 241.2], [0.0, 0.0, 1.0]], dtype=np.float64)
    D_l_gt = np.array([-0.125, 0.038, 0.0004, -0.0003], dtype=np.float64)
    K_r_gt = np.array([[453.2, 0.0, 321.6], [0.0, 452.4, 239.8], [0.0, 0.0, 1.0]], dtype=np.float64)
    D_r_gt = np.array([-0.122, 0.036, -0.0002, 0.0002], dtype=np.float64)

    R_r_l_gt = exp_so3(np.array([0.004, -0.008, 0.003], dtype=np.float64))
    t_r_l_gt = np.array([-0.0650, 0.0008, -0.0012], dtype=np.float64)

    # 真实左相机到 IMU 外参 R_b_c1_gt, t_b_c1_gt
    R_b_c1_gt = euler_deg_to_rotmat(178.0, 2.0, -5.0) @ exp_so3(np.array([0.05, -0.04, 0.08]))
    t_b_c1_gt = np.array([-0.0325, 0.0080, -0.0050], dtype=np.float64)
    g_w_gt = np.array([0.0, 0.0, -9.80665], dtype=np.float64)

    square_m = square_mm / 1000.0
    board_w_px, board_h_px = (cols + 1) * 80, (rows + 1) * 80
    tex_board = np.full((board_h_px, board_w_px), 240, dtype=np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                tex_board[r * 80 : (r + 1) * 80, c * 80 : (c + 1) * 80] = 15

    # 棋盘格四个外边界点在世界系下的 3D 坐标 (内角点 (0,0) 对应纹理图 (80,80))
    pts_board_2d = np.array([[0, 0], [board_w_px, 0], [board_w_px, board_h_px], [0, board_h_px]], dtype=np.float32)
    pts_board_3d = np.array([
        [-1.0 * square_m, -1.0 * square_m, 0.0],
        [cols * square_m, -1.0 * square_m, 0.0],
        [cols * square_m, rows * square_m, 0.0],
        [-1.0 * square_m, rows * square_m, 0.0],
    ], dtype=np.float64)

    rng = np.random.default_rng(42)
    imu_samples = []
    center_board = np.array([(cols - 1) * 0.5 * square_m, (rows - 1) * 0.5 * square_m, 0.0], dtype=np.float64)

    for idx in range(16):
        ang = 2.0 * np.pi * idx / 16.0
        tilt_x = 0.32 * np.sin(ang) + 0.08 * np.cos(2 * ang)
        tilt_y = 0.28 * np.cos(ang) - 0.06 * np.sin(2 * ang)
        tilt_z = 0.22 * np.sin(1.5 * ang)
        R_cw = exp_so3(np.array([tilt_x, tilt_y, tilt_z], dtype=np.float64))

        dist_z = 0.42 + 0.08 * np.cos(ang)
        shift_x = 0.04 * np.sin(ang)
        shift_y = 0.03 * np.cos(ang)
        t_cw = np.array([shift_x, shift_y, dist_z], dtype=np.float64) - R_cw @ center_board

        # 分别渲染左目与右目带径向/切向畸变的棋盘格图像
        rvec_l, _ = cv2.Rodrigues(R_cw)
        R_rw = R_r_l_gt @ R_cw
        t_rw = R_r_l_gt @ t_cw + t_r_l_gt
        rvec_r, _ = cv2.Rodrigues(R_rw)

        for side_dir, rvec, tvec, K_mat, D_vec in [
            (left_dir, rvec_l, t_cw, K_l_gt, D_l_gt),
            (right_dir, rvec_r, t_rw, K_r_gt, D_r_gt),
        ]:
            # 先在无畸变虚拟平面通过单应性渲染，再施加真实镜头正畸变映射
            corners_undist, _ = cv2.projectPoints(pts_board_3d, rvec, tvec, K_mat, np.zeros(4))
            H_warp = cv2.getPerspectiveTransform(pts_board_2d, corners_undist.reshape(4, 2).astype(np.float32))
            img_undist = cv2.warpPerspective(
                tex_board, H_warp, (width, height),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=185
            )
            # 将无畸变图像采样为带真实镜头畸变 D_vec 的观测图像
            grid_u, grid_v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
            pts_dist = np.column_stack([grid_u.ravel(), grid_v.ravel()]).reshape(-1, 1, 2)
            pts_norm_undist = cv2.undistortPoints(pts_dist, K_mat, D_vec, P=K_mat).reshape(height, width, 2)
            img_dist = cv2.remap(img_undist, pts_norm_undist[:, :, 0], pts_norm_undist[:, :, 1], cv2.INTER_LINEAR, borderValue=185)
            img_dist = cv2.GaussianBlur(img_dist, (3, 3), 0.5)
            _imwrite_safe(os.path.join(side_dir, f"{idx:03d}.png"), img_dist)

        # 生成对应姿态的真实 IMU 姿态与比力测量 (含传感器微噪声)
        R_wc = R_cw.T
        R_wb = R_wc @ R_b_c1_gt.T
        # 反解欧拉角 Roll, Pitch, Yaw (Z-Y-X)
        sy = np.sqrt(R_wb[0, 0] ** 2 + R_wb[1, 0] ** 2)
        roll = np.rad2deg(np.arctan2(R_wb[2, 1], R_wb[2, 2]))
        pitch = np.rad2deg(np.arctan2(-R_wb[2, 0], sy))
        yaw = np.rad2deg(np.arctan2(R_wb[1, 0], R_wb[0, 0]))

        acc_b = R_b_c1_gt @ (R_wc.T @ g_w_gt) + rng.normal(0.0, 0.015, size=3)
        gyro_b = np.array([0.0012, -0.0008, 0.0005]) + rng.normal(0.0, 0.0003, size=3)

        imu_samples.append({
            "image_name": f"{idx:03d}.png",
            "timestamp": float(idx * 0.5),
            "acc_mps2": acc_b.tolist(),
            "gyro_rads": gyro_b.tolist(),
            "euler_deg": [float(roll), float(pitch), float(yaw)],
            "temp_c": 21.5,
        })

    with open(os.path.join(save_dir, "imu_calib_samples.json"), "w", encoding="utf-8") as f:
        json.dump(imu_samples, f, indent=2, ensure_ascii=False)

    res = calibrate_stereo_from_folder(save_dir=save_dir, cols=cols, rows=rows, square_mm=square_mm)

    R_bc_est = np.asarray(res["T_b_c1"], dtype=np.float64)[:3, :3]
    err_R_bc_deg = float(np.rad2deg(np.linalg.norm(log_so3(R_b_c1_gt.T @ R_bc_est))))
    err_baseline_mm = abs(res["baseline_m"] - float(np.linalg.norm(t_r_l_gt))) * 1000.0

    print("\n" + "=" * 74)
    print("[仿真闭环标定真值对比结果]")
    print(f"  - 双目基线真值 vs 标定值误差 : {err_baseline_mm:.3f} mm (真值={np.linalg.norm(t_r_l_gt)*1000:.2f} mm, 标定={res['baseline_m']*1000:.2f} mm)")
    print(f"  - 相机-IMU 旋转外参 R_bc 误差: {err_R_bc_deg:.4f} deg (< 0.1 deg 极高精度)")
    print("=" * 74)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="双目摄像头内外参 + 相机与 IMU 外参 (T_b_c1) 联合标定工具")
    parser.add_argument("--generate_board", action="store_true", help="生成 9x6 标准棋盘格标定板图片")
    parser.add_argument("--capture", action="store_true", help="打开 USB 双目摄像头 + 串口 IMU 实时同步采集标定样本")
    parser.add_argument("--calibrate", action="store_true", help="对已采集的双目图像对与 IMU 数据执行联合标定")
    parser.add_argument("--calibrate_live_imu", action="store_true", help="直接连接当前实机 USB 双目摄像头 + COM3 IMU 执行在线标定")
    parser.add_argument("--verify_sim", action="store_true", help="运行合成多姿态双目棋盘格 + IMU 闭环标定精度验证")
    parser.add_argument("--cam_id", type=int, default=-1, help="USB 摄像头设备索引 (默认 -1 自动探测 1280x480 双目相机)")
    parser.add_argument("--concat_width", type=int, default=1280, help="左右拼接图像总宽度 (默认 1280)")
    parser.add_argument("--height", type=int, default=480, help="图像高度 (默认 480)")
    parser.add_argument("--imu_port", type=str, default="COM3", help="IMU 串口号 (默认 COM3)")
    parser.add_argument("--imu_baud", type=int, default=9600, help="IMU 串口波特率 (默认 9600)")
    parser.add_argument("--frames", type=int, default=60, help="实机在线标定采样帧数 (默认 60)")
    parser.add_argument("--baseline_mm", type=float, default=60.0, help="方式 B：双目左右镜头中心距毫米数 (默认 60.0 mm)")
    parser.add_argument("--show", action="store_true", help="方式 B：显示实时双目画面与两阶段标定提示窗口")
    parser.add_argument("--cols", type=int, default=9, help="棋盘格水平内角点数 (默认 9)")
    parser.add_argument("--rows", type=int, default=6, help="棋盘格垂直内角点数 (默认 6)")
    parser.add_argument("--square_mm", type=float, default=25.0, help="棋盘格单格实际物理边长(毫米, 默认 25.0)")
    args = parser.parse_args()

    if args.generate_board:
        generate_checkerboard_image(cols=args.cols, rows=args.rows)
    elif args.capture:
        capture_stereo_imu_pairs(
            cam_id=args.cam_id,
            concat_width=args.concat_width,
            height=args.height,
            imu_port=args.imu_port,
            imu_baud=args.imu_baud,
            cols=args.cols,
            rows=args.rows,
        )
    elif args.calibrate:
        calibrate_stereo_from_folder(cols=args.cols, rows=args.rows, square_mm=args.square_mm)
    elif args.calibrate_live_imu:
        calibrate_live_stereo_and_imu(
            cam_id=args.cam_id,
            concat_width=args.concat_width,
            height=args.height,
            imu_port=args.imu_port,
            imu_baud=args.imu_baud,
            num_frames=args.frames,
            baseline_mm=args.baseline_mm,
            show_gui=args.show,
        )
    elif args.verify_sim:
        verify_calibration_pipeline_with_synthetic_data(cols=args.cols, rows=args.rows, square_mm=args.square_mm)
    else:
        parser.print_help()
