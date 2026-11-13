# -*- coding: utf-8 -*-
"""
calibrate_stereo_camera.py
==========================
同步拼接双目摄像头 (Side-by-Side UVC Stereo) + IMU 一键标定与极线对齐检验工具

支持 3 种工作子命令:
1. 生成标准棋盘格标定板图片 (可打印在 A4 纸或直接全屏投放在平板/显示器上):
   `python calibrate_stereo_camera.py --generate_board`
2. 实时打开 USB 同步拼接双目摄像头交互式采集标定图片对:
   `python calibrate_stereo_camera.py --capture --cam_id 0 --concat_width 1280 --height 480`
3. 执行双目内参、畸变系数与左右外参 (R_r_l, t_r_l) 联合标定，导出 JSON 并生成极线对齐检验图:
   `python calibrate_stereo_camera.py --calibrate --square_mm 25.0`
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import cv2
import numpy as np


def generate_checkerboard_image(
    cols: int = 9,
    rows: int = 6,
    square_px: int = 120,
    out_path: str = "outputs/checkerboard_9x6.png"
) -> None:
    """
    生成 (cols x rows) 内角点的黑白棋盘格高清图像 (即 (cols+1) x (rows+1) 个方格)
    """
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

    cv2.imwrite(out_path, board)
    print(f"[标定板已生成] 路径: {os.path.abspath(out_path)} (内角点: {cols}x{rows})")
    print("  提示: 可按 100% 比例打印在硬质平板 A4 纸上，或直接全屏显示在副屏/iPad 上，并用尺子实测单个黑格边长 (毫米)。")


def capture_stereo_pairs(
    cam_id: int = 0,
    concat_width: int = 1280,
    height: int = 480,
    cols: int = 9,
    rows: int = 6,
    save_dir: str = "data/calib_images"
) -> None:
    """
    打开单 USB 同步左右拼接双目摄像头 (分辨率 concat_width x height，切分为左/右两张 width/2 x height)，
    实时检测棋盘格角点，按 [Space] 或 [S] 保存有效图像对，按 [Q] 或 [ESC] 结束采集。
    """
    left_dir = os.path.join(save_dir, "left")
    right_dir = os.path.join(save_dir, "right")
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, concat_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    half_w = concat_width // 2
    pattern_size = (cols, rows)
    saved_count = len(glob.glob(os.path.join(left_dir, "*.png")))

    print("=" * 70)
    print(f"[实时双目采集] 已打开摄像头 ID={cam_id}, 拼接分辨率={concat_width}x{height} (单目={half_w}x{height})")
    print("  操作说明: 当左右画面同时出现彩色棋盘格角点连线时，按 [Space/空格] 或 [S] 保存；采集 15~25 张后按 [Q] 退出。")
    print("=" * 70)

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

        vis = np.hstack([vis_l, vis_r])
        status_color = (0, 255, 0) if (ok_l and ok_r) else (0, 165, 255)
        status_txt = f"Saved: {saved_count} pairs | Corners: {'OK (Press SPACE to Save)' if (ok_l and ok_r) else 'Searching...'}"
        cv2.putText(vis, status_txt, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2, cv2.LINE_AA)
        cv2.imshow("Stereo Calibration Capture (Left | Right)", vis)

        key = cv2.waitKey(15) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        elif key in (ord(" "), ord("s"), ord("S")):
            if ok_l and ok_r:
                idx_str = f"{saved_count:03d}.png"
                cv2.imwrite(os.path.join(left_dir, idx_str), img_l)
                cv2.imwrite(os.path.join(right_dir, idx_str), img_r)
                saved_count += 1
                print(f"  [已保存第 {saved_count:2d} 对标定图] -> {idx_str}")
            else:
                print("  [跳过] 当前帧左/右目未能同时完整检测到棋盘格角点，请调整角度！")

    cap.release()
    cv2.destroyAllWindows()


def calibrate_stereo_from_folder(
    save_dir: str = "data/calib_images",
    cols: int = 9,
    rows: int = 6,
    square_mm: float = 25.0,
    out_json: str = "config/custom_stereo_imu.json",
    out_check_png: str = "outputs/stereo_rectification_check.png"
) -> dict:
    """
    读取 data/calib_images/left 与 right 下的图像对，完成：
    1. 左、右单目独立内参与畸变标定 (K_l, D_l, K_r, D_r)
    2. 双目联合外参标定 (R_r_l, t_r_l, 基线长度 baseline)
    3. 导出 JSON 标定文件并绘制带水平极线的对齐检验图
    """
    left_files = sorted(glob.glob(os.path.join(save_dir, "left", "*.png")))
    right_files = sorted(glob.glob(os.path.join(save_dir, "right", "*.png")))
    if len(left_files) == 0 or len(left_files) != len(right_files):
        raise FileNotFoundError(
            f"在 {save_dir}/left 和 right 下未找到成对的标定图片 (当前数量: {len(left_files)} vs {len(right_files)})。"
        )

    pattern_size = (cols, rows)
    square_m = float(square_mm) / 1000.0  # 转换为物理米 (m)，直接决定 SLAM 绝对物理尺度！

    objp = np.zeros((cols * rows, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m

    obj_points = []
    img_points_l = []
    img_points_r = []
    img_size = None
    sample_pair: tuple[np.ndarray, np.ndarray] | None = None

    subpix_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)

    for lf, rf in zip(left_files, right_files):
        im_l = cv2.imread(lf, cv2.IMREAD_GRAYSCALE)
        im_r = cv2.imread(rf, cv2.IMREAD_GRAYSCALE)
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
            if sample_pair is None:
                sample_pair = (im_l, im_r)

    n_valid = len(obj_points)
    if n_valid < 6 or img_size is None:
        raise RuntimeError(f"有效棋盘格图像对过少 ({n_valid} 对)，建议采集 15~25 对不同姿态的图像。")

    print(f"[开始标定] 共有效匹配 {n_valid} 对棋盘格图像, 分辨率={img_size[0]}x{img_size[1]}, 方格边长={square_mm:.2f} mm")

    # 1. 左右单目各自独立标定内参
    err_l, K_l, D_l, _, _ = cv2.calibrateCamera(obj_points, img_points_l, img_size, None, None)
    err_r, K_r, D_r, _, _ = cv2.calibrateCamera(obj_points, img_points_r, img_size, None, None)

    # 2. 双目联合外参标定 (求解右目相对于左目的旋转 R_r_l 与平移 t_r_l)
    stereo_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    err_stereo, K_l, D_l, K_r, D_r, R_r_l, t_r_l, _, _ = cv2.stereoCalibrate(
        obj_points, img_points_l, img_points_r,
        K_l, D_l, K_r, D_r, img_size,
        criteria=stereo_criteria,
        flags=cv2.CALIB_FIX_INTRINSIC
    )

    baseline_m = float(np.linalg.norm(t_r_l))
    print("-" * 70)
    print(f"  - 左目重投影误差 (RMS) : {err_l:.4f} px")
    print(f"  - 右目重投影误差 (RMS) : {err_r:.4f} px")
    print(f"  - 双目极线联合误差(RMS): {err_stereo:.4f} px (小于 0.5 px 为优秀)")
    print(f"  - 实测双目物理基线 (b) : {baseline_m * 100.0:.2f} cm ({baseline_m:.5f} m)")
    print("-" * 70)

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
        "K_l": K_l.tolist(),
        "D_l": D_l.ravel()[:4].tolist(),
        "K_r": K_r.tolist(),
        "D_r": D_r.ravel()[:4].tolist(),
        "R_r_l": R_r_l.tolist(),
        "t_r_l": t_r_l.ravel().tolist(),
        "T_b_c1": np.eye(4, dtype=np.float64).tolist(),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(calib_data, f, indent=2, ensure_ascii=False)
    print(f"[标定配置已保存] -> {os.path.abspath(out_json)}")

    # 3. 生成双目极线水平对齐检验图
    if sample_pair is not None:
        from src.core.camera import StereoCamera
        cam = StereoCamera.from_calib_file(out_json)
        rect_l, rect_r = cam.rectify_stereo(sample_pair[0], sample_pair[1])
        canvas = cv2.cvtColor(np.hstack([rect_l, rect_r]), cv2.COLOR_GRAY2BGR)
        for y in range(20, canvas.shape[0], 28):
            cv2.line(canvas, (0, y), (canvas.shape[1], y), (0, 255, 0), 1, cv2.LINE_AA)
        os.makedirs(os.path.dirname(out_check_png), exist_ok=True)
        cv2.imwrite(out_check_png, canvas)
        print(f"[极线对齐检验图已保存] -> {os.path.abspath(out_check_png)}")

    return calib_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="双目摄像头与 IMU 标定工具")
    parser.add_argument("--generate_board", action="store_true", help="生成 9x6 标准棋盘格标定板图片")
    parser.add_argument("--capture", action="store_true", help="打开 USB 同步拼接双目摄像头实时采集标定图")
    parser.add_argument("--calibrate", action="store_true", help="对已采集的图像对执行双目内外参标定")
    parser.add_argument("--cam_id", type=int, default=0, help="USB 摄像头设备索引 (默认 0)")
    parser.add_argument("--concat_width", type=int, default=1280, help="左右拼接图像总宽度 (默认 1280)")
    parser.add_argument("--height", type=int, default=480, help="图像高度 (默认 480)")
    parser.add_argument("--cols", type=int, default=9, help="棋盘格水平内角点数 (默认 9)")
    parser.add_argument("--rows", type=int, default=6, help="棋盘格垂直内角点数 (默认 6)")
    parser.add_argument("--square_mm", type=float, default=25.0, help="棋盘格单格实际物理边长(毫米, 默认 25.0)")
    args = parser.parse_args()

    if args.generate_board:
        generate_checkerboard_image(cols=args.cols, rows=args.rows)
    elif args.capture:
        capture_stereo_pairs(
            cam_id=args.cam_id,
            concat_width=args.concat_width,
            height=args.height,
            cols=args.cols,
            rows=args.rows
        )
    elif args.calibrate:
        calibrate_stereo_from_folder(cols=args.cols, rows=args.rows, square_mm=args.square_mm)
    else:
        parser.print_help()
