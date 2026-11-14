# -*- coding: utf-8 -*-
"""
calibrate_mono_camera.py
========================
单目摄像头棋盘格标定工具

使用方法:
  1. 准备一张打印的棋盘格标定板 (默认 9x6 内角点, 方格边长 25mm)
  2. 运行: python calibrate_mono_camera.py
  3. 将棋盘格对着摄像头，按 [空格] 拍照，按 [Q] 结束拍照并开始标定
  4. 标定完成后自动更新 config/custom_stereo_imu.json

拍摄要求 (15~25 张):
  - 棋盘格要占画面 1/4 ~ 3/4 大小
  - 覆盖画面四个角落 + 中间
  - 包含不同倾斜角度 (正面、左倾、右倾、上倾、下倾)
  - 棋盘格要完整出现在画面内 (不能被裁切)
"""

import os
import json
import time
import numpy as np
import cv2


# ============ 配置参数 ============
CAM_ID = 1                      # USB 摄像头编号
WIDTH, HEIGHT = 640, 480         # 分辨率
BOARD_COLS = 11                   # 棋盘格内角点列数 (12列方格 - 1 = 11)
BOARD_ROWS = 8                    # 棋盘格内角点行数 (9行方格 - 1 = 8)
SQUARE_SIZE_MM = 25.0            # 每个方格的边长 (毫米)
MIN_CAPTURES = 15                # 最少拍摄张数
MAX_CAPTURES = 30                # 最多拍摄张数
SAVE_DIR = "calib_images"        # 拍摄图片保存目录
CONFIG_PATH = "config/custom_stereo_imu.json"


def main():
    print("=" * 60)
    print("  单目摄像头棋盘格标定工具")
    print("=" * 60)
    print(f"  棋盘格规格: {BOARD_COLS} x {BOARD_ROWS} 内角点, 方格 {SQUARE_SIZE_MM}mm")
    print(f"  摄像头 ID: {CAM_ID}  |  分辨率: {WIDTH}x{HEIGHT}")
    print()
    print("  操作说明:")
    print("    [空格] = 拍照    [Q] = 结束拍照并标定")
    print()
    print("  拍摄技巧 (建议拍 15~25 张):")
    print("    1. 棋盘格占画面 1/4 ~ 3/4")
    print("    2. 在画面左上、右上、左下、右下、中间各拍几张")
    print("    3. 正面 + 向左/右/上/下倾斜各拍几张")
    print("    4. 棋盘格必须完整在画面内")
    print("=" * 60)

    os.makedirs(SAVE_DIR, exist_ok=True)

    cap = cv2.VideoCapture(CAM_ID)
    if not cap.isOpened():
        print(f"[错误] 无法打开摄像头 ID={CAM_ID}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    board_size = (BOARD_COLS, BOARD_ROWS)
    captured_images = []
    captured_corners = []

    # 棋盘格 3D 世界坐标 (Z=0 平面)
    objp = np.zeros((BOARD_ROWS * BOARD_COLS, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM  # 单位: mm

    print("\n[实时预览] 将棋盘格对准摄像头...")

    last_capture_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 实时检测棋盘格角点
        found, corners = cv2.findChessboardCorners(
            gray, board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
        )

        if found:
            # 亚像素精化
            corners_sub = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            cv2.drawChessboardCorners(display, board_size, corners_sub, found)
            status_text = f"[已检测到棋盘格] 按 [空格] 拍照 | 已拍: {len(captured_images)}/{MIN_CAPTURES}+"
            status_color = (0, 255, 0)
        else:
            status_text = f"[未检测到棋盘格] 已拍: {len(captured_images)}/{MIN_CAPTURES}+"
            status_color = (0, 0, 255)

        # 显示拍摄进度
        cv2.putText(display, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        if len(captured_images) >= MIN_CAPTURES:
            cv2.putText(display, "Press [Q] to finish and calibrate",
                        (10, HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Calibration - Press SPACE to capture, Q to finish", display)
        key = cv2.waitKey(30) & 0xFF

        now = time.time()

        if key == ord(' ') and found and (now - last_capture_time > 1.0):
            # 保存拍照
            idx = len(captured_images)
            img_path = os.path.join(SAVE_DIR, f"calib_{idx:03d}.png")
            cv2.imwrite(img_path, frame)
            captured_images.append(frame.copy())
            captured_corners.append((objp.copy(), corners_sub.copy()))
            last_capture_time = now
            print(f"  [拍照 {idx + 1}] 已保存 -> {img_path} (随时按 [Q] 结束拍照)")

        elif key == ord('q') or key == ord('Q'):
            print("\n  [用户退出] 检测到按键 'q'，正在关闭摄像头窗口...")
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(captured_images) < MIN_CAPTURES:
        print(f"\n[中止] 图片不足 {MIN_CAPTURES} 张，标定取消")
        return

    # ============ 开始标定 ============
    print(f"\n{'=' * 60}")
    print(f"  开始标定... ({len(captured_images)} 张图片)")
    print(f"{'=' * 60}")

    obj_points = [c[0] for c in captured_corners]
    img_points = [c[1] for c in captured_corners]

    rms, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (WIDTH, HEIGHT), None, None
    )

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    k1, k2, p1, p2 = [float(x) for x in dist_coeffs[0][:4]]

    print(f"\n  标定完成!")
    print(f"  RMS 重投影误差: {rms:.4f} px (< 0.5 优秀, < 1.0 良好)")
    print(f"\n  内参矩阵 K:")
    print(f"    fx = {fx:.2f}")
    print(f"    fy = {fy:.2f}")
    print(f"    cx = {cx:.2f}")
    print(f"    cy = {cy:.2f}")
    print(f"\n  畸变系数:")
    print(f"    k1 = {k1:.6f}")
    print(f"    k2 = {k2:.6f}")
    print(f"    p1 = {p1:.6f}")
    print(f"    p2 = {p2:.6f}")

    # 计算每张图的重投影误差
    print(f"\n  各图片重投影误差:")
    errors = []
    for i in range(len(obj_points)):
        proj_pts, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist_coeffs)
        err = float(cv2.norm(img_points[i], proj_pts, cv2.NORM_L2) / len(proj_pts))
        errors.append(err)
        marker = " ✗" if err > 1.5 else ""
        print(f"    图 {i + 1:2d}: {err:.4f} px{marker}")

    # ============ 更新配置文件 ============
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {}

    config["K_l"] = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    config["K_r"] = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    config["D_l"] = [k1, k2, p1, p2]
    config["D_r"] = [k1, k2, p1, p2]
    config["rms_left_px"] = round(rms, 4)
    config["rms_right_px"] = round(rms, 4)

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n  [已保存] 标定结果已写入 -> {CONFIG_PATH}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
