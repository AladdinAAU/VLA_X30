"""Camera intrinsics calibration with a checkerboard (OpenCV), writes configs/calibration/camera.yaml.

    calibrate_camera --images DIR --board 9x6 --square 0.025                  # photos of the board (jpg/png)
    calibrate_camera --rtsp rtsp://192.168.1.105:8554/test --capture 25 ...   # grab frames from the stream
        (stream mode shows a preview window; SPACE stores a frame when the board is found, ESC finishes;
         --auto stores a frame every --interval seconds when the board is detected, without a window)
--board is the number of INNER corners (columns x rows) of the printed board, --square the square size in metres.
Hold the board at many distances and tilts, fill the corners of the image, keep it still (use 20-30 views). Aim for an RMS
reprojection error below 0.5 px; the file is refused above 1.0 px unless --force.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import yaml


def find_corners(gray: np.ndarray, board: Tuple[int, int]):
    import cv2

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, board, flags)
    if not ok:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)


def calibrate(corner_sets: List[np.ndarray], size: Tuple[int, int], board: Tuple[int, int], square: float,
              camera_name: str = "x30_camera") -> Dict:
    """size = (width, height). Returns the dictionary written to camera.yaml (OpenCV pinhole + plumb_bob, 5 coefficients)."""
    import cv2

    objp = np.zeros((board[0] * board[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2) * square
    rms, k, dist, rvecs, tvecs = cv2.calibrateCamera([objp] * len(corner_sets), corner_sets, size, None, None)
    per_view = []
    for o, c, r, t in zip([objp] * len(corner_sets), corner_sets, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, k, dist)
        per_view.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - c.reshape(-1, 2)) ** 2, axis=1)))))
    p = np.hstack([k, np.zeros((3, 1))])
    return {
        "image_width": int(size[0]), "image_height": int(size[1]), "camera_name": camera_name,
        "camera_matrix": {"rows": 3, "cols": 3, "data": [float(x) for x in k.reshape(-1)]},
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {"rows": 1, "cols": int(dist.size), "data": [float(x) for x in dist.reshape(-1)]},
        "rectification_matrix": {"rows": 3, "cols": 3, "data": [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]},
        "projection_matrix": {"rows": 3, "cols": 4, "data": [float(x) for x in p.reshape(-1)]},
        "rms_reprojection_error": float(rms), "per_view_rms": per_view, "num_views": len(corner_sets),
        "board": {"inner_corners": list(board), "square_m": square}, "calibrated": True,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def corners_from_images(paths: List[str], board: Tuple[int, int]):
    import cv2

    sets, size, used = [], None, []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = (gray.shape[1], gray.shape[0])
        c = find_corners(gray, board)
        if c is not None:
            sets.append(c)
            used.append(p)
    return sets, size, used


def capture_from_stream(url: str, board, n: int, auto: bool, interval: float, backend: str = "ffmpeg"):
    import cv2

    from x30_hw.rtsp_source import open_capture

    cap = open_capture(url, backend, "tcp")
    if not cap.isOpened():
        raise SystemExit(f"cannot open {url}")
    sets, size, last = [], None, 0.0
    gui = not auto and bool(os.environ.get("DISPLAY"))
    while len(sets) < n:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = (gray.shape[1], gray.shape[0])
        c = find_corners(gray, board)
        if gui:
            vis = frame.copy()
            if c is not None:
                cv2.drawChessboardCorners(vis, board, c, True)
            cv2.putText(vis, f"{len(sets)}/{n} (SPACE store, ESC finish)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("calibrate_camera", vis)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                break
            if key == 32 and c is not None:
                sets.append(c)
        elif c is not None and time.time() - last > interval:
            sets.append(c)
            last = time.time()
            print(f"stored view {len(sets)}/{n}", flush=True)
    cap.release()
    return sets, size


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", help="directory with board photos")
    ap.add_argument("--rtsp", help="stream URL (or synthetic:// for tests)")
    ap.add_argument("--capture", type=int, default=25)
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--interval", type=float, default=1.5)
    ap.add_argument("--board", default="9x6", help="inner corners COLSxROWS")
    ap.add_argument("--square", type=float, required=True, help="square size in metres")
    ap.add_argument("--out", default=None)
    ap.add_argument("--camera-name", default="x30_camera")
    ap.add_argument("--min-views", type=int, default=10)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    board = tuple(int(x) for x in a.board.lower().split("x"))
    if a.images:
        paths = sorted(p for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp") for p in glob.glob(os.path.join(a.images, ext)))
        sets, size, used = corners_from_images(paths, board)
        print(f"board found in {len(sets)} of {len(paths)} images")
    elif a.rtsp:
        sets, size = capture_from_stream(a.rtsp, board, a.capture, a.auto, a.interval)
    else:
        ap.error("give --images or --rtsp")
    if len(sets) < a.min_views:
        print(f"only {len(sets)} usable views (< {a.min_views}); take more photos with different distances and tilts")
        return 2
    result = calibrate(sets, size, board, a.square, a.camera_name)
    print(f"RMS reprojection error {result['rms_reprojection_error']:.3f} px over {len(sets)} views; "
          f"fx={result['camera_matrix']['data'][0]:.1f} fy={result['camera_matrix']['data'][4]:.1f} "
          f"cx={result['camera_matrix']['data'][2]:.1f} cy={result['camera_matrix']['data'][5]:.1f}")
    if result["rms_reprojection_error"] > 1.0 and not a.force:
        print("RMS above 1.0 px: file NOT written (bad views? blurred board? wrong --square/--board?). Use --force to override.")
        return 3
    from x30_hw import config as cfgmod

    out = a.out or os.path.join(cfgmod.configs_dir(), "calibration", "camera.yaml")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(result, f, sort_keys=False)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
