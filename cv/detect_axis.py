import argparse
import json
import math
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def _noop(_: int) -> None:
    pass


def load_config(config_path: Path) -> dict:
    """Load JSON config from disk. Returns empty dict if absent/invalid."""
    if not config_path.exists():
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Config read error: {exc}")
        return {}
    if not isinstance(data, dict):
        print("Config root must be an object. Ignoring file.")
        return {}
    return data


def wrap_angle_deg(angle: float) -> float:
    """Normalize angle to [-90, 90)."""
    while angle >= 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    return angle


def detect_axis_pose(
    frame: np.ndarray,
    canny_low: int,
    canny_high: int,
    hough_threshold: int,
    min_line_length: int,
    max_line_gap: int,
    min_axis_length: int,
) -> Tuple[Optional[Tuple[int, int]], Optional[float], np.ndarray, np.ndarray]:
    """
    Detect elongated object by dominant line axis.
    Returns center(x,y), angle(deg), visualization and debug frame.
    """
    vis = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, canny_low, canny_high)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    lines = cv2.HoughLinesP(
        edges_closed,
        rho=1,
        theta=np.pi / 180.0,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    debug = cv2.cvtColor(edges_closed, cv2.COLOR_GRAY2BGR)
    if lines is None or len(lines) == 0:
        cv2.putText(
            vis,
            "Axis not found",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (20, 20, 200),
            2,
            cv2.LINE_AA,
        )
        return None, None, vis, debug

    # Collect long enough candidate segments.
    segments = []
    for item in lines:
        x1, y1, x2, y2 = item[0]
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = math.hypot(dx, dy)
        if length < min_axis_length:
            continue
        angle = wrap_angle_deg(math.degrees(math.atan2(dy, dx)))
        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0
        segments.append((length, angle, mx, my, x1, y1, x2, y2))

    for _, _, _, _, x1, y1, x2, y2 in segments:
        cv2.line(debug, (x1, y1), (x2, y2), (80, 80, 80), 1)

    if not segments:
        cv2.putText(
            vis,
            "No long lines",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (20, 20, 200),
            2,
            cv2.LINE_AA,
        )
        return None, None, vis, debug

    # Weighted average in doubled-angle space to avoid +/-180 ambiguity.
    sum_w = 0.0
    sum_cos = 0.0
    sum_sin = 0.0
    sum_mx = 0.0
    sum_my = 0.0
    for length, angle, mx, my, *_ in segments:
        w = length
        a2 = math.radians(2.0 * angle)
        sum_w += w
        sum_cos += w * math.cos(a2)
        sum_sin += w * math.sin(a2)
        sum_mx += w * mx
        sum_my += w * my

    if sum_w <= 1e-6:
        return None, None, vis, debug

    angle = wrap_angle_deg(0.5 * math.degrees(math.atan2(sum_sin, sum_cos)))
    cx = int(round(sum_mx / sum_w))
    cy = int(round(sum_my / sum_w))
    center = (cx, cy)

    # Draw dominant axis.
    axis_len = int(max(min_line_length, min_axis_length) * 1.2)
    theta = math.radians(angle)
    dx = int(math.cos(theta) * axis_len)
    dy = int(math.sin(theta) * axis_len)
    p1 = (cx - dx, cy - dy)
    p2 = (cx + dx, cy + dy)
    cv2.line(vis, p1, p2, (255, 0, 0), 3)
    cv2.circle(vis, center, 5, (0, 0, 255), -1)

    cv2.putText(
        vis,
        f"X: {cx}  Y: {cy}  Angle: {angle:.1f} deg",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )

    cv2.line(debug, p1, p2, (255, 0, 0), 2)
    cv2.circle(debug, center, 4, (0, 0, 255), -1)
    cv2.putText(
        debug,
        f"segments={len(segments)}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return center, angle, vis, debug


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect object pose by dominant line axis (HoughLinesP)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="axis_config.json",
        help="Path to JSON with startup parameters (read-only by script)",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--canny-low", type=int, default=60, help="Lower Canny threshold")
    parser.add_argument("--canny-high", type=int, default=160, help="Upper Canny threshold")
    parser.add_argument("--hough-th", type=int, default=45, help="HoughLinesP threshold")
    parser.add_argument("--min-line", type=int, default=50, help="Hough minLineLength")
    parser.add_argument("--max-gap", type=int, default=15, help="Hough maxLineGap")
    parser.add_argument("--min-axis", type=int, default=40, help="Min segment length to vote axis")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    camera = int(cfg.get("camera", args.camera))
    canny_low0 = int(cfg.get("canny_low", args.canny_low))
    canny_high0 = int(cfg.get("canny_high", args.canny_high))
    hough_th0 = int(cfg.get("hough_th", args.hough_th))
    min_line0 = int(cfg.get("min_line", args.min_line))
    max_gap0 = int(cfg.get("max_gap", args.max_gap))
    min_axis0 = int(cfg.get("min_axis", args.min_axis))

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera. Check camera index or permissions.")

    cv2.namedWindow("Axis pose detection", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Axis debug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Axis pose detection", 960, 720)
    cv2.resizeWindow("Axis debug", 640, 480)

    cv2.createTrackbar("canny_low", "Axis debug", max(0, min(255, canny_low0)), 255, _noop)
    cv2.createTrackbar("canny_high", "Axis debug", max(0, min(255, canny_high0)), 255, _noop)
    cv2.createTrackbar("hough_th", "Axis debug", max(1, min(255, hough_th0)), 255, _noop)
    cv2.createTrackbar("min_line", "Axis debug", max(5, min(400, min_line0)), 400, _noop)
    cv2.createTrackbar("max_gap", "Axis debug", max(0, min(100, max_gap0)), 100, _noop)
    cv2.createTrackbar("min_axis", "Axis debug", max(5, min(400, min_axis0)), 400, _noop)

    print("Press 'q' to quit.")
    print("Tune parameters in 'Axis debug' window.")
    print(f"Config file (read-only): {Path(args.config).resolve()}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera.")
                break

            canny_low = cv2.getTrackbarPos("canny_low", "Axis debug")
            canny_high = cv2.getTrackbarPos("canny_high", "Axis debug")
            hough_th = max(1, cv2.getTrackbarPos("hough_th", "Axis debug"))
            min_line = max(5, cv2.getTrackbarPos("min_line", "Axis debug"))
            max_gap = cv2.getTrackbarPos("max_gap", "Axis debug")
            min_axis = max(5, cv2.getTrackbarPos("min_axis", "Axis debug"))

            if canny_low >= canny_high:
                canny_high = min(255, canny_low + 1)
                cv2.setTrackbarPos("canny_high", "Axis debug", canny_high)

            center, angle, vis, debug = detect_axis_pose(
                frame=frame,
                canny_low=canny_low,
                canny_high=canny_high,
                hough_threshold=hough_th,
                min_line_length=min_line,
                max_line_gap=max_gap,
                min_axis_length=min_axis,
            )

            if center is not None and angle is not None:
                print(f"\rX={center[0]:4d}, Y={center[1]:4d}, angle={angle:6.2f} deg", end="")
            else:
                print("\rAxis not found. Adjust trackbars...                ", end="")

            cv2.putText(
                debug,
                f"Canny {canny_low}-{canny_high} hough={hough_th} min_line={min_line}",
                (10, max(40, debug.shape[0] - 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 200, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                debug,
                f"max_gap={max_gap} min_axis={min_axis}",
                (10, max(65, debug.shape[0] - 40)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 220),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Axis pose detection", vis)
            cv2.imshow("Axis debug", debug)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        print()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
