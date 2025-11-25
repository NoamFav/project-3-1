import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.signal import savgol_filter

# CONFIGURATION 
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_5X5_100
)  # change for appropriate ArUco code
MARKER_ID = None
PIXELS_PER_CM = 4.0   # adjust when camera calibration is known
SMOOTH_WINDOW = 9

app = FastAPI(
    title="Robot Speed Window Service",
    description="Computes a winning rate based on robot speed in a time window using ArUco markers.",
    version="1.0",
)


class WindowRequest(BaseModel):
    video_path: str
    window_start: float  # seconds
    window_end: float    # seconds


def compute_speed_for_window(video_path: str, window_start: float, window_end: float):
    """
    Process only the specified time window of the video and
    return average speed (cm/s) and number of detections.
    """
    if window_end <= window_start:
        raise ValueError("window_end must be greater than window_start")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    positions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Current timestamp in seconds
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # Skip frames before the window
        if t < window_start:
            continue
        # Stop once we are past the window
        if t >= window_end:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT)

        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if MARKER_ID is None or marker_id == MARKER_ID:
                    c = corners[i][0]
                    center = np.mean(c, axis=0)
                    x, y = center
                    positions.append((t, x, y))

    cap.release()

    if len(positions) <= 1:
        # Not enough detections in this window to compute speed
        return np.nan, 0

    positions = np.array(positions, dtype=float)
    t, x, y = positions[:, 0], positions[:, 1], positions[:, 2]

    # Smooth positions for stability (same as your original code) 
    if len(x) > SMOOTH_WINDOW:
        x = savgol_filter(x, SMOOTH_WINDOW, 3)
        y = savgol_filter(y, SMOOTH_WINDOW, 3)

    dt = np.diff(t)
    dx, dy = np.diff(x), np.diff(y)
    dt[dt == 0] = np.finfo(float).eps  # avoid division by zero

    vx = dx / dt
    vy = dy / dt
    v = np.sqrt(vx**2 + vy**2)

    # align lengths
    v = np.insert(v, 0, np.nan)

    # Convert to cm/s
    v_cm = v / PIXELS_PER_CM

    avg_speed_cm_s = float(np.nanmean(v_cm))
    num_detections = len(t)

    return avg_speed_cm_s, num_detections


def speed_to_winning_rate(avg_speed_cm_s: float) -> float:
    """
    Simple, monotonic winning-rate mapping.
    You can tune this later. For now:
    - 0 cm/s -> 0.0
    - 50 cm/s or more -> 1.0
    - linear in between
    """
    if np.isnan(avg_speed_cm_s):
        return 0.0

    target_speed = 50.0  # arbitrary scaling factor
    raw = avg_speed_cm_s / target_speed
    winning_rate = max(0.0, min(1.0, raw))
    return float(winning_rate)


@app.post("/winning_rate")
def compute_winning_rate(req: WindowRequest):
    """
    Compute robot winning rate for a given time window in a video.
    """
    try:
        avg_speed_cm_s, num_detections = compute_speed_for_window(
            req.video_path, req.window_start, req.window_end
        )

        winning_rate = speed_to_winning_rate(avg_speed_cm_s)

        return {
            "video_path": req.video_path,
            "window_start": req.window_start,
            "window_end": req.window_end,
            "avg_speed_cm_s": avg_speed_cm_s,
            "num_detections": num_detections,
            "winning_rate": winning_rate,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "healthy"}
