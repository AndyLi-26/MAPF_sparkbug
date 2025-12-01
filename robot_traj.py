from pickle import TRUE
import cv2, time, sys, math, csv, os
import numpy as np
from collections import deque
import database



# =================== GLOBAL SETTINGS ===================
DEBUG = False
SHOW  = True
LIGHT = False                 # set True if using saturation-based circle fit
R     = 45                  # ring radius in px (≈60 @ 1080p, ≈100 @ 4k)
MIN_DIST = R * 1.8



SAT_GAIN   = 2.5
SAT_CUTOFF = 130



# =================== LED RING CONFIG ===================
N_LEDS        = 16
ANN_INNER     = 0.75
ANN_OUTER     = 1.05
THETA0_DEG    = -90.0
DIR           = 1.0           # +1 clockwise (image coords)
BLOB_R        = 3
MIN_V         = 100
TINY_REFINE_R = 2
TH_STEP_DEG   = 2.0



# Hue bands (OpenCV hue 0..179)
HUE_BANDS = {
    'Y': [(0, 50), (145, 180)],
    'G': [(70, 95)],
    'C': [(100, 105)],
    'M': [(115, 140)],
}



# =================== SMOOTHING (runtime [ / ]) ===================
paused        = False
POSE_WIN      = 20
POSE_KEEP_FR  = 0.6
POSE_MIN_KEEP = 3
POSE_WIN_MAX  = 50
POSE_WIN_MIN  = 3



# =================== KALMAN FILTER SETTINGS ===================
USE_KALMAN_FILTER = True      # <-- Toggle Kalman filtering
KALMAN_PROCESS_NOISE = 0.01   # Process noise (lower = smoother but slower response)
KALMAN_MEASUREMENT_NOISE = 5.0  # Measurement noise (higher = more smoothing)



# =================== TRAJECTORY SETTINGS ===================
TRAJECTORY_ENABLED = True
SHOW_TRAJECTORY_CANVAS = False  # <-- Toggle separate trajectory window
TRAJECTORY_SAMPLE_INTERVAL = 6  # <-- NEW: Record trajectory every N frames
TRAJECTORY_MAX_ALPHA = 0.8    # max opacity for newest point
TRAJECTORY_MIN_ALPHA = 0.1    # min opacity for oldest point
TRAJECTORY_LINE_THICKNESS = 2
TRAJECTORY_FADE_POWER = 1.5   # exponential fade (higher = faster fade)


# Distinctive color palette for up to 20 robots (BGR format)
ROBOT_COLORS = [
    (255, 0, 0),      # Blue
    (0, 255, 0),      # Green
    (0, 0, 255),      # Red
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Yellow
    (255, 128, 0),    # Light Blue
    (128, 0, 255),    # Pink
    (0, 255, 128),    # Spring Green
    (255, 0, 128),    # Purple-ish
    (128, 255, 0),    # Lime
    (0, 128, 255),    # Orange
    (128, 255, 255),  # Light Yellow
    (255, 128, 255),  # Light Magenta
    (255, 255, 128),  # Light Cyan
    (200, 100, 50),   # Dark Blue
    (50, 200, 100),   # Teal
    (100, 50, 200),   # Purple
    (200, 200, 50),   # Olive
    (50, 100, 200),   # Brown-ish
]



# =================== RUNTIME STATE ===================
setup          = False
frame          = None
circles        = []           # user-confirmed centers (global coords)
new_circles    = []           # detector proposals (overlay only)
id_counts      = {}           # {idx: {robot_id: count}}
pose_raw_deg   = {}           # {idx: last raw pose}
pose_smooth    = {}           # {idx: last smoothed pose}
pose_hist      = {}           # {idx: deque([angles])}
last_centers   = {}           # {idx: (x,y)}
trajectories   = {}           # {idx: [(x, y, frame_num), ...]}
trajectory_frame_counter = {} # {idx: frame count since last trajectory point}
stable_ids     = {}           # {idx: stable_robot_id}
kalman_filters = {}           # {idx: KalmanFilter2D}



LABEL2I = {'C':0,'M':1,'Y':2,'G':3}



# =================== CSV LOGGING ===================
LOG_CSV  = False                     # <-- toggle logging here
CSV_PATH = "robot_frames_log.csv"   # output file
_csv_f   = None
_csv_w   = None



def csv_init(path):
    """Open CSV and write header."""
    global _csv_f, _csv_w
    if not LOG_CSV: return
    # ensure folder exists
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    _csv_f = open(path, "w", newline="")
    _csv_w = csv.writer(_csv_f)
    _csv_w.writerow([
        "frame", "circle_idx", "x", "y", "x_filtered", "y_filtered",
        "id_stable", "id_frame", "score16", "shift",
        "front_idx", "verified",
        "angle_raw_deg", "angle_smooth_deg",
        "front_v", "front_hue",
        "labels_string"
    ])



def csv_log_rows(rows):
    """Append rows to CSV."""
    if not LOG_CSV or _csv_w is None: return
    for r in rows:
        _csv_w.writerow([
            r.get("frame"),
            r.get("circle_idx"),
            r.get("x"),
            r.get("y"),
            r.get("x_filtered"),
            r.get("y_filtered"),
            r.get("id_stable"),
            r.get("id_frame"),
            r.get("score16"),
            r.get("shift"),
            r.get("front_idx"),
            r.get("verified"),
            r.get("angle_raw_deg"),
            r.get("angle_smooth_deg"),
            r.get("front_v"),
            r.get("front_hue"),
            r.get("labels_string")
        ])



def csv_close():
    """Close CSV cleanly."""
    global _csv_f, _csv_w
    if _csv_f:
        _csv_f.flush()
        _csv_f.close()
    _csv_f = None
    _csv_w = None



# ================================================================
# =================== KALMAN FILTER ==============================
class KalmanFilter2D:
    """
    Simple 2D Kalman filter for position smoothing.
    State: [x, y, vx, vy] (position and velocity)
    Measurement: [x, y] (position only)
    """
    
    def __init__(self, process_noise=0.01, measurement_noise=5.0):
        # State: [x, y, vx, vy]
        self.state = np.zeros((4, 1), dtype=np.float32)
        
        # State covariance matrix (uncertainty in state)
        self.P = np.eye(4, dtype=np.float32) * 1000.0
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ], dtype=np.float32)
        
        # Measurement matrix (we only measure position)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance
        self.Q = np.eye(4, dtype=np.float32) * process_noise
        
        # Measurement noise covariance
        self.R = np.eye(2, dtype=np.float32) * measurement_noise
        
        # Identity matrix
        self.I = np.eye(4, dtype=np.float32)
        
        self.initialized = False
    
    def predict(self):
        """Predict next state."""
        # Predict state: x = F * x
        self.state = self.F @ self.state
        
        # Predict covariance: P = F * P * F' + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, measurement):
        """Update state with new measurement [x, y]."""
        z = np.array([[measurement[0]], [measurement[1]]], dtype=np.float32)
        
        if not self.initialized:
            # Initialize state with first measurement
            self.state[0] = z[0]
            self.state[1] = z[1]
            self.state[2] = 0  # vx
            self.state[3] = 0  # vy
            self.initialized = True
            return (float(z[0]), float(z[1]))
        
        # Predict
        self.predict()
        
        # Innovation: y = z - H * x
        y = z - self.H @ self.state
        
        # Innovation covariance: S = H * P * H' + R
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain: K = P * H' * S^-1
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state: x = x + K * y
        self.state = self.state + K @ y
        
        # Update covariance: P = (I - K * H) * P
        self.P = (self.I - K @ self.H) @ self.P
        
        return (float(self.state[0]), float(self.state[1]))
    
    def get_position(self):
        """Get current filtered position."""
        return (float(self.state[0]), float(self.state[1]))



def get_or_create_kalman(idx):
    """Get or create Kalman filter for robot index."""
    if idx not in kalman_filters:
        kalman_filters[idx] = KalmanFilter2D(
            process_noise=KALMAN_PROCESS_NOISE,
            measurement_noise=KALMAN_MEASUREMENT_NOISE
        )
    return kalman_filters[idx]



def filter_position(idx, x, y):
    """Apply Kalman filter to position."""
    if not USE_KALMAN_FILTER:
        return (x, y)
    
    kf = get_or_create_kalman(idx)
    return kf.update((x, y))



# ================================================================
# =================== TRAJECTORY FUNCTIONS =======================
def get_robot_color(idx):
    """Get consistent color for a robot based on its index."""
    return ROBOT_COLORS[idx % len(ROBOT_COLORS)]



def add_trajectory_point(idx, x, y, frame_num):
    """Add a point to robot's trajectory (sampled at intervals)."""
    if idx not in trajectory_frame_counter:
        trajectory_frame_counter[idx] = 0
    
    # Only add point every TRAJECTORY_SAMPLE_INTERVAL frames
    trajectory_frame_counter[idx] += 1
    
    if trajectory_frame_counter[idx] >= TRAJECTORY_SAMPLE_INTERVAL:
        if idx not in trajectories:
            trajectories[idx] = []
        trajectories[idx].append((int(x), int(y), frame_num))
        trajectory_frame_counter[idx] = 0  # Reset counter



def reset_trajectory_counters():
    """Reset trajectory counters (call when enabling/disabling trajectory)."""
    global trajectory_frame_counter
    trajectory_frame_counter.clear()



def clear_trajectories():
    """Clear all trajectory history."""
    global trajectories, trajectory_frame_counter
    trajectories.clear()
    trajectory_frame_counter.clear()



def draw_trajectories(img, current_frame):
    """Draw all trajectories with fading effect on the main image."""
    if not TRAJECTORY_ENABLED:
        return
    
    for idx, traj in trajectories.items():
        if len(traj) < 2:
            continue
        
        color = get_robot_color(idx)
        n_points = len(traj)
        
        # Draw line segments with fading
        for i in range(len(traj) - 1):
            x1, y1, _ = traj[i]
            x2, y2, _ = traj[i + 1]
            
            # Calculate alpha based on position in trajectory (exponential fade)
            progress = (i + 1) / n_points
            alpha = TRAJECTORY_MIN_ALPHA + (TRAJECTORY_MAX_ALPHA - TRAJECTORY_MIN_ALPHA) * (progress ** TRAJECTORY_FADE_POWER)
            
            # Create overlay for transparency
            overlay = img.copy()
            cv2.line(overlay, (x1, y1), (x2, y2), color, TRAJECTORY_LINE_THICKNESS, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)



def draw_legend(img):
    """Draw legend showing robot ID to color mapping."""
    if not stable_ids:
        return
    
    # Legend settings
    legend_x = 10
    legend_y = 30
    line_height = 30
    circle_radius = 8
    text_offset_x = 25
    
    # Semi-transparent background
    max_robots = len(stable_ids)
    bg_height = max_robots * line_height + 20
    bg_width = 150
    overlay = img.copy()
    cv2.rectangle(overlay, (5, 5), (5 + bg_width, 5 + bg_height), (50, 50, 50), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    # Draw each robot entry
    for i, (idx, robot_id) in enumerate(sorted(stable_ids.items())):
        y_pos = legend_y + i * line_height
        color = get_robot_color(idx)
        
        # Draw colored circle
        cv2.circle(img, (legend_x + circle_radius, y_pos), circle_radius, color, -1)
        cv2.circle(img, (legend_x + circle_radius, y_pos), circle_radius, (255, 255, 255), 1)
        
        # Draw robot ID text
        cv2.putText(img, f"Robot {robot_id + 1:02d}", 
                    (legend_x + text_offset_x, y_pos + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)



def create_trajectory_canvas(width, height):
    """Create a separate canvas for trajectory visualization."""
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background
    return canvas



def draw_trajectory_canvas(width, height, current_frame):
    """Draw trajectory canvas with robot icons and paths."""
    canvas = create_trajectory_canvas(width, height)
    
    # Draw grid (optional)
    grid_color = (220, 220, 220)
    grid_spacing = 50
    for x in range(0, width, grid_spacing):
        cv2.line(canvas, (x, 0), (x, height), grid_color, 1)
    for y in range(0, height, grid_spacing):
        cv2.line(canvas, (0, y), (width, y), grid_color, 1)
    
    # Draw all trajectories
    for idx, traj in trajectories.items():
        if len(traj) < 2:
            continue
        
        color = get_robot_color(idx)
        n_points = len(traj)
        
        # Draw path with fading
        for i in range(len(traj) - 1):
            x1, y1, _ = traj[i]
            x2, y2, _ = traj[i + 1]
            
            # Fading effect
            progress = (i + 1) / n_points
            alpha = TRAJECTORY_MIN_ALPHA + (TRAJECTORY_MAX_ALPHA - TRAJECTORY_MIN_ALPHA) * (progress ** TRAJECTORY_FADE_POWER)
            
            overlay = canvas.copy()
            cv2.line(overlay, (x1, y1), (x2, y2), color, TRAJECTORY_LINE_THICKNESS, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
        
        # Draw robot icon at current position (last point)
        if traj:
            x, y, _ = traj[-1]
            robot_radius = 12
            
            # Draw robot body (circle)
            cv2.circle(canvas, (x, y), robot_radius, color, -1)
            cv2.circle(canvas, (x, y), robot_radius, (0, 0, 0), 2)
            
            # Draw orientation arrow
            if idx in pose_smooth or idx in pose_raw_deg:
                angle = pose_smooth.get(idx, pose_raw_deg.get(idx))
                if angle is not None:
                    arrow_len = int(robot_radius * 1.5)
                    dx = int(arrow_len * math.cos(math.radians(angle)))
                    dy = int(arrow_len * math.sin(math.radians(angle)))
                    arrow_end = (x + dx, y + dy)
                    cv2.arrowedLine(canvas, (x, y), arrow_end, (0, 0, 0), 2, tipLength=0.3)
            
            # Draw robot ID
            robot_id = stable_ids.get(idx, -1)
            if robot_id >= 0:
                cv2.putText(canvas, f"{robot_id + 1}", 
                           (x - 8, y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Draw legend on trajectory canvas
    draw_legend_trajectory_canvas(canvas)
    
    return canvas



def draw_legend_trajectory_canvas(canvas):
    """Draw legend on trajectory canvas."""
    if not stable_ids:
        return
    
    # Legend at top-right
    canvas_height, canvas_width = canvas.shape[:2]
    legend_width = 150
    legend_x = canvas_width - legend_width - 10
    legend_y = 30
    line_height = 30
    circle_radius = 8
    text_offset_x = 25
    
    # Semi-transparent background
    max_robots = len(stable_ids)
    bg_height = max_robots * line_height + 20
    overlay = canvas.copy()
    cv2.rectangle(overlay, (legend_x - 5, 5), 
                  (legend_x + legend_width, 5 + bg_height), 
                  (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.8, canvas, 0.2, 0, canvas)
    
    # Draw border
    cv2.rectangle(canvas, (legend_x - 5, 5), 
                  (legend_x + legend_width, 5 + bg_height), 
                  (100, 100, 100), 2)
    
    # Draw each robot entry
    for i, (idx, robot_id) in enumerate(sorted(stable_ids.items())):
        y_pos = legend_y + i * line_height
        color = get_robot_color(idx)
        
        # Draw colored circle
        cv2.circle(canvas, (legend_x + circle_radius, y_pos), circle_radius, color, -1)
        cv2.circle(canvas, (legend_x + circle_radius, y_pos), circle_radius, (0, 0, 0), 1)
        
        # Draw robot ID text
        cv2.putText(canvas, f"Robot {robot_id + 1:02d}", 
                    (legend_x + text_offset_x, y_pos + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)



# ================================================================
# =================== CORE GEOM / ANGLES =========================
def ang_norm(a):
    a %= 360.0
    return a + 360.0 if a < 0 else a



def ang_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0



def circ_mean_deg(angles):
    if not angles: return None
    s = sum(math.sin(math.radians(a)) for a in angles)
    c = sum(math.cos(math.radians(a)) for a in angles)
    if s == 0 and c == 0:     # degenerate
        return ang_norm(angles[-1])
    return ang_norm(math.degrees(math.atan2(s, c)))



def angle_of(cx, cy, x, y):
    a = math.degrees(math.atan2(y - cy, x - cx))
    return a + 360.0 if a < 0 else a



# ================================================================
# =================== CIRCLE FINDING =============================
def fit_circle(img):
    if img is None or img.size == 0:
        return None
    return fit_bright(img) if LIGHT else fit_dark(img)



def fit_bright(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * SAT_GAIN, 0, 255)
    hsv[:, :, 1] = np.where(hsv[:, :, 1] < SAT_CUTOFF, 0, hsv[:, :, 1])
    gray = hsv[:, :, 1]
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    if SHOW: cv2.imshow("saturation", blur)
    return cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=MIN_DIST,
        param1=30, param2=60, minRadius=10, maxRadius=150
    )



def fit_dark(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = np.where(g < SAT_CUTOFF, 0, g)
    blur = cv2.GaussianBlur(g, (7, 7), 2)
    return cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=MIN_DIST,
        param1=10, param2=25, minRadius=8, maxRadius=150
    )



def crop_bounds(center, img):
    cx, cy = int(center[0]), int(center[1])
    h, w   = img.shape[:2]
    half   = int(R * 1.5)
    x1, x2 = max(0, cx - half), min(w, cx + half)
    y1, y2 = max(0, cy - half), min(h, cy + half)
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return y1, y2, x1, x2



# ================================================================
# =================== LED SAMPLING / CLASSIFY ====================
def sample_v_stat(V, x0, y0, r, stat='median'):
    h, w = V.shape
    if x0 < 0 or y0 < 0 or x0 >= w or y0 >= h: return -1.0
    y_min, y_max = max(0, y0 - r), min(h - 1, y0 + r)
    x_min, x_max = max(0, x0 - r), min(w - 1, x0 + r)
    r2, vals = r * r, []
    for y in range(y_min, y_max + 1):
        dy2 = (y - y0) * (y - y0)
        if dy2 > r2: continue
        dx  = int((r2 - dy2) ** 0.5)
        xs  = max(x_min, x0 - dx)
        xe  = min(x_max, x0 + dx)
        row = V[y, xs:xe + 1]
        if row.size: vals.extend(row.tolist())
    if not vals: return -1.0
    arr = np.asarray(vals, dtype=np.float32)
    if   stat == 'median': return float(np.median(arr))
    elif stat == 'max':    return float(np.max(arr))
    else:                  return float(np.mean(arr))



def find_bright_theta(V, cx, cy, r_mid, step_deg):
    best_t, best_v = None, -1.0
    for t in np.arange(0.0, 360.0, step_deg):
        x = int(round(cx + r_mid * math.cos(math.radians(t))))
        y = int(round(cy + r_mid * math.sin(math.radians(t))))
        v = sample_v_stat(V, x, y, BLOB_R, stat='median')
        if v > best_v: best_v, best_t = v, t
    return best_t if best_t is not None else (THETA0_DEG % 360.0)



def refine_local_max(V, x0, y0, r, bounds_wh):
    w, h = bounds_wh
    best_v, best_xy = -1, (x0, y0)
    y_min, y_max = max(0, y0 - r), min(h - 1, y0 + r)
    x_min, x_max = max(0, x0 - r), min(w - 1, x0 + r)
    r2 = r * r
    for y in range(y_min, y_max + 1):
        dy2 = (y - y0) * (y - y0)
        if dy2 > r2: continue
        dx  = int((r2 - dy2) ** 0.5)
        xs  = max(x_min, x0 - dx)
        xe  = min(x_max, x0 + dx)
        row = V[y, xs:xe + 1]
        if row.size == 0: continue
        col = int(np.argmax(row))
        v   = int(row[col])
        x   = xs + col
        if v > best_v:
            best_v, best_xy = v, (x, y)
    return best_xy[0], best_xy[1], best_v



def hue_median(H, V, x0, y0, r, v_min):
    h, w = H.shape
    y_min, y_max = max(0, y0 - r), min(h - 1, y0 + r)
    x_min, x_max = max(0, x0 - r), min(w - 1, x0 + r)
    r2, vals = r * r, []
    for y in range(y_min, y_max + 1):
        dy2 = (y - y0) * (y - y0)
        if dy2 > r2: continue
        dx  = int((r2 - dy2) ** 0.5)
        xs  = max(x_min, x0 - dx)
        xe  = min(x_max, x0 + dx)
        vrow, hrow = V[y, xs:xe + 1], H[y, xs:xe + 1]
        m = vrow >= v_min
        if np.any(m): vals.extend(hrow[m].tolist())
    if not vals: return -1
    return float(np.median(np.asarray(vals, dtype=np.float32)))



def classify_hue(h):
    if h < 0: return 'C'
    for lab, ranges in HUE_BANDS.items():
        for a, b in ranges:
            if a <= h < b: return lab
    centers = {'Y':30, 'G':70, 'C':95, 'M':150}
    best_lab, best_d = None, 1e9
    for lab, c in centers.items():
        d = min(abs(h - c), 180 - abs(h - c))
        if d < best_d: best_d, best_lab = d, lab
    return best_lab or 'C'



def analyze_ring(crop_bgr, center_xy, approx_r):
    cx, cy = map(int, center_xy)
    hsv    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    H, V   = hsv[:, :, 0], hsv[:, :, 2]


    r_in, r_out = ANN_INNER * approx_r, ANN_OUTER * approx_r
    r_mid       = 0.5 * (r_in + r_out)


    theta0 = find_bright_theta(V, cx, cy, r_mid, TH_STEP_DEG)
    step   = (360.0 / N_LEDS) * DIR


    out, h, w = [], *crop_bgr.shape[:2]
    for k in range(N_LEDS):
        th = math.radians(theta0 + k * step)
        nx = int(round(cx + r_mid * math.cos(th)))
        ny = int(round(cy + r_mid * math.sin(th)))


        rx, ry, vpk = refine_local_max(V, nx, ny, TINY_REFINE_R, (w, h))
        hmed        = hue_median(H, V, rx, ry, BLOB_R, MIN_V)
        lab         = classify_hue(hmed)
        out.append({'pt':(nx,ny), 'refined':(rx,ry), 'label':lab, 'hue':hmed, 'v':vpk})
    return out



# ================================================================
# =================== ROTATION-AWARE ID & POSE ====================
def rot(seq, k): k %= len(seq); return seq[k:] + seq[:k]
def score(a, b): return sum(1 for x, y in zip(a, b) if x == y)



def best_match(labels, sequences):
    obs = [LABEL2I.get(ch, 0) for ch in labels]
    n   = len(obs)
    best_id, best_sc, best_sh = -1, -1, 0
    for sid, cand in enumerate(sequences):
        if len(cand) != n: continue
        for sh in range(n):
            sc = score(obs, rot(cand, sh))
            if sc > best_sc: best_id, best_sc, best_sh = sid, sc, sh
    return best_id, best_sc, best_sh



def front_from_shift(shift, n): return (-shift) % n
def _wrap(i, n): return (i % n + n) % n



def neighbor_score(obs, canon, idx, win=2):
    n, s = len(obs), 0
    for off in range(-win, win + 1):
        if obs[_wrap(idx + off, n)] == canon[_wrap(0 + off, n)]:
            s += 1
    return s



def best_front_index(labels, canon, win=2):
    obs = [LABEL2I.get(ch, 0) for ch in labels]
    best_i, best_s = 0, -1
    for j in range(len(obs)):
        s = neighbor_score(obs, canon, j, win)
        if s > best_s: best_i, best_s = j, s
    return best_i, best_s



def compute_pose(led_info, labels, rid, shift, center_xy, verify=True, min_ok=3):
    if rid is None or rid < 0 or (led_info is None) or len(led_info) != N_LEDS:
        return None, None, False


    cx, cy  = map(int, center_xy)
    canon   = database.SEQUENCES[rid]          # ints 0..3
    obs_int = [LABEL2I.get(ch, 0) for ch in labels]


    front = front_from_shift(shift, N_LEDS)
    ok    = True
    if verify:
        if neighbor_score(obs_int, canon, front, win=2) < min_ok:
            j2, s2 = best_front_index(labels, canon, win=2)
            front  = j2
            ok     = s2 >= min_ok


    rx, ry = led_info[front]['refined']
    pose   = angle_of(cx, cy, rx, ry)
    return pose, front, ok



# ================================================================
# =================== POSE STABILIZATION =========================
def pose_push(i, pose_deg):
    if i not in pose_hist:
        pose_hist[i] = deque(maxlen=POSE_WIN)
    dq = pose_hist[i]
    if dq.maxlen != POSE_WIN:
        pose_hist[i] = deque(list(dq)[-POSE_WIN:], maxlen=POSE_WIN)
        dq = pose_hist[i]
    dq.append(pose_deg)



def pose_get(i):
    if i not in pose_hist or len(pose_hist[i]) == 0: return None
    arr    = list(pose_hist[i])
    latest = arr[-1]
    arr.sort(key=lambda a: abs(ang_diff(a, latest)))
    k = max(1, min(len(arr), max(POSE_MIN_KEEP, int(math.ceil(POSE_KEEP_FR * len(arr))))))
    return circ_mean_deg(arr[:k])



# ================================================================
# =================== UI: CLICKS & OVERLAY =======================
def on_click(event, x, y, *_):
    global setup
    if setup: return
    if event == cv2.EVENT_LBUTTONDOWN:
        circles.append((int(x), int(y)))
        draw_main()
        if DEBUG: print("Clicked:", (x, y))



def draw_main():
    if frame is None or frame.size == 0: return
    vis = frame.copy()


    for x, y in new_circles:
        cv2.circle(vis, (x, y), int(R), (255, 255, 0), 2)
        cv2.circle(vis, (x, y), 2, (0, 0, 255), 3)


    for idx, (x, y) in enumerate(circles):
        cv2.circle(vis, (x, y), int(R), (0, 255, 0), 2)
        cv2.circle(vis, (x, y), 2, (0, 0, 255), 3)


        if idx in id_counts and id_counts[idx]:
            sid = max(id_counts[idx], key=id_counts[idx].get) + 1
            tl  = (x - R, y - R)
            br  = (x + R, y + R)
            cv2.rectangle(vis, tl, br, (0, 255, 0), 2)
            cv2.putText(vis, f"ID:{sid}", (x - R, y - R - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)


        ang = pose_smooth.get(idx, pose_raw_deg.get(idx))
        if ang is not None:
            L  = int(R * 0.9)
            dx = int(round(L * math.cos(math.radians(ang))))
            dy = int(round(L * math.sin(math.radians(ang))))
            p0 = (x, y)
            p1 = (x + dx, y + dy)
            cv2.arrowedLine(vis, p0, p1, (0, 255, 255), 2, tipLength=0.2)
            cv2.putText(vis, f"{ang:5.1f} deg", (p1[0] + 8, p1[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)


    # Draw trajectories with fading
    draw_trajectories(vis, 0)  # frame number not used currently
    
    # Draw legend
    draw_legend(vis)


    cv2.imshow("Circle Detection", vis)



# ================================================================
# =================== PER-FRAME UPDATE ===========================
def update(frame_idx):
    """Process all circles, return list of per-circle dicts for logging."""
    global pose_raw_deg, pose_smooth, last_centers, stable_ids
    rows = []


    for i, c in enumerate(circles):
        y1, y2, x1, x2 = crop_bounds(c, frame)
        if (y2 - y1) <= 1 or (x2 - x1) <= 1:
            if DEBUG: print("Skip tiny crop"); continue


        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            if DEBUG: print("Empty crop"); continue


        det = fit_circle(crop)
        local_center = None
        gx = gy = None
        gx_filtered = gy_filtered = None


        if det is not None:
            det = np.uint16(np.around(det))
            x, y, _ = det[0][0]
            local_center = (int(x), int(y))
            gx, gy = int(x) + x1, int(y) + y1
            
            # Apply Kalman filter to raw position
            gx_filtered, gy_filtered = filter_position(i, gx, gy)
            
            circles[i]      = (int(gx_filtered), int(gy_filtered))
            last_centers[i] = (int(gx_filtered), int(gy_filtered))
            
            # Add filtered position to trajectory (sampled)
            add_trajectory_point(i, gx_filtered, gy_filtered, frame_idx)
        else:
            if i in last_centers:
                gx_filtered, gy_filtered = last_centers[i]
                add_trajectory_point(i, gx_filtered, gy_filtered, frame_idx)


        led = None
        if local_center is not None:
            try:
                led = analyze_ring(crop, local_center, R)
                if DEBUG:
                    labs = ''.join(p['label'] for p in led)
                    print(f"LED[{i}] -> {labs}")
            except Exception as e:
                if DEBUG: print("LED ring error:", e)


        # Defaults for logging in case we can't compute this frame
        row = {
            "frame": frame_idx, "circle_idx": i,
            "x": gx if gx is not None else "",
            "y": gy if gy is not None else "",
            "x_filtered": gx_filtered if gx_filtered is not None else "",
            "y_filtered": gy_filtered if gy_filtered is not None else "",
            "id_stable": "", "id_frame": "", "score16": "", "shift": "",
            "front_idx": "", "verified": "",
            "angle_raw_deg": pose_raw_deg.get(i, ""),
            "angle_smooth_deg": pose_smooth.get(i, ""),
            "front_v": "", "front_hue": "", "labels_string": ""
        }


        if led is not None:
            labels = [p['label'] for p in led]
            row["labels_string"] = ''.join(labels)
            try:
                rid, sc, sh = best_match(labels, database.SEQUENCES)


                if i not in id_counts: id_counts[i] = {}
                id_counts[i][rid] = id_counts[i].get(rid, 0) + 1
                stable_id = max(id_counts[i], key=id_counts[i].get)
                stable_ids[i] = stable_id  # Store for legend


                pose_deg, front_idx, ok = compute_pose(
                    led, labels, rid, sh, local_center, verify=True, min_ok=3
                )
                if pose_deg is not None:
                    pose_raw_deg[i] = pose_deg
                    pose_push(i, pose_deg)
                    sm = pose_get(i)
                    if sm is not None:
                        pose_smooth[i] = sm


                # Terminal status
                gx_s = "None" if gx is not None else str(int(gx))
                gy_s = "None" if gy is not None else str(int(gy))
                gxf_s = f"{gx_filtered:.1f}" if gx_filtered is not None else "None"
                gyf_s = f"{gy_filtered:.1f}" if gy_filtered is not None else "None"
                
                kalman_status = "K" if USE_KALMAN_FILTER else ""
                print(f"[circle {i}] ID(stable)={stable_id+1:02d}  "
                      f"pos=({gxf_s}, {gyf_s}){kalman_status}  "
                      f"(frame id={rid+1:02d}, score={sc}/16, shift={sh})")


                # Populate row for CSV
                row.update({
                    "id_stable": stable_id + 1,
                    "id_frame": rid + 1,
                    "score16": sc,
                    "shift": sh,
                    "front_idx": front_idx if front_idx is not None else "",
                    "verified": int(bool(ok)),
                    "angle_raw_deg": pose_raw_deg.get(i, ""),
                    "angle_smooth_deg": pose_smooth.get(i, "")
                })


                if front_idx is not None:
                    row["front_v"]   = led[front_idx]['v']
                    row["front_hue"] = led[front_idx]['hue']


            except Exception as e:
                print("Robot ID/pose error:", e)


        rows.append(row)


    return rows



# ================================================================
# =================== CAMERA / MAIN LOOP =========================
def list_cameras(max_tested=10):
    found = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found.append(i)
            cap.release()
    return found



def run(camera_index=0):
    global frame, setup, new_circles, paused, POSE_WIN, SHOW_TRAJECTORY_CANVAS, USE_KALMAN_FILTER, TRAJECTORY_ENABLED
    csv_init(CSV_PATH)


    if camera_index == -1:
        cap = cv2.VideoCapture("bright.mov" if LIGHT else "dark_multiple.mov")
    else:
        cap = cv2.VideoCapture(camera_index)


    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_index}")
        csv_close()
        return


    # Get video dimensions for trajectory canvas
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    print(f"[smoothing] window={POSE_WIN}, keep_frac={POSE_KEEP_FR}, min_keep={POSE_MIN_KEEP}")
    print(f"[trajectory] Enabled with fading (alpha: {TRAJECTORY_MIN_ALPHA}-{TRAJECTORY_MAX_ALPHA})")
    print(f"[trajectory sampling] Every {TRAJECTORY_SAMPLE_INTERVAL} frames")
    print(f"[trajectory canvas] {'Enabled' if SHOW_TRAJECTORY_CANVAS else 'Disabled'} (toggle with 'c' key)")
    print(f"[kalman filter] {'Enabled' if USE_KALMAN_FILTER else 'Disabled'} (toggle with 'k' key)")
    print(f"   Process noise: {KALMAN_PROCESS_NOISE}, Measurement noise: {KALMAN_MEASUREMENT_NOISE}")
    print(f"[controls] 't'=toggle trajectory, 'c'=canvas, 'k'=kalman, 'r'=clear history, '['']'=smoothing, Space=pause, Esc=exit")


    total, t0 = 0, 0
    while True:
        if not setup:
            ret, f = cap.read()
            if not ret:
                if t0:
                    dt = time.time() - t0
                    if dt > 0: print(f"fps: {total/dt}")
                print("Failed to grab frame"); break


            frame = f
            total += 1
            new_circles = []
            det = fit_circle(frame)
            if det is not None:
                det = np.uint16(np.around(det))
                for (x, y, r) in det[0, :]:
                    nx, ny = int(x), int(y)
                    new_circles.append((nx, ny))
                    cv2.circle(frame, (nx, ny), int(R), (255, 255, 0), 2)
                    cv2.circle(frame, (nx, ny), 2, (0, 0, 255), 3)


            cv2.imshow("Circle Detection", frame)
            cv2.setMouseCallback("Circle Detection", on_click)


            key = cv2.waitKey(0) & 0xFF
            if key == 13:      # Enter
                setup = True
                t0    = time.time()
                new_circles = []
                if DEBUG: print("Enter -> tracking")
            elif key == 8:     # Backspace
                if circles: circles.pop(-1)
            elif key == 27:    # Esc
                break
            elif key == ord('n'):
                continue
        else:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    if t0:
                        dt = time.time() - t0
                        if dt > 0: print(f"fps: {total/dt}")
                    print("Failed to grab frame"); break
                total += 1


            # ==== process + CSV log ====
            rows = update(total)
            if LOG_CSV and rows:
                csv_log_rows(rows)


            if SHOW: 
                draw_main()
                
                # Draw trajectory canvas only if enabled
                if SHOW_TRAJECTORY_CANVAS:
                    traj_canvas = draw_trajectory_canvas(video_width, video_height, total)
                    cv2.imshow("Trajectory View", traj_canvas)
                else:
                    # Close trajectory window if it exists
                    try:
                        cv2.destroyWindow("Trajectory View")
                    except:
                        pass


            key = cv2.waitKey(1) & 0xFF
            if   key == 27: break                 # Esc
            elif key == ord(' '):
                paused = not paused
            elif key == ord(']'):
                POSE_WIN = min(POSE_WIN_MAX, POSE_WIN + 1)
                print(f"[smoothing] window -> {POSE_WIN}")
            elif key == ord('['):
                POSE_WIN = max(POSE_WIN_MIN, POSE_WIN - 1)
                print(f"[smoothing] window -> {POSE_WIN}")
            elif key == ord('\\'):
                print(f"[smoothing] window={POSE_WIN}, keep_frac={POSE_KEEP_FR}, min_keep={POSE_MIN_KEEP}")
            elif key == ord('t'):
                TRAJECTORY_ENABLED = not TRAJECTORY_ENABLED
                reset_trajectory_counters()
                print(f"[trajectory] {'Enabled' if TRAJECTORY_ENABLED else 'Disabled'}")
            elif key == ord('c'):
                SHOW_TRAJECTORY_CANVAS = not SHOW_TRAJECTORY_CANVAS
                print(f"[trajectory canvas] {'Enabled' if SHOW_TRAJECTORY_CANVAS else 'Disabled'}")
            elif key == ord('r'):
                clear_trajectories()
                print("[trajectory] History cleared")
            elif key == ord('k'):
                USE_KALMAN_FILTER = not USE_KALMAN_FILTER
                # Reset Kalman filters when toggling
                if USE_KALMAN_FILTER:
                    kalman_filters.clear()
                print(f"[kalman filter] {'Enabled' if USE_KALMAN_FILTER else 'Disabled'}")


    cap.release()
    cv2.destroyAllWindows()
    csv_close()



# ================================================================
# =================== ENTRY =====================================
if __name__ == "__main__":
    cams = list_cameras()
    if not cams:
        print("No cameras found!")
        choice = 0
    else:
        print("Available cameras:")
        for idx in cams: print(f"  {idx}")
        try:    choice = int(input("Select camera index: "))
        except: choice = cams[0]
    run(choice)