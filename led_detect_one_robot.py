# LED ring detector (48 samples from 12 o'clock CW)
# Pipeline: color pre-process -> ring (circle) detect -> annulus-only color votes -> collapse -> drop 'U' -> limit to 16
# Prints robot centre (red dot) coordinates every frame.
# Shows image origin (0,0) in the preview with x→ and y↓ arrows.
# Keeps per-ring "streaks" (how long a 16-LED sequence stays identical), and shows the best streak on HUD.
# Keys: q=quit, f=fullscreen, m=toggle overlays

import cv2
import numpy as np
import time
import math

# ---------------------- Basic camera & frame setup ----------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30

# ---------------------- Circle detector (Hough) settings ----------------------
# These are geometric constraints used to find the physical LED ring as a circle.
HOUGH_DP = 1.2            # inverse accumulator resolution
HOUGH_MIN_CENTER_DIST = 30  # minimum distance between circle centers (pixels)
HOUGH_CANNY_HIGH = 50       # Canny high threshold (lower is implied)
HOUGH_ACCUMULATOR_THRESH = 90  # circle vote threshold (higher = fewer circles)
HOUGH_MIN_RADIUS = 10
HOUGH_MAX_RADIUS = 200

# ---------------------- Annulus sampling (where we look for LEDs) ----------------------
SAMPLES_AROUND_RING = 48            # 48 samples evenly spaced around the ring
ANNULUS_INNER_RATIO = 0.85          # inner radius as a fraction of circle radius
ANNULUS_OUTER_RATIO = 1.05          # outer radius as a fraction of circle radius
LOCAL_PATCH_RADIUS_PX = 4           # half-size of the square patch for local votes
NUM_RADII_TO_PROBE = 3              # how many radii to probe between inner & outer bands

# ---------------------- Base HSV thresholds (will adapt per ring) ----------------------
BASE_SAT_MIN = 120
BASE_VAL_MIN = 130
RED_WRAP_TOL = 14                   # red wraps around hue=0/180
GREEN_HUE_CENTER = 70               # OpenCV hue scale [0..179]
GREEN_HUE_TOL = 22
BLUE_HUE_CENTER = 115
BLUE_HUE_TOL = 18

# ---------------------- Light color pre-processing (no UI sliders) ----------------------
USE_GRAY_WORLD_WHITE_BALANCE = True
LINEAR_GAIN = 0.9
LINEAR_OFFSET = -5
GAMMA_VALUE = 1.8
SATURATION_GAIN = 2.0
VALUE_GAIN = 1.2

# ---------------------- Tracking & smoothing ----------------------
# We smooth circle center/radius frame-to-frame for visual stability.
SMOOTHING_ALPHA = 0.5               # 0=no smoothing, 1=frozen
MAX_CENTER_MATCH_DIST = 40.0        # px: match detections to existing tracks

# ---------------------- Preview flags ----------------------
DRAW_OVERLAYS = True
SHOW_PREVIEW_WINDOW = True


# ---------------------- Small helpers ----------------------
def clip8(x):
    """Clamp to [0,255] and cast to uint8 (OpenCV-friendly)."""
    return np.clip(x, 0, 255).astype(np.uint8)

def gray_world_white_balance(bgr_img):
    """Simple 'gray world' WB to balance channels by their means."""
    b, g, r = cv2.split(bgr_img.astype(np.float32))
    mean_b, mean_g, mean_r = b.mean()+1e-6, g.mean()+1e-6, r.mean()+1e-6
    gray = (mean_b + mean_g + mean_r) / 3.0
    gain_b, gain_g, gain_r = gray/mean_b, gray/mean_g, gray/mean_r
    return clip8(cv2.merge([b*gain_b, g*gain_g, r*gain_r]))

def apply_gamma(bgr_img, gamma):
    """Apply gamma via lookup table (fast, stable)."""
    if abs(gamma - 1.0) < 1e-3:
        return bgr_img
    inv = 1.0 / max(gamma, 1e-6)
    lut = np.array([((i/255.0)**inv)*255 for i in range(256)], np.uint8)
    return cv2.LUT(bgr_img, lut)

def red_mask_hsv(hsv_img, red_tol, sat_min, val_min):
    """Binary mask for 'red' in HSV, accounting for hue wrap at 0/180."""
    h, s, v = cv2.split(hsv_img)
    hue_is_red = (h <= red_tol) | (h >= 180 - red_tol)
    return ((hue_is_red) & (s >= sat_min) & (v >= val_min)).astype(np.uint8) * 255

def color_mask_hsv(hsv_img, hue_center, hue_tol, sat_min, val_min):
    """Binary mask for a color band centered at hue_center +/- hue_tol."""
    h, s, v = cv2.split(hsv_img)
    lo = (hue_center - hue_tol) % 180
    hi = (hue_center + hue_tol) % 180
    hue_ok = (h >= lo) & (h <= hi) if lo <= hi else ((h >= lo) | (h <= hi))
    return ((hue_ok) & (s >= sat_min) & (v >= val_min)).astype(np.uint8) * 255

def collapse_runs(seq):
    """Collapse consecutive duplicates: R,R,G,G,B -> R,G,B."""
    if not seq:
        return []
    out = [seq[0]]
    for c in seq[1:]:
        if c != out[-1]:
            out.append(c)
    return out

def collapse_and_drop_unknowns(seq):
    """Collapse runs and remove 'U' (unknown) entries entirely."""
    return [c for c in collapse_runs(seq) if c in ('R', 'G', 'B')]

def limit_to_16(seq):
    """After collapse+drop-U, keep only the first 16 tokens."""
    return seq[:16]


# ---------------------- Core sampling logic ----------------------
def sample_around_ring(
    hsv_img, center_x, center_y, radius,
    num_samples, inner_ratio, outer_ratio, patch_half, num_radii,
    sat_min, val_min
):
    """
    Sample colors at 'num_samples' angles around the ring.
    Angles start at 12 o'clock (theta=pi/2) and go clockwise.
    For each angle, probe multiple radii inside the annulus, pick the strongest color.
    Returns: (raw_48, collapsed_no_U, inner_radius_px, outer_radius_px)
    """
    height, width = hsv_img.shape[:2]

    # Angles: 12 o'clock, clockwise (subtract as we go around)
    angles = (np.pi/2) - (2.0 * np.pi * np.arange(num_samples) / float(num_samples))

    inner_radius = radius * inner_ratio
    outer_radius = radius * outer_ratio
    probe_radii = np.linspace(inner_radius, outer_radius, max(1, num_radii))

    # Precompute masks once per ring (faster than re-thresholding per patch)
    red_mask = red_mask_hsv(hsv_img, RED_WRAP_TOL, sat_min, val_min)
    green_mask = color_mask_hsv(hsv_img, GREEN_HUE_CENTER, GREEN_HUE_TOL, sat_min, val_min)
    blue_mask = color_mask_hsv(hsv_img, BLUE_HUE_CENTER, BLUE_HUE_TOL, sat_min, val_min)

    raw_sequence = []

    for theta in angles:
        best_color = 'U'
        best_score = -1

        # Try several radii along this angle; keep the most confident color
        for rr in probe_radii:
            px = int(round(center_x + rr * math.cos(theta)))
            py = int(round(center_y - rr * math.sin(theta)))  # y axis is downwards in images

            if 0 <= px < width and 0 <= py < height:
                x0, x1 = max(0, px - patch_half), min(width, px + patch_half + 1)
                y0, y1 = max(0, py - patch_half), min(height, py + patch_half + 1)

                red_votes = int(np.count_nonzero(red_mask[y0:y1, x0:x1]))
                green_votes = int(np.count_nonzero(green_mask[y0:y1, x0:x1]))
                blue_votes = int(np.count_nonzero(blue_mask[y0:y1, x0:x1]))

                local_best = max(red_votes, green_votes, blue_votes)
                if local_best > best_score and local_best > 0:
                    best_score = local_best
                    if local_best == red_votes:
                        best_color = 'R'
                    elif local_best == green_votes:
                        best_color = 'G'
                    else:
                        best_color = 'B'

        raw_sequence.append(best_color)

    collapsed_no_u = collapse_and_drop_unknowns(raw_sequence)
    return raw_sequence, collapsed_no_u, int(inner_radius), int(outer_radius)


def adapt_thresholds_from_annulus(hsv_img, center_x, center_y, radius, inner_ratio, outer_ratio):
    """
    Look only inside the ring's annulus and adapt S/V thresholds from its histogram.
    This helps when exposure/white-balance drifts.
    """
    height, width = hsv_img.shape[:2]
    r_in = int(max(1, radius * inner_ratio))
    r_out = int(max(r_in + 1, radius * outer_ratio))

    Y, X = np.ogrid[:height, :width]
    rsq = (X - center_x) ** 2 + (Y - center_y) ** 2
    annulus = (rsq >= r_in * r_in) & (rsq <= r_out * r_out)

    sat = hsv_img[:, :, 1][annulus].astype(np.float32)
    val = hsv_img[:, :, 2][annulus].astype(np.float32)
    if sat.size == 0:
        # Fallback to base thresholds if annulus is empty (shouldn't happen usually)
        return BASE_SAT_MIN, BASE_VAL_MIN

    # We're permissive but not too low: use upper percentiles with a margin
    sat_75 = np.percentile(sat, 75)
    val_98 = np.percentile(val, 98)
    sat_min = int(max(BASE_SAT_MIN * 0.8, min(255, sat_75 * 0.8)))
    val_min = int(max(BASE_VAL_MIN * 0.7, min(255, val_98 * 0.5)))
    return sat_min, val_min


# ---------------------- Simple ring tracking ----------------------
class RingTrack:
    """
    One track = one ring across frames.
    We smooth its center/radius (EMA), and keep the last 16-LED sequence + how long it stayed unchanged.
    """
    def __init__(self, x, y, r):
        self.x = float(x)
        self.y = float(y)
        self.r = float(r)
        self.missed = 0

        self.last_seq16 = None   # last 16-token collapsed (no 'U') sequence
        self.streak16 = 0        # how many consecutive frames last_seq16 stayed identical

    def update_geometry(self, x, y, r, alpha=SMOOTHING_ALPHA):
        """Exponential moving average for visual stability."""
        self.x = alpha * self.x + (1 - alpha) * float(x)
        self.y = alpha * self.y + (1 - alpha) * float(y)
        self.r = alpha * self.r + (1 - alpha) * float(r)
        self.missed = 0


def match_and_update_tracks(existing_tracks, detected_circles):
    """
    Associate each detected circle to the nearest existing track (within a distance),
    update the track, and create new tracks for unmatched circles.
    """
    used = [False] * len(detected_circles)

    # Try to update existing tracks
    for tr in existing_tracks:
        best_idx = -1
        best_dist = 1e9
        for j, (cx, cy, rr) in enumerate(detected_circles):
            if used[j]:
                continue
            d = math.hypot(tr.x - cx, tr.y - cy)
            if d < best_dist:
                best_dist = d
                best_idx = j

        if best_idx >= 0 and best_dist <= MAX_CENTER_MATCH_DIST:
            px, py, pr = detected_circles[best_idx]
            tr.update_geometry(px, py, pr)
            used[best_idx] = True
        else:
            tr.missed += 1

    # Create new tracks for any unmatched circles
    for j, (cx, cy, rr) in enumerate(detected_circles):
        if not used[j]:
            existing_tracks.append(RingTrack(cx, cy, rr))

    # Drop stale tracks
    return [tr for tr in existing_tracks if tr.missed <= 5]


# ---------------------- Main program ----------------------
def main():
    # Open the camera and set the desired capture properties.
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError("Camera open failed")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    show_overlays = DRAW_OVERLAYS
    tracks = []

    # We still keep the "best" 16 by streak for the HUD, but we don't print it in the terminal.
    best_seq_overall = None
    best_seq_streak = 0

    if SHOW_PREVIEW_WINDOW:
        cv2.namedWindow("LED Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("LED Preview", 1280, 720)

    last_time = time.time()

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            time.sleep(0.005)
            continue

        # ----------- Light color pre-processing (kept small to stay real-time) -----------
        img_bgr = frame_bgr
        if USE_GRAY_WORLD_WHITE_BALANCE:
            img_bgr = gray_world_white_balance(img_bgr)

        # Linear gain/offset + gamma to steady exposure/contrast a bit
        img_bgr = cv2.convertScaleAbs(img_bgr, alpha=LINEAR_GAIN, beta=LINEAR_OFFSET)
        img_bgr = apply_gamma(img_bgr, GAMMA_VALUE)

        # Boost saturation/value slightly to help color masks
        hsv_float = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv_float[:, :, 1] = np.clip(hsv_float[:, :, 1] * SATURATION_GAIN, 0, 255)
        hsv_float[:, :, 2] = np.clip(hsv_float[:, :, 2] * VALUE_GAIN, 0, 255)
        hsv_img = hsv_float.astype(np.uint8)
        img_proc = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

        # ----------- Detect circles (rings) via Hough transform -----------
        gray = cv2.cvtColor(img_proc, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=HOUGH_DP,
            minDist=HOUGH_MIN_CENTER_DIST,
            param1=HOUGH_CANNY_HIGH,
            param2=HOUGH_ACCUMULATOR_THRESH,
            minRadius=HOUGH_MIN_RADIUS,
            maxRadius=HOUGH_MAX_RADIUS
        )

        # Keep only de-duplicated circles
        detections = []
        if circles is not None:
            candidates = np.round(circles[0]).astype(np.int32)  # int32 avoids subtraction overflow
            kept = []
            for (cx, cy, rr) in candidates:
                # Simple dedupe: discard if it's very close to an already kept circle
                too_close = any(
                    math.hypot(cx - ox, cy - oy) <= 0.25 * min(rr, orr)
                    for (ox, oy, orr) in kept
                )
                if not too_close:
                    kept.append((cx, cy, rr))
            detections = kept

        # Update ring tracks with current detections (adds smoothing)
        tracks = match_and_update_tracks(tracks, detections)

        # Prepare preview image (copy so we can draw on it)
        vis = img_proc.copy()
        hsv_for_masks = cv2.cvtColor(img_proc, cv2.COLOR_BGR2HSV)

        # ----------- Draw image origin & axes (OpenCV origin is top-left) -----------
        # Origin: (0,0) at top-left, x increases to right, y increases downwards.
        if SHOW_PREVIEW_WINDOW and show_overlays:
            # Small red square at exact origin
            cv2.rectangle(vis, (0, 0), (6, 6), (0, 0, 255), -1)
            # Axes
            cv2.arrowedLine(vis, (0, 0), (70, 0), (255, 255, 255), 1, tipLength=0.25)  # +x →
            cv2.arrowedLine(vis, (0, 0), (0, 70), (255, 255, 255), 1, tipLength=0.25)  # +y ↓
            cv2.putText(vis, "origin (0,0)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, "x->", (74, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(vis, "y v", (6, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # ----------- For each tracked ring: sample, collapse, print centre, update streaks -----------
        for tr in tracks:
            center_x = int(round(tr.x))
            center_y = int(round(tr.y))
            radius = float(tr.r)

            # Adapt S/V thresholds using only pixels in the annulus
            sat_min, val_min = adapt_thresholds_from_annulus(
                hsv_for_masks, center_x, center_y, radius, ANNULUS_INNER_RATIO, ANNULUS_OUTER_RATIO
            )

            # Sample 48 angles from 12 o'clock, clockwise, within the ring band only
            raw48, collapsed_no_u, inner_radius_px, outer_radius_px = sample_around_ring(
                hsv_for_masks, center_x, center_y, radius,
                SAMPLES_AROUND_RING, ANNULUS_INNER_RATIO, ANNULUS_OUTER_RATIO,
                LOCAL_PATCH_RADIUS_PX, NUM_RADII_TO_PROBE,
                sat_min, val_min
            )

            # Final 16-token sequence used for ID (drop 'U', collapse runs, then limit)
            seq16 = limit_to_16(collapsed_no_u)

            # --- Print the robot centre every frame (kept simple and consistent) ---
            print(f"robot centre: ({center_x}, {center_y})")

            # Maintain per-ring streaks of identical seq16
            if len(seq16) == 16:
                if tr.last_seq16 is not None and tr.last_seq16 == seq16:
                    tr.streak16 += 1
                else:
                    tr.last_seq16 = seq16[:]
                    tr.streak16 = 1

                # Update global "best by streak" (for HUD only; we do not print this to terminal)
                if tr.streak16 > best_seq_streak:
                    best_seq_streak = tr.streak16
                    best_seq_overall = seq16[:]

            # ------------- Draw overlays for this ring (optional) -------------
            if SHOW_PREVIEW_WINDOW and show_overlays:
                # Circle & centre
                cv2.circle(vis, (center_x, center_y), int(radius), (0, 255, 0), 2)
                cv2.circle(vis, (center_x, center_y), 2, (0, 0, 255), -1)  # red dot = robot centre

                # Annulus bounds (where sampling happens)
                cv2.circle(vis, (center_x, center_y), inner_radius_px, (255, 255, 0), 1)
                cv2.circle(vis, (center_x, center_y), outer_radius_px, (255, 255, 0), 1)

                # Tick marks for the 48 angles (white dots)
                for i in range(SAMPLES_AROUND_RING):
                    theta = (math.pi/2) - (2.0 * math.pi * i / float(SAMPLES_AROUND_RING))
                    px = int(round(center_x + radius * math.cos(theta)))
                    py = int(round(center_y - radius * math.sin(theta)))
                    cv2.circle(vis, (px, py), 1, (255, 255, 255), -1)

                # Live per-ring 16 if available (shown near the ring)
                text_16 = ' '.join(seq16) if len(seq16) == 16 else "-"
                cv2.putText(
                    vis, text_16, (center_x + 12, center_y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
                )

        # ----------- HUD (FPS, #rings, and best streak info) -----------
        if SHOW_PREVIEW_WINDOW:
            now = time.time()
            dt = now - last_time
            last_time = now
            fps_txt = f"FPS~{int(1.0/dt) if dt > 0 else '--'} Rings:{len(tracks)}"
            cv2.putText(vis, fps_txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

            if best_seq_overall:
                cv2.putText(
                    vis, f"Best16 ({best_seq_streak}): {' '.join(best_seq_overall)}",
                    (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA
                )

            # Show window and handle keys
            cv2.imshow("LED Preview", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                prop = cv2.WND_PROP_FULLSCREEN
                mode = cv2.WINDOW_FULLSCREEN if cv2.getWindowProperty("LED Preview", prop) != cv2.WINDOW_FULLSCREEN else cv2.WINDOW_NORMAL
                cv2.setWindowProperty("LED Preview", prop, mode)
            elif key == ord('m'):
                show_overlays = not show_overlays
        else:
            # Headless mode: still allow 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()