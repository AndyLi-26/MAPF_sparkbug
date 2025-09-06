"""
LED Ring Detector – Multi-Robot, Rotation-Invariant ID

What this does (high level):
- Captures frames from an overhead USB camera.
- Finds circular LED rings via HoughCircles (geometry-first — robust to color noise).
- Samples a thin annulus around each detected circle at 48 angles (starting at 12 o'clock, going clockwise).
- Classifies each sample patch as R/G/B/Unknown (U) using HSV thresholds that adapt per ring.
- Collapses consecutive duplicates and drops Unknown to get a clean sequence, then keeps the first 16 colors.
- Converts that 16-color sequence to a rotation-invariant “canonical” signature and assigns a stable Robot ID.
- Tracks rings across frames (simple nearest-neighbor with exponential smoothing).
- Draws per-ring overlays (circle, sample ticks, live sequence, Robot ID) and a small HUD.
- Prints per-frame compact summaries you can pipe to your controller layer.

Controls:
- q : Quit (closes the preview window)
- f : Toggle fullscreen
- m : Toggle overlays (useful to see raw camera feed)

Notes:
- All thresholds are intentionally conservative. Tune RED_TOL / GREEN_CENTER/TOL / BLUE_CENTER/TOL and the adaptive S/V logic to your lighting.
- Hough parameters determine how aggressively circles are proposed; start from defaults and adjust on your scene.
- If multiple robots share the exact same 16-LED sequence (up to rotation), they’ll get the same Robot ID (by design).
- If you want “unique per individual” even when sequences are identical, use `track_id` instead (that is per-visual-track).
"""

import cv2
import numpy as np
import time
import math

# ---------------------- Camera & Frame ----------------------
CAMERA_INDEX = 0
FRAME_WIDTH, FRAME_HEIGHT, FRAME_FPS = 1280, 720, 30

# ---------------------- Circle Detection (Hough) ----------------------
# dp: Inverse ratio of accumulator resolution to image resolution (1.2 = accumulator is ~83% size of image)
HOUGH_DP = 1.2
# minDist: Minimum distance between centers of detected circles (pixels)
HOUGH_MIN_DIST = 30
# param1: Canny upper threshold (lower is half internally); affects edge detection for Hough
HOUGH_PARAM1 = 50
# param2: Accumulator threshold for circle centers (higher -> fewer false circles)
HOUGH_PARAM2 = 90
# min/max radius (in pixels) for rings in your scene (rough bounds are fine)
HOUGH_RADIUS_MIN = 10
HOUGH_RADIUS_MAX = 200

# ---------------------- Annulus Sampling ----------------------
# We sample patches along a thin ring (annulus) just inside/outside the detected circle
SAMPLES_PER_RING = 48                  # angular samples per ring (12 o'clock CW)
ANNULUS_INNER_SCALE = 0.85             # inner radius = scale * detected radius
ANNULUS_OUTER_SCALE = 1.05             # outer radius = scale * detected radius
PATCH_HALF_SIZE_PX = 4                 # half-size of square patch for voting (patch = (2h+1)^2)
NUM_SAMPLED_RADII = 3                  # sample that many radii uniformly within the annulus

# ---------------------- Color Thresholds (HSV) ----------------------
# Base S and V mins (will be adapted per ring by local percentiles)
BASE_S_MIN = 120
BASE_V_MIN = 130

# Hue windows (0..179 in OpenCV). Tune to your LEDs.
RED_TOLERANCE = 14                     # red is special (wraps around 0/179)
GREEN_HUE_CENTER, GREEN_HUE_TOL = 70, 22
BLUE_HUE_CENTER,  BLUE_HUE_TOL  = 115, 18

# ---------------------- Preprocessing ----------------------
APPLY_GRAY_WORLD_WB = True             # simple per-frame gray-world white balance
LINEAR_GAIN = 0.9                      # brightness/contrast gain (alpha)
LINEAR_OFFSET = -5                     # brightness offset (beta)
GAMMA = 1.8                            # gamma correction (>1 = darker mids, helps de-white saturated LEDs)
SAT_GAIN = 2.0                         # multiply HSV S (saturation)
VAL_GAIN = 1.2                         # multiply HSV V (value)

# ---------------------- Tracking & Smoothing ----------------------
SMOOTH_ALPHA = 0.5                     # EMA weight for center & radius (higher = smoother, slower)
MATCH_DISTANCE_THRESHOLD = 40.0        # pixels; max distance to link detection to existing track

# ---------------------- Visualization ----------------------
DRAW_OVERLAYS = True
SHOW_PREVIEW = True

# ---------------------- Utilities ----------------------
def clip_u8(arr):
    """Clamp to [0,255] and convert to uint8."""
    return np.clip(arr, 0, 255).astype(np.uint8)

def gray_world_white_balance(bgr_img):
    """Simple gray-world white balance: scale channels so their means are equal."""
    b, g, r = cv2.split(bgr_img.astype(np.float32))
    mb, mg, mr = b.mean() + 1e-6, g.mean() + 1e-6, r.mean() + 1e-6
    mean_all = (mb + mg + mr) / 3.0
    gb, gg, gr = mean_all / mb, mean_all / mg, mean_all / mr
    return clip_u8(cv2.merge([b * gb, g * gg, r * gr]))

def apply_gamma_u8(bgr_img, gamma):
    """Apply gamma curve using a lookup table (fast)."""
    if abs(gamma - 1.0) < 1e-3:
        return bgr_img
    inv_gamma = 1.0 / max(gamma, 1e-6)
    lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], np.uint8)
    return cv2.LUT(bgr_img, lut)

# ---------------------- Color Masks (HSV) ----------------------
def red_mask_from_hsv(hsv_img, tol, s_min, v_min):
    """Binary mask for RED using wrap-around at hue=0/179."""
    h, s, v = cv2.split(hsv_img)
    hue_hit = (h <= tol) | (h >= 180 - tol)
    return ((hue_hit) & (s >= s_min) & (v >= v_min)).astype(np.uint8) * 255

def color_mask_from_hsv(hsv_img, center, tol, s_min, v_min):
    """Binary mask for a hue window centered at `center` with ±tol."""
    h, s, v = cv2.split(hsv_img)
    lo = (center - tol) % 180
    hi = (center + tol) % 180
    if lo <= hi:
        hue_hit = (h >= lo) & (h <= hi)
    else:
        # window crosses 0: accept high or low wrap
        hue_hit = (h >= lo) | (h <= hi)
    return ((hue_hit) & (s >= s_min) & (v >= v_min)).astype(np.uint8) * 255

def collapse_consecutive(seq_chars):
    """Collapse consecutive duplicates: e.g., R R G G U R -> R G U R."""
    if not seq_chars:
        return []
    out = [seq_chars[0]]
    for c in seq_chars[1:]:
        if c != out[-1]:
            out.append(c)
    return out

def drop_unknown_and_collapse(seq_chars):
    """Drop 'U' (unknown) and collapse runs; result only has 'R','G','B'."""
    return [c for c in collapse_consecutive(seq_chars) if c in ('R', 'G', 'B')]

def first_16(seq_chars):
    """Take the first 16 tokens (or fewer if not available)."""
    return seq_chars[:16]

# ---------------------- Sampling & Adaptation ----------------------
def sample_ring_sequence(
    hsv_img, cx, cy, radius,
    num_angles, inner_scale, outer_scale, patch_half_size, num_radii,
    s_min, v_min
):
    """
    For one ring:
    - Sample `num_angles` around the circle from 12 o'clock clockwise.
    - For each angle, evaluate multiple radii within the annulus.
    - Each sample votes R/G/B by counting pixels within that patch from the 3 masks.
    - Choose the color with the highest count; 'U' if no count > 0.
    Returns:
    - raw_48:     raw sequence length `num_angles` incl. 'U'
    - collapsed:  collapsed/no-U sequence
    - inner_px:   inner radius (for drawing)
    - outer_px:   outer radius (for drawing)
    """
    height, width = hsv_img.shape[:2]
    # Angle list: start at 12 o'clock (pi/2), go clockwise
    thetas = (np.pi / 2.0) - (2.0 * np.pi * np.arange(num_angles) / float(num_angles))

    inner_radius = radius * inner_scale
    outer_radius = radius * outer_scale
    radii_to_sample = np.linspace(inner_radius, outer_radius, max(1, num_radii))

    mask_red   = red_mask_from_hsv(hsv_img, RED_TOLERANCE, s_min, v_min)
    mask_green = color_mask_from_hsv(hsv_img, GREEN_HUE_CENTER, GREEN_HUE_TOL, s_min, v_min)
    mask_blue  = color_mask_from_hsv(hsv_img, BLUE_HUE_CENTER,  BLUE_HUE_TOL,  s_min, v_min)

    seq_raw = []
    for theta in thetas:
        best_color = 'U'
        best_vote = -1
        for r_test in radii_to_sample:
            px = int(round(cx + r_test * math.cos(theta)))
            py = int(round(cy - r_test * math.sin(theta)))  # screen y grows downward
            if 0 <= px < width and 0 <= py < height:
                x0, x1 = max(0, px - patch_half_size), min(width,  px + patch_half_size + 1)
                y0, y1 = max(0, py - patch_half_size), min(height, py + patch_half_size + 1)
                vote_r = int(np.count_nonzero(mask_red[y0:y1,   x0:x1]))
                vote_g = int(np.count_nonzero(mask_green[y0:y1, x0:x1]))
                vote_b = int(np.count_nonzero(mask_blue[y0:y1,  x0:x1]))
                vote = max(vote_r, vote_g, vote_b)
                if vote > best_vote and vote > 0:
                    best_vote = vote
                    best_color = 'R' if vote == vote_r else ('G' if vote == vote_g else 'B')
        seq_raw.append(best_color)

    collapsed_no_u = drop_unknown_and_collapse(seq_raw)
    return seq_raw, collapsed_no_u, int(inner_radius), int(outer_radius)

def adapt_sv_thresholds_within_annulus(hsv_img, cx, cy, radius, inner_scale, outer_scale):
    """
    Compute S/V thresholds per ring from local annulus statistics.
    Idea: LEDs are among the brightest/saturated pixels near the ring.
    - S_min ~ 0.8 * 75th percentile of saturation
    - V_min ~ 0.5 * 98th percentile of value
    Clamped to a floor relative to BASE_S_MIN/BASE_V_MIN.
    """
    height, width = hsv_img.shape[:2]
    inner_r = int(max(1, radius * inner_scale))
    outer_r = int(max(inner_r + 1, radius * outer_scale))

    Y, X = np.ogrid[:height, :width]
    rsq = (X - cx) ** 2 + (Y - cy) ** 2
    annulus_mask = (rsq >= inner_r * inner_r) & (rsq <= outer_r * outer_r)

    s_vals = hsv_img[:, :, 1][annulus_mask].astype(np.float32)
    v_vals = hsv_img[:, :, 2][annulus_mask].astype(np.float32)
    if s_vals.size == 0:
        return BASE_S_MIN, BASE_V_MIN

    s75 = np.percentile(s_vals, 75)
    v98 = np.percentile(v_vals, 98)
    s_min = int(max(BASE_S_MIN * 0.8, min(255, s75 * 0.8)))
    v_min = int(max(BASE_V_MIN * 0.7, min(255, v98 * 0.5)))
    return s_min, v_min

# ---------------------- Rotation-Invariant Signature & IDs ----------------------
def canonical_rotation(seq16):
    """
    Make sequence rotation-invariant by choosing the lexicographically smallest rotation.
    If not exactly 16, return tuple as-is (not considered stable yet).
    """
    if not seq16 or len(seq16) != 16:
        return tuple(seq16)
    s = seq16[:]
    rotations = [tuple(s[i:] + s[:i]) for i in range(len(s))]
    return min(rotations)

# Registry that maps canonical 16-token sequences -> incremental Robot ID
ROBOT_REGISTRY = {}     # { canonical_tuple : robot_id }
NEXT_ROBOT_ID = 1       # increment as new unique sequences appear

# ---------------------- Track Class (per visual circle) ----------------------
NEXT_TRACK_ID = 0

class Track:
    """A visual track for one detected ring (smoothed center/radius + last sequence info)."""
    def __init__(self, center_x, center_y, radius):
        global NEXT_TRACK_ID
        self.track_id = NEXT_TRACK_ID; NEXT_TRACK_ID += 1

        # State (smoothed)
        self.center_x = float(center_x)
        self.center_y = float(center_y)
        self.radius   = float(radius)

        # Bookkeeping
        self.missed_frames = 0

        # Sequence/stability
        self.last_seq16 = None   # last 16-length sequence list
        self.seq_streak = 0      # how many consecutive frames the 16-seq stayed identical

        # Identity
        self.robot_id = None     # assigned from rotation-invariant signature

    def update_ema(self, center_x, center_y, radius, alpha=SMOOTH_ALPHA):
        """EMA smoothing for center/radius; resets missed counter."""
        self.center_x = alpha * self.center_x + (1.0 - alpha) * float(center_x)
        self.center_y = alpha * self.center_y + (1.0 - alpha) * float(center_y)
        self.radius   = alpha * self.radius   + (1.0 - alpha) * float(radius)
        self.missed_frames = 0

# ---------------------- Simple Nearest-Neighbor Tracker ----------------------
def match_and_update_tracks(existing_tracks, current_detections):
    """
    Greedily match each existing track to the nearest unclaimed detection within a distance threshold.
    Unmatched detections create new tracks. Tracks that go missing for a few frames are dropped.
    """
    used = [False] * len(current_detections)

    # Try to update existing tracks with closest detection
    for tr in existing_tracks:
        best_j = -1
        best_dist = 1e9
        for j, (x, y, r) in enumerate(current_detections):
            if used[j]:
                continue
            d = math.hypot(tr.center_x - x, tr.center_y - y)
            if d < best_dist:
                best_dist = d
                best_j = j

        if best_j >= 0 and best_dist <= MATCH_DISTANCE_THRESHOLD:
            x, y, r = current_detections[best_j]
            tr.update_ema(x, y, r)
            used[best_j] = True
        else:
            tr.missed_frames += 1

    # Add new tracks for unmatched detections
    for j, (x, y, r) in enumerate(current_detections):
        if not used[j]:
            existing_tracks.append(Track(x, y, r))

    # Keep only active tracks (allow up to 5 consecutive misses)
    active_tracks = [tr for tr in existing_tracks if tr.missed_frames <= 5]
    return active_tracks

# ---------------------- Main Loop ----------------------
def main():
    global NEXT_ROBOT_ID, ROBOT_REGISTRY

    # --- Open camera ---
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError("Camera open failed")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          FRAME_FPS)

    show_overlays = DRAW_OVERLAYS
    tracks = []

    # HUD “best” (just for the top line; not used for logic)
    hud_best_seq = None
    hud_best_streak = 0

    # Prepare preview window
    if SHOW_PREVIEW:
        cv2.namedWindow("LED Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("LED Preview", 1280, 720)

    last_time = time.time()

    # --- Frame loop ---
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            # If the camera starves, don’t crash; just wait a moment.
            time.sleep(0.005)
            continue

        # ----- Preprocess (WB -> gain/offset -> gamma -> HSV adapt S/V -> back to BGR for Hough) -----
        img_bgr = frame_bgr
        if APPLY_GRAY_WORLD_WB:
            img_bgr = gray_world_white_balance(img_bgr)

        img_bgr = cv2.convertScaleAbs(img_bgr, alpha=LINEAR_GAIN, beta=LINEAR_OFFSET)
        img_bgr = apply_gamma_u8(img_bgr, GAMMA)

        hsv_tmp = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv_tmp[:, :, 1] = np.clip(hsv_tmp[:, :, 1] * SAT_GAIN, 0, 255)
        hsv_tmp[:, :, 2] = np.clip(hsv_tmp[:, :, 2] * VAL_GAIN, 0, 255)
        hsv_for_color = hsv_tmp.astype(np.uint8)

        # Convert back to BGR for circle detection (Hough likes grayscale of BGR)
        bgr_for_hough = cv2.cvtColor(hsv_for_color, cv2.COLOR_HSV2BGR)

        # ----- Circle detection (Hough) -----
        gray = cv2.cvtColor(bgr_for_hough, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            gray_blur, cv2.HOUGH_GRADIENT,
            dp=HOUGH_DP, minDist=HOUGH_MIN_DIST,
            param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
            minRadius=HOUGH_RADIUS_MIN, maxRadius=HOUGH_RADIUS_MAX
        )

        # Keep circles that are not extremely overlapping (rudimentary NMS)
        current_detections = []
        if circles is not None:
            proposed = np.round(circles[0]).astype(np.int32)
            kept = []
            for (x, y, r) in proposed:
                too_close = any(math.hypot(x - x0, y - y0) <= 0.25 * min(r, r0) for (x0, y0, r0) in kept)
                if not too_close:
                    kept.append((x, y, r))
            current_detections = kept

        # ----- Update tracks -----
        tracks = match_and_update_tracks(tracks, current_detections)

        # Visualization buffer
        preview_bgr = bgr_for_hough.copy()
        hsv_for_sampling = cv2.cvtColor(bgr_for_hough, cv2.COLOR_BGR2HSV)

        # Draw origin & axes (OpenCV origin is top-left)
        if SHOW_PREVIEW and show_overlays:
            cv2.rectangle(preview_bgr, (0, 0), (6, 6), (0, 0, 255), -1)  # small red square at origin
            cv2.arrowedLine(preview_bgr, (0, 0), (70, 0), (255, 255, 255), 1, tipLength=0.25)  # +x to the right
            cv2.arrowedLine(preview_bgr, (0, 0), (0, 70), (255, 255, 255), 1, tipLength=0.25)  # +y downward
            cv2.putText(preview_bgr, "origin (0,0)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(preview_bgr, "x->", (74, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(preview_bgr, "y v", (6, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # ----- Per-ring sampling & identification -----
        per_frame_output = []  # collect info for this frame (for printing/logging/IPC)
        for tr in tracks:
            cx = int(round(tr.center_x))
            cy = int(round(tr.center_y))
            radius = float(tr.radius)

            # Local S/V thresholds from the annulus
            s_min, v_min = adapt_sv_thresholds_within_annulus(
                hsv_for_sampling, cx, cy, radius,
                ANNULUS_INNER_SCALE, ANNULUS_OUTER_SCALE
            )

            # Sample around the ring
            raw_48, collapsed_no_u, inner_px, outer_px = sample_ring_sequence(
                hsv_for_sampling, cx, cy, radius,
                SAMPLES_PER_RING, ANNULUS_INNER_SCALE, ANNULUS_OUTER_SCALE,
                PATCH_HALF_SIZE_PX, NUM_SAMPLED_RADII,
                s_min, v_min
            )
            seq16 = first_16(collapsed_no_u)

            # Log robot center every frame (helps you line up geometry & intrinsics)
            print(f"robot center: ({cx}, {cy})")

            # Track-internal stability: how long the seq16 stayed identical
            if len(seq16) == 16:
                if tr.last_seq16 is not None and tr.last_seq16 == seq16:
                    tr.seq_streak += 1
                else:
                    tr.last_seq16 = seq16[:]
                    tr.seq_streak = 1

                # Update HUD-best (purely cosmetic)
                if tr.seq_streak > hud_best_streak:
                    hud_best_streak = tr.seq_streak
                    hud_best_seq = seq16[:]

            # Rotation-invariant signature -> Robot ID
            robot_id_to_draw = tr.robot_id
            if len(seq16) == 16:
                canon = canonical_rotation(seq16)
                if canon in ROBOT_REGISTRY:
                    robot_id_to_draw = ROBOT_REGISTRY[canon]
                else:
                    robot_id_to_draw = NEXT_ROBOT_ID
                    ROBOT_REGISTRY[canon] = robot_id_to_draw
                    NEXT_ROBOT_ID += 1
                tr.robot_id = robot_id_to_draw  # persist

            # ----- Overlays -----
            if SHOW_PREVIEW and show_overlays:
                # Circle + center
                cv2.circle(preview_bgr, (cx, cy), int(radius), (0, 255, 0), 2)
                cv2.circle(preview_bgr, (cx, cy), 2, (0, 0, 255), -1)

                # Annulus bounds (for your reference)
                cv2.circle(preview_bgr, (cx, cy), inner_px, (255, 255, 0), 1)
                cv2.circle(preview_bgr, (cx, cy), outer_px, (255, 255, 0), 1)

                # Angular ticks (where samples are taken)
                for i in range(SAMPLES_PER_RING):
                    theta = (np.pi / 2.0) - (2.0 * np.pi * i / float(SAMPLES_PER_RING))
                    px = int(round(cx + radius * math.cos(theta)))
                    py = int(round(cy - radius * math.sin(theta)))
                    preview_bgr = cv2.circle(preview_bgr, (px, py), 1, (255, 255, 255), -1)

                # Live per-ring 16-sequence (shows '-' until 16 available)
                live_text = ' '.join(seq16) if len(seq16) == 16 else "-"
                cv2.putText(preview_bgr, live_text, (cx + 12, cy - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                # Robot ID (by rotation-invariant signature)
                id_text = f"ID:{robot_id_to_draw}" if robot_id_to_draw is not None else "ID:-"
                cv2.putText(preview_bgr, id_text, (cx + 12, cy + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            # Collect for programmatic use
            per_frame_output.append({
                "robot_id": robot_id_to_draw,
                "track_id": tr.track_id,
                "center": (cx, cy),
                "radius": radius,
                "seq16": seq16[:] if len(seq16) == 16 else []
            })

        # Optional: print a compact per-frame summary (comment out if too chatty)
        if per_frame_output:
            printable = []
            for d in per_frame_output:
                if d["robot_id"] is not None and len(d["seq16"]) == 16:
                    printable.append({
                        "robot_id": d["robot_id"],
                        "track_id": d["track_id"],
                        "center": d["center"],
                        "seq16": ''.join(d["seq16"])
                    })
            if printable:
                print("robots:", printable)

        # ----- HUD & key handling -----
        if SHOW_PREVIEW:
            now = time.time()
            dt = now - last_time
            last_time = now
            fps_text = f"FPS~{int(1.0/dt) if dt > 0 else '--'} Rings:{len(tracks)}"
            cv2.putText(preview_bgr, fps_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

            if hud_best_seq:
                cv2.putText(preview_bgr, f"Best16 ({hud_best_streak}): {' '.join(hud_best_seq)}",
                            (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("LED Preview", preview_bgr)
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
            # Even without a window, allow 'q' to bail out if any event loop is active
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Clean shutdown: release camera and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# ---------------------- Entry ----------------------
if __name__ == "__main__":
    main()
