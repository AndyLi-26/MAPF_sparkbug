"""
LED-first robot detector (with robust 16-LED sequence):
- LED masks (HSV) -> DBSCAN clusters -> RANSAC+LS circle fit
- 48-point annulus sampling from 12 o'clock CW
- Per-ring adaptive S/V thresholds + hue-distance color classifier
- 48 -> exact 16 bins (majority per 3 samples), symbols in C/M/Y/G
- Rotation-invariant ID match (window=6) via external codebook

Changes:
- No 'U' symbols produced anywhere (default fallback -> 'C')
- Merge near/overlapping circles to avoid duplicate robots
"""

import cv2
import numpy as np
import math, time, random

# === External codebook & helpers ===
import codebook_helpers as cb

# ---------------------- Camera / frame ----------------------
CAM_INDEX = 0
FRAME_W, FRAME_H, FRAME_FPS = 1280, 720, 30

# ---------------------- Color model (HSV) ----------------------
HUE_CENTERS = {'C': 90, 'M': 150, 'Y': 30, 'G': 60}  # cyan, magenta, yellow, soft-green
HUE_TOL     = {'C': 15, 'M': 15, 'Y': 10, 'G': 12}
SAT_MIN = 70
VAL_MIN = 70

# ---------------------- Pre-processing ----------------------
USE_GRAY_WORLD_WB = True
LINEAR_GAIN = 0.9
LINEAR_OFFSET = -5
GAMMA = 1.8
SAT_GAIN = 2.0
VAL_GAIN = 1.0
# --- Extra LED saliency preproc params ---
CLAHE_CLIP = 2.0                 # 1.5–3.0 is reasonable
CLAHE_TILE = (8, 8)               # local equalization grid
TOPHAT_KERNEL = 9                 # size of structuring element for top-hat (odd)
BG_DIM = 20                     # global dim factor for non-LED regions
LED_BOOST = 8.0                   # how much to boost LED regions
LED_V_PERCENTILE = 98             # percentile to define "very bright" pixels
LED_S_GATE = 120                  # min saturation for saliency (HSV S channel)


# ---------------------- Clustering ----------------------
DBSCAN_EPS = 40
DBSCAN_MIN_SAMPLES = 6

# ---------------------- Circle fit & sampling ----------------------
SAMPLES_AROUND = 48
ANNULUS_INNER = 0.82
ANNULUS_OUTER = 1.08
PATCH_RADIUS  = 5
RADII_PROBES  = 3
INLIER_DIST_THRESH = 5.0

# Expected ring size band (px)
MAX_RING_RADIUS_PX = 70
MIN_RING_RADIUS_PX = 45

# ------ Duplicate circle suppression (tunable) ------
MERGE_CENTER_DIST_PX = 15      # if centers closer than this, treat as same robot
MERGE_OVERLAP_FRAC   = 0.60    # or if center distance < frac * min(radius)

# ---------------------- Tracking ----------------------
SMOOTH_ALPHA = 0.3
MATCH_DIST = 50.0

# ---------------------- Visualization ----------------------
SHOW_PREVIEW = True
DRAW_OVERLAYS = True


# ==== Utilities ===============================================================
def enhance_for_leds(img_bgr):
    """
    Strong, LED-focused preprocessing:
      1) Gray-world WB (optional in your global flag)
      2) Linear gain/offset + gamma (your current settings)
      3) HSV: boost S and V (your SAT_GAIN / VAL_GAIN)
      4) CLAHE on V + Top-Hat to emphasize small bright blobs
      5) Build a saliency mask using (V >= p98) & (S >= LED_S_GATE)
      6) Blend: dim background, boost LED regions
    Returns:
      - bgr_view: boosted image for visualization / downstream
      - hsv_boost: HSV image aligned with bgr_view (for masks/sampling)
    """
    # Step 1–2: (we leave gray-world to your global flag, so we only linear + gamma here)
    base = cv2.convertScaleAbs(img_bgr, alpha=LINEAR_GAIN, beta=LINEAR_OFFSET)
    base = apply_gamma(base, GAMMA)

    # Step 3: HSV + global color gains
    hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * SAT_GAIN, 0, 255)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * VAL_GAIN, 0, 255)

    # Step 4: CLAHE on V + Top-Hat (on equalized V)
    V8 = hsv[:,:,2].astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    V_eq = clahe.apply(V8)

    ksz = TOPHAT_KERNEL if TOPHAT_KERNEL % 2 == 1 else TOPHAT_KERNEL + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    V_tophat = cv2.morphologyEx(V_eq, cv2.MORPH_TOPHAT, kernel)

    # Combine V components: keep some equalized V, add tophat to highlight LEDs
    V_boost = np.clip(0.7 * V_eq + 1.3 * V_tophat, 0, 255).astype(np.uint8)

    # Step 5: LED saliency mask (using robust V threshold + saturation gate)
    V_thresh = np.percentile(V_boost, LED_V_PERCENTILE)
    S8 = hsv[:,:,1].astype(np.uint8)
    led_mask = ((V_boost >= V_thresh) & (S8 >= LED_S_GATE)).astype(np.uint8) * 255
    # optional cleanup
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    led_mask = cv2.morphologyEx(led_mask, cv2.MORPH_OPEN, k_small, iterations=1)

    # Step 6: Compose boosted view: dim background, boost LED regions
    hsv_boost = hsv.copy()
    hsv_boost[:,:,2] = V_boost

    bgr_boost = cv2.cvtColor(hsv_boost.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    base_dim  = (base.astype(np.float32) * BG_DIM)

    mask_f = (led_mask.astype(np.float32) / 255.0)[:,:,None]
    # On LED regions: boost further; elsewhere: dim
    bgr_view = base_dim * (1.0 - mask_f) + (bgr_boost * LED_BOOST) * mask_f
    bgr_view = clip8(bgr_view)

    # Keep an HSV aligned with bgr_view for downstream color masks/sampling
    hsv_view = cv2.cvtColor(bgr_view, cv2.COLOR_BGR2HSV)

    return bgr_view, hsv_view


def clip8(x): return np.clip(x, 0, 255).astype(np.uint8)

def gray_world(img_bgr):
    b,g,r = cv2.split(img_bgr.astype(np.float32))
    mb, mg, mr = b.mean()+1e-6, g.mean()+1e-6, r.mean()+1e-6
    m = (mb + mg + mr)/3.0
    return clip8(cv2.merge([b*(m/mb), g*(m/mg), r*(m/mr)]))

def apply_gamma(img_bgr, gamma):
    if abs(gamma - 1.0) < 1e-3: return img_bgr
    inv = 1.0 / max(gamma, 1e-6)
    lut = np.array([((i/255.0)**inv)*255 for i in range(256)], np.uint8)
    return cv2.LUT(img_bgr, lut)

def collapse_runs(seq):
    if not seq: return []
    out=[seq[0]]
    for c in seq[1:]:
        if c != out[-1]: out.append(c)
    return out

# ==== Masks for finding LED blobs ============================================
def hue_band_mask(hsv_img, hue_center, hue_tol, s_min, v_min):
    h,s,v = cv2.split(hsv_img)
    lo = (hue_center - hue_tol) % 180
    hi = (hue_center + hue_tol) % 180
    hue_ok = (h >= lo) & (h <= hi) if lo <= hi else ((h >= lo) | (h <= hi))
    return ((hue_ok) & (s >= s_min) & (v >= v_min)).astype(np.uint8) * 255

def build_color_masks(hsv_img):
    masks = {}
    k = np.ones((3,3), np.uint8)
    for sym, hc in HUE_CENTERS.items():
        m = hue_band_mask(hsv_img, hc, HUE_TOL[sym], SAT_MIN, VAL_MIN)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
        masks[sym] = m
    return masks

def find_blob_centroids(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts=[]
    for c in cnts:
        if len(c) < 3: continue
        if cv2.contourArea(c) < 3: continue
        M = cv2.moments(c)
        if M['m00'] > 0:
            cx = int(M['m10']/M['m00']); cy = int(M['m01']/M['m00'])
            pts.append((cx, cy))
    return pts

def detect_led_blobs(hsv_img):
    masks = build_color_masks(hsv_img)
    all_pts=[]; all_syms=[]
    for sym, m in masks.items():
        pts = find_blob_centroids(m)
        all_pts.extend(pts)
        all_syms.extend([sym]*len(pts))
    return np.array(all_pts, dtype=np.int32), all_syms

# ==== DBSCAN ==================================================================
def dbscan(points, eps, min_pts):
    if len(points) == 0: return []
    P = points.astype(np.float32)
    n = len(P)
    visited = np.zeros(n, dtype=bool)
    labels  = np.full(n, -1, dtype=int)
    cid = 0
    for i in range(n):
        if visited[i]: continue
        visited[i] = True
        d = np.hypot(P[:,0]-P[i,0], P[:,1]-P[i,1])
        neigh = np.where(d <= eps)[0]
        if len(neigh) < min_pts:
            labels[i] = -1; continue
        labels[neigh] = cid
        queue = list(neigh)
        for j in queue:
            if not visited[j]:
                visited[j] = True
                d2 = np.hypot(P[:,0]-P[j,0], P[:,1]-P[j,1])
                neigh2 = np.where(d2 <= eps)[0]
                if len(neigh2) >= min_pts:
                    for k in neigh2:
                        if labels[k] == -1: labels[k] = cid
                        if k not in queue: queue.append(k)
        cid += 1
    return [np.where(labels==k)[0].tolist() for k in range(cid)]

# ==== Circle fitting ==========================================================
def circle_from_3pts(p1, p2, p3):
    (x1,y1),(x2,y2),(x3,y3) = p1, p2, p3
    temp = x2**2 + y2**2
    bc = (x1**2 + y1**2 - temp)/2.0
    cd = (temp - x3**2 - y3**2)/2.0
    det = (x1-x2)*(y2-y3) - (x2-x3)*(y1-y2)
    if abs(det) < 1e-6: return None
    cx = (bc*(y2-y3) - cd*(y1-y2)) / det
    cy = ((x1-x2)*cd - (x2-x3)*bc) / det
    r  = math.hypot(x1-cx, y1-cy)
    return (cx, cy, r)

def fit_circle_least_squares(pts):
    x = pts[:,0].astype(np.float64); y = pts[:,1].astype(np.float64)
    xm, ym = x.mean(), y.mean()
    u = x - xm; v = y - ym
    Suu = (u*u).sum(); Svv = (v*v).sum(); Suv = (u*v).sum()
    Suuu = (u*u*u).sum(); Svvv = (v*v*v).sum()
    Suvv = (u*v*v).sum(); Svuu = (v*u*u).sum()
    A = np.array([[Suu, Suv],[Suv, Svv]], np.float64)
    b = np.array([0.5*(Suuu + Suvv), 0.5*(Svvv + Svuu)], np.float64)
    if abs(np.linalg.det(A)) < 1e-9: return None
    uc, vc = np.linalg.solve(A, b)
    cx = xm + uc; cy = ym + vc
    r  = np.mean(np.hypot(x - cx, y - cy))
    return (cx, cy, r)

def fit_circle_ransac(pts_xy, trials=64, inlier_thresh=INLIER_DIST_THRESH):
    pts = np.asarray(pts_xy, dtype=np.float32)
    if len(pts) < 3: return None, None
    best = None; best_inliers = []
    n = len(pts)
    for _ in range(trials):
        i1, i2, i3 = random.sample(range(n), 3)
        circ = circle_from_3pts(pts[i1], pts[i2], pts[i3])
        if circ is None: continue
        cx, cy, r = circ
        d = np.abs(np.hypot(pts[:,0]-cx, pts[:,1]-cy) - r)
        inliers = np.where(d <= inlier_thresh)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best = (cx, cy, r)
    if best is None or len(best_inliers) < 3:
        return None, None
    refined = fit_circle_least_squares(pts[best_inliers])
    return refined, best_inliers

# === Adaptive thresholds from the ring annulus ===============================
def estimate_ring_thresholds(hsv_img, cx, cy, r, rin_pct=ANNULUS_INNER, rout_pct=ANNULUS_OUTER):
    H, W = hsv_img.shape[:2]
    rin = max(1, int(r * rin_pct))
    rout = max(rin+1, int(r * rout_pct))
    Y, X = np.ogrid[:H, :W]
    rsq = (X - cx)**2 + (Y - cy)**2
    mask = (rsq >= rin*rin) & (rsq <= rout*rout)
    if not np.any(mask):
        return SAT_MIN, VAL_MIN
    s = hsv_img[:,:,1][mask].astype(np.float32)
    v = hsv_img[:,:,2][mask].astype(np.float32)
    s_gate = int(np.clip(np.percentile(s, 65) * 0.9, 60, 255))
    v_gate = int(np.clip(np.percentile(v, 95) * 0.8, 60, 255))
    return s_gate, v_gate

# === Hue-distance classifier; NEVER returns 'U' ==============================
def classify_patch_hsv(hsv_patch, s_gate, v_gate):
    """
    Return 'C','M','Y','G'. If patch too dim/unsaturated or far from all hues,
    we default to 'C' to avoid any 'U' in downstream logic.
    """
    h = hsv_patch[...,0].astype(np.float32)
    s = hsv_patch[...,1].astype(np.float32)
    v = hsv_patch[...,2].astype(np.float32)

    mS = float(np.mean(s)); mV = float(np.mean(v))
    # If below gates, fallback to 'C'
    if mS < s_gate or mV < v_gate:
        return 'C'

    # circular mean hue
    ang = h * (np.pi/90.0)
    c = np.cos(ang); s_ = np.sin(ang)
    mean_ang = math.atan2(np.mean(s_), np.mean(c))
    if mean_ang < 0: mean_ang += 2*np.pi
    mean_h = mean_ang * (90.0/np.pi)

    # nearest hue center
    best_sym='C'; best_d=1e9
    for sym, hc in HUE_CENTERS.items():
        d = abs(mean_h - hc)
        d = min(d, 180 - d)
        if d < best_d:
            best_d = d; best_sym = sym

    # Guard: if exceedingly far, still default to 'C'
    return best_sym if best_d <= max(HUE_TOL.values()) else 'C'

# === 48-sample sequence with multi-radius voting; NEVER yields 'U' ===========
def sample_sequence_symbols(hsv_img, cx, cy, r):
    H, W = hsv_img.shape[:2]
    thetas = (np.pi/2) - (2.0*np.pi*np.arange(SAMPLES_AROUND)/float(SAMPLES_AROUND))
    rin = r * ANNULUS_INNER
    rout = r * ANNULUS_OUTER
    radii = np.linspace(rin, rout, max(1, RADII_PROBES))

    s_gate, v_gate = estimate_ring_thresholds(hsv_img, int(cx), int(cy), r)

    seq=[]
    for th in thetas:
        votes = {'C':0, 'M':0, 'Y':0, 'G':0}
        for rr in radii:
            px = int(round(cx + rr*math.cos(th)))
            py = int(round(cy - rr*math.sin(th)))
            if 0 <= px < W and 0 <= py < H:
                x0, x1 = max(0, px-PATCH_RADIUS), min(W, px+PATCH_RADIUS+1)
                y0, y1 = max(0, py-PATCH_RADIUS), min(H, py+PATCH_RADIUS+1)
                patch = hsv_img[y0:y1, x0:x1]
                sym = classify_patch_hsv(patch, s_gate, v_gate)  # returns C/M/Y/G
                votes[sym] += 1
        # choose best; default remains 'C'
        best_sym, best_count = 'C', -1
        for sym in ('C','M','Y','G'):
            if votes[sym] > best_count:
                best_count = votes[sym]; best_sym = sym
        seq.append(best_sym)
    return seq

# === 48 -> exact 16; NEVER yields 'U' ========================================
def samples48_to_exact16(seq48):
    out=[]
    for i in range(16):
        chunk = seq48[3*i : 3*i + 3]
        counts = {'C':0,'M':0,'Y':0,'G':0}
        for c in chunk:
            if c in counts: counts[c] += 1
        # if all zero (shouldn't happen now), default to 'C'
        best_sym = max(counts.items(), key=lambda kv: kv[1])[0]
        out.append(best_sym if counts[best_sym] > 0 else 'C')
    return out

# ==== Rotation-invariant matching (fallback, not used if your cb has it) =====
def match_id_rotation_window(seq16_syms, codebook_num, window=6):
    if len(seq16_syms) < window: return (None, 0)
    try:
        obs_int = cb.syms_to_ints(seq16_syms)
    except AttributeError:
        sym2idx = {'C':0,'M':1,'Y':2,'G':3}
        obs_int = [sym2idx[s] for s in seq16_syms]

    best_idx, best_score = None, -1
    for idx, code in enumerate(codebook_num):
        for rot in range(16):
            rotcode = code[rot:] + code[:rot]
            for j in range(16 - window + 1):
                score = sum(1 for a,b in zip(obs_int[j:j+window], rotcode[j:j+window]) if a==b)
                if score > best_score:
                    best_score = score
                    best_idx = idx
    return (best_idx, best_score)

# ==== Track ===================================================================
class Track:
    def __init__(self, cx, cy, r):
        self.cx = float(cx); self.cy = float(cy); self.r = float(r)
        self.missed = 0
        self.last_seq16 = None
        self.streak16 = 0

    def update(self, cx, cy, r, alpha=SMOOTH_ALPHA):
        self.cx = alpha*self.cx + (1-alpha)*float(cx)
        self.cy = alpha*self.cy + (1-alpha)*float(cy)
        self.r  = alpha*self.r  + (1-alpha)*float(r)
        self.missed = 0

def associate_tracks(tracks, circles):
    used = [False]*len(circles)
    for tr in tracks:
        best_j=-1; best_d=1e9
        for j,(cx,cy,r) in enumerate(circles):
            if used[j]: continue
            d=math.hypot(tr.cx-cx, tr.cy-cy)
            if d < best_d:
                best_d=d; best_j=j
        if best_j>=0 and best_d<=MATCH_DIST:
            cx,cy,r = circles[best_j]
            tr.update(cx,cy,r)
            used[best_j]=True
        else:
            tr.missed += 1
    for j,(cx,cy,r) in enumerate(circles):
        if not used[j]:
            tracks.append(Track(cx,cy,r))
    return [t for t in tracks if t.missed <= 5]

# ==== Merge near/overlapping circles to avoid duplicates =====================
def merge_circles(circles, center_dist_px=MERGE_CENTER_DIST_PX, overlap_frac=MERGE_OVERLAP_FRAC):
    """
    Greedy NMS-style merge:
    - Sort by radius descending (keep larger ring as representative)
    - Discard any circle whose center is very close to a kept one
      or whose center distance is less than overlap_frac * min(radii)
    """
    if not circles: return []
    # sort larger first
    circles_sorted = sorted(circles, key=lambda c: c[2], reverse=True)
    kept = []
    for (cx,cy,r) in circles_sorted:
        dup = False
        for (kx,ky,kr) in kept:
            d = math.hypot(cx-kx, cy-ky)
            if d <= center_dist_px or d <= overlap_frac * min(r, kr):
                dup = True
                break
        if not dup:
            kept.append((cx,cy,r))
    return kept

# ==== Main ===================================================================
def main():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError("Camera open failed")

    if SHOW_PREVIEW:
        cv2.namedWindow("LED Robots", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("LED Robots", 1280, 720)

    tracks=[]
    last_t=time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.005); continue

        # ---- Preproc ----
        img = frame
        if USE_GRAY_WORLD_WB:
            img = gray_world(img)

        bgr_for_view, hsv = enhance_for_leds(img)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * SAT_GAIN, 0, 255)
        hsv[:,:,2] = np.clip(hsv[:,:,2] * VAL_GAIN, 0, 255)
        hsv = hsv.astype(np.uint8)
        bgr_for_view = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # ---- LED blobs ----
        pts, _ = detect_led_blobs(hsv)

        # ---- Clusters -> circles ----
        circles=[]
        if len(pts) > 0:
            cluster_indices = dbscan(pts, eps=DBSCAN_EPS, min_pts=DBSCAN_MIN_SAMPLES)
            for idxs in cluster_indices:
                if len(idxs) < 3: continue
                cl_pts = pts[idxs]

                circ, inliers = fit_circle_ransac(cl_pts, trials=64, inlier_thresh=INLIER_DIST_THRESH)
                if circ is None: continue
                cx, cy, r = circ

                refined = fit_circle_least_squares(cl_pts[inliers]) if inliers is not None else circ
                if refined is None: continue
                cx, cy, r = refined

                if r < MIN_RING_RADIUS_PX or r > MAX_RING_RADIUS_PX:
                    continue

                circles.append((cx,cy,r))

        # --- Merge duplicates before tracking ---
        circles = merge_circles(circles)

        # ---- Track robots across frames ----
        tracks = associate_tracks(tracks, circles)

        # ---- Per-robot sequence & ID ----
        vis = bgr_for_view.copy()
        hsv_for_sampling = cv2.cvtColor(bgr_for_view, cv2.COLOR_BGR2HSV)

        for tr in tracks:
            cx, cy, r = tr.cx, tr.cy, tr.r

            # 48 raw samples with classifier (C/M/Y/G only)
            raw48 = sample_sequence_symbols(hsv_for_sampling, cx, cy, r)

            # exact 16 symbols, no 'U'
            seq16_exact = samples48_to_exact16(raw48)
            display_seq = seq16_exact  # keep positional info

            # streak (for stable label)
            if tr.last_seq16 == display_seq:
                tr.streak16 += 1
            else:
                tr.last_seq16 = display_seq[:]
                tr.streak16 = 1

            # Optional: rotation-invariant match (window=6)
            if len(display_seq) >= 6:
                if hasattr(cb, 'match_id_rotation_window'):
                    best_idx, score = cb.match_id_rotation_window(display_seq, cb.CODEBOOK_NUM, window=6)
                else:
                    best_idx, score = match_id_rotation_window(display_seq, cb.CODEBOOK_NUM, window=6)
            else:
                best_idx, score = (None, 0)

            # ---- Draw overlays ----
            if SHOW_PREVIEW and DRAW_OVERLAYS:
                icx, icy, ir = int(round(cx)), int(round(cy)), int(round(r))
                cv2.circle(vis, (icx, icy), ir, (0,255,0), 2)
                cv2.circle(vis, (icx, icy), 2, (0,0,255), -1)
                cv2.circle(vis, (icx, icy), int(round(r*ANNULUS_INNER)), (255,255,0), 1)
                cv2.circle(vis, (icx, icy), int(round(r*ANNULUS_OUTER)), (255,255,0), 1)

                # optional tick marks
                for i in range(SAMPLES_AROUND):
                    th = (math.pi/2) - (2.0*math.pi*i/float(SAMPLES_AROUND))
                    px = int(round(cx + r*math.cos(th)))
                    py = int(round(cy - r*math.sin(th)))
                    cv2.circle(vis, (px, py), 1, (255,255,255), -1)

                live = ' '.join(display_seq) if display_seq else '-'
                label = f"{live}"
                if best_idx is not None:
                    label += f"   ID:{best_idx}  w6:{score}"
                cv2.putText(vis, label, (icx+12, icy-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)

        # ---- HUD ----
        if SHOW_PREVIEW:
            t=time.time(); dt=t-last_t; last_t=t
            hud = f"FPS~{int(1.0/dt) if dt>0 else '--'}  robots:{len(tracks)}  r∈[{MIN_RING_RADIUS_PX},{MAX_RING_RADIUS_PX}]px"
            cv2.putText(vis, hud, (10,22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1, cv2.LINE_AA)
            cv2.imshow("LED Robots", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
