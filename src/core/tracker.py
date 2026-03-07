import cv2
import math
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from settings import DEVICE

def is_overlay_frame(frame, num_detections, rolling_det_median, h, w):
    """Detect leaderboard/scoreboard overlay frames useless for training.
    
    Multiple signals are checked — any one triggers a skip:
    1. Detection count anomaly vs rolling baseline  
    2. Low color variance in top/bottom strips (solid-color leaderboard bars)
    3. High horizontal edge density in strips (text-heavy leaderboards)
    4. Single dominant color covering center of frame (graphic overlays)
    """
    # 1. Detection anomaly — sudden loss of people in mid-video
    if rolling_det_median >= 2 and num_detections <= 0:
        return True
    
    # 2. Color variance in top/bottom strips (leaderboard zones)
    top_strip = frame[0:int(h * 0.18), :]
    bot_strip = frame[int(h * 0.82):, :]
    
    top_var = np.var(top_strip.astype(np.float32)) if top_strip.size > 0 else 999
    bot_var = np.var(bot_strip.astype(np.float32)) if bot_strip.size > 0 else 999
    
    # Low-variance bands = graphic overlay (raised threshold to catch more)
    if top_var < 1200 and bot_var < 1200:
        return True
    
    # 3. Horizontal edge density in strips — text creates strong Sobel-Y edges
    for strip in [top_strip, bot_strip]:
        if strip.size > 0:
            gray_strip = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
            sobel_y = cv2.Sobel(gray_strip, cv2.CV_64F, 0, 1, ksize=3)
            h_edge_density = np.mean(np.abs(sobel_y)) 
            # High horizontal edge density = text-heavy leaderboard
            if h_edge_density > 15.0 and num_detections <= 2:
                return True
    
    # 4. Dominant color check — if center is mostly one color, it's a graphic
    center = frame[int(h * 0.25):int(h * 0.75), int(w * 0.15):int(w * 0.85)]
    if center.size > 0:
        hsv_center = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
        # Check if >50% of pixels share a narrow hue range (within 20 bins)
        hue_hist = cv2.calcHist([hsv_center], [0], None, [18], [0, 180])
        total_px = hsv_center.shape[0] * hsv_center.shape[1]
        max_bin_pct = float(np.max(hue_hist)) / (total_px + 1e-5)
        if max_bin_pct > 0.55 and num_detections <= 1:
            return True
    
    # 5. Low edge complexity in center + few detections = title card
    if center.size > 0:
        gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / (edges.shape[0] * edges.shape[1] + 1e-5)
        if edge_density < 0.02 and num_detections <= 1:
            return True
    
    return False


def is_camera_blocked(detections, h, w):
    """Detect when someone is standing right in front of the camera, dominating the view.
    
    A single person occupying >45% of frame area or a person touching the bottom
    of the frame and taller than 55% of frame height is likely blocking the camera.
    """
    frame_area = h * w
    for det in detections:
        b = det['box']
        det_w = b[2] - b[0]
        det_h = b[3] - b[1]
        det_area = det_w * det_h
        
        # Single detection covers >45% of frame
        if det_area > frame_area * 0.45:
            return True
        
        # Person touching bottom of frame and very tall (standing right in front)
        if b[3] > h * 0.95 and det_h > h * 0.55:
            return True
    
    return False

def compute_z_depth_score(box, h):
    """Compute faux-3D Z-depth score from 2D bounding box.
    
    Z-Depth = normalized_area * foot_y_normalized³
    
    MATHEMATICAL RATIONALE:
    In broadcast video, camera perspective creates a natural depth cue:
    - Objects closer to the camera appear BOTH larger AND lower in frame
    - foot_y = bottom of bounding box = where feet touch the floor
    - The CUBIC exponent on foot_y aggressively weights proximity:
        y=0.9 (near camera) → 0.729, y=0.5 (mid-field) → 0.125 → 5.8x difference
    - Multiplying by area adds a second depth cue (larger = closer)
    - This 2D→3D projection is cheap but effective for filtering background matches
    
    Returns: float score in range [0, 1]. Higher = closer to camera.
    """
    bw = box[2] - box[0]
    bh = box[3] - box[1]
    area_norm = (bw * bh) / (h * h)  # Normalize by frame height² for scale invariance
    foot_y_norm = box[3] / h          # Bottom of box = foot position, normalized [0,1]
    return area_norm * (foot_y_norm ** 3)


def classify_ref_arm_signal(kpts):
    """Classify referee arm signals from COCO pose keypoints.
    
    COCO keypoints used:
      5,6 = left/right shoulder
      7,8 = left/right elbow
      9,10 = left/right wrist
      0 = nose (head reference)
    
    Detects arm postures for Judo, BJJ, and Wrestling.
    Returns: dict with 'signal' (str), 'confidence' (float), 'sport_signals' (dict)
             or None if no clear signal detected.
    
    ARM ANGLE GEOMETRY:
    - Angle measured between shoulder→wrist vector and horizontal
    - >70° from horizontal = arm raised high (IPPON territory)
    - 30-70° = arm at moderate angle (WAZA-ARI / ADVANTAGE territory) 
    - <30° = arm roughly horizontal (OSAEKOMI / CAUTION territory)
    - Both arms raised >70° simultaneously = strongest IPPON signal
    """
    if kpts is None or len(kpts) < 11:
        return None
    
    # Extract keypoints with confidence threshold
    CONF_MIN = 0.3
    nose = kpts[0] if kpts[0][2] > CONF_MIN else None
    l_shoulder = kpts[5] if kpts[5][2] > CONF_MIN else None
    r_shoulder = kpts[6] if kpts[6][2] > CONF_MIN else None
    l_elbow = kpts[7] if kpts[7][2] > CONF_MIN else None
    r_elbow = kpts[8] if kpts[8][2] > CONF_MIN else None
    l_wrist = kpts[9] if kpts[9][2] > CONF_MIN else None
    r_wrist = kpts[10] if kpts[10][2] > CONF_MIN else None
    
    # Need at least one full arm chain
    has_left = all(p is not None for p in [l_shoulder, l_elbow, l_wrist])
    has_right = all(p is not None for p in [r_shoulder, r_elbow, r_wrist])
    
    if not has_left and not has_right:
        return None
    
    def arm_angle_from_horizontal(shoulder, wrist):
        """Angle of shoulder→wrist vector from horizontal. 0°=horizontal, 90°=vertical up."""
        dx = wrist[0] - shoulder[0]
        dy = shoulder[1] - wrist[1]  # Inverted Y (screen coords: y increases downward)
        return math.degrees(math.atan2(dy, abs(dx) + 1e-5))
    
    def arm_extension_ratio(shoulder, elbow, wrist):
        """How straight is the arm? 1.0 = perfectly straight, 0.0 = fully bent."""
        full_len = math.hypot(wrist[0]-shoulder[0], wrist[1]-shoulder[1])
        seg_len = (math.hypot(elbow[0]-shoulder[0], elbow[1]-shoulder[1]) + 
                   math.hypot(wrist[0]-elbow[0], wrist[1]-elbow[1]))
        return full_len / (seg_len + 1e-5)
    
    l_angle = arm_angle_from_horizontal(l_shoulder, l_wrist) if has_left else None
    r_angle = arm_angle_from_horizontal(r_shoulder, r_wrist) if has_right else None
    l_ext = arm_extension_ratio(l_shoulder, l_elbow, l_wrist) if has_left else None
    r_ext = arm_extension_ratio(r_shoulder, r_elbow, r_wrist) if has_right else None
    
    # Check if wrists are above head (nose)
    l_above_head = (l_wrist is not None and nose is not None and l_wrist[1] < nose[1] - 20)
    r_above_head = (r_wrist is not None and nose is not None and r_wrist[1] < nose[1] - 20)
    
    # Check if arms are crossed (wrists cross over center)
    arms_crossed = False
    if l_wrist is not None and r_wrist is not None and l_shoulder is not None and r_shoulder is not None:
        shoulder_mid_x = (l_shoulder[0] + r_shoulder[0]) / 2.0
        l_side = l_wrist[0] - shoulder_mid_x
        r_side = r_wrist[0] - shoulder_mid_x
        arms_crossed = (l_side > 0 and r_side < 0)  # Left wrist on right side and vice versa

    # Classify signals across sports
    signal = None
    confidence = 0.0
    sport_signals = {}
    
    # ═══ BOTH ARMS RAISED HIGH (>70°) — strongest signal ═══
    both_high = ((l_angle is not None and l_angle > 70) and 
                 (r_angle is not None and r_angle > 70))
    one_high = ((l_angle is not None and l_angle > 70) or 
                (r_angle is not None and r_angle > 70))
    one_above = l_above_head or r_above_head
    both_above = l_above_head and r_above_head
    
    # Get the higher angle and its extension
    max_angle = max(l_angle or -90, r_angle or -90)
    max_ext = max(l_ext or 0, r_ext or 0)
    
    if both_high and both_above:
        # IPPON (Judo) — both arms straight up above head
        signal = "ARMS_UP"
        confidence = min(0.95, 0.7 + max_ext * 0.25)
        sport_signals = {
            'judo': ('IPPON', confidence),
            'bjj': ('SUBMISSION_SIGNAL', confidence * 0.6),
            'wrestling': ('PIN_CALLED', confidence * 0.7),
        }
    elif one_high and one_above and max_ext > 0.75:
        # One arm raised high — WAZA-ARI (Judo) or POINTS (BJJ/Wrestling)
        signal = "ARM_RAISED"
        confidence = min(0.85, 0.6 + max_ext * 0.2)
        sport_signals = {
            'judo': ('WAZA-ARI', confidence),
            'bjj': ('POINTS', confidence * 0.8),
            'wrestling': ('POINTS', confidence * 0.8),
        }
    elif max_angle > 30 and max_angle < 70 and max_ext > 0.6:
        # Arm at moderate angle — ADVANTAGE/MATTE
        signal = "ARM_ANGLED"
        confidence = min(0.7, 0.4 + max_ext * 0.2)
        sport_signals = {
            'judo': ('MATTE', confidence * 0.7),
            'bjj': ('ADVANTAGE', confidence * 0.8),
            'wrestling': ('CAUTION', confidence * 0.6),
        }
    elif max_angle >= -10 and max_angle <= 30 and max_ext > 0.7:
        # Arm roughly horizontal — OSAEKOMI/POINTS
        signal = "ARM_HORIZONTAL"
        confidence = min(0.65, 0.4 + max_ext * 0.15)
        sport_signals = {
            'judo': ('OSAEKOMI', confidence * 0.7),
            'bjj': ('POINTS', confidence * 0.5),
            'wrestling': ('CAUTION', confidence * 0.6),
        }
    elif arms_crossed:
        # Arms crossed — SUBMISSION (BJJ) / MATTE (Judo)
        signal = "ARMS_CROSSED"
        confidence = 0.6
        sport_signals = {
            'judo': ('MATTE', confidence * 0.8),
            'bjj': ('SUBMISSION', confidence * 0.7),
            'wrestling': ('STOP', confidence * 0.5),
        }
    
    if signal is None:
        return None
    
    return {
        'signal': signal,
        'confidence': confidence,
        'sport_signals': sport_signals,
        'l_angle': l_angle,
        'r_angle': r_angle,
    }

def bb_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

def get_color_hist(frame, box, w, h):
    x1, y1, x2, y2 = map(int, box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 10 or bh < 10: return None
    
    cx1, cx2 = int(x1 + bw*0.35), int(x1 + bw*0.65)
    cy1, cy2 = int(y1 + bh*0.20), int(y1 + bh*0.50)
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0: return None
    
    try:
        # Calculate HSV Histogram
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        
        # Calculate Black & White / Low Saturation Volume (Bins 0-3 on the Saturation Axis)
        # This completely ignores Hue and just looks for gray/black/white clothing
        bw_score = np.sum(hist[:, 0:3]) 
        
        return {'hist': hist, 'bw_score': bw_score}
    except Exception: return None

class MatchTracker:
    def __init__(self, model, use_mojo=False):
        self.model = model
        self.use_mojo = use_mojo
        self._mojo = None
        if use_mojo:
            try:
                from experiments.mojo_core.mojo_adapter import MojoAccelerator
                self._mojo = MojoAccelerator()
                if not self._mojo.available:
                    self._mojo = None
                    self.use_mojo = False
            except ImportError:
                self.use_mojo = False

    def extract_raw_data(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        raw_data = {}
        frame_idx = 0
        half_opt = (DEVICE != 'cpu')
        
        skip = 3
        hist_interval = 3
        processed_count = 0
        last_dets = []
        overlay_count = 0
        
        # Rolling detection count for overlay detection baseline
        recent_det_counts = []
        rolling_det_median = 0
        
        effective_fps = fps / skip
        print(f"\n🚀 [PHASE 1] BOT-SORT CENSUS: Mapping all entities (skip={skip}, ~{effective_fps:.0f} effective fps)...")
        print(f"   ⚡ Hardware Engine: {DEVICE.upper()} | imgsz=640 | processing ~{total_frames // skip} of {total_frames} frames")
        
        while cap.isOpened():
            if frame_idx % skip != 0:
                grabbed = cap.grab()
                if not grabbed:
                    break
                raw_data[frame_idx] = last_dets
                frame_idx += 1
                continue
            
            ret, frame = cap.read()
            if not ret: break
            
            results = self.model.track(
                frame, persist=True, tracker='botsort.yaml', 
                device=DEVICE, half=half_opt, imgsz=640, 
                verbose=False, conf=0.25, classes=[0]
            )
            
            dets = []
            compute_hist = (processed_count % hist_interval == 0)
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().copy()
                ids = results[0].boxes.id.cpu().numpy().astype(int).copy()
                kpts = results[0].keypoints.data.cpu().numpy().copy() if results[0].keypoints is not None else [None]*len(boxes)
                
                for b, tid, k in zip(boxes, ids, kpts):
                    if compute_hist:
                        color_data = get_color_hist(frame, b, w, h)
                        hist = color_data['hist'] if color_data else None
                        bw_score = color_data['bw_score'] if color_data else 0.0
                    else:
                        hist = None
                        bw_score = 0.0
                    dets.append({'box': b, 'id': tid, 'kpt': k, 'hist': hist, 'bw_score': bw_score})
            
            # Overlay frame detection — flag leaderboard/graphic frames
            if is_overlay_frame(frame, len(dets), rolling_det_median, h, w):
                raw_data[frame_idx] = []
                last_dets = []
                overlay_count += 1
            # Camera-blocker detection — person dominating the view
            elif len(dets) > 0 and is_camera_blocked(dets, h, w):
                raw_data[frame_idx] = []
                last_dets = []
                overlay_count += 1
            else:
                raw_data[frame_idx] = dets
                last_dets = dets
            
            # Update rolling detection baseline
            recent_det_counts.append(len(dets))
            if len(recent_det_counts) > 30:
                recent_det_counts.pop(0)
            rolling_det_median = np.median(recent_det_counts) if recent_det_counts else 0
            
            processed_count += 1
            frame_idx += 1
            
            if processed_count % max(1, int(effective_fps * 3)) == 0:
                print(f"   ... Processed {frame_idx}/{total_frames} frames ({(frame_idx/total_frames)*100:.1f}%)")
                
        cap.release()
        print(f"   ✅ Census complete: {processed_count} YOLO frames, {overlay_count} overlay frames filtered, {frame_idx - processed_count} skipped")
        return raw_data, total_frames, fps, w, h


    def find_foreground_anchor(self, raw_data, w, h, fps):
        """PHASE 2a: Find the two foreground athletes using Z-depth scoring and the Opening Bell Signature.
        
        THE OPENING BELL SIGNATURE:
        In competitive grappling, matches begin with a ritualized sequence:
        1. Athletes start on opposite sides (L/R of X-axis)
        2. They step toward each other (centroids converge)
        3. They bow/handshake/grip (bounding boxes intersect)
        
        We scan the first 15 seconds for this geometry. The last frame where they're
        still separated becomes the anchor (cleanest visual for histogram locking).
        """
        print("\n🔍 [PHASE 2a] ENGAGEMENT MATRIX: Isolating Absolute Foreground Athletes...")
        search_frames = min(int(fps * 15), len(raw_data))
        
        pair_stats = defaultdict(list)
        # Track per-entity Z-depth EMA for foreground supremacy filtering
        z_depth_ema = defaultdict(float)  # id -> smoothed Z-depth
        Z_ALPHA = 0.85  # EMA decay: α=0.85 means ~3s half-life at 30fps — smooths jitter, keeps fast transitions
        
        for f in range(search_frames):
            dets = raw_data.get(f, [])
            
            # Update per-entity Z-depth EMA
            for d in dets:
                z_score = compute_z_depth_score(d['box'], h)
                tid = d['id']
                if tid in z_depth_ema:
                    z_depth_ema[tid] = Z_ALPHA * z_depth_ema[tid] + (1.0 - Z_ALPHA) * z_score
                else:
                    z_depth_ema[tid] = z_score
            
            if len(dets) < 2: continue
            for i in range(len(dets)):
                for j in range(i+1, len(dets)):
                    b1, b2 = dets[i]['box'], dets[j]['box']
                    h1, h2 = b1[3]-b1[1], b2[3]-b2[1]
                    
                    if max(h1, h2) / (min(h1, h2) + 1e-5) > 1.8: continue 
                    score = 0.0
                    should_skip = False
                    if self.use_mojo and self._mojo:
                        score, should_skip = self._mojo.score_foreground_pair(b1, b2, w, h)
                    else:
                        if max(h1, h2) / (min(h1, h2) + 1e-5) > 1.8: continue 
                        
                        cx1, cy1 = (b1[0]+b1[2])/2, (b1[1]+b1[3])/2
                        cx2, cy2 = (b2[0]+b2[2])/2, (b2[1]+b2[3])/2
                        if math.hypot(cx1-cx2, cy1-cy2) > max(h1, h2) * 2.5: continue 
                        
                        overlap_x = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
                        overlap_y = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
                        
                        # Z-DEPTH HEURISTIC: Use the faux-3D Z-depth score
                        # Combined depth = minimum of both athletes' Z-depth EMA
                        # Both must be in the foreground to score high
                        z1 = z_depth_ema.get(dets[i]['id'], 0)
                        z2 = z_depth_ema.get(dets[j]['id'], 0)
                        depth_score = min(z1, z2)
                        
                        size_h = max(h1, h2) / h
                        center = 1.0 - (abs(((cx1+cx2)/2) - w/2) / (w/2))
                        
                        # ASPECT RATIO: Refs are tall/skinny (AR < 0.6). Grapplers are wider.
                        aspect1 = (b1[2]-b1[0]) / (h1 + 1e-5)
                        aspect2 = (b2[2]-b2[0]) / (h2 + 1e-5)
                        ref_penalty = -50.0 if (aspect1 < 0.6 or aspect2 < 0.6) else 0.0
                        
                        # LEFT/RIGHT APPROACH: Players typically come from opposite sides
                        mid_x = w / 2.0
                        lr_bonus = 25.0 if (cx1 < mid_x and cx2 > mid_x) or (cx1 > mid_x and cx2 < mid_x) else 0.0
                        
                        # Combined score: Z-depth replaces the old fg_y heuristic
                        # depth_score is in [0,1], cubed weighting already baked in
                        score = (depth_score * 150.0) + (size_h ** 2) * 15.0 + (center * 75.0) + ref_penalty + lr_bonus
                        if overlap_x * overlap_y > 0:
                            score += 15.0
                    
                    if should_skip:
                        continue
                    
                    pair_id = tuple(sorted([dets[i]['id'], dets[j]['id']]))
                    pair_stats[pair_id].append(score)
                        
        if not pair_stats: return None
        
        best_pair_id = None
        best_med_score = -float('inf')
        
        for pid, scores in pair_stats.items():
            if len(scores) < int(fps * 0.5): continue
            med_score = np.median(scores)
            if med_score > best_med_score:
                best_med_score = med_score
                best_pair_id = pid
                
        if not best_pair_id: 
            best_pair_id = max(pair_stats.keys(), key=lambda k: np.median(pair_stats[k]))
            
        id_A, id_B = best_pair_id
        
        # =====================================================================
        # OPENING BELL SIGNATURE: Detect approach→converge pattern
        # Scan for the sequence: separated → approaching → contact (bow/grip)
        # Lock Athlete_1 = leftmost starting position, Athlete_2 = rightmost
        # =====================================================================
        approach_frames = []  # (frame, d1, d2, centroid_dist, iou)
        for f in range(search_frames):
            dets = raw_data.get(f, [])
            d1 = next((d for d in dets if d['id'] == id_A), None)
            d2 = next((d for d in dets if d['id'] == id_B), None)
            if d1 and d2:
                b1, b2 = d1['box'], d2['box']
                cx1 = (b1[0] + b1[2]) / 2.0
                cx2 = (b2[0] + b2[2]) / 2.0
                dist = abs(cx1 - cx2)
                iou = bb_iou(b1, b2)
                approach_frames.append((f, d1, d2, dist, iou, cx1, cx2))
        
        # Look for the approach→converge pattern
        # Phase 1: Find frames where athletes are separated (dist > 30% frame width)
        # Phase 2: Distance decreases over subsequent frames
        # Phase 3: Boxes intersect (iou > 0.05) = bow/handshake/grip
        best_anchor = None
        best_anchor_score = -float('inf')
        converge_detected = False
        
        if len(approach_frames) >= int(fps * 1.0):  # Need at least 1 second of data
            # Find the convergence point (first sustained contact)
            for idx in range(len(approach_frames)):
                _, _, _, dist, iou, _, _ = approach_frames[idx]
                if iou > 0.05 and dist < w * 0.15:
                    # Found first contact — look backwards for the separated state
                    for back_idx in range(max(0, idx - int(fps * 5)), idx):
                        _, d1_back, d2_back, back_dist, back_iou, cx1_b, cx2_b = approach_frames[back_idx]
                        if back_dist > w * 0.20 and back_iou < 0.01:
                            # This is a clean separation frame — ideal anchor
                            b1_b, b2_b = d1_back['box'], d2_back['box']
                            fg_score = max(b1_b[3], b2_b[3]) / h
                            center_score = 1.0 - (abs(((b1_b[0]+b1_b[2]+b2_b[0]+b2_b[2])/4) - w/2) / (w/2))
                            score = fg_score + center_score
                            if score > best_anchor_score:
                                best_anchor_score = score
                                # Lock Athlete_1 = left, Athlete_2 = right
                                if cx1_b <= cx2_b:
                                    best_anchor = {'f': approach_frames[back_idx][0], 'p1': d1_back, 'p2': d2_back}
                                else:
                                    best_anchor = {'f': approach_frames[back_idx][0], 'p1': d2_back, 'p2': d1_back}
                                converge_detected = True
                    if converge_detected:
                        break
        
        if converge_detected:
            print(f"   🔔 OPENING BELL DETECTED: Approach→Converge at frame {best_anchor['f']}")
            print(f"   🎯 Athlete_1 (Left): ID {best_anchor['p1']['id']} | Athlete_2 (Right): ID {best_anchor['p2']['id']}")
        else:
            # Fallback: find the cleanest separation frame
            print(f"   🎯 Foreground Athletes Verified: IDs {id_A} & {id_B}")
            lowest_overlap = float('inf')
            
            for f, d1, d2, dist, iou, cx1, cx2 in approach_frames:
                b1, b2 = d1['box'], d2['box']
                overlap_area = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0])) * max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
                fg_score = max(b1[3], b2[3]) / h
                score = fg_score + (1.0 - (abs(((b1[0]+b1[2]+b2[0]+b2[2])/4) - w/2) / (w/2)))
                
                if overlap_area == 0:
                    if lowest_overlap > 0: 
                        lowest_overlap = 0
                        best_anchor_score = score
                        # Lock L/R by initial X position
                        if cx1 <= cx2:
                            best_anchor = {'f': f, 'p1': d1, 'p2': d2}
                        else:
                            best_anchor = {'f': f, 'p1': d2, 'p2': d1}
                    elif score > best_anchor_score: 
                        best_anchor_score = score
                        if cx1 <= cx2:
                            best_anchor = {'f': f, 'p1': d1, 'p2': d2}
                        else:
                            best_anchor = {'f': f, 'p1': d2, 'p2': d1}
                elif lowest_overlap > 0 and overlap_area < lowest_overlap:
                    lowest_overlap = overlap_area
                    best_anchor_score = score
                    if cx1 <= cx2:
                        best_anchor = {'f': f, 'p1': d1, 'p2': d2}
                    else:
                        best_anchor = {'f': f, 'p1': d2, 'p2': d1}
                    
        if not best_anchor:
            print("   ⚠️ No separation frame found. Defaulting to highest foreground state.")
            best_anchor_score = -float('inf')
            for data in approach_frames:
                f, d1, d2, dist, iou, cx1, cx2 = data
                b1, b2 = d1['box'], d2['box']
                fg_score = max(b1[3], b2[3]) / h
                if fg_score > best_anchor_score:
                    best_anchor_score = fg_score
                    if cx1 <= cx2:
                        best_anchor = {'f': f, 'p1': d1, 'p2': d2}
                    else:
                        best_anchor = {'f': f, 'p1': d2, 'p2': d1}
                        
        return best_anchor

    def build_global_blacklist(self, raw_data, anchor, w, h, fps):
        """SENTINEL REF CLASSIFIER: Behavioral profiling to classify and blacklist referees.
        
        Uses 5 independent signals to identify the referee:
        1. POSTURE (standing_pct): Refs maintain vertical AR (height > width) >60% of the time
        2. KINEMATICS (head_stability): Refs have stable head altitude (σ < 30px)
           Athletes experience 200-400px Y-drops during takedowns.
           WHY σ<30px: At 1080p, 30px ≈ 2.8% of frame height — normal head-bob while walking.
        3. ORBIT (orbit_radius): Refs maintain safe buffer distance from athletes' center-of-gravity
           WHY 1.5x: Refs must be close enough to intervene but far enough to not interfere.
        4. MAX ASPECT RATIO: If max AR across all frames < 0.85, the entity never goes horizontal = ref.
           Athletes go horizontal (AR > 1.2) during takedowns/groundwork.
        5. BETWEEN-RATIO + BW-SCORE: Legacy signals — refs start between players, wear dark clothing.
        """
        print("🔍 [PHASE 2b] BEHAVIORAL PRE-SCAN: Executing Spatial & Spectral Referee Matrix...")
        global_stats = defaultdict(lambda: {
            'boxes': [], 'between_frames': 0, 'interact_frames': 0, 
            'bw_scores': [], 'head_ys': [], 'kpts_available': False
        })

        start_match_frames = anchor['f'] + int(fps * 8.0)
        early_frames_counted = 0
        # Track athletes' combined center-of-gravity for orbit calculation
        athlete_cog_positions = []  # [(cog_x, cog_y), ...]

        for f, dets in raw_data.items():
            p1_cand, p2_cand = None, None
            for d in dets:
                tid = d['id']
                global_stats[tid]['boxes'].append(d['box'])
                if d.get('bw_score'): global_stats[tid]['bw_scores'].append(d['bw_score'])
                
                # HEAD-ALTITUDE: Extract head keypoint Y if available (keypoint 0 = nose)
                kpt = d.get('kpt')
                if kpt is not None and hasattr(kpt, '__len__') and len(kpt) > 0:
                    nose_y = float(kpt[0][1]) if kpt[0][2] > 0.3 else None  # Only if confidence > 0.3
                    if nose_y is not None:
                        global_stats[tid]['head_ys'].append(nose_y)
                        global_stats[tid]['kpts_available'] = True
                
                if tid == anchor['p1']['id']: p1_cand = d['box']
                if tid == anchor['p2']['id']: p2_cand = d['box']
                
            if p1_cand is not None and p2_cand is not None:
                c1x = (p1_cand[0] + p1_cand[2]) / 2.0
                c1y = (p1_cand[1] + p1_cand[3]) / 2.0
                c2x = (p2_cand[0] + p2_cand[2]) / 2.0
                c2y = (p2_cand[1] + p2_cand[3]) / 2.0
                
                athlete_cog_positions.append(((c1x + c2x) / 2.0, (c1y + c2y) / 2.0))
                
                left_bound = min(c1x, c2x) - (w * 0.10)
                right_bound = max(c1x, c2x) + (w * 0.10)
                
                if anchor['f'] <= f < start_match_frames:
                    early_frames_counted += 1
                
                for d in dets:
                    tid = d['id']
                    if tid in (anchor['p1']['id'], anchor['p2']['id']): continue
                    
                    b = d['box']
                    dcx = (b[0] + b[2]) / 2.0
                    
                    if anchor['f'] <= f < start_match_frames:
                        if left_bound <= dcx <= right_bound:
                            global_stats[tid]['between_frames'] += 1
                            
                    if bb_iou(b, p1_cand) > 0 or bb_iou(b, p2_cand) > 0:
                        global_stats[tid]['interact_frames'] += 1

        anchor_h, anchor_y = 0, 0
        for tid in (anchor['p1']['id'], anchor['p2']['id']):
            if tid in global_stats and global_stats[tid]['boxes']:
                boxes = np.array(global_stats[tid]['boxes'])
                anchor_h = max(anchor_h, np.percentile(boxes[:, 3] - boxes[:, 1], 80))
                anchor_y = max(anchor_y, np.median(boxes[:, 3]))

        # Pre-compute median athlete COG for orbit-radius
        med_cog_x, med_cog_y = w / 2.0, h / 2.0
        if athlete_cog_positions:
            cogs = np.array(athlete_cog_positions)
            med_cog_x, med_cog_y = np.median(cogs[:, 0]), np.median(cogs[:, 1])

        true_coach_id = None
        bg_ids, spec_ids = set(), set()
        ref_candidates = []
        
        for tid, stats in global_stats.items():
            if tid in (anchor['p1']['id'], anchor['p2']['id']): continue
            if len(stats['boxes']) == 0: continue
            
            boxes = np.array(stats['boxes'])
            heights = boxes[:, 3] - boxes[:, 1]
            aspect_ratios = (boxes[:, 2] - boxes[:, 0]) / (heights + 1e-5)
            
            med_y = np.median(boxes[:, 3])
            standing_pct = np.mean(aspect_ratios < 0.85)
            between_ratio = stats['between_frames'] / float(max(1, early_frames_counted))
            interaction_rate = stats['interact_frames'] / float(max(1, len(boxes)))
            avg_bw = np.median(stats['bw_scores']) if stats['bw_scores'] else 0.0
            
            # Z-DEPTH FILTER: entities consistently in far background → bg
            if med_y < anchor_y - (anchor_h * 0.45):
                bg_ids.add(tid); continue
            
            # ── NEW SIGNAL 1: HEAD-ALTITUDE STABILITY ──
            # Refs maintain a steady head height. Athletes drop 200-400px during takedowns.
            # σ < 30px at 1080p ≈ 2.8% of frame height = normal head-bob
            head_stability_score = 0.0
            if stats['head_ys'] and len(stats['head_ys']) > 10:
                head_std = np.std(stats['head_ys'])
                # Normalize to frame height for resolution independence
                head_std_norm = head_std / h
                if head_std_norm < 0.028:  # σ < 2.8% of frame height
                    head_stability_score = 30.0  # Strong ref signal
                elif head_std_norm < 0.05:
                    head_stability_score = 15.0  # Moderate ref signal
            
            # ── NEW SIGNAL 2: ORBIT-RADIUS ──
            # Refs maintain a safe buffer distance from athletes' combined center-of-gravity
            centroids_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
            centroids_y = (boxes[:, 1] + boxes[:, 3]) / 2.0
            dists_to_cog = np.sqrt((centroids_x - med_cog_x)**2 + (centroids_y - med_cog_y)**2)
            med_orbit = np.median(dists_to_cog)
            # Refs orbit at ~1.5-3x the athlete group radius
            orbit_score = 10.0 if (anchor_h * 0.8 < med_orbit < anchor_h * 3.5) else 0.0
            
            # ── NEW SIGNAL 3: MAX ASPECT RATIO ──
            # If the entity NEVER goes horizontal (max AR < 0.85), it's a strong ref signal
            # Athletes go horizontal (AR > 1.2) during takedowns/groundwork
            max_ar = np.max(aspect_ratios)
            max_ar_score = 15.0 if max_ar < 0.85 else 0.0
                
            is_upright = standing_pct > 0.60
            # Relaxed from 0.35 → 0.60: Refs DO interact with athletes (resets, matework, 
            # standing close during groundwork). The old 0.35 threshold was filtering real
            # refs into the spectator bucket. Refs interact less than 60% of the time.
            is_non_interactive = interaction_rate < 0.60
            
            # Combined ref scoring: all 5 signals
            if is_upright and is_non_interactive:
                ref_score = (
                    (between_ratio * 30.0) +   # Legacy: starts between players
                    (avg_bw * 30.0) +           # Legacy: dark clothing
                    head_stability_score +       # NEW: steady head altitude
                    orbit_score +                # NEW: stable orbit around action
                    max_ar_score                 # NEW: never goes horizontal
                )
                ref_candidates.append({
                    'id': tid, 'score': ref_score, 'bw': avg_bw, 
                    'ratio': between_ratio, 'head_σ': np.std(stats['head_ys']) if stats['head_ys'] else -1,
                    'orbit': med_orbit, 'max_ar': max_ar,
                    'standing_pct': standing_pct, 'interaction_rate': interaction_rate
                })
            else:
                spec_ids.add(tid)

        if ref_candidates:
            ref_candidates.sort(key=lambda x: x['score'], reverse=True)
            true_coach_id = ref_candidates[0]['id']
            rc = ref_candidates[0]
            print(f"   👔 IGNORE_REF SECURED: ID {true_coach_id} | BW:{rc['bw']:.2f} Between:{rc['ratio']:.2f} Head-σ:{rc['head_σ']:.1f}px Orbit:{rc['orbit']:.0f}px MaxAR:{rc['max_ar']:.2f}")
            for cand in ref_candidates[1:]: spec_ids.add(cand['id'])
        else:
            # FALLBACK: If no ref found with strict gates, pick the most upright
            # foreground entity that isn't an athlete. In a competition, there's
            # ALWAYS a ref — we just need to relax the gate.
            fallback_candidates = []
            for tid, stats in global_stats.items():
                if tid in (anchor['p1']['id'], anchor['p2']['id']): continue
                if tid in bg_ids: continue
                if len(stats['boxes']) < 5: continue
                
                boxes = np.array(stats['boxes'])
                heights = boxes[:, 3] - boxes[:, 1]
                aspect_ratios = (boxes[:, 2] - boxes[:, 0]) / (heights + 1e-5)
                sp = np.mean(aspect_ratios < 0.85)
                med_y = np.median(boxes[:, 3])
                
                # Must be upright >50% and in the foreground
                if sp > 0.50 and med_y > anchor_y - (anchor_h * 0.5):
                    fallback_candidates.append((tid, sp, med_y))
            
            if fallback_candidates:
                # Pick the most consistently upright one
                fallback_candidates.sort(key=lambda x: x[1], reverse=True)
                true_coach_id = fallback_candidates[0][0]
                print(f"   👔 IGNORE_REF (fallback): ID {true_coach_id} (upright {fallback_candidates[0][1]:.0%})")
                # Remove from spec_ids if it was there
                spec_ids.discard(true_coach_id)
            else:
                print("   ⚠️ WARNING: No referee detected in this segment")
                
        return true_coach_id, spec_ids, bg_ids

    def resolve_timeline(self, raw_data, total_frames, anchor, true_coach_id, spec_ids, bg_ids, w, h):
        print("\n🧠 [PHASE 2c] SCIPY TRACKING: Enforcing Immutable DNA & Anti-Swap Physics...")
        timeline = {}
        
        def step_tracker(f, prev_dets, current_dets, p1_prof, p2_prof, y_ema, h_ema):
            valid_cands, bg, ref, spec, unk = [], [], [], [], []
            melded = False
            
            if len(current_dets) == 0:
                p1_ret, p2_ret = p1_prof.copy(), p2_prof.copy()
                p1_ret['kpt'], p2_ret['kpt'] = None, None
                return {'p1': p1_ret, 'p2': p2_ret, 'melded': True, 'bg': [], 'ref': [], 'spec': [], 'unk': [], 'z_horizon': y_ema}, p1_prof, p2_prof, y_ema, h_ema

            shift_x, shift_y = 0.0, 0.0
            if prev_dets:
                prev_by_id = {d['id']: d for d in prev_dets}
                sx, sy = [], []
                for cd in current_dets:
                    pd = prev_by_id.get(cd['id'])
                    if pd:
                        sx.append(((cd['box'][0]+cd['box'][2])/2) - ((pd['box'][0]+pd['box'][2])/2))
                        sy.append(((cd['box'][1]+cd['box'][3])/2) - ((pd['box'][1]+pd['box'][3])/2))
                if sx: shift_x, shift_y = np.median(sx), np.median(sy)
                
            for prof in [p1_prof, p2_prof]:
                prof['box'][0] += shift_x; prof['box'][2] += shift_x
                prof['box'][1] += shift_y; prof['box'][3] += shift_y
            y_ema += shift_y
            
            action_cx = (p1_prof['box'][0] + p1_prof['box'][2] + p2_prof['box'][0] + p2_prof['box'][2]) / 4.0
            action_cy = (p1_prof['box'][1] + p1_prof['box'][3] + p2_prof['box'][1] + p2_prof['box'][3]) / 4.0

            for det in current_dets:
                tid, b = det['id'], det['box']
                if tid == true_coach_id: ref.append(det); continue
                if tid in spec_ids: spec.append(det); continue
                if tid in bg_ids: bg.append(det); continue
                
                dcx, dcy = (b[0]+b[2])/2, (b[1]+b[3])/2
                dist_to_action = math.hypot(dcx - action_cx, dcy - action_cy)
                
                if b[3] < y_ema - (h_ema * 0.35): bg.append(det); continue
                if dist_to_action > h_ema * 2.0 and tid not in (p1_prof['id'], p2_prof['id']) and tid not in (p1_prof['pure_id'], p2_prof['pure_id']):
                    spec.append(det); continue
                    
                valid_cands.append(det)

            targets = [p1_prof, p2_prof]
            if len(valid_cands) > 0:
                cost_matrix = np.zeros((2, len(valid_cands)))
                for i, target in enumerate(targets):
                    other = targets[1 - i]
                    for j, cand in enumerate(valid_cands):
                        cost = 0.0
                        
                        # ID match bonuses
                        if cand['id'] == target['id']: cost -= 200.0 
                        if cand['id'] == target.get('pure_id'): cost -= 300.0 
                        
                        tcx, tcy = (target['box'][0]+target['box'][2])/2, (target['box'][1]+target['box'][3])/2
                        ccx, ccy = (cand['box'][0]+cand['box'][2])/2, (cand['box'][1]+cand['box'][3])/2
                        dist = math.hypot(tcx-ccx, tcy-ccy)
                        
                        # ANTI-TELEPORT: massive penalty if jump > 30% of frame width
                        # People don't teleport between frames
                        if dist > w * 0.30: cost += 5000.0
                        elif dist > w * 0.25: cost += 2000.0 
                        cost += (dist / w) * 200.0
                        
                        # CROSS-ASSIGNMENT PENALTY: if candidate is closer to the OTHER
                        # profile than to this target, penalize to prevent swaps
                        ocx, ocy = (other['box'][0]+other['box'][2])/2, (other['box'][1]+other['box'][3])/2
                        dist_to_other = math.hypot(ocx-ccx, ocy-ccy)
                        if dist_to_other < dist * 0.5:
                            cost += 500.0  # Much closer to the other player = likely a swap
                        
                        iou = bb_iou(target['box'], cand['box'])
                        cost -= (iou * 200.0)
                        
                        if target.get('base_hist') is not None and cand.get('hist') is not None:
                            base_diff = cv2.compareHist(target['base_hist'], cand['hist'], cv2.HISTCMP_BHATTACHARYYA)
                            cost += (base_diff * 600.0) 
                            
                        if target.get('hist') is not None and cand.get('hist') is not None:
                            hist_diff = cv2.compareHist(target['hist'], cand['hist'], cv2.HISTCMP_BHATTACHARYYA)
                            cost += (hist_diff * 100.0)
                            
                        cost_matrix[i, j] = cost
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
            else:
                row_ind, col_ind, cost_matrix = [], [], np.zeros((2, 0))

            p1_det, p2_det = None, None
            used = set()
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 800.0:
                    if r == 0: p1_det = valid_cands[c]
                    else: p2_det = valid_cands[c]
                    used.add(c)
                    
            for j, cand in enumerate(valid_cands):
                if j not in used: unk.append(cand)
                
            if p1_det and not p2_det:
                melded = True
                p2_det = p1_det.copy(); p2_det['kpt'] = None; p2_det['id'] = p2_prof['id']
            elif p2_det and not p1_det:
                melded = True
                p1_det = p2_det.copy(); p1_det['kpt'] = None; p1_det['id'] = p1_prof['id']
            elif not p1_det and not p2_det:
                melded = True
                p1_det = p1_prof.copy(); p2_det = p2_prof.copy()
                p1_det['kpt'], p2_det['kpt'] = None, None

            current_overlap = 0.0
            if not melded and p1_det and p2_det:
                current_overlap = bb_iou(p1_det['box'], p2_det['box'])
            
            # ── THE ECLIPSE: Referee occlusion handling ──
            # If the referee walks in front (higher Z-depth than athletes), freeze profiles.
            # This prevents the ref's visual features from contaminating athlete profiles.
            ref_eclipse = False
            if ref:
                for r in ref:
                    ref_z = compute_z_depth_score(r['box'], h)
                    p1_z = compute_z_depth_score(p1_prof['box'], h) if p1_det else 0
                    p2_z = compute_z_depth_score(p2_prof['box'], h) if p2_det else 0
                    ref_iou_p1 = bb_iou(r['box'], p1_prof['box']) if p1_det else 0
                    ref_iou_p2 = bb_iou(r['box'], p2_prof['box']) if p2_det else 0
                    # Eclipse condition: ref closer to camera AND overlapping an athlete
                    # WHY IoU 0.3: Partial occlusion starts polluting features at 30%
                    if ref_z > max(p1_z, p2_z) and (ref_iou_p1 > 0.3 or ref_iou_p2 > 0.3):
                        ref_eclipse = True
                        break
                
            k1 = p1_det.get('kpt')
            k2 = p2_det.get('kpt')
            
            # ── THE PRETZEL: Heavy overlap → freeze visual features ──
            # WHY IoU > 0.60: At 60% overlap, bounding boxes are essentially one blob.
            # Visual features (histograms) from torso crops will be contaminated by the
            # other athlete's gi/skin. Freezing prevents color-bleed in profile memory.
            # ALSO freeze during ref eclipse to prevent ref's features from bleeding in.
            pretzel_state = current_overlap > 0.60
            
            if not ref_eclipse:  # Only update box/id if ref isn't eclipsing
                p1_prof.update({'box': p1_det['box'].copy(), 'id': p1_det['id'], 'kpt': k1.copy() if k1 is not None else None})
                p2_prof.update({'box': p2_det['box'].copy(), 'id': p2_det['id'], 'kpt': k2.copy() if k2 is not None else None})
            else:
                # Eclipse: hold position trajectories, don't update IDs
                # Only update box position (for tracking continuity) but NOT histogram/color
                p1_prof['kpt'] = k1.copy() if k1 is not None else p1_prof.get('kpt')
                p2_prof['kpt'] = k2.copy() if k2 is not None else p2_prof.get('kpt')
            
            # Histogram update: ONLY when NOT in pretzel AND NOT in eclipse
            # Old threshold was IoU < 0.15 — far too aggressive, froze updates during normal grappling
            if not melded and not pretzel_state and not ref_eclipse:
                if p1_det.get('hist') is not None and p1_prof.get('hist') is not None:
                    p1_prof['hist'] = cv2.addWeighted(p1_prof['hist'], 0.95, p1_det['hist'], 0.05, 0)
                if p2_det.get('hist') is not None and p2_prof.get('hist') is not None:
                    p2_prof['hist'] = cv2.addWeighted(p2_prof['hist'], 0.95, p2_det['hist'], 0.05, 0)
                
            curr_y = max(p1_prof['box'][3], p2_prof['box'][3])
            curr_h = max(p1_prof['box'][3]-p1_prof['box'][1], p2_prof['box'][3]-p2_prof['box'][1])
            y_ema = 0.9 * y_ema + 0.1 * curr_y
            h_ema = 0.9 * h_ema + 0.1 * curr_h

            res = {
                'p1': p1_det, 'p2': p2_det, 'melded': melded,
                'bg': bg, 'ref': ref, 'spec': spec, 'unk': unk,
                'z_horizon': y_ema, 'ref_signal': None
            }
            # Classify ref arm signal if ref detected with keypoints
            if ref:
                for r in ref:
                    r_kpt = r.get('kpt')
                    if r_kpt is not None:
                        sig = classify_ref_arm_signal(r_kpt)
                        if sig and sig['confidence'] > 0.5:
                            res['ref_signal'] = sig
                            break
            return res, p1_prof, p2_prof, y_ema, h_ema

        import copy
        p1_profile, p2_profile = copy.deepcopy(anchor['p1']), copy.deepcopy(anchor['p2'])
        p1_profile['base_hist'] = p1_profile.get('hist').copy() if p1_profile.get('hist') is not None else None
        p2_profile['base_hist'] = p2_profile.get('hist').copy() if p2_profile.get('hist') is not None else None
        p1_profile['pure_id'] = p1_profile['id']
        p2_profile['pure_id'] = p2_profile['id']
        
        action_y_ema = max(p1_profile['box'][3], p2_profile['box'][3])
        action_h_ema = max(p1_profile['box'][3]-p1_profile['box'][1], p2_profile['box'][3]-p2_profile['box'][1])

        timeline[anchor['f']] = {'p1': p1_profile.copy(), 'p2': p2_profile.copy(), 'melded': False, 'bg': [], 'ref': [], 'spec': [], 'unk': [], 'z_horizon': action_y_ema, 'ref_signal': None}

        p1_f, p2_f, y_f, h_f = copy.deepcopy(p1_profile), copy.deepcopy(p2_profile), action_y_ema, action_h_ema
        for f in range(anchor['f'] + 1, total_frames):
            cd = raw_data.get(f, [])
            pd = raw_data.get(f-1, [])
            res, p1_f, p2_f, y_f, h_f = step_tracker(f, pd, cd, p1_f, p2_f, y_f, h_f)
            timeline[f] = res

        p1_b, p2_b, y_b, h_b = copy.deepcopy(p1_profile), copy.deepcopy(p2_profile), action_y_ema, action_h_ema
        for f in range(anchor['f'] - 1, -1, -1):
            cd = raw_data.get(f, [])
            nd = raw_data.get(f+1, []) 
            res, p1_b, p2_b, y_b, h_b = step_tracker(f, nd, cd, p1_b, p2_b, y_b, h_b)
            timeline[f] = res
            
        return timeline
