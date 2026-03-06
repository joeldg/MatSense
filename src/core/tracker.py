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
        print("\n🔍 [PHASE 2a] ENGAGEMENT MATRIX: Isolating Absolute Foreground Athletes...")
        search_frames = min(int(fps * 15), len(raw_data))
        
        pair_stats = defaultdict(list)
        
        for f in range(search_frames):
            dets = raw_data.get(f, [])
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
                        
                        # 1. DEPTH HEURISTIC: Use MIN depth instead of MAX so BOTH feet must be close to camera
                        fg_y = min(b1[3], b2[3]) / h
                        
                        # 2. SIZE HEURISTIC: Prefer larger objects slightly, but prioritize centering
                        size_h = max(h1, h2) / h
                        
                        # 3. CENTER HEURISTIC: Heavily prioritize pairs interacting near the center of the frame
                        center = 1.0 - (abs(((cx1+cx2)/2) - w/2) / (w/2))
                        
                        # 4. ASPECT RATIO PENALTY: Referees are usually tall/skinny (<0.5 aspect). Grapplers are wider.
                        aspect1 = (b1[2]-b1[0]) / (h1 + 1e-5)
                        aspect2 = (b2[2]-b2[0]) / (h2 + 1e-5)
                        ref_penalty = -50.0 if (aspect1 < 0.6 or aspect2 < 0.6) else 0.0
                        
                        # 5. LEFT/RIGHT APPROACH: Players typically come from opposite sides
                        # Bonus when one is left-of-center and other is right-of-center
                        mid_x = w / 2.0
                        lr_bonus = 25.0 if (cx1 < mid_x and cx2 > mid_x) or (cx1 > mid_x and cx2 < mid_x) else 0.0
                        
                        score = (fg_y ** 3) * 75.0 + (size_h ** 2) * 15.0 + (center * 75.0) + ref_penalty + lr_bonus
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
        print(f"   🎯 Foreground Athletes Verified: IDs {id_A} & {id_B}")
        
        best_anchor = None
        lowest_overlap = float('inf')
        best_score = -float('inf')
        
        for f in range(search_frames):
            dets = raw_data.get(f, [])
            d1 = next((d for d in dets if d['id'] == id_A), None)
            d2 = next((d for d in dets if d['id'] == id_B), None)
            
            if d1 and d2:
                b1, b2 = d1['box'], d2['box']
                overlap_area = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0])) * max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
                
                fg_score = max(b1[3], b2[3]) / h
                score = fg_score + (1.0 - (abs(((b1[0]+b1[2]+b2[0]+b2[2])/4) - w/2) / (w/2)))
                
                if overlap_area == 0:
                    if lowest_overlap > 0: 
                        lowest_overlap = 0
                        best_score = score
                        best_anchor = {'f': f, 'p1': d1, 'p2': d2}
                    elif score > best_score: 
                        best_score = score
                        best_anchor = {'f': f, 'p1': d1, 'p2': d2}
                elif lowest_overlap > 0 and overlap_area < lowest_overlap:
                    lowest_overlap = overlap_area
                    best_score = score
                    best_anchor = {'f': f, 'p1': d1, 'p2': d2}
                    
        if not best_anchor:
            print("   ⚠️ No pure separation frame found. Defaulting to highest foreground state.")
            best_score = -float('inf')
            for f in range(search_frames):
                dets = raw_data.get(f, [])
                d1 = next((d for d in dets if d['id'] == id_A), None)
                d2 = next((d for d in dets if d['id'] == id_B), None)
                if d1 and d2:
                    b1, b2 = d1['box'], d2['box']
                    fg_score = max(b1[3], b2[3]) / h
                    if fg_score > best_score:
                        best_score = fg_score
                        best_anchor = {'f': f, 'p1': d1, 'p2': d2}
                        
        return best_anchor

    def build_global_blacklist(self, raw_data, anchor, w, h, fps):
        print("🔍 [PHASE 2b] BEHAVIORAL PRE-SCAN: Executing Spatial & Spectral Referee Matrix...")
        global_stats = defaultdict(lambda: {'boxes': [], 'between_frames': 0, 'interact_frames': 0, 'bw_scores': []})

        start_match_frames = anchor['f'] + int(fps * 8.0)
        early_frames_counted = 0

        for f, dets in raw_data.items():
            p1_cand, p2_cand = None, None
            for d in dets:
                tid = d['id']
                global_stats[tid]['boxes'].append(d['box'])
                if d.get('bw_score'): global_stats[tid]['bw_scores'].append(d['bw_score'])
                
                if tid == anchor['p1']['id']: p1_cand = d['box']
                if tid == anchor['p2']['id']: p2_cand = d['box']
                
            if p1_cand is not None and p2_cand is not None:
                c1x = (p1_cand[0] + p1_cand[2]) / 2.0
                c2x = (p2_cand[0] + p2_cand[2]) / 2.0
                
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
            
            if med_y < anchor_y - (anchor_h * 0.45):
                bg_ids.add(tid); continue
                
            is_upright = standing_pct > 0.60
            is_non_interactive = interaction_rate < 0.35
            
            # The Referee is usually standing, rarely interacts, and starts between the players
            if is_upright and is_non_interactive:
                ref_score = (between_ratio * 50.0) + (avg_bw * 50.0)
                ref_candidates.append({'id': tid, 'score': ref_score, 'bw': avg_bw, 'ratio': between_ratio})
            else:
                spec_ids.add(tid)

        if ref_candidates:
            # Rank strictly by the newly calculated ref_score
            ref_candidates.sort(key=lambda x: x['score'], reverse=True)
            true_coach_id = ref_candidates[0]['id']
            print(f"   👔 TRUE REF/COACH SECURED: ID {true_coach_id} (Sat-Score: {ref_candidates[0]['bw']:.2f}, Between-Ratio: {ref_candidates[0]['ratio']:.2f})")
            for cand in ref_candidates[1:]: spec_ids.add(cand['id'])
                
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
                
            k1 = p1_det.get('kpt')
            p1_prof.update({'box': p1_det['box'].copy(), 'id': p1_det['id'], 'kpt': k1.copy() if k1 is not None else None})
            
            k2 = p2_det.get('kpt')
            p2_prof.update({'box': p2_det['box'].copy(), 'id': p2_det['id'], 'kpt': k2.copy() if k2 is not None else None})
            
            if not melded and current_overlap < 0.15:
                if p1_det.get('hist') is not None and p1_prof.get('hist') is not None:
                    p1_prof['hist'] = cv2.addWeighted(p1_prof['hist'], 0.95, p1_det['hist'], 0.05, 0)
                if p2_det.get('hist') is not None and p2_prof.get('hist') is not None:
                    p2_prof['hist'] = cv2.addWeighted(p2_prof['hist'], 0.95, p2_det['hist'], 0.05, 0)
                
            curr_y = max(p1_prof['box'][3], p2_prof['box'][3])
            curr_h = max(p1_prof['box'][3]-p1_prof['box'][1], p2_prof['box'][3]-p2_prof['box'][1])
            y_ema = 0.9 * y_ema + 0.1 * curr_y
            h_ema = 0.9 * h_ema + 0.1 * curr_h

            res = {'p1': p1_det, 'p2': p2_det, 'melded': melded, 'bg': bg, 'ref': ref, 'spec': spec, 'unk': unk, 'z_horizon': y_ema}
            return res, p1_prof, p2_prof, y_ema, h_ema

        import copy
        p1_profile, p2_profile = copy.deepcopy(anchor['p1']), copy.deepcopy(anchor['p2'])
        p1_profile['base_hist'] = p1_profile.get('hist').copy() if p1_profile.get('hist') is not None else None
        p2_profile['base_hist'] = p2_profile.get('hist').copy() if p2_profile.get('hist') is not None else None
        p1_profile['pure_id'] = p1_profile['id']
        p2_profile['pure_id'] = p2_profile['id']
        
        action_y_ema = max(p1_profile['box'][3], p2_profile['box'][3])
        action_h_ema = max(p1_profile['box'][3]-p1_profile['box'][1], p2_profile['box'][3]-p2_profile['box'][1])

        timeline[anchor['f']] = {'p1': p1_profile.copy(), 'p2': p2_profile.copy(), 'melded': False, 'bg': [], 'ref': [], 'spec': [], 'unk': [], 'z_horizon': action_y_ema}

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
