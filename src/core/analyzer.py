import numpy as np
from scipy.signal import savgol_filter

def calculate_fast_kuzushi(kpts):
    """Fast 2D biomechanics projected directly onto the broadcast video"""
    try:
        # YOLO Indices: 11/12 (Hips), 5/6 (Shoulders), 15/16 (Ankles)
        pelvis = (kpts[11][:2] + kpts[12][:2]) / 2.0
        neck = (kpts[5][:2] + kpts[6][:2]) / 2.0
        r_ankle, l_ankle = kpts[16][:2], kpts[15][:2]
        
        # Center of Mass (Weighted 60% lower, 40% upper)
        com = (pelvis * 0.6) + (neck * 0.4)
        stance_width = np.linalg.norm(r_ankle - l_ankle)
        
        ab = l_ankle - r_ankle
        ap = com - r_ankle
        t = 0 if np.dot(ab, ab) == 0 else max(0.0, min(1.0, np.dot(ap, ab) / np.dot(ab, ab)))
        closest_point = r_ankle + t * ab
        distance_to_base = np.linalg.norm(com - closest_point)
        
        dynamic_threshold = 15.0 + (stance_width * 0.30)
        is_kuzushi = distance_to_base > dynamic_threshold
        
        return com, closest_point, distance_to_base, is_kuzushi
    except Exception:
        return None, None, 0, False

class MatchAnalyzer:
    def __init__(self, fps, height):
        self.fps = fps
        self.h = height

    def detect_events_from_timeline(self, timeline, total_frames):
        print("\n📊 [PHASE 3] EXECUTING KINEMATIC TRANSITION MATRIX...")
        raw_heights, raw_tops, max_ar_arr = [], [], []
        
        for f in range(total_frames):
            if f in timeline and not timeline[f]['melded'] and timeline[f]['p1'] and timeline[f]['p2']:
                b1, b2 = timeline[f]['p1']['box'], timeline[f]['p2']['box']
                raw_heights.append(max(b1[3], b2[3]) - min(b1[1], b2[1]))
                raw_tops.append(min(b1[1], b2[1]))
                w1, h1 = b1[2]-b1[0], b1[3]-b1[1]
                w2, h2 = b2[2]-b2[0], b2[3]-b2[1]
                max_ar_arr.append(max(w1/(h1+1e-5), w2/(h2+1e-5)))
            else:
                raw_heights.append(raw_heights[-1] if len(raw_heights) > 0 else self.h * 0.4)
                raw_tops.append(raw_tops[-1] if len(raw_tops) > 0 else self.h * 0.2)
                max_ar_arr.append(max_ar_arr[-1] if len(max_ar_arr) > 0 else 1.0)
                
        window = int(self.fps * 1.5) | 1
        smooth_heights = savgol_filter(raw_heights, window_length=window, polyorder=2) if total_frames > window else np.array(raw_heights)
        smooth_tops = savgol_filter(raw_tops, window_length=window, polyorder=2) if total_frames > window else np.array(raw_tops)
                   
        standing_h = np.percentile(smooth_heights, 85)
        print(f"   📏 Calibrated Standing Height: {int(standing_h)}px")
        
        impacts = []
        is_standing = True
        ground_frames = 0
        
        for f in range(total_frames):
            c_h = smooth_heights[f]
            max_ar = max_ar_arr[f]
            melded = timeline.get(f, {}).get('melded', False)
            
            if c_h < standing_h * 0.60 or max_ar > 1.35 or melded:
                ground_frames += 1
            else:
                if ground_frames < self.fps * 1.0: 
                    ground_frames = 0
                if c_h > standing_h * 0.8 and max_ar < 1.0 and not melded:
                    is_standing = True
                    
            if is_standing and ground_frames >= int(self.fps * 1.5):
                impact_frame = f - int(self.fps * 1.5)
                
                transition_f = impact_frame
                for fb in range(impact_frame, max(0, impact_frame - int(self.fps * 6.0)), -1):
                    if smooth_heights[fb] > standing_h * 0.8 and max_ar_arr[fb] < 1.0:
                        transition_f = fb
                        break
                        
                severity = smooth_tops[impact_frame] - smooth_tops[transition_f]
                impacts.append({
                    'impact_frame': impact_frame,
                    'transition_frame': transition_f,
                    'severity': severity
                })
                is_standing = False
                
        if impacts:
            best_event = max(impacts, key=lambda x: x['severity'])
            impact_f = best_event['impact_frame']
            trans_f = best_event['transition_frame']
            
            print(f"      💥 TAKEDOWN VERIFIED! Transition: {trans_f/self.fps:.1f}s -> Impact: {impact_f/self.fps:.1f}s")
            return [{
                "transition_frame": trans_f,
                "impact_frame": impact_f,
                "start_frame": max(0, int(trans_f - (8.0 * self.fps))), 
                "end_frame": min(total_frames - 1, int(impact_f + (6.0 * self.fps))),
                "severity": best_event['severity']
            }]
                
        print("   ⚠️ No definitive throw found. Defaulting to lowest altitude...")
        lowest_f = int(np.argmax(smooth_tops))
        return [{
            "transition_frame": lowest_f,
            "impact_frame": lowest_f, 
            "start_frame": max(0, int(lowest_f - (8.0 * self.fps))), 
            "end_frame": min(total_frames - 1, int(lowest_f + (6.0 * self.fps))),
            "severity": 0
        }]
