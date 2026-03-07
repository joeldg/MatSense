import math
import numpy as np
from scipy.signal import savgol_filter


def calculate_fast_kuzushi(kpts):
    """Fast 2D biomechanics projected directly onto the broadcast video.
    
    Computes Center-of-Mass displacement from the support base (stance line).
    
    Returns: (com, closest_point, distance_to_base, is_kuzushi, direction)
        direction: 'forward' | 'backward' | 'left' | 'right' | 'forward_left' etc.
    """
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
        ab_dot = np.dot(ab, ab)
        t = 0 if ab_dot == 0 else max(0.0, min(1.0, np.dot(ap, ab) / ab_dot))
        closest_point = r_ankle + t * ab
        diff = com - closest_point
        distance_sq = np.dot(diff, diff)
        
        dynamic_threshold = 15.0 + (stance_width * 0.30)
        is_kuzushi = distance_sq > dynamic_threshold * dynamic_threshold
        distance_to_base = np.sqrt(distance_sq)
        
        # Determine displacement direction
        direction = _compute_kuzushi_direction(diff)
        
        return com, closest_point, distance_to_base, is_kuzushi, direction
    except Exception:
        return None, None, 0, False, None


def _compute_kuzushi_direction(diff_vector):
    """Compute the kuzushi direction from the COM displacement vector.
    
    In screen coordinates:
    - Positive X = rightward  
    - Positive Y = downward (into camera = forward in mat-space)
    
    Returns: string like 'forward', 'forward_left', 'backward_right', etc.
    """
    dx, dy = diff_vector[0], diff_vector[1]
    if abs(dx) < 3 and abs(dy) < 3:
        return None  # No significant displacement
    
    parts = []
    if dy > 5:
        parts.append('forward')   # Falling toward camera
    elif dy < -5:
        parts.append('backward')  # Falling away from camera
    
    if dx < -5:
        parts.append('left')
    elif dx > 5:
        parts.append('right')
    
    return '_'.join(parts) if parts else None


# ==============================================================================
# MATCH PHASE DEFINITIONS
# ==============================================================================
PHASE_STANDING = "STANDING"
PHASE_GRIP_FIGHT = "GRIP_FIGHT"
PHASE_TRANSITION = "TRANSITION" 
PHASE_GROUND = "GROUND"
PHASE_RESET = "RESET"  # Athletes standing back up / being reset by ref

# Event types
EVENT_THROW = "THROW"
EVENT_TAKEDOWN = "TAKEDOWN"
EVENT_GUARD_PULL = "GUARD_PULL"
EVENT_SWEEP = "SWEEP"


class MatchAnalyzer:
    def __init__(self, fps, height, use_mojo=False):
        self.fps = fps
        self.h = height
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

    def detect_events_from_timeline(self, timeline, total_frames):
        """Multi-stage event detection pipeline.
        
        Stage 1: Per-frame feature extraction
        Stage 2: Phase segmentation (Standing/GripFight/Transition/Ground)
        Stage 3: Transition classification (Throw/Takedown/GuardPull/Sweep)
        Stage 4: Kuzushi analysis
        Stage 5: Ref signal correlation
        """
        print("\n📊 [PHASE 3] EXECUTING KINEMATIC EVENT CLASSIFIER...")
        
        # ═══ STAGE 1: Per-frame feature extraction ═══
        features = self._extract_per_frame_features(timeline, total_frames)
        if not features:
            print("   ⚠️ Insufficient data for event detection.")
            return []
        
        # ═══ STAGE 2: Phase segmentation ═══
        phases = self._segment_phases(features, total_frames)
        
        # ═══ STAGE 3: Find and classify transitions ═══
        events = self._classify_transitions(features, phases, timeline, total_frames)
        
        if events:
            print(f"   📊 Total events detected: {len(events)}")
        else:
            print("   ⚠️ No definitive events found in this window.")
        
        return events

    def _extract_per_frame_features(self, timeline, total_frames):
        """Extract per-frame kinematic features from the resolved timeline.
        
        Returns dict of arrays: heights, tops, max_ar, overlap_iou, 
        kuzushi_dist, kuzushi_dir, velocity_y, ref_signals
        """
        raw_heights = []
        raw_tops = []
        max_ar_arr = []
        overlap_iou = []
        kuzushi_dists = []
        kuzushi_dirs = []
        ref_signals = []
        
        # Per-athlete features for asymmetric detection (guard pull = one side drops)
        p1_bottoms = []  # Athlete 1 foot Y
        p2_bottoms = []  # Athlete 2 foot Y
        p1_ars = []      # Athlete 1 aspect ratio
        p2_ars = []      # Athlete 2 aspect ratio
        
        for f in range(total_frames):
            data = timeline.get(f)
            if data and not data.get('melded', True) and data.get('p1') and data.get('p2'):
                b1, b2 = data['p1']['box'], data['p2']['box']
                k1 = data['p1'].get('kpt')
                
                combined_h = max(b1[3], b2[3]) - min(b1[1], b2[1])
                raw_heights.append(combined_h)
                raw_tops.append(min(b1[1], b2[1]))
                
                w1, h1 = b1[2]-b1[0], b1[3]-b1[1]
                w2, h2 = b2[2]-b2[0], b2[3]-b2[1]
                max_ar_arr.append(max(w1/(h1+1e-5), w2/(h2+1e-5)))
                
                p1_ars.append(w1/(h1+1e-5))
                p2_ars.append(w2/(h2+1e-5))
                p1_bottoms.append(b1[3])
                p2_bottoms.append(b2[3])
                
                # Bounding box overlap (IoU)
                from src.core.tracker import bb_iou
                iou = bb_iou(b1, b2)
                overlap_iou.append(iou)
                
                # Kuzushi from keypoints
                if k1 is not None and len(k1) >= 17:
                    _, _, dist, is_kuz, direction = calculate_fast_kuzushi(k1)
                    kuzushi_dists.append(dist if is_kuz else 0.0)
                    kuzushi_dirs.append(direction if is_kuz else None)
                else:
                    kuzushi_dists.append(0.0)
                    kuzushi_dirs.append(None)
                
                # Ref signal
                sig = data.get('ref_signal')
                ref_signals.append(sig)
            else:
                # Carry forward or default
                raw_heights.append(raw_heights[-1] if raw_heights else self.h * 0.4)
                raw_tops.append(raw_tops[-1] if raw_tops else self.h * 0.2)
                max_ar_arr.append(max_ar_arr[-1] if max_ar_arr else 1.0)
                overlap_iou.append(overlap_iou[-1] if overlap_iou else 0.0)
                kuzushi_dists.append(0.0)
                kuzushi_dirs.append(None)
                ref_signals.append(None)
                p1_bottoms.append(p1_bottoms[-1] if p1_bottoms else self.h * 0.5)
                p2_bottoms.append(p2_bottoms[-1] if p2_bottoms else self.h * 0.5)
                p1_ars.append(p1_ars[-1] if p1_ars else 0.5)
                p2_ars.append(p2_ars[-1] if p2_ars else 0.5)
        
        if not raw_heights:
            return None
        
        # Smooth signals
        window = max(5, int(self.fps * 1.5) | 1)
        if total_frames > window:
            smooth_heights = savgol_filter(raw_heights, window_length=window, polyorder=2)
            smooth_tops = savgol_filter(raw_tops, window_length=window, polyorder=2)
        else:
            smooth_heights = np.array(raw_heights)
            smooth_tops = np.array(raw_tops)
        
        # Velocity (Y-axis rate of change per frame)
        velocity_y = np.gradient(smooth_tops)
        
        standing_h = np.percentile(smooth_heights, 85)
        print(f"   📏 Calibrated Standing Height: {int(standing_h)}px")
        
        return {
            'heights': smooth_heights,
            'raw_heights': np.array(raw_heights),
            'tops': smooth_tops,
            'max_ar': np.array(max_ar_arr),
            'overlap_iou': np.array(overlap_iou),
            'kuzushi_dists': np.array(kuzushi_dists),
            'kuzushi_dirs': kuzushi_dirs,
            'ref_signals': ref_signals,
            'velocity_y': velocity_y,
            'standing_h': standing_h,
            'p1_bottoms': np.array(p1_bottoms),
            'p2_bottoms': np.array(p2_bottoms),
            'p1_ars': np.array(p1_ars),
            'p2_ars': np.array(p2_ars),
        }

    def _segment_phases(self, features, total_frames):
        """Segment the match timeline into phases: STANDING, GRIP_FIGHT, TRANSITION, GROUND.
        
        STANDING: Both athletes upright, not entangled
        GRIP_FIGHT: Both upright + high overlap → engaged but no state change
        TRANSITION: Height actively dropping (rate of descent)
        GROUND: Sustained low posture / horizontal aspect ratios
        """
        standing_h = features['standing_h']
        phases = []
        
        for f in range(total_frames):
            h = features['heights'][f]
            ar = features['max_ar'][f]
            iou = features['overlap_iou'][f]
            vel = features['velocity_y'][f]
            
            if h < standing_h * 0.55 or ar > 1.35:
                phases.append(PHASE_GROUND)
            elif vel > 3.0 and h < standing_h * 0.80:
                # Active descent = transition happening RIGHT NOW
                phases.append(PHASE_TRANSITION)
            elif h > standing_h * 0.75 and ar < 1.0:
                if iou > 0.25:
                    phases.append(PHASE_GRIP_FIGHT)
                else:
                    phases.append(PHASE_STANDING)
            elif h > standing_h * 0.80 and ar < 0.95:
                phases.append(PHASE_STANDING)
            else:
                # Intermediate — check if coming from ground (reset) or going to ground (transition)
                if phases and phases[-1] == PHASE_GROUND and vel < -1.0:
                    phases.append(PHASE_RESET)
                elif phases and phases[-1] in (PHASE_STANDING, PHASE_GRIP_FIGHT):
                    phases.append(PHASE_TRANSITION)
                else:
                    phases.append(phases[-1] if phases else PHASE_STANDING)
        
        return phases

    def _classify_transitions(self, features, phases, timeline, total_frames):
        """Find STANDING→GROUND transitions and classify them.
        
        For each transition:
        1. Was kuzushi detected? (throw indicator)
        2. How fast was the descent? (throw = fast, takedown = moderate, guard_pull = slow)
        3. Was it asymmetric? (guard pull = only one athlete drops)
        4. Was there a ref signal nearby? (points confirmation)
        """
        standing_h = features['standing_h']
        events = []
        
        # Find sustained ground segments (>1.5s)
        min_ground_frames = int(self.fps * 1.5)
        ground_start = None
        ground_count = 0
        was_standing = True
        
        for f in range(total_frames):
            if phases[f] == PHASE_GROUND:
                if ground_start is None:
                    ground_start = f
                ground_count += 1
            else:
                if ground_count >= min_ground_frames and was_standing:
                    # We have a STANDING → GROUND transition
                    impact_frame = ground_start
                    
                    # Find the transition start (look back for last standing frame)
                    transition_frame = impact_frame
                    for fb in range(impact_frame, max(0, impact_frame - int(self.fps * 6.0)), -1):
                        if phases[fb] in (PHASE_STANDING, PHASE_GRIP_FIGHT):
                            transition_frame = fb
                            break
                    
                    event = self._classify_single_transition(
                        features, phases, timeline,
                        transition_frame, impact_frame, total_frames
                    )
                    if event:
                        events.append(event)
                
                ground_count = 0
                ground_start = None
                
                if phases[f] in (PHASE_STANDING, PHASE_GRIP_FIGHT, PHASE_RESET):
                    was_standing = True
        
        # Check final ground segment
        if ground_count >= min_ground_frames and was_standing and ground_start is not None:
            impact_frame = ground_start
            transition_frame = impact_frame
            for fb in range(impact_frame, max(0, impact_frame - int(self.fps * 6.0)), -1):
                if phases[fb] in (PHASE_STANDING, PHASE_GRIP_FIGHT):
                    transition_frame = fb
                    break
            event = self._classify_single_transition(
                features, phases, timeline,
                transition_frame, impact_frame, total_frames
            )
            if event:
                events.append(event)
        
        # Deduplicate nearby events (within 3s)
        if len(events) > 1:
            events.sort(key=lambda x: x['impact_frame'])
            deduped = [events[0]]
            for ev in events[1:]:
                if ev['impact_frame'] - deduped[-1]['impact_frame'] > self.fps * 3:
                    deduped.append(ev)
                elif ev['severity'] > deduped[-1]['severity']:
                    deduped[-1] = ev
            events = deduped
        
        return events

    def _classify_single_transition(self, features, phases, timeline,
                                     transition_frame, impact_frame, total_frames):
        """Classify a single STANDING→GROUND transition.
        
        PHYSICS SIGNATURES:
        
        THROW: Kuzushi detected + rapid descent (>200px/s) + descent <1.5s
          - COM leaves support base BEFORE the fall
          - Shoulder rotation often present
          - Both athletes change elevation simultaneously
        
        TAKEDOWN: Moderate descent (1-3s) + horizontal convergence + no kuzushi
          - One athlete shoots/penetrates (their center drops while opponent stays high initially)
          - Lower velocity than throw
        
        GUARD PULL: One athlete voluntarily descends + no kuzushi + no impact force
          - Only ONE athlete's AR changes (goes horizontal)
          - Other stays upright initially
          - Very low vertical velocity
        
        SWEEP: Already on ground + position reversal (detected elsewhere)
        """
        trans_window = slice(transition_frame, impact_frame + 1)
        window_len = impact_frame - transition_frame
        
        if window_len < 2:
            return None
        
        # ── Extract window features ──
        max_kuzushi = np.max(features['kuzushi_dists'][trans_window])
        kuz_frames = features['kuzushi_dists'][trans_window] > 0
        kuz_count = np.sum(kuz_frames)
        
        # Find dominant kuzushi direction in the window
        kuz_dirs = [d for d in features['kuzushi_dirs'][transition_frame:impact_frame+1] if d is not None]
        kuz_direction = max(set(kuz_dirs), key=kuz_dirs.count) if kuz_dirs else None
        
        # Descent velocity (pixels per frame → pixels per second)
        descent = features['tops'][impact_frame] - features['tops'][transition_frame]
        descent_duration_sec = window_len / self.fps
        descent_velocity = descent / max(0.01, descent_duration_sec)  # px/s
        
        # Height drop ratio
        height_drop = 1.0 - (features['heights'][impact_frame] / features['standing_h'])
        
        # Asymmetry: did only ONE athlete go down?
        p1_ar_change = features['p1_ars'][impact_frame] - features['p1_ars'][transition_frame]
        p2_ar_change = features['p2_ars'][impact_frame] - features['p2_ars'][transition_frame]
        asymmetric = abs(p1_ar_change - p2_ar_change) > 0.4
        one_went_horizontal = (features['p1_ars'][impact_frame] > 1.2) != (features['p2_ars'][impact_frame] > 1.2)
        
        # ── CLASSIFY ──
        has_kuzushi = max_kuzushi > 20.0 and kuz_count >= 2
        fast_descent = descent_velocity > 150.0  # px/s
        moderate_descent = 50.0 < descent_velocity <= 150.0
        slow_descent = descent_velocity <= 50.0
        
        event_type = EVENT_TAKEDOWN  # Default
        confidence = 0.5
        mechanism = None
        
        if has_kuzushi and fast_descent and descent_duration_sec < 2.0:
            # THROW: kuzushi + fast + short duration
            event_type = EVENT_THROW
            confidence = min(0.95, 0.6 + (max_kuzushi / 100.0) + (0.1 if descent_duration_sec < 1.0 else 0))
            mechanism = _infer_throw_mechanism(kuz_direction, descent_velocity, height_drop)
            print(f"      🥋 THROW detected! Kuzushi: {max_kuzushi:.1f}px {kuz_direction} | "
                  f"Descent: {descent_velocity:.0f}px/s in {descent_duration_sec:.1f}s | "
                  f"Mechanism: {mechanism}")
        
        elif asymmetric and one_went_horizontal and slow_descent and not has_kuzushi:
            # GUARD PULL: one athlete goes horizontal voluntarily, no kuzushi
            event_type = EVENT_GUARD_PULL
            confidence = min(0.80, 0.5 + (0.2 if not has_kuzushi else 0))
            print(f"      🛡️ GUARD PULL detected! Asymmetric descent, no kuzushi | "
                  f"Descent: {descent_velocity:.0f}px/s in {descent_duration_sec:.1f}s")
        
        elif moderate_descent and descent_duration_sec > 0.8:
            # TAKEDOWN: moderate speed, driven to ground
            event_type = EVENT_TAKEDOWN
            confidence = min(0.85, 0.5 + (descent_velocity / 300.0))
            print(f"      💥 TAKEDOWN detected! Descent: {descent_velocity:.0f}px/s in {descent_duration_sec:.1f}s | "
                  f"Kuzushi: {'yes' if has_kuzushi else 'no'}")
        
        elif has_kuzushi and moderate_descent:
            # THROW (but slower — maybe a trip or foot sweep)
            event_type = EVENT_THROW
            confidence = min(0.75, 0.5 + (max_kuzushi / 150.0))
            mechanism = _infer_throw_mechanism(kuz_direction, descent_velocity, height_drop)
            print(f"      🥋 THROW (trip/sweep) detected! Kuzushi: {max_kuzushi:.1f}px {kuz_direction} | "
                  f"Mechanism: {mechanism}")
        
        else:
            # Generic takedown
            event_type = EVENT_TAKEDOWN
            confidence = 0.5
            print(f"      💥 TRANSITION detected! Descent: {descent_velocity:.0f}px/s in {descent_duration_sec:.1f}s")
        
        # ── REF SIGNAL CORRELATION ──
        ref_signal_event = None
        search_start = max(0, impact_frame - int(self.fps * 1))
        search_end = min(total_frames, impact_frame + int(self.fps * 4))
        
        for f in range(search_start, search_end):
            sig = features['ref_signals'][f]
            if sig and sig.get('confidence', 0) > 0.5:
                ref_signal_event = {
                    'signal': sig['signal'],
                    'sport_signals': sig.get('sport_signals', {}),
                    'frame': f,
                    'confidence': sig['confidence']
                }
                # Boost confidence when ref confirms the event
                confidence = min(0.98, confidence + 0.10)
                
                # IPPON detection
                if sig['signal'] == 'ARMS_UP' and event_type == EVENT_THROW:
                    confidence = 0.95
                    print(f"      🏁 REF SIGNAL: ARMS_UP → Possible IPPON!")
                elif sig['signal'] == 'ARM_RAISED':
                    print(f"      🏁 REF SIGNAL: ARM_RAISED → Points scored!")
                break
        
        severity = features['tops'][impact_frame] - features['tops'][transition_frame]
        
        # Build phase timeline for this event
        phase_tl = []
        for f in range(transition_frame, min(impact_frame + int(self.fps * 3), total_frames)):
            p = phases[f]
            if not phase_tl or p != phase_tl[-1]:
                phase_tl.append(p)
        
        return {
            "type": event_type,
            "confidence": round(confidence, 2),
            "transition_frame": transition_frame,
            "impact_frame": impact_frame,
            "start_frame": max(0, int(transition_frame - (8.0 * self.fps))),
            "end_frame": min(total_frames - 1, int(impact_frame + (9.0 * self.fps))),
            "severity": severity,
            "kuzushi": {
                "direction": kuz_direction,
                "severity_px": round(float(max_kuzushi), 1),
                "kuzushi_frames": int(kuz_count),
                "mechanism": mechanism,
            },
            "descent": {
                "velocity_px_s": round(float(descent_velocity), 1),
                "duration_sec": round(float(descent_duration_sec), 2),
                "height_drop_pct": round(float(height_drop * 100), 1),
            },
            "ref_signal": ref_signal_event,
            "phase_timeline": phase_tl,
        }


def _infer_throw_mechanism(kuz_direction, velocity, height_drop):
    """Infer the throw mechanism from kuzushi direction and descent physics.
    
    This is a rough heuristic — precise technique classification would need
    the --cognitive VideoMAE engine.
    """
    if kuz_direction is None:
        return "unknown"
    
    if 'forward' in kuz_direction and velocity > 200:
        if height_drop > 0.5:
            return "hip_throw"     # O-goshi / Harai-goshi family
        else:
            return "body_drop"     # Tai-otoshi family
    elif 'backward' in kuz_direction:
        return "leg_reap"          # Osoto-gari / Ouchi-gari family
    elif 'left' in kuz_direction or 'right' in kuz_direction:
        if velocity < 150:
            return "foot_sweep"    # De-ashi-barai / Okuri-ashi family
        else:
            return "lateral_throw" # Seoi-nage / Sode family
    
    return "projection"  # Generic throw
