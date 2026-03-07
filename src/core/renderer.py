import cv2
import math
import numpy as np
from settings import HUD_COLORS, DASHBOARD_WIDTH

class SkeletonEMA:
    def __init__(self, alpha=0.75, use_mojo=False):
        self.history = None; self.alpha = alpha
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
        
    def update(self, kpts):
        if self.use_mojo and self._mojo:
            result = self._mojo.update_skeleton_ema(self.history, kpts, self.alpha)
            if result is not None:
                self.history = result.copy()
            return self.history

        if kpts is None:
            if self.history is not None: self.history[:, 2] *= 0.8
            return self.history
        if self.history is None:
            self.history = kpts.copy(); return kpts
            
        res = np.zeros_like(kpts)
        for i in range(len(kpts)):
            if kpts[i][2] > 0.30 and self.history[i][2] > 0.30:
                jump = np.hypot(kpts[i][0] - self.history[i][0], kpts[i][1] - self.history[i][1])
                dynamic_alpha = self.alpha if jump < 50.0 else 1.0
                res[i][0] = (dynamic_alpha * kpts[i][0]) + ((1 - dynamic_alpha) * self.history[i][0])
                res[i][1] = (dynamic_alpha * kpts[i][1]) + ((1 - dynamic_alpha) * self.history[i][1])
                res[i][2] = kpts[i][2]
            elif kpts[i][2] > 0.30: res[i] = kpts[i]
            else: res[i] = self.history[i].copy(); res[i][2] *= 0.8 
        self.history = res.copy()
        return res

class BroadcastRenderer:
    def __init__(self, fps, w, h, use_mojo=False):
        self.fps = fps
        self.w = w
        self.h = h
        self.use_mojo = use_mojo
        self.dash_w = DASHBOARD_WIDTH
        self.perspective_lines = None  # Cached vanishing lines
        self.ref_signal_display = None  # Current ref signal to display
        self.ref_signal_timer = 0       # Frames remaining for signal flash
        
        # Dynamic Scaling
        self.scale = max(0.4, min(1.0, w / 1280.0))
        self.th_thin = max(1, int(1 * self.scale))
        self.th_thick = max(1, int(2 * self.scale))
        self.th_bold = max(2, int(4 * self.scale))
        self.r_small = max(2, int(4 * self.scale))
        self.r_large = max(3, int(6 * self.scale))

    def compute_perspective_lines(self, frame):
        """Detect actual mat boundary lines for perspective visualization.
        
        Strategy:
        1. Focus on the mat region (lower 70% of frame, where the mat is)
        2. Use bilateral filtering to preserve strong edges (mat lines) while
           reducing texture noise (gi fabric, skin)
        3. Run Canny + HoughLinesP to find dominant straight lines
        4. Group lines by angle into 2 clusters (longitudinal + lateral mat edges)
        5. Keep strongest lines from each cluster for clean perspective
        6. Extend detected lines to frame edges
        """
        h, w = frame.shape[:2]
        
        # Focus on mat region — the mat is typically in the lower 70% of the frame
        # The top 30% is usually crowd/ceiling/scoreboard
        mat_top = int(h * 0.30)
        mat_region = frame[mat_top:, :]
        
        # Bilateral filter: preserves strong edges (mat lines) while smoothing texture
        filtered = cv2.bilateralFilter(mat_region, 9, 75, 75)
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        
        # Adaptive Canny thresholds based on image contrast
        median_val = np.median(gray)
        low_thresh = int(max(30, 0.5 * median_val))
        high_thresh = int(min(250, 1.5 * median_val))
        edges = cv2.Canny(gray, low_thresh, high_thresh, apertureSize=3)
        
        # HoughLinesP — tuned for mat lines:
        # - minLineLength: 20% of frame width (mat lines are long)
        # - maxLineGap: 5% (lines can be interrupted by athletes/objects)
        # - threshold: 80 votes (less strict to catch mat lines)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                                minLineLength=int(w * 0.12),
                                maxLineGap=int(w * 0.06))
        
        if lines is None or len(lines) < 2:
            return []  # No lines found — draw nothing rather than fake lines
        
        # Collect all candidate lines with metadata
        candidates = []
        for line in lines:
            x1, y1_local, x2, y2_local = line[0]
            # Convert back to full-frame coordinates
            y1 = y1_local + mat_top
            y2 = y2_local + mat_top
            
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            length = math.hypot(x2 - x1, y2 - y1)
            
            # Skip very short lines and nearly vertical lines (usually people standing)
            if length < w * 0.08:
                continue
            if abs(angle) > 85 or abs(angle) < 2:
                continue  # Skip purely horizontal and purely vertical
                
            candidates.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'angle': angle, 'length': length
            })
        
        if len(candidates) < 2:
            return []
        
        # Group lines into angle clusters using simple binning
        # Mat lines typically form two families: ~parallel to each other
        # One family for longitudinal edges, one for lateral edges
        angles = [c['angle'] for c in candidates]
        
        # Sort candidates by angle for cluster detection
        candidates.sort(key=lambda c: c['angle'])
        
        # Find the two dominant angle clusters
        cluster_a = []  # First angle family
        cluster_b = []  # Second angle family
        
        # Simple approach: partition into positive and negative angles
        # (mat lines going left-to-right vs right-to-left)
        for c in candidates:
            if c['angle'] >= 0:
                cluster_a.append(c)
            else:
                cluster_b.append(c)
        
        # If one cluster is empty, try splitting the larger one at the median
        if len(cluster_a) < 2 and len(cluster_b) >= 4:
            mid = len(cluster_b) // 2
            cluster_a = cluster_b[mid:]
            cluster_b = cluster_b[:mid]
        elif len(cluster_b) < 2 and len(cluster_a) >= 4:
            mid = len(cluster_a) // 2
            cluster_b = cluster_a[mid:]
            cluster_a = cluster_a[:mid]
        
        def extend_line_to_edges(x1, y1, x2, y2, w, h):
            """Extend a line segment to reach the frame edges."""
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) < 1:
                return (x1, 0, x1, h)  # Vertical line
            slope = dy / dx
            intercept = y1 - slope * x1
            # Find intersections with frame edges
            y_at_x0 = intercept
            y_at_xw = slope * w + intercept
            x_at_y0 = -intercept / slope if abs(slope) > 0.01 else x1
            x_at_yh = (h - intercept) / slope if abs(slope) > 0.01 else x1
            
            # Pick two intersection points that are within frame bounds
            pts = []
            if 0 <= y_at_x0 <= h: pts.append((0, int(y_at_x0)))
            if 0 <= y_at_xw <= h: pts.append((w, int(y_at_xw)))
            if 0 <= x_at_y0 <= w: pts.append((int(x_at_y0), 0))
            if 0 <= x_at_yh <= w: pts.append((int(x_at_yh), h))
            
            if len(pts) >= 2:
                return (pts[0][0], pts[0][1], pts[1][0], pts[1][1])
            return (x1, y1, x2, y2)  # Couldn't extend
        
        # Take top lines from each cluster (by length), extend them
        result = []
        for cluster in [cluster_a, cluster_b]:
            cluster.sort(key=lambda c: c['length'], reverse=True)
            for c in cluster[:3]:  # Top 3 from each angle family
                ext = extend_line_to_edges(c['x1'], c['y1'], c['x2'], c['y2'], w, h)
                result.append(ext)
        
        return result

    def draw_perspective_grid(self, canvas):
        """Draw green perspective lines along detected mat boundaries."""
        if self.perspective_lines is None:
            self.perspective_lines = self.compute_perspective_lines(canvas[:, :self.w])
        
        if not self.perspective_lines:
            return  # No lines detected — don't draw fake ones
        
        overlay = canvas[:, :self.w].copy()
        color = HUD_COLORS["PERSPECTIVE"]
        for line_data in self.perspective_lines:
            x1, y1, x2, y2 = line_data[0], line_data[1], line_data[2], line_data[3]
            # Draw dashed lines for subtlety
            self._draw_dashed_line(overlay, (x1, y1), (x2, y2), color, self.th_thin, dash_len=15, gap_len=10)
        
        # Blend at 25% opacity — subtle depth cue, not distracting
        cv2.addWeighted(overlay, 0.25, canvas[:, :self.w], 0.75, 0, canvas[:, :self.w])
    
    def _draw_dashed_line(self, img, pt1, pt2, color, thickness, dash_len=15, gap_len=10):
        """Draw a dashed line between two points."""
        x1, y1 = pt1
        x2, y2 = pt2
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist < 1:
            return
        dx = (x2 - x1) / dist
        dy = (y2 - y1) / dist
        pos = 0
        drawing = True
        while pos < dist:
            seg_len = dash_len if drawing else gap_len
            end_pos = min(pos + seg_len, dist)
            if drawing:
                sx = int(x1 + dx * pos)
                sy = int(y1 + dy * pos)
                ex = int(x1 + dx * end_pos)
                ey = int(y1 + dy * end_pos)
                cv2.line(img, (sx, sy), (ex, ey), color, thickness)
            pos = end_pos
            drawing = not drawing

    def draw_custom_skeleton(self, canvas, kpts, color):
        if kpts is None: return 
        skeleton = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
        for p1, p2 in skeleton:
            if p1 < len(kpts) and p2 < len(kpts):
                if kpts[p1][2] > 0.25 and kpts[p2][2] > 0.25 and kpts[p1][0] > 0 and kpts[p2][0] > 0: 
                    cv2.line(canvas, (int(kpts[p1][0]), int(kpts[p1][1])), (int(kpts[p2][0]), int(kpts[p2][1])), color, self.th_bold)
        for pt in kpts:
            if pt[2] > 0.25 and pt[0] > 0:
                cv2.circle(canvas, (int(pt[0]), int(pt[1])), self.r_large, HUD_COLORS["TEXT"], -1)
                cv2.circle(canvas, (int(pt[0]), int(pt[1])), self.r_small, color, -1)

    def render_event_clip(self, video_path, event, timeline, match_num, output_dir="."):
        import os
        start_frame, end_frame = event['start_frame'], event['end_frame']
        cap = cv2.VideoCapture(video_path)
        
        # Calculate human readable timestamps for start and end
        def _format_time(total_seconds):
            hrs = total_seconds // 3600
            mins = (total_seconds % 3600) // 60
            secs = total_seconds % 60
            if hrs > 0:
                return f"{hrs}h{mins:02d}m{secs:02d}s"
            return f"{mins:02d}m{secs:02d}s"
        
        start_ts = _format_time(int(start_frame / self.fps))
        end_ts = _format_time(int(end_frame / self.fps))
        
        current_frame = 0
        print(f"   🎥 Synchronizing video to exact frame {start_frame}...")
        while cap.isOpened() and current_frame < start_frame:
            cap.grab()
            current_frame += 1
            
        realtime_file = f"{output_dir}/match_{match_num}_{start_ts}-{end_ts}_RealSpeed.mp4"
        slowmo_file = f"{output_dir}/match_{match_num}_{start_ts}-{end_ts}_SlowMo.mp4"
        
        out_real = cv2.VideoWriter(realtime_file, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.w + self.dash_w, self.h))
        
        ema1, ema2 = SkeletonEMA(alpha=0.75, use_mojo=self.use_mojo), SkeletonEMA(alpha=0.75, use_mojo=self.use_mojo)
        clip_timeline = []
        total_rendered_frames = 0
        
        # Pre-allocate reusable buffers to avoid per-frame allocation
        mask_buffer = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        black_buffer = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        
        while cap.isOpened() and current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret: break
            
            data = timeline.get(current_frame)
            if not data: 
                current_frame += 1; continue
            
            canvas = np.zeros((self.h, self.w + self.dash_w, 3), dtype=np.uint8)
            canvas[:, :self.w] = frame.copy(); canvas[:, self.w:] = (25, 25, 25) 
            new_state = "STANDING (TACHI-WAZA)"
            
            b1, k1 = data['p1']['box'], data['p1'].get('kpt')
            b2, k2 = data['p2']['box'], data['p2'].get('kpt')
            z_horizon = data.get('z_horizon', self.h*0.8)
            ref_signal = data.get('ref_signal')
            
            if b1 is not None or b2 is not None:
                f_min_x = min(b1[0] if b1 is not None else 9999, b2[0] if b2 is not None else 9999)
                f_min_y = min(b1[1] if b1 is not None else 9999, b2[1] if b2 is not None else 9999)
                f_max_x = max(b1[2] if b1 is not None else 0, b2[2] if b2 is not None else 0)
                f_max_y = max(b1[3] if b1 is not None else 0, b2[3] if b2 is not None else 0)
                
                mask_buffer.fill(0)
                if f_min_x != 9999:
                    cv2.rectangle(mask_buffer, (int(f_min_x-80), int(f_min_y-80)), (int(f_max_x+80), int(f_max_y+80)), (255,255,255), -1)
                    dimmed = cv2.addWeighted(canvas[:, :self.w], 0.35, black_buffer, 0, 0)
                    canvas[:, :self.w] = np.where(mask_buffer == 255, canvas[:, :self.w], dimmed)

                # ── PERSPECTIVE GRID: Green 3D depth lines ──
                self.draw_perspective_grid(canvas)

                cv2.line(canvas[:, :self.w], (0, int(z_horizon)), (self.w, int(z_horizon)), (0, 0, 255), self.th_bold)
                cv2.putText(canvas[:, :self.w], "Z-AXIS DEPTH HORIZON", (20, int(z_horizon) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * self.scale, (0, 0, 255), self.th_thick)

                # ── BACKGROUND entities (grey) ──
                for bg in data.get('bg', []):
                    b = bg['box']
                    cv2.rectangle(canvas[:, :self.w], (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), HUD_COLORS["BG"], self.th_thin)
                
                # ── SPECTATORS (orange) ──
                for spec in data.get('spec', []):
                    b = spec['box']
                    cv2.rectangle(canvas[:, :self.w], (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), HUD_COLORS["SPEC"], self.th_thick) 
                    cv2.putText(canvas[:, :self.w], "SPEC", (int(b[0]), int(b[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * self.scale, HUD_COLORS["SPEC"], self.th_thin)

                # ── REFEREE (red, must exist) ──
                for ref in data.get('ref', []):
                    b = ref['box']
                    cv2.rectangle(canvas[:, :self.w], (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), HUD_COLORS["REF"], self.th_bold) 
                    ref_label = "REF"
                    if ref_signal and ref_signal.get('signal'):
                        ref_label = f"REF [{ref_signal['signal']}]"
                    cv2.putText(canvas[:, :self.w], ref_label, (int(b[0]), int(b[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * self.scale, HUD_COLORS["REF"], self.th_thick)

                # ── UNKNOWN entities ──
                for unk in data.get('unk', []):
                    b = unk['box']
                    cv2.rectangle(canvas[:, :self.w], (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), HUD_COLORS["UNK"], self.th_thin)

                # ── ATHLETE SKELETONS & BOUNDING BOXES ──
                k1_smooth, k2_smooth = ema1.update(k1), ema2.update(k2)
                self.draw_custom_skeleton(canvas[:, :self.w], k1_smooth, HUD_COLORS["ATHLETE_1"])
                self.draw_custom_skeleton(canvas[:, :self.w], k2_smooth, HUD_COLORS["ATHLETE_2"])
                
                # Athlete 1 — WHITE
                if b1 is not None:
                    cv2.rectangle(canvas[:, :self.w], (int(b1[0]), int(b1[1])), (int(b1[2]), int(b1[3])), HUD_COLORS["ATHLETE_1"], self.th_thick)
                    lbl = "ATHLETE 1 (MELD)" if data['melded'] else "ATHLETE 1"
                    cv2.putText(canvas[:, :self.w], lbl, (int(b1[0]), int(b1[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * self.scale, HUD_COLORS["ATHLETE_1"], self.th_thick)
                # Athlete 2 — BLUE
                if b2 is not None and not np.array_equal(b1, b2):
                    cv2.rectangle(canvas[:, :self.w], (int(b2[0]), int(b2[1])), (int(b2[2]), int(b2[3])), HUD_COLORS["ATHLETE_2"], self.th_thick)
                    lbl = "ATHLETE 2 (MELD)" if data['melded'] else "ATHLETE 2"
                    cv2.putText(canvas[:, :self.w], lbl, (int(b2[0]), int(b2[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * self.scale, HUD_COLORS["ATHLETE_2"], self.th_thick)

                c_top = min(b1[1] if b1 is not None else 9999, b2[1] if b2 is not None else 9999)
                if data['melded'] and c_top > (self.h * 0.45): new_state = "NE-WAZA / PIN DETECTED"
                elif (f_max_x - f_min_x) > ((f_max_y - f_min_y) * 1.25): new_state = "NE-WAZA / GROUNDWORK"

            if not clip_timeline or new_state != clip_timeline[-1]: clip_timeline.append(new_state)

            # ── HUD DASHBOARD ──
            hud_color = HUD_COLORS["NE-WAZA"] if "NE-WAZA" in new_state else HUD_COLORS["STANDING"]
            dash_x = self.w + int(30 * self.scale)
            cv2.putText(canvas, "AI GRAPPLING RADAR", (dash_x, int(60 * self.scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * self.scale, HUD_COLORS["TEXT"], self.th_thick)
            cv2.putText(canvas, f"MEDIAN FOREGROUND LOCK", (dash_x, int(100 * self.scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * self.scale, HUD_COLORS["STANDING"], self.th_thin)
            
            # Entity legend
            legend_y = int(130 * self.scale)
            cv2.circle(canvas, (dash_x + 10, legend_y), 6, HUD_COLORS["ATHLETE_1"], -1)
            cv2.putText(canvas, "ATHLETE 1", (dash_x + 25, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * self.scale, HUD_COLORS["ATHLETE_1"], self.th_thin)
            cv2.circle(canvas, (dash_x + 140, legend_y), 6, HUD_COLORS["ATHLETE_2"], -1)
            cv2.putText(canvas, "ATHLETE 2", (dash_x + 155, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * self.scale, HUD_COLORS["ATHLETE_2"], self.th_thin)
            cv2.circle(canvas, (dash_x + 270, legend_y), 6, HUD_COLORS["REF"], -1)
            cv2.putText(canvas, "REF", (dash_x + 285, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * self.scale, HUD_COLORS["REF"], self.th_thin)
            
            cv2.putText(canvas, "CURRENT PHASE:", (dash_x, int(170 * self.scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * self.scale, HUD_COLORS["TEXT"], self.th_thin)
            cv2.rectangle(canvas, (dash_x, int(190 * self.scale)), (dash_x + int(350 * self.scale), int(240 * self.scale)), hud_color, self.th_thick)
            cv2.putText(canvas, new_state, (dash_x + int(10 * self.scale), int(225 * self.scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * self.scale, hud_color, self.th_thick)

            # ── REF ARM SIGNAL DISPLAY ──
            if ref_signal and ref_signal.get('signal'):
                self.ref_signal_display = ref_signal
                self.ref_signal_timer = int(self.fps * 2)  # Flash for 2 seconds
            
            if self.ref_signal_timer > 0 and self.ref_signal_display:
                sig = self.ref_signal_display
                sig_y = int(280 * self.scale)
                # Flash color (red) for first second, then settle to yellow
                sig_color = HUD_COLORS["SIGNAL_FLASH"] if self.ref_signal_timer > self.fps else HUD_COLORS["SIGNAL"]
                cv2.putText(canvas, "REF SIGNAL:", (dash_x, sig_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * self.scale, HUD_COLORS["TEXT"], self.th_thin)
                # Show all sport interpretations
                sport_y = sig_y + int(25 * self.scale)
                for sport, (name, conf) in sig['sport_signals'].items():
                    label = f"{sport.upper()}: {name} ({conf:.0%})"
                    cv2.putText(canvas, label, (dash_x + 10, sport_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * self.scale, sig_color, self.th_thin)
                    sport_y += int(20 * self.scale)
                self.ref_signal_timer -= 1

            out_real.write(canvas)
            total_rendered_frames += 1
            current_frame += 1

        cap.release()
        out_real.release()
        
        print(f"   ✅ Saved Real-Time Output: {realtime_file}")
        
        # Generate slow-mo by re-reading the saved real-time file (avoids storing all frames in RAM)
        if total_rendered_frames > 0:
            rel_transition = int(event['transition_frame'] - event['start_frame'])
            rel_impact = int(event['impact_frame'] - event['start_frame'])
            out_slow = cv2.VideoWriter(slowmo_file, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.w + self.dash_w, self.h))
            
            start_slm = max(0, rel_transition - int(self.fps * 5.0))
            end_slm = min(total_rendered_frames, rel_impact + int(self.fps * 4.0))
            
            # Re-read only the slow-mo slice from the saved real-time file
            cap_rt = cv2.VideoCapture(realtime_file)
            rt_frame_idx = 0
            while cap_rt.isOpened():
                ret, frm = cap_rt.read()
                if not ret: break
                if start_slm <= rt_frame_idx < end_slm:
                    cv2.putText(frm, "SLOW MOTION REPLAY", (int(40*self.scale), int(80*self.scale)), cv2.FONT_HERSHEY_SIMPLEX, 1.2*self.scale, HUD_COLORS["MATE"], self.th_bold)
                    out_slow.write(frm); out_slow.write(frm); out_slow.write(frm)
                elif rt_frame_idx >= end_slm:
                    break
                rt_frame_idx += 1
            cap_rt.release()
            out_slow.release()
            
            if os.path.exists(slowmo_file):
                print(f"   🎥 Saved Cinematic Slow-Mo: {slowmo_file}")

        return {"filename": realtime_file, "slow_mo_file": slowmo_file, "phases": clip_timeline}
