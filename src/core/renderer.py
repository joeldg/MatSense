import cv2
import numpy as np
from settings import HUD_COLORS, DASHBOARD_WIDTH

class SkeletonEMA:
    def __init__(self, alpha=0.75):
        self.history = None; self.alpha = alpha
        
    def update(self, kpts):
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
    def __init__(self, fps, w, h):
        self.fps = fps
        self.w = w
        self.h = h
        self.dash_w = DASHBOARD_WIDTH

    def draw_custom_skeleton(self, canvas, kpts, color):
        if kpts is None: return 
        skeleton = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
        for p1, p2 in skeleton:
            if p1 < len(kpts) and p2 < len(kpts):
                if kpts[p1][2] > 0.25 and kpts[p2][2] > 0.25 and kpts[p1][0] > 0 and kpts[p2][0] > 0: 
                    cv2.line(canvas, (int(kpts[p1][0]), int(kpts[p1][1])), (int(kpts[p2][0]), int(kpts[p2][1])), color, 4)
        for pt in kpts:
            if pt[2] > 0.25 and pt[0] > 0:
                cv2.circle(canvas, (int(pt[0]), int(pt[1])), 6, HUD_COLORS["TEXT"], -1)
                cv2.circle(canvas, (int(pt[0]), int(pt[1])), 4, color, -1)

    def render_event_clip(self, video_path, event, timeline, clip_id, output_dir="."):
        start_frame, end_frame = event['start_frame'], event['end_frame']
        cap = cv2.VideoCapture(video_path)
        
        current_frame = 0
        print(f"   🎥 Synchronizing video to exact frame {start_frame}...")
        while cap.isOpened() and current_frame < start_frame:
            cap.grab()
            current_frame += 1
            
        realtime_file = f"{output_dir}/Analyzed_Event_{clip_id}_RealSpeed.mp4"
        slowmo_file = f"{output_dir}/Analyzed_Event_{clip_id}_SlowMo.mp4"
        
        out_real = cv2.VideoWriter(realtime_file, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.w + self.dash_w, self.h))
        
        ema1, ema2 = SkeletonEMA(alpha=0.75), SkeletonEMA(alpha=0.75)
        all_rendered_frames, clip_timeline = [], []
        
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
            
            if b1 is not None or b2 is not None:
                f_min_x = min(b1[0] if b1 is not None else 9999, b2[0] if b2 is not None else 9999)
                f_min_y = min(b1[1] if b1 is not None else 9999, b2[1] if b2 is not None else 9999)
                f_max_x = max(b1[2] if b1 is not None else 0, b2[2] if b2 is not None else 0)
                f_max_y = max(b1[3] if b1 is not None else 0, b2[3] if b2 is not None else 0)
                
                mask = np.zeros((self.h, self.w, 3), dtype=np.uint8)
                if f_min_x != 9999:
                    cv2.rectangle(mask, (int(f_min_x-80), int(f_min_y-80)), (int(f_max_x+80), int(f_max_y+80)), (255,255,255), -1)
                    dimmed = cv2.addWeighted(canvas[:, :self.w], 0.35, np.zeros_like(canvas[:, :self.w]), 0, 0)
                    canvas[:, :self.w] = np.where(mask == 255, canvas[:, :self.w], dimmed)

                cv2.line(canvas[:, :self.w], (0, int(z_horizon)), (self.w, int(z_horizon)), (0, 0, 255), 4)
                cv2.putText(canvas[:, :self.w], "Z-AXIS DEPTH HORIZON", (20, int(z_horizon) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                for bg in data.get('bg', []):
                    b = bg['box']
                    cv2.rectangle(canvas[:, :self.w], (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), HUD_COLORS["BG"], 2)
                    cv2.putText(canvas[:, :self.w], f"[BG ID:{bg.get('id','?')}]", (int(b[0]), int(b[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, HUD_COLORS["TEXT_DIM"], 1)
                
                for spec in data.get('spec', []):
                    b = spec['box']
                    cv2.rectangle(canvas[:, :self.w], (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), HUD_COLORS["SPEC"], 3) 
                    cv2.putText(canvas[:, :self.w], f"[SPECTATOR ID:{spec.get('id','?')}]", (int(b[0]), int(b[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, HUD_COLORS["SPEC"], 2)

                for ref in data.get('ref', []):
                    b = ref['box']
                    cv2.rectangle(canvas[:, :self.w], (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), HUD_COLORS["REF"], 3) 
                    cv2.putText(canvas[:, :self.w], f"[REF/COACH ID:{ref.get('id','?')}]", (int(b[0]), int(b[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, HUD_COLORS["REF"], 2)

                for unk in data.get('unk', []):
                    b = unk['box']
                    cv2.rectangle(canvas[:, :self.w], (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), HUD_COLORS["UNK"], 2) 
                    cv2.putText(canvas[:, :self.w], f"[BYSTANDER ID:{unk.get('id','?')}]", (int(b[0]), int(b[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, HUD_COLORS["UNK"], 1)

                k1_smooth, k2_smooth = ema1.update(k1), ema2.update(k2)
                self.draw_custom_skeleton(canvas[:, :self.w], k1_smooth, HUD_COLORS["SKELETON_0"])
                self.draw_custom_skeleton(canvas[:, :self.w], k2_smooth, HUD_COLORS["SKELETON_1"])
                
                if b1 is not None:
                    cv2.rectangle(canvas[:, :self.w], (int(b1[0]), int(b1[1])), (int(b1[2]), int(b1[3])), HUD_COLORS["SKELETON_0"], 2)
                    lbl = f"ID:{data['p1']['id']} (MELD)" if data['melded'] else f"ID:{data['p1']['id']} (PLAYER 1)"
                    cv2.putText(canvas[:, :self.w], lbl, (int(b1[0]), int(b1[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HUD_COLORS["SKELETON_0"], 2)
                if b2 is not None and not np.array_equal(b1, b2):
                    cv2.rectangle(canvas[:, :self.w], (int(b2[0]), int(b2[1])), (int(b2[2]), int(b2[3])), HUD_COLORS["SKELETON_1"], 2)
                    lbl = f"ID:{data['p2']['id']} (MELD)" if data['melded'] else f"ID:{data['p2']['id']} (PLAYER 2)"
                    cv2.putText(canvas[:, :self.w], lbl, (int(b2[0]), int(b2[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HUD_COLORS["SKELETON_1"], 2)

                c_top = min(b1[1] if b1 is not None else 9999, b2[1] if b2 is not None else 9999)
                if data['melded'] and c_top > (self.h * 0.45): new_state = "NE-WAZA / PIN DETECTED"
                elif (f_max_x - f_min_x) > ((f_max_y - f_min_y) * 1.25): new_state = "NE-WAZA / GROUNDWORK"

            if not clip_timeline or new_state != clip_timeline[-1]: clip_timeline.append(new_state)

            hud_color = HUD_COLORS["NE-WAZA"] if "NE-WAZA" in new_state else HUD_COLORS["STANDING"]
            dash_x = self.w + 30
            cv2.putText(canvas, "AI GRAPPLING RADAR", (dash_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, HUD_COLORS["TEXT"], 2)
            cv2.putText(canvas, f"MEDIAN FOREGROUND LOCK", (dash_x, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, HUD_COLORS["STANDING"], 1)
            cv2.putText(canvas, "CURRENT PHASE:", (dash_x, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, HUD_COLORS["TEXT"], 1)
            cv2.rectangle(canvas, (dash_x, 190), (dash_x + 350, 240), hud_color, 2)
            cv2.putText(canvas, new_state, (dash_x + 10, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 2)

            out_real.write(canvas)
            all_rendered_frames.append(canvas.copy())
            current_frame += 1

        cap.release()
        out_real.release()
        print(f"   ✅ Saved Real-Time Output: {realtime_file}")
        
        if len(all_rendered_frames) > 0:
            rel_transition = int(event['transition_frame'] - event['start_frame'])
            rel_impact = int(event['impact_frame'] - event['start_frame'])
            out_slow = cv2.VideoWriter(slowmo_file, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.w + self.dash_w, self.h))
            
            start_slm = max(0, rel_transition - int(self.fps * 5.0))
            end_slm = min(len(all_rendered_frames), rel_impact + int(self.fps * 4.0))
            
            for idx in range(start_slm, end_slm):
                frm = all_rendered_frames[idx].copy()
                cv2.putText(frm, "SLOW MOTION REPLAY", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, HUD_COLORS["MATE"], 3)
                out_slow.write(frm); out_slow.write(frm); out_slow.write(frm)
                
            out_slow.release()
            print(f"   🎥 Saved Cinematic Slow-Mo: {slowmo_file}")

        return {"filename": realtime_file, "slow_mo_file": slowmo_file, "phases": clip_timeline}
