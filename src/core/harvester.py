import cv2
import numpy as np
import os

class DataHarvester:
    def __init__(self, tensor_size=224, output_dir="dataset/raw_clips"):
        """
        tensor_size=224 is the industry standard input size for Hugging Face VideoMAE.
        """
        self.tensor_size = tensor_size
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def harvest_kinematic_tensor(self, video_path, event, timeline, fps, w, h, clip_id):
        """
        Dynamically tracks, crops, and stabilizes the two fighters during an event.
        """
        start_frame = event['start_frame']
        end_frame = event['end_frame']
        
        vid_name = os.path.basename(video_path).split('.')[0]
        out_filename = os.path.join(self.output_dir, f"tensor_{vid_name}_event_{clip_id}.mp4")
        
        out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (self.tensor_size, self.tensor_size))
        cap = cv2.VideoCapture(video_path)
        
        # Prevent VFR Desync by manually stepping to the start frame
        current_frame = 0
        while cap.isOpened() and current_frame < start_frame:
            cap.grab()
            current_frame += 1
            
        cam_x_ema, cam_y_ema, cam_size_ema = None, None, None
        print(f"   🌾 Harvesting pure kinematic tensor -> {out_filename}")
        
        while cap.isOpened() and current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret: break
            
            data = timeline.get(current_frame)
            if data and (data['p1']['box'] is not None or data['p2']['box'] is not None):
                b1 = data['p1']['box'] if data['p1']['box'] is not None else [9999, 9999, 0, 0]
                b2 = data['p2']['box'] if data['p2']['box'] is not None else [9999, 9999, 0, 0]
                
                if b1[0] == 9999 and b2[0] != 9999: b1 = b2
                elif b2[0] == 9999 and b1[0] != 9999: b2 = b1
                
                # Find the bounding box that encompasses BOTH fighters
                min_x, min_y = min(b1[0], b2[0]), min(b1[1], b2[1])
                max_x, max_y = max(b1[2], b2[2]), max(b1[3], b2[3])
                
                cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
                box_w, box_h = max_x - min_x, max_y - min_y
                target_size = max(box_w, box_h) * 1.45 # 45% padding for flying limbs
                
                # Smooth virtual camera movement
                if cam_x_ema is None:
                    cam_x_ema, cam_y_ema, cam_size_ema = cx, cy, target_size
                else:
                    cam_x_ema = (0.85 * cam_x_ema) + (0.15 * cx)
                    cam_y_ema = (0.85 * cam_y_ema) + (0.15 * cy)
                    cam_size_ema = (0.95 * cam_size_ema) + (0.05 * target_size)
                    
            if cam_x_ema is not None:
                half_s = cam_size_ema / 2.0
                c_x1, c_x2 = int(cam_x_ema - half_s), int(cam_x_ema + half_s)
                c_y1, c_y2 = int(cam_y_ema - half_s), int(cam_y_ema + half_s)
                
                # Handle out-of-bounds screen edges with black padding
                pad_left, pad_top = max(0, -c_x1), max(0, -c_y1)
                pad_right, pad_bot = max(0, c_x2 - w), max(0, c_y2 - h)
                
                c_x1_s, c_x2_s = max(0, c_x1), min(w, c_x2)
                c_y1_s, c_y2_s = max(0, c_y1), min(h, c_y2)
                
                crop = frame[c_y1_s:c_y2_s, c_x1_s:c_x2_s]
                
                if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bot > 0:
                    crop = cv2.copyMakeBorder(crop, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                
                if crop.size > 0:
                    out.write(cv2.resize(crop, (self.tensor_size, self.tensor_size)))
                    
            current_frame += 1
            
        out.release()
        cap.release()
        return out_filename