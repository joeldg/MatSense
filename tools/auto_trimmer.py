import cv2
import numpy as np
from ultralytics import YOLO
import os
from settings import DEVICE

def find_all_takedowns(video_path):
    print(f"🔍 [INTAKE FILTER V5] Center-Gravity Skeletal Tracking: {video_path}")
    
    # Upgrade to the blazing-fast Pose model to ignore the Hovering Coach
    model = YOLO('yolov8n-pose.pt') 
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open {video_path}")
        return [], 30.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 30.0
    
    # Dynamic Skip Architecture
    base_skip = max(1, int(fps / 5))      # ~0.2s check when action is hot
    empty_skip = max(1, int(fps * 2.5))   # ~2.5s jump when the mat is dead
    current_skip = empty_skip
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    center_x = w / 2.0
    
    posture_history = []
    frame_indices = []
    
    print(f"⏩ Analyzing Match Dynamics (Dynamic Frame Skip Enabled)...")
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        if frame_idx % current_skip == 0:
            # Force CPU so it doesn't interrupt your M4 overnight training!
            results = model.predict(frame, classes=[0], device=DEVICE, verbose=False) 
            
            if results[0].keypoints is not None and len(results[0].keypoints.data) >= 2:
                # Mat is active - Reset to fine-grained inspection
                current_skip = base_skip
                
                kpts = results[0].keypoints.data.cpu().numpy()
                boxes = results[0].boxes.xyxy.cpu().numpy()
                
                # 1. Isolate the Main Subject (Center of the screen)
                scores = []
                for i, b in enumerate(boxes):
                    cx = (b[0] + b[2]) / 2.0
                    dist_to_center = abs(cx - center_x)
                    # Heavily prioritize center-frame existence, ignore area size entirely for distant recordings
                    scores.append((i, 1000.0 / (dist_to_center + 1.0)))
                
                scores.sort(key=lambda x: x[1], reverse=True)
                main_idx = scores[0][0]
                
                # 2. Find their Opponent (Person closest to the Main Subject)
                best_dist = float('inf')
                opp_idx = -1
                main_cx = (boxes[main_idx][0] + boxes[main_idx][2]) / 2.0
                main_cy = (boxes[main_idx][1] + boxes[main_idx][3]) / 2.0
                
                for i in range(len(boxes)):
                    if i == main_idx: continue
                    cx = (boxes[i][0] + boxes[i][2]) / 2.0
                    cy = (boxes[i][1] + boxes[i][3]) / 2.0
                    dist = np.sqrt((main_cx - cx)**2 + (main_cy - cy)**2)
                    if dist < best_dist:
                        best_dist = dist
                        opp_idx = i
                
                if opp_idx != -1:
                    p1, p2 = kpts[main_idx], kpts[opp_idx]
                    
                    # 3. Extract Y-Coordinates of Hips (11/12) and Ankles (15/16)
                    def get_y(kp, idx1, idx2):
                        pts = []
                        if kp[idx1][2] > 0.4: pts.append(kp[idx1][1])
                        if kp[idx2][2] > 0.4: pts.append(kp[idx2][1])
                        return np.mean(pts) if pts else None

                    p1_hip, p1_ankle = get_y(p1, 11, 12), get_y(p1, 15, 16)
                    p2_hip, p2_ankle = get_y(p2, 11, 12), get_y(p2, 15, 16)
                    
                    spreads = []
                    if p1_hip is not None and p1_ankle is not None: spreads.append(p1_ankle - p1_hip)
                    if p2_hip is not None and p2_ankle is not None: spreads.append(p2_ankle - p2_hip)
                    
                    if spreads:
                        # Track whoever is the lowest/most grounded
                        min_spread = min(spreads)
                        
                        # Normalize against their bounding box height so it works on kids and adults equally
                        h1 = boxes[main_idx][3] - boxes[main_idx][1]
                        h2 = boxes[opp_idx][3] - boxes[opp_idx][1]
                        avg_height = (h1 + h2) / 2.0
                        
                        normalized_spread = min_spread / (avg_height + 1e-5)
                        posture_history.append(normalized_spread)
                        frame_indices.append(frame_idx)
            else:
                # Nobody on the mat - Engage fast forward protocol
                current_skip = empty_skip
                
        frame_idx += 1
        if frame_idx % int(fps * 60) == 0:
            print(f"   ...Scanned {int((frame_idx/fps)/60)} minutes")

    cap.release()
    
    if len(posture_history) < 2: return [], fps
    
    # --- MULTI-EVENT TIMELINE ENGINE ---
    takedown_frames = []
    is_standing = False
    last_takedown_frame = -9999
    cooldown_frames = int(fps * 6.0) # 6 second cooldown between events
    
    # Smooth data to prevent 1-frame glitches
    smoothed_posture = np.convolve(posture_history, np.ones(3)/3, mode='same') if len(posture_history) > 3 else posture_history
    
    for i in range(len(smoothed_posture)):
        spread = smoothed_posture[i]
        current_frame = frame_indices[i]
        
        # Standing: Hips are generally > 20% of body height above ankles
        if spread > 0.20: 
            is_standing = True 
            
        # Grounded: Hips hit the mat, vertical gap collapses
        if is_standing and spread < 0.10:
            if (current_frame - last_takedown_frame) > cooldown_frames:
                takedown_frames.append(current_frame)
                last_takedown_frame = current_frame
                is_standing = False
                print(f"💥 TAKEDOWN DETECTED at {current_frame/fps:.1f} seconds!")

    print(f"\n📊 Match Analysis Complete: Found {len(takedown_frames)} total events.")
    return takedown_frames, fps

def trim_all_highlights(video_path, impact_frames, fps, pre_buffer=4.0, post_buffer=3.0):
    if not impact_frames:
        print("❌ No takedowns found to trim.")
        return
        
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Extract Base Filename safely
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Sort impact frames to enable single forward pass
    sorted_impacts = sorted(enumerate(impact_frames), key=lambda x: x[1])
    
    print(f"\n🎥 Auto-Generating {len(impact_frames)} Highlight Reels from {base_name}...")
    
    for orig_idx, impact_frame in sorted_impacts:
        clip_num = orig_idx + 1
        start_frame = max(0, int(impact_frame - (pre_buffer * fps)))
        end_frame = int(impact_frame + (post_buffer * fps))
        
        # Calculate Human Readable Timestamp
        total_seconds = int(impact_frame / fps)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        if hours > 0:
            timestamp_str = f"{hours}h_{minutes:02d}m_{seconds:02d}s"
        else:
            timestamp_str = f"{minutes:02d}m_{seconds:02d}s"
            
        out_path = f'{base_name}_takedown_at_{timestamp_str}.mp4'
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        print(f"✂️  Extracting Match Event {clip_num}: {start_frame/fps:.1f}s to {end_frame/fps:.1f}s...")
        
        # Direct seek — no tracking IDs needed for trimming
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame
            
        while cap.isOpened() and current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret: break
            out.write(frame)
            current_frame += 1
            
        out.release()
        print(f"✅ Saved '{out_path}'")

    cap.release()

if __name__ == '__main__':
    target_video = 'long_gym_match.mp4' 
    if os.path.exists(target_video):
        impacts, video_fps = find_all_takedowns(target_video)
        trim_all_highlights(target_video, impacts, video_fps)
    else:
        print(f"❌ Video not found: {target_video}")
