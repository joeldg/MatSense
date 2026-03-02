import json
import os
import glob
from PIL import Image

def convert_vicos_to_yolo(json_path):
    print("1. Loading ViCoS academic annotations...")
    with open(json_path, 'r') as f: data = json.load(f) 
        
    print("2. Grouping poses...")
    poses_by_img = {}
    for ann in data:
        img_name = str(ann.get('image', ''))
        if not img_name: continue
        if not img_name.endswith('.jpg'): img_name += '.jpg'
        if img_name not in poses_by_img: poses_by_img[img_name] = []
        if 'pose1' in ann and ann['pose1']: poses_by_img[img_name].append(ann['pose1'])
        if 'pose2' in ann and ann['pose2']: poses_by_img[img_name].append(ann['pose2'])
        
    print("3. Locating images on M4 SSD...")
    # Finds images anywhere inside your current folder recursively
    jpg_files = glob.glob('./**/*.jpg', recursive=True)
    image_paths = {os.path.basename(p): p for p in jpg_files}
    print(f"✅ Found {len(image_paths)} images.")
    
    processed = 0
    print("4. Translating targets to YOLO format...")
    
    for img_name, poses in poses_by_img.items():
        source_path = image_paths.get(img_name)
        if not source_path: continue
            
        try:
            with Image.open(source_path) as img: w, h = img.size 
        except Exception: continue
            
        txt_path = os.path.splitext(source_path)[0] + '.txt'
        lines = []
        
        for pose in poses:
            valid_x = [pt[0] for pt in pose if pt[2] > 0.05] 
            valid_y = [pt[1] for pt in pose if pt[2] > 0.05]
            if not valid_x or not valid_y: continue
                
            min_x, max_x = min(valid_x), max(valid_x)
            min_y, max_y = min(valid_y), max(valid_y)
            box_w, box_h = (max_x - min_x) * 1.15, (max_y - min_y) * 1.15
            center_x, center_y = min_x + (max_x - min_x) / 2.0, min_y + (max_y - min_y) / 2.0
            
            n_cx, n_cy = max(0.0, min(center_x / w, 1.0)), max(0.0, min(center_y / h, 1.0))
            n_bw, n_bh = max(0.0, min(box_w / w, 1.0)), max(0.0, min(box_h / h, 1.0))
            
            yolo_kpts = []
            for pt in pose:
                if pt[2] < 0.05: yolo_kpts.extend([0.0, 0.0, 0]) 
                else: yolo_kpts.extend([max(0.0, min(pt[0] / w, 1.0)), max(0.0, min(pt[1] / h, 1.0)), 2]) 
                    
            lines.append(f"0 {n_cx:.6f} {n_cy:.6f} {n_bw:.6f} {n_bh:.6f} " + " ".join([f"{val:.6f}" for val in yolo_kpts]))
                
        if lines:
            with open(txt_path, 'w') as out: out.write("\n".join(lines) + "\n")
            processed += 1
            
        if processed > 0 and processed % 20000 == 0:
            print(f"   ...Processed {processed} YOLO labels...")
            
    print(f"\n✅ Successfully generated {processed} YOLO labels!")
    
    yaml_content = """
path: .
train: .
val: .

names:
  0: grappler

kpt_shape: [17, 3] 
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
"""
    with open('grappling.yaml', 'w') as f: f.write(yaml_content.strip())
    print("✅ Dynamically created grappling.yaml mission briefing!")

if __name__ == '__main__':
    convert_vicos_to_yolo('annotations.json')
