import os
import torch
import argparse
from tqdm import tqdm
from src.core.cognitive_engine_3d import WHAMCognitiveEngine

def precompute_wham_tensors(dataset_dir="dataset/train", output_dir="dataset/3d_tensors"):
    """
    Crawls the labeled `dataset/train/` directory, extracts the 3D WHAM tensor `(Time, Joints, 3)`
    for every `.mp4` takedown clip, and saves it as a `.pt` PyTorch file.
    
    This saves massive computational overhead during Volumetric LSTM training by pre-calculating the 3D geometry.
    """
    
    if not os.path.exists(dataset_dir):
        print(f"❌ Error: Dataset directory {dataset_dir} not found.")
        return
        
    print(f"🧠 Booting WHAM Neural Extraction Subsystem...")
    lifter = WHAMCognitiveEngine()
    
    # Organize by existing technique classes
    classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    
    total_files = 0
    for c in classes:
        total_files += len([f for f in os.listdir(os.path.join(dataset_dir, c)) if f.endswith('.mp4')])
        
    if total_files == 0:
        print(f"❌ No .mp4 clips found in {dataset_dir}")
        return
        
    print(f"🚜 Initiating Offline Pre-Computation for {total_files} tensors across {len(classes)} classes...")
    
    processed = 0
    for class_name in classes:
        input_class_dir = os.path.join(dataset_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        
        os.makedirs(output_class_dir, exist_ok=True)
        
        clips = [f for f in os.listdir(input_class_dir) if f.endswith('.mp4')]
        for clip in tqdm(clips, desc=f"Lifting {class_name}"):
            video_path = os.path.join(input_class_dir, clip)
            tensor_filename = clip.replace('.mp4', '.pt')
            tensor_output_path = os.path.join(output_class_dir, tensor_filename)
            
            # Skip if already computed
            if os.path.exists(tensor_output_path):
                processed += 1
                continue
                
            try:
                # Expected output: (T, 23, 3)
                mesh_sequence = lifter.extract_3d_mesh(video_path)
                
                if mesh_sequence is not None:
                    # Save the raw tensor directly to disk
                    torch.save(mesh_sequence, tensor_output_path)
                    processed += 1
                else:
                    print(f"\n⚠️ WHAM failed to lift: {clip}")
                    
            except Exception as e:
                print(f"\n❌ Error pre-computing {clip}: {e}")
                
    print(f"\n✅ Pre-computation phase complete. {processed}/{total_files} tensors saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute WHAM 3D Tensors offline to accelerate training.")
    parser.add_argument("-d", "--dataset", default="dataset/train", help="Directory containing labeled .mp4 training classes.")
    parser.add_argument("-o", "--output", default="dataset/3d_tensors", help="Output directory for the .pt tensors.")
    
    args = parser.parse_args()
    precompute_wham_tensors(dataset_dir=args.dataset, output_dir=args.output)
