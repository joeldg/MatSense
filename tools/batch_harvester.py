import os
import argparse
import glob
from src.pipeline import GrapplingPipeline
from tools.auto_trimmer import find_all_takedowns, trim_all_highlights

def harvest_directory(input_dir, output_dir, mode="trim"):
    """
    Crawls a directory of grappling matches and automatically harvests 
    4-second cinematic action tensors from every clip found.

    mode="trim": Just extracts the original 1080p clips (fast, no HUD).
    mode="analyze": Runs the full tracker, draws the cinematic HUD, and extracts the 224x224 tensor for AI.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Find all video files
    extensions = ('*.mp4', '*.mov', '*.avi', '*.mkv')
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))

    if not video_files:
        print(f"❌ No videos found in {input_dir}")
        return

    print(f"🚜 Batch Harvester Initialized. Found {len(video_files)} match videos.")

    pipeline = GrapplingPipeline()

    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {os.path.basename(video_path)}")
        try:
            if mode == "analyze":
                # Run the full heavy pipeline (Generates Datasets + Broadcast UI)
                pipeline.analyze_match(video_path, output_dir=output_dir)
            else:
                # Fast extraction (Only gets the raw footage, good for manual reviews)
                og_dir = os.getcwd()
                target_out = os.path.join(output_dir, os.path.splitext(os.path.basename(video_path))[0])
                if not os.path.exists(target_out):
                    os.makedirs(target_out)
                os.chdir(target_out)
                
                impacts, video_fps = find_all_takedowns(video_path)
                trim_all_highlights(video_path, impacts, video_fps)
                os.chdir(og_dir)
                
        except Exception as e:
            print(f"⚠️ Failed to harvest {video_path}: {e}")

    print(f"\n✅ Harvester complete. Output saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process multiple Grappling Matches to harvest datasets.")
    parser.add_argument("-d", "--directory", required=True, help="Input directory containing match videos.")
    parser.add_argument("-o", "--output", default="dataset/raw_clips", help="Output directory for harvested tensors.")
    parser.add_argument("--mode", choices=["trim", "analyze"], default="trim", help="Mode: 'trim' (fast video cuts) or 'analyze' (full AI tracking and 224x224 tensor harvesting).")
    
    args = parser.parse_args()
    harvest_directory(args.directory, args.output, mode=args.mode)
