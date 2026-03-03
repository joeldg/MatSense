import argparse
import sys
import os

from src.media_handler import VideoFetcher
from src.pipeline import GrapplingPipeline
# Import tools
from tools.auto_trimmer import find_all_takedowns, trim_all_highlights
from tools.train import forge_on_apple_silicon
from tools.resume import resume_forge
from tools.prep_data import convert_vicos_to_yolo

def main():
    parser = argparse.ArgumentParser(description="Grappling AI: Cinematic Takedown Analytics")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 1. Analyze Command
    analyze_parser = subparsers.add_parser("analyze", help="Run full pipeline on a local file, direct URL, or YouTube link.")
    analyze_parser.add_argument("-i", "--input", required=True, help="Path to local video or a URL/YouTube link.")
    analyze_parser.add_argument("-o", "--output", default=".", help="Directory to save output files.")

    # 2. Trim Command
    trim_parser = subparsers.add_parser("trim", help="Quickly find and extract takedowns from a long video.")
    trim_parser.add_argument("-i", "--input", required=True, help="Path to local video or a URL/YouTube link.")
    trim_parser.add_argument("-o", "--output", default=".", help="Directory to save highlight clips.")

    # 3. Classify Command
    classify_parser = subparsers.add_parser("classify", help="Run the VideoMAE Cognitive Engine to classify a takedown clip.")
    classify_parser.add_argument("-i", "--input", required=True, help="Path to the trimmed takedown clip.")

    # 4. Classify 3D Command
    classify3d_parser = subparsers.add_parser("classify-3d", help="Lifts video to 3D via WHAM/SMPL-X and classifies trajectories.")
    classify3d_parser.add_argument("-i", "--input", required=True, help="Path to the trimmed takedown clip.")
    classify3d_parser.add_argument("--engine", choices=["wham", "smplx"], default="wham", help="Which 3D engine to use.")

    # 5. Train Command
    train_parser = subparsers.add_parser("train", help="Train the model using Apple MPS / CUDA.")
    train_parser.add_argument("--resume", action="store_true", help="Resume from the last checkpoint.")

    # 6. Train Cognitive Command
    train_cog_parser = subparsers.add_parser("train-cognitive", help="Fine-tune the VideoMAE Cognitive Engine.")
    train_cog_parser.add_argument("-d", "--dataset", default="dataset/train", help="Directory containing the training classes.")
    train_cog_parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of training epochs.")

    # 7. Train 3D Cognitive Command
    train_3d_parser = subparsers.add_parser("train-3d", help="Train the Volumetric LSTM Engine on pre-computed WHAM 3D tensors.")
    train_3d_parser.add_argument("-d", "--dataset", default="dataset/3d_tensors", help="Directory containing the pre-computed .pt tensors.")
    train_3d_parser.add_argument("-e", "--epochs", type=int, default=25, help="Number of training epochs.")

    # 8. Prep Command
    prep_parser = subparsers.add_parser("prep", help="Convert ViCoS annotations into YOLO text files.")
    prep_parser.add_argument("-d", "--data", default="annotations.json", help="Path to annotations.json")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    fetcher = VideoFetcher()

    if args.command == "analyze":
        output_dir = args.output
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        try:
            video_path = fetcher.get_video_path(args.input)
            pipeline = GrapplingPipeline()
            pipeline.analyze_match(video_path, output_dir=output_dir)
        except Exception as e:
            print(f"❌ Error during analysis: {e}")

    elif args.command == "trim":
        output_dir = args.output
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        try:
            video_path = fetcher.get_video_path(args.input)
            # Temporarily cd into output dir so auto_trimmer dumps highlights there
            og_dir = os.getcwd()
            os.chdir(output_dir)
            impacts, video_fps = find_all_takedowns(video_path)
            trim_all_highlights(video_path, impacts, video_fps)
            os.chdir(og_dir)
            print("✅ Trimming complete!")
        except Exception as e:
            print(f"❌ Error during trimming: {e}")

    elif args.command == "classify":
        try:
            video_path = fetcher.get_video_path(args.input)
            from src.core.cognitive_engine import GrapplingCognitiveEngine
            engine = GrapplingCognitiveEngine()
            result = engine.predict_technique(video_path)
            print(f"\n🧠 Classification Result for {video_path}:")
            import json
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"❌ Error during classification: {e}")

    elif args.command == "classify-3d":
        try:
            video_path = fetcher.get_video_path(args.input)
            from src.core.cognitive_engine_3d import WHAMCognitiveEngine, SMPLXCognitiveEngine
            from src.core.volumetric_classifier import VolumetricTechniqueClassifier
            import json
            
            # Select 3D lifter
            if args.engine == "smplx":
                lifter = SMPLXCognitiveEngine()
                num_joints = 55
            else:
                lifter = WHAMCognitiveEngine()
                num_joints = 23
                
            mesh_sequence = lifter.extract_3d_mesh(video_path)
            
            classifier = VolumetricTechniqueClassifier(num_joints=num_joints)
            result = classifier.predict(mesh_sequence, engine_name=args.engine)
            
            print(f"\n🧠 3D Classification Result for {video_path}:")
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"❌ Error during 3D classification: {e}")

    elif args.command == "train":
        if args.resume:
            print("🔄 Attempting to resume training...")
            resume_forge()
        else:
            print("🔥 Starting new training session...")
            forge_on_apple_silicon()

    elif args.command == "train-cognitive":
        from src.core.cognitive_engine import GrapplingCognitiveEngine
        try:
            GrapplingCognitiveEngine.train_model(dataset_dir=args.dataset, epochs=args.epochs)
        except Exception as e:
            print(f"❌ Error during cognitive engine fine-tuning: {e}")

    elif args.command == "train-3d":
        from src.core.volumetric_classifier import VolumetricTechniqueClassifier
        try:
            VolumetricTechniqueClassifier.train_model(dataset_dir=args.dataset, epochs=args.epochs)
        except Exception as e:
            print(f"❌ Error during 3D Volumetric training: {e}")

    elif args.command == "prep":
        if os.path.exists(args.data):
            convert_vicos_to_yolo(args.data)
        else:
            print(f"❌ Annotation file not found: {args.data}")

if __name__ == "__main__":
    main()
