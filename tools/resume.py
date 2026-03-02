from ultralytics import YOLO
import os

def resume_forge():
    # Based on your terminal screenshot yesterday, Ultralytics nested your project inside 'runs/pose/'
    paths_to_check = [
        'runs/pose/Grappling_AI_Project/v1_sensor/weights/last.pt',
        'Grappling_AI_Project/v1_sensor/weights/last.pt'
    ]
    
    checkpoint_path = None
    for path in paths_to_check:
        if os.path.exists(path):
            checkpoint_path = path
            break
            
    if not checkpoint_path:
        print("❌ Could not find 'last.pt'.")
        print("If the script was killed before it could finish Epoch 1, there is no checkpoint to load.")
        return

    print(f"✅ Intact Checkpoint Found! Loading {checkpoint_path}...")
    
    # 1. Load the exact state of the interrupted model
    model = YOLO(checkpoint_path)

    print("\n🔄 RE-IGNITING M4 MAX FORGE FROM LAST CHECKPOINT 🔄")

    # 2. Resume training (It automatically picks up exactly where it left off)
    model.train(resume=True)
    
    print("\n🎉 M4 FORGE COMPLETE!")

if __name__ == '__main__':
    resume_forge()
