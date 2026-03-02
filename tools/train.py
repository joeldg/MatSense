from ultralytics import YOLO
import torch

def forge_on_apple_silicon():
    # 1. Verify Apple Metal (MPS) is active
    if torch.backends.mps.is_available():
        print("✅ Apple M-Series GPU (MPS) Detected & Activated!")
        compute_device = 'mps'
    else:
        print("⚠️ MPS not detected. Falling back to CPU.")
        compute_device = 'cpu'
        
    # 2. Load the base AI model
    model = YOLO('yolov8n-pose.pt') 

    print("\n🔥 IGNITING M4 MAX METAL GPU 🔥")

    # 3. Train using Apple MPS (Metal Performance Shaders)
    results = model.train(
        data='grappling.yaml',
        epochs=15,              
        imgsz=640,              
        batch=32,               # M4 Unified Memory will eat 32 for breakfast (bump to 64 if you have 64GB+ RAM) 
        device=compute_device,  # TARGET APPLE SILICON GPU
        workers=8,              # Unleash your M4 CPU cores
        amp=False,              # CRITICAL for Macs: Disables Automatic Mixed Precision to prevent NaN crashes
        
        # Defense AI Augmentations
        degrees=90.0,           
        fliplr=0.5,             
        erasing=0.3,            
        mosaic=1.0,             
        
        project='Grappling_AI_Project',
        name='v1_sensor'
    )
    
    print("\n🎉 M4 FORGE COMPLETE! IP saved locally in Grappling_AI_Project/v1_sensor/weights/best.pt")

if __name__ == '__main__':
    forge_on_apple_silicon()
