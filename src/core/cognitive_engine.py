import torch
import numpy as np
import cv2
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from settings import DEVICE, DEFAULT_VIDEOMAE_MODEL, VIDEOMAE_NUM_FRAMES

class GrapplingCognitiveEngine:
    """
    VideoMAE-based action recognition engine.
    Ingests cropped video clips of grappling techniques and outputs classification probabilities.
    """
    def __init__(self, model_id=DEFAULT_VIDEOMAE_MODEL):
        print(f"🧠 Initializing Cognitive Engine with {model_id} on {DEVICE.upper()}...")
        self.processor = VideoMAEImageProcessor.from_pretrained(model_id)
        self.model = VideoMAEForVideoClassification.from_pretrained(model_id)
        self.model.to(DEVICE)
        self.model.eval()
        self.num_frames = VIDEOMAE_NUM_FRAMES

    def _sample_frames(self, video_path):
        """Uniformly samples frames from the video using OpenCV."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video for classification: {video_path}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            raise ValueError(f"Video {video_path} has 0 frames")
            
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            else:
                # Fallback if frame read fails; duplicate last valid frame
                if frames:
                    frames.append(frames[-1])
                else: 
                     # Create dummy blank frame if even the first read fails
                     frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
                     
        cap.release()
        return frames

    def predict_technique(self, video_path) -> dict:
        """
        Runs the sampled frames through VideoMAE.
        Returns the top predicted class and confidence score.
        """
        try:
            frames = self._sample_frames(video_path)
            
            # Prepare inputs
            inputs = self.processor(list(frames), return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
            top_prob, top_idx = torch.max(probs, dim=-1)
            predicted_class_idx = top_idx.item()
            confidence = top_prob.item()
            
            predicted_label = self.model.config.id2label[predicted_class_idx]
            
            return {
                "technique": predicted_label,
                "confidence": round(confidence, 4)
            }
            
        except Exception as e:
            print(f"❌ Cognitive Engine Error: {e}")
            return {"technique": "Unknown", "confidence": 0.0}

if __name__ == "__main__":
    # Test stub
    import sys
    if len(sys.argv) > 1:
        engine = GrapplingCognitiveEngine()
        result = engine.predict_technique(sys.argv[1])
        print(result)
