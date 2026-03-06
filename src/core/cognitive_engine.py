import torch
import numpy as np
import cv2
import os
from torch.utils.data import Dataset
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification, TrainingArguments, Trainer
import evaluate
from settings import DEVICE, DEFAULT_VIDEOMAE_MODEL, VIDEOMAE_NUM_FRAMES


def sample_video_frames(video_path, num_frames=16):
    """Uniformly sample frames using efficient forward-pass reading (grab + selective retrieve)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise ValueError(f"Video {video_path} has 0 frames")
    
    indices = set(np.linspace(0, total_frames - 1, num_frames, dtype=int))
    sorted_indices = sorted(indices)
    
    frames = []
    frame_idx = 0
    target_ptr = 0
    
    while cap.isOpened() and target_ptr < len(sorted_indices):
        if frame_idx == sorted_indices[target_ptr]:
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
            target_ptr += 1
            # Handle duplicate indices (when total_frames < num_frames)
            while target_ptr < len(sorted_indices) and sorted_indices[target_ptr] == frame_idx:
                frames.append(frames[-1])
                target_ptr += 1
        else:
            cap.grab()  # Skip without decoding
        frame_idx += 1
    
    # Pad if we didn't get enough frames
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
    
    cap.release()
    return frames

class VideoClipDataset(Dataset):
    """
    Custom PyTorch Dataset for loading 16-frame VideoMAE tensors from .mp4 clips.
    Expected structure: dataset_dir / class_name / video.mp4
    """
    def __init__(self, dataset_dir, processor, num_frames=16):
        self.processor = processor
        self.num_frames = num_frames
        self.samples = []
        
        self.classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
        self.label2id = {label: i for i, label in enumerate(self.classes)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        
        for label in self.classes:
            class_dir = os.path.join(dataset_dir, label)
            for file in os.listdir(class_dir):
                if file.endswith('.mp4'):
                    self.samples.append((os.path.join(class_dir, file), self.label2id[label]))
                    
        print(f"📦 Found {len(self.samples)} clips across {len(self.classes)} classes.")

    def __len__(self):
        return len(self.samples)

    def _sample_frames(self, video_path):
        return sample_video_frames(video_path, self.num_frames)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self._sample_frames(video_path)
        
        # The processor expects a list of 3D arrays and returns pixel_values of shape (num_frames, num_channels, height, width)
        inputs = self.processor(list(frames), return_tensors="pt")
        # remove the batch dim added by processor
        pixel_values = inputs["pixel_values"].squeeze(0) 
        
        return {"pixel_values": pixel_values, "labels": torch.tensor(label, dtype=torch.long)}


class GrapplingCognitiveEngine:
    """
    VideoMAE-based action recognition engine.
    Ingests cropped video clips of grappling techniques and outputs classification probabilities.
    """
    def __init__(self, model_id=DEFAULT_VIDEOMAE_MODEL, num_labels=None, id2label=None, label2id=None):
        print(f"🧠 Initializing Cognitive Engine with {model_id} on {DEVICE.upper()}...")
        self.processor = VideoMAEImageProcessor.from_pretrained(model_id)
        
        if num_labels is not None:
             # Load for fine-tuning with custom heads
             self.model = VideoMAEForVideoClassification.from_pretrained(
                 model_id, 
                 ignore_mismatched_sizes=True,
                 num_labels=num_labels,
                 id2label=id2label,
                 label2id=label2id
             )
        else:
             # Load existing zero-shot weights (e.g. Kinetics)
             self.model = VideoMAEForVideoClassification.from_pretrained(model_id, ignore_mismatched_sizes=True)
             
        self.model.to(DEVICE)
        self.num_frames = VIDEOMAE_NUM_FRAMES

    def _sample_frames(self, video_path):
        """Uniformly samples frames from the video."""
        return sample_video_frames(video_path, self.num_frames)

    def predict_technique(self, video_path) -> dict:
        """
        Runs the sampled frames through VideoMAE.
        Returns the top predicted class and confidence score.
        """
        self.model.eval()
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

    @classmethod
    def train_model(cls, dataset_dir, output_dir="models/videomae-grappling", epochs=10, batch_size=4):
        """
        Sets up the Hugging Face Trainer to fine-tune the VideoMAE model on our auto-labeled dataset.
        """
        print(f"\n🚀 Initiating Cognitive Engine Fine-Tuning Pipeline...")
        print(f"📂 Scanning dataset at: {dataset_dir}")
        
        # 1. Initialize Processor to build Dataset
        processor = VideoMAEImageProcessor.from_pretrained(DEFAULT_VIDEOMAE_MODEL)
        
        full_dataset = VideoClipDataset(dataset_dir, processor, num_frames=VIDEOMAE_NUM_FRAMES)
        
        if len(full_dataset) == 0:
            print("❌ No .mp4 files found in dataset directories. Aborting training.")
            return

        id2label = full_dataset.id2label
        label2id = full_dataset.label2id
        num_labels = len(id2label)
        
        # 2. Split dataset into Train / Eval (80/20)
        train_size = int(0.8 * len(full_dataset))
        eval_size = len(full_dataset) - train_size
        train_dataset, eval_dataset = torch.utils.data.random_split(full_dataset, [train_size, eval_size])
        
        print(f"📊 Dataset Split: {train_size} Train | {eval_size} Eval")

        # 3. Initialize Model with new classification head
        engine = cls(
            model_id=DEFAULT_VIDEOMAE_MODEL, 
            num_labels=num_labels, 
            id2label=id2label, 
            label2id=label2id
        )

        # 4. Define Evaluation Metric
        metric = evaluate.load("accuracy")
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return metric.compute(predictions=predictions, references=labels)

        # 5. Training Arguments optimized for local compute (MPS/CUDA)
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=5e-5,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            fp16=False, # MPS traditionally has issues with HF FP16 mixed precision
            use_mps_device=(DEVICE == 'mps'),
            remove_unused_columns=False # Required when passing custom pixel_values
        )

        # 6. Initialize Trainer
        trainer = Trainer(
            model=engine.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        # 7. Start Training!
        print("🔥 Starting VideoMAE Transfer Learning...")
        trainer.train()

        # 8. Save final artifacts
        print(f"💾 Saving finetuned model and processor to {output_dir}...")
        trainer.save_model(output_dir)
        engine.processor.save_pretrained(output_dir)
        print("✅ Training complete.")

if __name__ == "__main__":
    # Test stub
    import sys
    if len(sys.argv) > 1:
        engine = GrapplingCognitiveEngine()
        result = engine.predict_technique(sys.argv[1])
        print(result)
