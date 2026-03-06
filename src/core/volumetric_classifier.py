import torch
import torch.nn as nn
import os
from settings import DEVICE

class LightweightTrajectoryClassifier(nn.Module):
    """
    A fast 1D-CNN or Transformer that ingests a tensor of (Time, Joints, 3D Coordinates).
    Because it only looks at 3D geometry, it is immune to camera angle or lighting changes.
    """
    def __init__(self, num_joints=23, num_classes=10):
        super().__init__()
        # Input shape per frame: num_joints * 3
        input_dim = num_joints * 3
        
        # Simple LSTM sequence classifier
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, num_classes)
        
        # Dummy labels for testing
        self.id2label = {
            0: "Osoto Gari",
            1: "Uchi Mata",
            2: "Seoi Nage",
            3: "Single Leg",
            4: "Double Leg",
            5: "Guard Pull",
            6: "Sprawl",
            7: "Triangle Choke",
            8: "Armbar",
            9: "Unknown"
        }

    def forward(self, x):
        # x shape: (Batch, Time, Joints, 3)
        b, t, j, d = x.shape
        x = x.view(b, t, j * d) # Flatten joints into a single feature vector per time step
        
        lstm_out, (hn, cn) = self.lstm(x)
        # Take the last hidden state
        last_hidden = lstm_out[:, -1, :] 
        
        logits = self.fc(last_hidden)
        return logits


class TensorDataset3D(torch.utils.data.Dataset):
    """
    Loads pre-computed (Time, Joints, 3) tensors from disk.
    Expected structure: dataset_dir / class_name / tensor.pt
    """
    def __init__(self, dataset_dir, max_frames=60):
        self.dataset_dir = dataset_dir
        self.max_frames = max_frames
        self.samples = []
        
        # Build mapping based on VolumetricTechniqueClassifier's ID to Label 
        self.classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
        self.label2id = {label: i for i, label in enumerate(self.classes)}
        
        for label in self.classes:
            class_dir = os.path.join(dataset_dir, label)
            for file in os.listdir(class_dir):
                if file.endswith('.pt'):
                    self.samples.append((os.path.join(class_dir, file), self.label2id[label]))
                    
        print(f"📦 Found {len(self.samples)} pre-computed tensors across {len(self.classes)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tensor_path, label = self.samples[idx]
        
        # Load the pre-computed (T, J, 3) tensor
        tensor = torch.load(tensor_path, weights_only=False)
        
        # Sequence Padding/Truncation to normalize LSTM input length
        t_current = tensor.shape[0]
        if t_current > self.max_frames:
            tensor = tensor[:self.max_frames]
        elif t_current < self.max_frames:
            pad_size = self.max_frames - t_current
            padding = torch.zeros((pad_size, tensor.shape[1], tensor.shape[2]), dtype=tensor.dtype)
            tensor = torch.cat([tensor, padding], dim=0)

        return tensor, torch.tensor(label, dtype=torch.long)


class VolumetricTechniqueClassifier:
    """
    Classifier that orchestrates the lightweight trajectory network.
    """
    def __init__(self, num_joints=23):
        print(f"🧠 Initializing 3D Volumetric Classifier on {DEVICE.upper()}...")
        self.device = DEVICE
        self.model = LightweightTrajectoryClassifier(num_joints=num_joints).to(self.device)
        self.model.eval()
        
    def predict(self, mesh_sequence_tensor: torch.Tensor, engine_name="WHAM") -> dict:
        """
        mesh_sequence_tensor: (Time, Joints, 3)
        """
        try:
            # Add batch dimension and force to the correct hardware device
            x = mesh_sequence_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
            top_prob, top_idx = torch.max(probs, dim=-1)
            predicted_class_idx = top_idx.item()
            confidence = top_prob.item()
            
            predicted_label = self.model.id2label.get(predicted_class_idx, "Unknown")
            
            return {
                "technique": predicted_label,
                "confidence": round(confidence, 4),
                "engine": f"Volumetric/{engine_name}"
            }
            
        except Exception as e:
            print(f"❌ 3D Classification Error: {e}")
            return {"technique": "Error", "confidence": 0.0, "engine": "Error"}

    @classmethod
    def train_model(cls, dataset_dir="dataset/3d_tensors", output_dir="models/volumetric_lstm", epochs=25, batch_size=16):
        """
        Trains the LightweightTrajectoryClassifier natively on extracted WHAM tensors.
        """
        print(f"\n🚀 Initiating Volumetric Engine Training Pipeline on {DEVICE.upper()}...")
        
        dataset = TensorDataset3D(dataset_dir)
        if len(dataset) == 0:
            print("❌ No .pt files found. Did you run `tools/bulk_extract_wham_tensors.py`?")
            return
        if len(dataset) < 3:
            print("❌ Need at least 3 samples to split into train/eval. Aborting.")
            return
            
        # 1. Map dynamic dataset classes back into the architecture
        num_classes = len(dataset.classes)
        model = LightweightTrajectoryClassifier(num_classes=num_classes).to(DEVICE)
        
        # Inject dynamic mapping for future inference
        model.id2label = {v: k for k, v in dataset.label2id.items()}
        
        # 2. Split dataset Train/Eval
        train_size = int(0.8 * len(dataset))
        eval_size = len(dataset) - train_size
        train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        
        # 3. Setup Optimizer and Loss
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # 4. Training Loop
        best_acc = 0.0
        os.makedirs(output_dir, exist_ok=True)
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_x.size(0)
                
            train_loss = train_loss / train_size
            
            # 5. Evaluation Phase
            model.eval()
            correct = 0
            with torch.no_grad():
                for batch_x, batch_y in eval_loader:
                    batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                    outputs = model(batch_x)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == batch_y).sum().item()
                    
            val_acc = correct / max(1, eval_size)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'id2label': model.id2label
                }, os.path.join(output_dir, 'best_volumetric_model.pth'))
                
        print(f"\n✅ Training Complete. Best Validation Accuracy: {best_acc:.4f}")
        print(f"💾 Model saved to: {output_dir}/best_volumetric_model.pth")
