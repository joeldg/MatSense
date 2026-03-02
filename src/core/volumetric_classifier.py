import torch
import torch.nn as nn
from settings import DEVICE

class LightweightTrajectoryClassifier(nn.Module):
    """
    A fast 1D-CNN or Transformer that ingests a tensor of (Time, Joints, 3D Coordinates).
    Because it only looks at 3D geometry, it is immune to camera angle or lighting changes.
    """
    def __init__(self, num_joints=24, num_classes=10):
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


class VolumetricTechniqueClassifier:
    """
    Classifier that orchestrates the lightweight trajectory network.
    """
    def __init__(self, num_joints=24):
        print(f"🧠 Initializing 3D Volumetric Classifier on {DEVICE.upper()}...")
        self.device = DEVICE
        self.model = LightweightTrajectoryClassifier(num_joints=num_joints).to(self.device)
        self.model.eval()
        
    def predict(self, mesh_sequence_tensor: torch.Tensor, engine_name="WHAM") -> dict:
        """
        mesh_sequence_tensor: (Time, Joints, 3)
        """
        try:
            # Add batch dimension
            x = mesh_sequence_tensor.unsqueeze(0)
            
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
