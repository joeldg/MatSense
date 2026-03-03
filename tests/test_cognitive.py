import pytest
import torch
import numpy as np

from src.core.cognitive_engine import GrapplingCognitiveEngine
from src.core.volumetric_classifier import VolumetricTechniqueClassifier, LightweightTrajectoryClassifier

@pytest.fixture
def mock_wham_tensor():
    # Simulate the output of WHAM for a 2-second clip at 30fps
    # (Time=60, Joints=23, Dim=3)
    return torch.randn((60, 23, 3))

def test_volumetric_classifier_structure(mock_wham_tensor):
    """
    Verifies the `VolumetricTechniqueClassifier` can digest a 3D geometry matrix
    without any size mismatch errors on its internal LSTM.
    """
    classifier = VolumetricTechniqueClassifier(num_joints=23)
    
    # Forward Pass
    result = classifier.predict(mock_wham_tensor, engine_name="PyTest_Mock")
    
    assert "technique" in result
    assert "confidence" in result
    assert "engine" in result
    assert result["technique"] != "Error", "Volumetric Classifier threw an exception during prediction."

def test_lightweight_trajectory_lstm_batch(mock_wham_tensor):
    """
    Verifies the standalone PyTorch nn.Module supports batched training inputs properly.
    """
    model = LightweightTrajectoryClassifier(num_joints=23, num_classes=10)
    
    # Create a batch of 4 tensors (Batch=4, Time=60, Joints=23, Dim=3)
    batch_input = torch.stack([mock_wham_tensor for _ in range(4)])
    
    logits = model(batch_input)
    
    # We expect an output of (Batch=4, Classes=10)
    assert logits.shape == (4, 10), f"Expected shape (4, 10), got {logits.shape}"

@pytest.mark.skip(reason="VideoMAE requires internet models unless cached; skipping in fast unit tests.")
def test_videomae_cognitive_engine():
    """
    Stub for testing the 2D VideoMAE image processor.
    """
    engine = GrapplingCognitiveEngine()
    assert engine.model is not None
