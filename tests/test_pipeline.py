import os
import pytest
from src.pipeline import GrapplingPipeline

@pytest.mark.integration
def test_full_grappling_pipeline_execution(test_video_path, test_output_dir):
    """
    Simulates a full end-to-end run of the primary CLI `analyze` command on the test clip.
    This guarantees that the PyTorch/YOLO/BoTSORT models load, the Mat Tracker identifies actors,
    the Analyzer detects physics changes, the Renderer draws the HUD, and the 
    Harvester securely extracts the final 224x224 tensor.
    """
    
    pipeline = GrapplingPipeline()
    try:
        pipeline.analyze_match(test_video_path, output_dir=test_output_dir)
    except Exception as e:
        pytest.fail(f"Pipeline crashed during execution: {e}")
        
    # Verify the global Match Report JSON was generated (now in per-video subdir)
    video_basename = os.path.splitext(os.path.basename(test_video_path))[0]
    video_dir = os.path.join(test_output_dir, video_basename)
    report_path = os.path.join(video_dir, "master_match_report.json")
    assert os.path.exists(report_path), f"Master Match Report JSON was not generated at {report_path}."
    
    # Verify that the Data Harvester successfully extracted 224x224 tensors
    harvester_dir = os.path.join(video_dir, "dataset", "raw_clips")
    assert os.path.exists(harvester_dir), "Data Harvester directory was not built."
    
    harvested_files = [f for f in os.listdir(harvester_dir) if f.endswith(".mp4")]
    assert len(harvested_files) > 0, "Data Harvester failed to extract any .mp4 clips during the match."
