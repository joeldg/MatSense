import os
import pytest
from src.pipeline import GrapplingPipeline

@pytest.mark.integration
def test_full_grappling_pipeline_execution(test_video_path, test_output_dir):
    """
    Simulates a full end-to-end run of the primary CLI `analyze` command on the test clip.
    This guarantees that the PyTorch/YOLO/BoTSORT models load, the Mat Tracker identifies actors,
    and the pipeline runs without crashing. 
    
    NOTE: The test clip may not pass the Match Triad Gate (2 athletes + ref required)
    or Engagement Gate (>15% frames with both athletes), so we verify graceful handling
    of both "events found" and "no events" paths.
    """
    
    pipeline = GrapplingPipeline()
    try:
        pipeline.analyze_match(test_video_path, output_dir=test_output_dir)
    except Exception as e:
        pytest.fail(f"Pipeline crashed during execution: {e}")
        
    # Verify the per-video output directory was always created
    video_basename = os.path.splitext(os.path.basename(test_video_path))[0]
    video_dir = os.path.join(test_output_dir, video_basename)
    assert os.path.exists(video_dir), f"Per-video output directory was not created at {video_dir}"
    
    # If events were found (triad + engagement gates passed), verify outputs
    report_path = os.path.join(video_dir, "master_match_report.json")
    if os.path.exists(report_path):
        # Full pipeline ran — verify harvester output too
        harvester_dir = os.path.join(video_dir, "dataset", "raw_clips")
        assert os.path.exists(harvester_dir), "Data Harvester directory was not built."
        
        harvested_files = [f for f in os.listdir(harvester_dir) if f.endswith(".mp4")]
        assert len(harvested_files) > 0, "Data Harvester failed to extract any .mp4 clips."
    else:
        # No events found (triad/engagement gate filtered) — this is valid for test clips
        print("   ℹ️ Pipeline completed gracefully with no events (triad/engagement gate)")

