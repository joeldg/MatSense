import pytest
import os
import shutil

# This fixture provides the path to the real test video.
@pytest.fixture(scope="session")
def test_video_path():
    path = "tmp_downloads/test_clip.mp4"
    if not os.path.exists(path):
        pytest.skip(f"Live integration tests require {path} to exist.")
    return path

# This fixture creates a safe, isolated output directory for tests that is cleaned up afterward.
@pytest.fixture(scope="function")
def test_output_dir():
    out_dir = "tests/test_output"
    os.makedirs(out_dir, exist_ok=True)
    yield out_dir
    # Teardown: Remove the output dir after the test completes
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
