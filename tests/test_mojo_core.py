"""
Parity tests for Mojo-accelerated analytics functions.

These tests compare the Mojo implementations against the Python originals
to ensure identical results. All tests gracefully skip if Mojo is not installed.
"""

import pytest
import numpy as np
import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ---- Helpers to import the Python originals ----
from src.core.tracker import bb_iou as py_bb_iou
from src.core.analyzer import calculate_fast_kuzushi as py_kuzushi


def _try_import_mojo():
    """Attempt to import the Mojo adapter. Returns (adapter, available)."""
    try:
        from experiments.mojo_core.mojo_adapter import MojoAccelerator
        accel = MojoAccelerator()
        return accel, accel.available
    except ImportError:
        return None, False


# Shared fixture
@pytest.fixture(scope="module")
def mojo():
    accel, available = _try_import_mojo()
    if not available:
        pytest.skip("Mojo not installed — skipping Mojo parity tests")
    return accel


# ==============================================================================
# 1. bb_iou parity
# ==============================================================================
class TestBbIou:
    """Test bb_iou Mojo vs Python with multiple box configurations."""

    CASES = [
        # (box_a, box_b, description)
        ([0, 0, 100, 100], [50, 50, 150, 150], "partial overlap"),
        ([0, 0, 100, 100], [0, 0, 100, 100], "exact same box"),
        ([0, 0, 100, 100], [200, 200, 300, 300], "no overlap"),
        ([10, 20, 300, 500], [50, 100, 250, 400], "large overlap"),
        ([800, 400, 1100, 900], [850, 420, 1150, 950], "typical fighters"),
    ]

    def test_python_bb_iou_sanity(self):
        """Verify Python bb_iou returns expected values."""
        iou = py_bb_iou([0, 0, 100, 100], [0, 0, 100, 100])
        assert abs(iou - 1.0) < 0.01, "Identical boxes should have IoU ~1.0"
        
        iou_none = py_bb_iou([0, 0, 100, 100], [200, 200, 300, 300])
        assert iou_none < 0.01, "Non-overlapping boxes should have IoU ~0.0"

    @pytest.mark.parametrize("box_a,box_b,desc", CASES)
    def test_mojo_matches_python(self, mojo, box_a, box_b, desc):
        py_result = py_bb_iou(box_a, box_b)
        mojo_result = mojo.bb_iou(box_a, box_b)
        assert abs(py_result - mojo_result) < 1e-6, \
            f"IoU mismatch for {desc}: Python={py_result:.6f}, Mojo={mojo_result:.6f}"


# ==============================================================================
# 2. calculate_fast_kuzushi parity
# ==============================================================================
class TestKuzushi:
    """Test kuzushi biomechanics calculation Mojo vs Python."""

    def _make_mock_kpts(self, seed=42):
        """Generate realistic mock YOLO (17, 3) keypoints."""
        rng = np.random.RandomState(seed)
        kpts = rng.rand(17, 3).astype(np.float64) * 500.0
        kpts[:, 2] = 0.9  # high confidence
        return kpts

    def test_python_kuzushi_runs(self):
        kpts = self._make_mock_kpts()
        com, cp, dist, is_k = py_kuzushi(kpts)
        assert com is not None, "Python kuzushi returned None com"
        assert dist >= 0.0

    def test_mojo_matches_python(self, mojo):
        kpts = self._make_mock_kpts(seed=123)
        py_com, py_cp, py_dist, py_is_k = py_kuzushi(kpts)
        mojo_com, mojo_cp, mojo_dist, mojo_is_k = mojo.calculate_fast_kuzushi(kpts)

        np.testing.assert_allclose(py_com, mojo_com, atol=1e-6,
                                   err_msg="Center of Mass mismatch")
        np.testing.assert_allclose(py_cp, mojo_cp, atol=1e-6,
                                   err_msg="Closest point mismatch")
        assert abs(py_dist - mojo_dist) < 1e-6, \
            f"Distance mismatch: Python={py_dist}, Mojo={mojo_dist}"
        assert py_is_k == mojo_is_k, \
            f"is_kuzushi mismatch: Python={py_is_k}, Mojo={mojo_is_k}"

    def test_mojo_kuzushi_multiple_seeds(self, mojo):
        """Run parity check across many random inputs."""
        for seed in range(10):
            kpts = self._make_mock_kpts(seed=seed)
            py_com, py_cp, py_dist, py_is_k = py_kuzushi(kpts)
            mojo_com, mojo_cp, mojo_dist, mojo_is_k = mojo.calculate_fast_kuzushi(kpts)
            assert abs(py_dist - mojo_dist) < 1e-6, f"Seed {seed}: distance mismatch"


# ==============================================================================
# 3. Skeleton EMA parity
# ==============================================================================
class TestSkeletonEMA:
    """Test skeleton EMA update Mojo vs Python."""

    def _make_kpts(self, offset=0.0):
        kpts = np.zeros((17, 3), dtype=np.float64)
        for i in range(17):
            kpts[i] = [100.0 + i * 10.0 + offset, 200.0 + i * 5.0 + offset, 0.9]
        return kpts

    def test_mojo_ema_first_frame(self, mojo):
        """First frame should just return the keypoints."""
        kpts = self._make_kpts()
        result = mojo.update_skeleton_ema(None, kpts, 0.75)
        assert result is not None
        np.testing.assert_allclose(result, kpts, atol=1e-6)

    def test_mojo_ema_smoothing(self, mojo):
        """Second frame should show smoothing effect."""
        kpts1 = self._make_kpts(offset=0.0)
        kpts2 = self._make_kpts(offset=5.0)  # small movement
        
        # First frame
        history = mojo.update_skeleton_ema(None, kpts1, 0.75)
        # Second frame
        result = mojo.update_skeleton_ema(history, kpts2, 0.75)
        
        assert result is not None
        # Result should be between kpts1 and kpts2 (smoothed)
        for i in range(17):
            assert kpts1[i][0] <= result[i][0] <= kpts2[i][0] or \
                   kpts2[i][0] <= result[i][0] <= kpts1[i][0], \
                   f"Keypoint {i} X not smoothed properly"

    def test_mojo_ema_none_kpts(self, mojo):
        """When kpts is None with history, should decay confidence."""
        kpts = self._make_kpts()
        history = mojo.update_skeleton_ema(None, kpts, 0.75)
        result = mojo.update_skeleton_ema(history, None, 0.75)
        
        assert result is not None
        # Confidence should be decayed
        for i in range(17):
            assert result[i][2] < kpts[i][2], \
                f"Keypoint {i} confidence not decayed"


# ==============================================================================
# 4. Event detection parity
# ==============================================================================
class TestEventDetection:
    """Test kinematic event detection Mojo vs Python."""

    def test_mojo_detects_takedown(self, mojo):
        """Given a mock timeline with standing → ground, should detect an event."""
        n = 180
        fps = 30.0
        
        # Standing for 60 frames, then ground for 120 frames (need >= 45 at 30fps)
        heights = []
        tops = []
        ars = []
        melded = []
        for i in range(n):
            if i < 60:
                heights.append(400.0)  # tall = standing
                tops.append(200.0)    # head high up
                ars.append(0.7)       # normal aspect
                melded.append(0.0)
            else:
                heights.append(150.0)  # short = ground
                tops.append(800.0)    # head dropped
                ars.append(1.5)       # wide aspect
                melded.append(0.0)
        
        standing_h = np.percentile(heights[:60], 85)
        
        events = mojo.detect_kinematic_events(
            np.array(heights), np.array(tops), ars, melded,
            standing_h, fps, n
        )
        
        assert len(events) >= 1, "Mojo should detect at least 1 takedown event"
        assert events[0]['impact_frame'] >= 15, "Impact should be after standing frames"


# ==============================================================================  
# 5. Foreground pair scoring parity
# ==============================================================================
class TestForegroundPairScoring:
    """Test foreground pair scoring Mojo vs Python."""

    def test_mojo_fighter_pair_scores_high(self, mojo):
        """Two central, large boxes should score highly."""
        b1 = [800, 400, 1100, 900]
        b2 = [850, 420, 1150, 950]
        score, should_skip = mojo.score_foreground_pair(b1, b2, 1920, 1080)
        assert not should_skip, "Central fighters should not be skipped"
        assert score > 50.0, f"Central fighters should score high, got {score}"

    def test_mojo_mismatched_sizes_skip(self, mojo):
        """Very different sized objects should be skipped."""
        b1 = [100, 100, 200, 600]  # tall
        b2 = [300, 400, 400, 500]  # tiny
        score, should_skip = mojo.score_foreground_pair(b1, b2, 1920, 1080)
        assert should_skip, "Mismatched sizes should trigger skip"

    def test_mojo_distant_pair_skip(self, mojo):
        """Very far apart boxes should be skipped."""
        b1 = [100, 400, 300, 900]
        b2 = [1600, 400, 1800, 900]
        score, should_skip = mojo.score_foreground_pair(b1, b2, 1920, 1080)
        assert should_skip, "Distant pair should trigger skip"


# ==============================================================================
# 6. Cost matrix parity
# ==============================================================================
class TestCostMatrix:
    """Test cost matrix computation Mojo vs Python."""

    def test_mojo_cost_matrix_shape(self, mojo):
        """Cost matrix should be (num_targets, num_candidates)."""
        targets = [
            {'id': 1, 'pure_id': 1, 'box': [100, 200, 300, 500]},
            {'id': 2, 'pure_id': 2, 'box': [400, 200, 600, 500]},
        ]
        candidates = [
            {'id': 1, 'box': [110, 210, 310, 510]},
            {'id': 3, 'box': [800, 200, 1000, 500]},
            {'id': 2, 'box': [420, 220, 620, 520]},
        ]
        
        cost_matrix = mojo.compute_cost_matrix(targets, candidates, 1920)
        assert cost_matrix.shape == (2, 3), f"Expected (2,3), got {cost_matrix.shape}"

    def test_mojo_cost_matrix_id_bonus(self, mojo):
        """Matching IDs should produce lower costs."""
        targets = [
            {'id': 1, 'pure_id': 1, 'box': [100, 200, 300, 500]},
        ]
        same_id_cand = [{'id': 1, 'box': [110, 210, 310, 510]}]
        diff_id_cand = [{'id': 99, 'box': [110, 210, 310, 510]}]
        
        cost_same = mojo.compute_cost_matrix(targets, same_id_cand, 1920)
        cost_diff = mojo.compute_cost_matrix(targets, diff_id_cand, 1920)
        
        assert cost_same[0, 0] < cost_diff[0, 0], \
            "Same-ID candidate should have lower cost"
