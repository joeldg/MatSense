"""
Mojo Adapter — Drop-in replacement functions that delegate to the Mojo
compiled analytics module when available, with graceful fallback to Python.

Usage:
    from experiments.mojo_core.mojo_adapter import MojoAccelerator
    accel = MojoAccelerator()
    if accel.available:
        iou = accel.bb_iou(box_a, box_b)
"""

import os
import sys
import importlib.util
import numpy as np

MOJO_AVAILABLE = False
_mojo_mod = None

def _try_load_mojo():
    """Attempt to load the pre-compiled mojo_analytics.so shared library."""
    global MOJO_AVAILABLE, _mojo_mod
    
    mojo_dir = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(mojo_dir, "mojo_analytics.so")
    
    if not os.path.exists(so_path):
        print(f"⚠️ Mojo Accelerator: OFFLINE (mojo_analytics.so not found at {so_path})")
        print(f"   Build it with: cd {mojo_dir} && pixi run mojo build mojo_analytics.mojo --emit shared-lib -o mojo_analytics.so")
        return
    
    try:
        spec = importlib.util.spec_from_file_location("mojo_analytics", so_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _mojo_mod = mod
        MOJO_AVAILABLE = True
        print("🔥 Mojo Accelerator: ONLINE (mojo_analytics.so loaded)")
    except Exception as e:
        MOJO_AVAILABLE = False
        print(f"⚠️ Mojo Accelerator: OFFLINE ({e})")

_try_load_mojo()


class MojoAccelerator:
    """Wrapper providing Mojo-accelerated functions with matching Python signatures."""
    
    def __init__(self):
        self.available = MOJO_AVAILABLE
        self._mod = _mojo_mod
    
    # ------------------------------------------------------------------
    # 1. bb_iou
    # ------------------------------------------------------------------
    def bb_iou(self, box_a, box_b):
        """IoU between two bounding boxes [x1, y1, x2, y2]."""
        return float(self._mod.bb_iou(
            list(map(float, box_a)),
            list(map(float, box_b))
        ))
    
    # ------------------------------------------------------------------
    # 2. calculate_fast_kuzushi
    # ------------------------------------------------------------------
    def calculate_fast_kuzushi(self, kpts):
        """Takes a (17, 3) numpy array. Returns (com, closest_point, distance, is_kuzushi)."""
        flat = kpts.flatten().tolist()
        result = self._mod.calculate_fast_kuzushi(flat)
        com = np.array([result[0], result[1]])
        cp = np.array([result[2], result[3]])
        distance = float(result[4])
        is_kuzushi = bool(result[5] > 0.5)
        return com, cp, distance, is_kuzushi
    
    # ------------------------------------------------------------------
    # 3. compute_cost_matrix
    # ------------------------------------------------------------------
    def compute_cost_matrix(self, targets, candidates, w):
        """
        targets: list of dicts with 'id', 'pure_id', 'box' keys
        candidates: list of dicts with 'id', 'box' keys
        Returns: numpy cost matrix of shape (len(targets), len(candidates))
        """
        targets_flat = []
        for t in targets:
            targets_flat.extend([
                float(t['id']),
                float(t.get('pure_id', t['id'])),
                float(t['box'][0]), float(t['box'][1]),
                float(t['box'][2]), float(t['box'][3])
            ])
        
        candidates_flat = []
        for c in candidates:
            candidates_flat.extend([
                float(c['id']),
                float(c['box'][0]), float(c['box'][1]),
                float(c['box'][2]), float(c['box'][3])
            ])
        
        nt = len(targets)
        nc = len(candidates)
        
        # Single-arg pattern: pass as a list of [targets_flat, candidates_flat, nt, nc, w]
        costs_flat = self._mod.compute_cost_matrix([
            targets_flat, candidates_flat, nt, nc, float(w)
        ])
        
        cost_matrix = np.zeros((nt, nc))
        for i in range(nt):
            for j in range(nc):
                cost_matrix[i, j] = float(costs_flat[i * nc + j])
        return cost_matrix
    
    # ------------------------------------------------------------------
    # 4. score_foreground_pair
    # ------------------------------------------------------------------
    def score_foreground_pair(self, b1, b2, w, h):
        """Score a pair of bounding boxes. Returns (score, should_skip)."""
        # Single-arg pattern
        result = self._mod.score_foreground_pair([
            float(b1[0]), float(b1[1]), float(b1[2]), float(b1[3]),
            float(b2[0]), float(b2[1]), float(b2[2]), float(b2[3]),
            float(w), float(h)
        ])
        return float(result[0]), bool(result[1] > 0.5)
    
    # ------------------------------------------------------------------
    # 5. detect_kinematic_events
    # ------------------------------------------------------------------
    def detect_kinematic_events(self, smooth_heights, smooth_tops, max_ar_arr,
                                 melded_flags, standing_h, fps, total_frames):
        """Detect takedown events from smoothed kinematic data.
        Returns list of dicts with 'impact_frame', 'transition_frame', 'severity'.
        """
        # Single-arg pattern
        result = self._mod.detect_kinematic_events([
            smooth_heights.tolist(),
            smooth_tops.tolist(),
            max_ar_arr if isinstance(max_ar_arr, list) else list(max_ar_arr),
            melded_flags if isinstance(melded_flags, list) else list(melded_flags),
            float(standing_h),
            float(fps),
            int(total_frames)
        ])
        
        events = []
        result_list = list(result)
        for i in range(0, len(result_list), 3):
            events.append({
                'impact_frame': int(result_list[i]),
                'transition_frame': int(result_list[i + 1]),
                'severity': float(result_list[i + 2])
            })
        return events
    
    # ------------------------------------------------------------------
    # 6. update_skeleton_ema
    # ------------------------------------------------------------------
    def update_skeleton_ema(self, history, kpts, alpha=0.75):
        """EMA update for skeleton keypoints. Returns smoothed (N, 3) numpy array or None."""
        has_history = history is not None
        kpts_is_none = kpts is None
        num_kp = 17
        
        if has_history:
            history_flat = history.flatten().tolist()
            num_kp = len(history_flat) // 3
        else:
            history_flat = [0.0] * (num_kp * 3)
        
        if not kpts_is_none:
            kpts_flat = kpts.flatten().tolist()
            num_kp = len(kpts_flat) // 3
        else:
            kpts_flat = [0.0] * (num_kp * 3)
        
        # Single-arg pattern
        result = self._mod.update_skeleton_ema([
            history_flat, kpts_flat, num_kp, float(alpha),
            1.0 if has_history else 0.0,
            1.0 if kpts_is_none else 0.0
        ])
        
        result_list = list(result)
        if len(result_list) == 0:
            return None
        
        return np.array(result_list, dtype=np.float64).reshape(-1, 3)
