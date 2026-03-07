"""Mat Homography — Perspective Transform for Mat-Space Coordinates.

Computes a homography matrix from detected mat boundary lines, enabling
pixel coordinates to be mapped to real-world mat-plane coordinates.

This AUGMENTS existing pixel-space heuristics (Z-depth, orbit-radius, etc.)
with true spatial reasoning: on-mat/off-mat classification, real distances,
and perspective-corrected position tracking.

Usage:
    mat_H = MatHomography()
    mat_H.compute(first_frame, w, h)   # Compute once per match segment
    
    if mat_H.available:
        pos = mat_H.pixel_to_mat(foot_x, foot_y)   # → (mat_x, mat_y) in meters
        on_mat = mat_H.is_on_mat(foot_x, foot_y)    # → True/False
        dist = mat_H.mat_distance(x1, y1, x2, y2)   # → meters
"""

import cv2
import math
import numpy as np


# Standard competition mat sizes (meters)
DEFAULT_MAT_WIDTH = 10.0   # 10m wide
DEFAULT_MAT_HEIGHT = 10.0  # 10m deep


class MatHomography:
    """Perspective transform from pixel space to mat-plane coordinates."""
    
    def __init__(self, mat_width=DEFAULT_MAT_WIDTH, mat_height=DEFAULT_MAT_HEIGHT):
        self.mat_w = mat_width
        self.mat_h = mat_height
        self.H = None            # 3×3 homography matrix (pixel → mat)
        self.H_inv = None        # 3×3 inverse (mat → pixel) for visualization
        self.quad = None         # Detected mat corners in pixel space (4×2)
        self.available = False   # True if homography was successfully computed
    
    def compute(self, frame, w=None, h=None):
        """Compute the homography from a video frame.
        
        1. Detect mat boundary lines (HoughLinesP on mat region)
        2. Find 4 mat corners by intersecting line families
        3. Build perspective transform mapping quad → rectified rectangle
        
        Returns: True if successful, False if detection failed.
        """
        if w is None:
            h, w = frame.shape[:2]
        
        quad = self._detect_mat_quad(frame, w, h)
        if quad is None:
            self.available = False
            return False
        
        self.quad = quad
        
        # Rectified mat corners: a rectangle in mat-space (meters)
        # Order: top-left, top-right, bottom-right, bottom-left
        rectified = np.float32([
            [0, 0],
            [self.mat_w, 0],
            [self.mat_w, self.mat_h],
            [0, self.mat_h]
        ])
        
        self.H = cv2.getPerspectiveTransform(quad, rectified)
        self.H_inv = cv2.getPerspectiveTransform(rectified, quad)
        self.available = True
        return True
    
    def pixel_to_mat(self, px, py):
        """Transform pixel coordinates to mat-plane coordinates (meters).
        
        Args:
            px, py: pixel coordinates (typically foot position = bottom of bbox)
        
        Returns:
            (mat_x, mat_y) in meters, or None if homography unavailable.
        """
        if not self.available:
            return None
        
        pt = np.float32([[[px, py]]])
        transformed = cv2.perspectiveTransform(pt, self.H)
        return (float(transformed[0][0][0]), float(transformed[0][0][1]))
    
    def is_on_mat(self, px, py, margin=0.5):
        """Check if a pixel position maps to a point on the mat.
        
        Args:
            px, py: pixel coordinates (foot position)
            margin: meters of tolerance outside mat boundary (default 0.5m)
                    Allows for athletes near the edge.
        
        Returns:
            True if position is on/near the mat, False if clearly off-mat,
            None if homography unavailable.
        """
        pos = self.pixel_to_mat(px, py)
        if pos is None:
            return None
        
        mx, my = pos
        return (-margin <= mx <= self.mat_w + margin and 
                -margin <= my <= self.mat_h + margin)
    
    def mat_distance(self, px1, py1, px2, py2):
        """Compute real-world distance between two pixel positions (in meters).
        
        Returns: distance in meters, or None if homography unavailable.
        """
        p1 = self.pixel_to_mat(px1, py1)
        p2 = self.pixel_to_mat(px2, py2)
        if p1 is None or p2 is None:
            return None
        
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
    
    def mat_edge_distance(self, px, py):
        """Compute minimum distance from a point to the mat edge (in meters).
        
        Returns: distance to nearest edge (positive = inside, negative = outside),
                 or None if unavailable.
        """
        pos = self.pixel_to_mat(px, py)
        if pos is None:
            return None
        
        mx, my = pos
        # Distance to each edge
        d_left = mx
        d_right = self.mat_w - mx
        d_top = my
        d_bottom = self.mat_h - my
        
        min_dist = min(d_left, d_right, d_top, d_bottom)
        return min_dist  # Negative if outside the mat
    
    def on_mat_percentage(self, boxes, frame_h):
        """Compute what percentage of frames an entity's feet are on the mat.
        
        Args:
            boxes: list of bounding boxes [(x1,y1,x2,y2), ...]
            frame_h: frame height (for foot position = y2)
        
        Returns: float 0.0-1.0, or None if homography unavailable.
        """
        if not self.available or not boxes:
            return None
        
        on_count = 0
        for box in boxes:
            foot_x = (box[0] + box[2]) / 2.0  # Center X
            foot_y = box[3]                     # Bottom of bbox = feet
            if self.is_on_mat(foot_x, foot_y):
                on_count += 1
        
        return on_count / len(boxes)
    
    def _detect_mat_quad(self, frame, w, h):
        """Detect 4 corners of the mat from HoughLinesP line families.
        
        Strategy:
        1. Edge-detect the mat region (lower 70% of frame)
        2. Find two dominant line families (by angle clustering)
        3. Intersect the families to find 4 corner points
        4. Order corners: TL, TR, BR, BL (counter-clockwise from top-left)
        
        Returns: np.float32 array of shape (4,2), or None if failed.
        """
        # Focus on mat region
        mat_top = int(h * 0.25)
        mat_region = frame[mat_top:, :]
        
        filtered = cv2.bilateralFilter(mat_region, 9, 75, 75)
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        
        median_val = np.median(gray)
        low_thresh = int(max(30, 0.5 * median_val))
        high_thresh = int(min(250, 1.5 * median_val))
        edges = cv2.Canny(gray, low_thresh, high_thresh, apertureSize=3)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                                minLineLength=int(w * 0.12),
                                maxLineGap=int(w * 0.06))
        
        if lines is None or len(lines) < 4:
            return None
        
        # Collect candidate lines
        candidates = []
        for line in lines:
            x1, y1_local, x2, y2_local = line[0]
            y1 = y1_local + mat_top
            y2 = y2_local + mat_top
            
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            length = math.hypot(x2 - x1, y2 - y1)
            
            if length < w * 0.08:
                continue
            if abs(angle) > 85 or abs(angle) < 2:
                continue
            
            candidates.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'angle': angle, 'length': length
            })
        
        if len(candidates) < 4:
            return None
        
        # Split into two angle families
        candidates.sort(key=lambda c: c['angle'])
        family_a = [c for c in candidates if c['angle'] >= 0]
        family_b = [c for c in candidates if c['angle'] < 0]
        
        # Need at least 2 lines in each family
        if len(family_a) < 2 and len(family_b) >= 4:
            mid = len(family_b) // 2
            family_a = family_b[mid:]
            family_b = family_b[:mid]
        elif len(family_b) < 2 and len(family_a) >= 4:
            mid = len(family_a) // 2
            family_b = family_a[mid:]
            family_a = family_a[:mid]
        
        if len(family_a) < 2 or len(family_b) < 2:
            return None
        
        # Take 2 strongest from each family (by length)
        family_a.sort(key=lambda c: c['length'], reverse=True)
        family_b.sort(key=lambda c: c['length'], reverse=True)
        
        lines_a = family_a[:2]  # Two lines from family A
        lines_b = family_b[:2]  # Two lines from family B
        
        # Find 4 intersection points (each A line × each B line)
        intersections = []
        for la in lines_a:
            for lb in lines_b:
                pt = self._line_intersection(
                    la['x1'], la['y1'], la['x2'], la['y2'],
                    lb['x1'], lb['y1'], lb['x2'], lb['y2']
                )
                if pt is not None:
                    px, py = pt
                    # Reject intersections far outside the frame
                    if -w * 0.5 < px < w * 1.5 and -h * 0.5 < py < h * 1.5:
                        intersections.append(pt)
        
        if len(intersections) < 4:
            return None
        
        # Take the 4 intersections closest to the frame center as mat corners
        cx, cy = w / 2.0, h / 2.0
        intersections.sort(key=lambda p: math.hypot(p[0] - cx, p[1] - cy))
        corners = intersections[:4]
        
        # Order corners: TL, TR, BR, BL
        corners = self._order_corners(corners)
        
        return np.float32(corners)
    
    @staticmethod
    def _line_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
        """Find intersection point of two line segments (extended to infinity).
        
        Returns: (x, y) tuple, or None if lines are parallel.
        """
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None  # Parallel lines
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        return (px, py)
    
    @staticmethod
    def _order_corners(corners):
        """Order 4 corner points as: top-left, top-right, bottom-right, bottom-left.
        
        Uses centroid-based angle sorting for robust ordering.
        """
        # Compute centroid
        cx = sum(p[0] for p in corners) / 4.0
        cy = sum(p[1] for p in corners) / 4.0
        
        # Sort by angle from centroid
        def angle_from_center(p):
            return math.atan2(p[1] - cy, p[0] - cx)
        
        sorted_pts = sorted(corners, key=angle_from_center)
        
        # atan2 gives angles -π to π, with:
        # top-left ≈ -135° (−3π/4), top-right ≈ -45° (−π/4)
        # bottom-right ≈ 45° (π/4), bottom-left ≈ 135° (3π/4)
        # After sorting: TL, TR, BR, BL
        return sorted_pts
