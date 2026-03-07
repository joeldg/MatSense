from math import sqrt
from collections import List
from python import PythonObject, Python
from python.bindings import PythonModuleBuilder
from os import abort


# ==============================================================================
# 1. BB_IOU — Bounding Box Intersection-Over-Union
# ==============================================================================
fn _bb_iou(
    a_x1: Float64, a_y1: Float64, a_x2: Float64, a_y2: Float64,
    b_x1: Float64, b_y1: Float64, b_x2: Float64, b_y2: Float64,
) -> Float64:
    var xA = a_x1 if a_x1 > b_x1 else b_x1
    var yA = a_y1 if a_y1 > b_y1 else b_y1
    var xB = a_x2 if a_x2 < b_x2 else b_x2
    var yB = a_y2 if a_y2 < b_y2 else b_y2

    var inter_w = xB - xA
    if inter_w < 0.0:
        inter_w = 0.0
    var inter_h = yB - yA
    if inter_h < 0.0:
        inter_h = 0.0
    var inter_area = inter_w * inter_h

    var area_a = (a_x2 - a_x1) * (a_y2 - a_y1)
    var area_b = (b_x2 - b_x1) * (b_y2 - b_y1)
    return inter_area / (area_a + area_b - inter_area + 1e-5)


fn bb_iou(box_a: PythonObject, box_b: PythonObject) raises -> PythonObject:
    """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
    var a_x1 = Float64(box_a[0])
    var a_y1 = Float64(box_a[1])
    var a_x2 = Float64(box_a[2])
    var a_y2 = Float64(box_a[3])
    var b_x1 = Float64(box_b[0])
    var b_y1 = Float64(box_b[1])
    var b_x2 = Float64(box_b[2])
    var b_y2 = Float64(box_b[3])
    return _bb_iou(a_x1, a_y1, a_x2, a_y2, b_x1, b_y1, b_x2, b_y2)


# ==============================================================================
# 2. CALCULATE_FAST_KUZUSHI — Biomechanics Balance Detection
# ==============================================================================
fn _kuzushi_core(
    p11x: Float64, p11y: Float64, p12x: Float64, p12y: Float64,
    p5x: Float64, p5y: Float64, p6x: Float64, p6y: Float64,
    p15x: Float64, p15y: Float64, p16x: Float64, p16y: Float64,
) -> List[Float64]:
    """Returns [com_x, com_y, cp_x, cp_y, distance_to_base, is_kuzushi (0/1)]."""
    var pelvis_x = (p11x + p12x) / 2.0
    var pelvis_y = (p11y + p12y) / 2.0
    var neck_x = (p5x + p6x) / 2.0
    var neck_y = (p5y + p6y) / 2.0

    var com_x = (pelvis_x * 0.6) + (neck_x * 0.4)
    var com_y = (pelvis_y * 0.6) + (neck_y * 0.4)

    var dx_stance = p15x - p16x
    var dy_stance = p15y - p16y
    var stance_width = sqrt((dx_stance * dx_stance) + (dy_stance * dy_stance))

    var ab_x = p15x - p16x
    var ab_y = p15y - p16y
    var ap_x = com_x - p16x
    var ap_y = com_y - p16y

    var dot_ab = (ab_x * ab_x) + (ab_y * ab_y)
    var t: Float64 = 0.0
    if dot_ab != 0.0:
        t = ((ap_x * ab_x) + (ap_y * ab_y)) / dot_ab
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0

    var cp_x = p16x + (t * ab_x)
    var cp_y = p16y + (t * ab_y)

    var dist_x = com_x - cp_x
    var dist_y = com_y - cp_y
    var dist_sq = (dist_x * dist_x) + (dist_y * dist_y)

    var dynamic_threshold = 15.0 + (stance_width * 0.30)
    var is_kuzushi: Float64 = 0.0
    if dist_sq > dynamic_threshold * dynamic_threshold:
        is_kuzushi = 1.0

    var distance = sqrt(dist_sq)

    var result = List[Float64]()
    result.append(com_x)
    result.append(com_y)
    result.append(cp_x)
    result.append(cp_y)
    result.append(distance)
    result.append(is_kuzushi)
    return result


fn calculate_fast_kuzushi(flat_kpts: PythonObject) raises -> PythonObject:
    """Takes a flat list of 51 Float64s (17 YOLO joints * 3).
    Returns [com_x, com_y, cp_x, cp_y, distance, is_kuzushi].
    """
    var p11x = Float64(flat_kpts[11 * 3 + 0])
    var p11y = Float64(flat_kpts[11 * 3 + 1])
    var p12x = Float64(flat_kpts[12 * 3 + 0])
    var p12y = Float64(flat_kpts[12 * 3 + 1])
    var p5x = Float64(flat_kpts[5 * 3 + 0])
    var p5y = Float64(flat_kpts[5 * 3 + 1])
    var p6x = Float64(flat_kpts[6 * 3 + 0])
    var p6y = Float64(flat_kpts[6 * 3 + 1])
    var p15x = Float64(flat_kpts[15 * 3 + 0])
    var p15y = Float64(flat_kpts[15 * 3 + 1])
    var p16x = Float64(flat_kpts[16 * 3 + 0])
    var p16y = Float64(flat_kpts[16 * 3 + 1])

    var result = _kuzushi_core(p11x, p11y, p12x, p12y, p5x, p5y, p6x, p6y, p15x, p15y, p16x, p16y)

    var py = Python.import_module("builtins")
    var py_list = py.list()
    for i in range(len(result)):
        py_list.append(result[i])
    return py_list


# ==============================================================================
# 3. COMPUTE_COST_MATRIX — With anti-teleport & cross-assignment penalties
# ==============================================================================
fn compute_cost_matrix(args: PythonObject) raises -> PythonObject:
    """Compute cost matrix for Hungarian assignment.
    
    args is a Python list: [targets_flat, candidates_flat, num_targets, num_candidates, frame_width]
    targets_flat: [id, pure_id, x1, y1, x2, y2] * num_targets (flat list)
    candidates_flat: [id, x1, y1, x2, y2] * num_candidates (flat list)
    Returns: flat list of costs, row-major [num_targets * num_candidates].
    """
    var targets_flat = args[0]
    var candidates_flat = args[1]
    var nt = Int(args[2])
    var nc = Int(args[3])
    var w = Float64(args[4])

    var py = Python.import_module("builtins")
    var costs = py.list()

    for i in range(nt):
        var t_offset = i * 6
        var t_id = Float64(targets_flat[t_offset + 0])
        var t_pure_id = Float64(targets_flat[t_offset + 1])
        var t_x1 = Float64(targets_flat[t_offset + 2])
        var t_y1 = Float64(targets_flat[t_offset + 3])
        var t_x2 = Float64(targets_flat[t_offset + 4])
        var t_y2 = Float64(targets_flat[t_offset + 5])
        var tcx = (t_x1 + t_x2) / 2.0
        var tcy = (t_y1 + t_y2) / 2.0

        # Get the OTHER target's centroid for cross-assignment penalty
        var other_idx = 1 - i
        if other_idx < 0:
            other_idx = 0
        if other_idx >= nt:
            other_idx = nt - 1
        var o_offset = other_idx * 6
        var ocx = (Float64(targets_flat[o_offset + 2]) + Float64(targets_flat[o_offset + 4])) / 2.0
        var ocy = (Float64(targets_flat[o_offset + 3]) + Float64(targets_flat[o_offset + 5])) / 2.0

        for j in range(nc):
            var c_offset = j * 5
            var c_id = Float64(candidates_flat[c_offset + 0])
            var c_x1 = Float64(candidates_flat[c_offset + 1])
            var c_y1 = Float64(candidates_flat[c_offset + 2])
            var c_x2 = Float64(candidates_flat[c_offset + 3])
            var c_y2 = Float64(candidates_flat[c_offset + 4])
            var ccx = (c_x1 + c_x2) / 2.0
            var ccy = (c_y1 + c_y2) / 2.0

            var cost: Float64 = 0.0

            # ID match bonuses
            if c_id == t_id:
                cost = cost - 200.0
            if c_id == t_pure_id:
                cost = cost - 300.0

            var dx = tcx - ccx
            var dy = tcy - ccy
            var dist = sqrt(dx * dx + dy * dy)

            # ANTI-TELEPORT: massive penalty if jump > 30% of frame width
            if dist > w * 0.30:
                cost = cost + 5000.0
            elif dist > w * 0.25:
                cost = cost + 2000.0
            cost = cost + (dist / w) * 200.0

            # CROSS-ASSIGNMENT PENALTY: if candidate is closer to the OTHER
            # profile than to this target, penalize to prevent swaps
            var odx = ocx - ccx
            var ody = ocy - ccy
            var dist_to_other = sqrt(odx * odx + ody * ody)
            if dist_to_other < dist * 0.5:
                cost = cost + 500.0

            var iou = _bb_iou(t_x1, t_y1, t_x2, t_y2, c_x1, c_y1, c_x2, c_y2)
            cost = cost - (iou * 200.0)

            costs.append(cost)

    return costs


# ==============================================================================
# 4. SCORE_FOREGROUND_PAIR — Single args list approach
# ==============================================================================
fn score_foreground_pair(args: PythonObject) raises -> PythonObject:
    """Score a pair of bounding boxes for foreground dominance.
    args: [b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2, frame_w, frame_h]
    Returns [score, should_skip (0.0 or 1.0)]
    """
    var x1a = Float64(args[0])
    var y1a = Float64(args[1])
    var x2a = Float64(args[2])
    var y2a = Float64(args[3])
    var x1b = Float64(args[4])
    var y1b = Float64(args[5])
    var x2b = Float64(args[6])
    var y2b = Float64(args[7])
    var w = Float64(args[8])
    var h = Float64(args[9])

    var h1 = y2a - y1a
    var h2 = y2b - y1b

    var py = Python.import_module("builtins")
    var result = py.list()

    var max_h = h1 if h1 > h2 else h2
    var min_h = h1 if h1 < h2 else h2
    if max_h / (min_h + 1e-5) > 1.8:
        result.append(0.0)
        result.append(1.0)
        return result

    var cx1 = (x1a + x2a) / 2.0
    var cy1 = (y1a + y2a) / 2.0
    var cx2 = (x1b + x2b) / 2.0
    var cy2 = (y1b + y2b) / 2.0
    var dx = cx1 - cx2
    var dy = cy1 - cy2
    var dist = sqrt(dx * dx + dy * dy)
    if dist > max_h * 2.5:
        result.append(0.0)
        result.append(1.0)
        return result

    var overlap_x: Float64 = 0.0
    var min_x2 = x2a if x2a < x2b else x2b
    var max_x1 = x1a if x1a > x1b else x1b
    if min_x2 > max_x1:
        overlap_x = min_x2 - max_x1
    var overlap_y: Float64 = 0.0
    var min_y2 = y2a if y2a < y2b else y2b
    var max_y1 = y1a if y1a > y1b else y1b
    if min_y2 > max_y1:
        overlap_y = min_y2 - max_y1

    var fg_y = y2a if y2a < y2b else y2b
    fg_y = fg_y / h
    var size_h = max_h / h
    var avg_cx = (cx1 + cx2) / 2.0
    var center_dist = avg_cx - (w / 2.0)
    if center_dist < 0.0:
        center_dist = -center_dist
    var center = 1.0 - (center_dist / (w / 2.0))

    var aspect1 = (x2a - x1a) / (h1 + 1e-5)
    var aspect2 = (x2b - x1b) / (h2 + 1e-5)
    var ref_penalty: Float64 = 0.0
    if aspect1 < 0.6 or aspect2 < 0.6:
        ref_penalty = -50.0

    # LEFT/RIGHT APPROACH: Players typically come from opposite sides
    var mid_x = w / 2.0
    var lr_bonus: Float64 = 0.0
    if (cx1 < mid_x and cx2 > mid_x) or (cx1 > mid_x and cx2 < mid_x):
        lr_bonus = 25.0

    var score = (fg_y * fg_y * fg_y) * 75.0 + (size_h * size_h) * 15.0 + (center * 75.0) + ref_penalty + lr_bonus
    if overlap_x * overlap_y > 0.0:
        score = score + 15.0

    result.append(score)
    result.append(0.0)
    return result


# ==============================================================================
# 5. DETECT_KINEMATIC_EVENTS — Single args list approach
# ==============================================================================
fn detect_kinematic_events(args: PythonObject) raises -> PythonObject:
    """Scan smoothed kinematic data for takedown events.
    args: [smooth_heights, smooth_tops, max_ar_arr, melded_flags, standing_h, fps, total_frames]
    Returns flat list: [impact_frame, transition_frame, severity, ...] for each event.
    """
    var smooth_heights = args[0]
    var smooth_tops = args[1]
    var max_ar_arr = args[2]
    var melded_flags = args[3]
    var s_h = Float64(args[4])
    var f = Float64(args[5])
    var n = Int(args[6])

    var py = Python.import_module("builtins")
    var impacts = py.list()

    var is_standing: Bool = True
    var ground_frames: Int = 0

    for i in range(n):
        var c_h = Float64(smooth_heights[i])
        var max_ar = Float64(max_ar_arr[i])
        var melded_val = Float64(melded_flags[i])
        var melded: Bool = melded_val > 0.5

        if c_h < s_h * 0.60 or max_ar > 1.35 or melded:
            ground_frames += 1
        else:
            if c_h > s_h * 0.8 and max_ar < 1.0 and not melded:
                is_standing = True
                ground_frames = 0
            elif ground_frames < Int(f * 1.0):
                ground_frames = 0

        if is_standing and ground_frames >= Int(f * 1.5):
            var impact_frame = i - Int(f * 1.5)

            var transition_f = impact_frame
            var lookback_limit = impact_frame - Int(f * 6.0)
            if lookback_limit < 0:
                lookback_limit = 0
            var fb = impact_frame
            while fb > lookback_limit:
                var fb_h = Float64(smooth_heights[fb])
                var fb_ar = Float64(max_ar_arr[fb])
                if fb_h > s_h * 0.8 and fb_ar < 1.0:
                    transition_f = fb
                    break
                fb -= 1

            var severity = Float64(smooth_tops[impact_frame]) - Float64(smooth_tops[transition_f])

            impacts.append(impact_frame)
            impacts.append(transition_f)
            impacts.append(severity)
            is_standing = False

    return impacts


# ==============================================================================
# 6. UPDATE_SKELETON_EMA — Single args list approach
# ==============================================================================
fn update_skeleton_ema(args: PythonObject) raises -> PythonObject:
    """EMA update for skeleton keypoints.
    args: [history_flat, kpts_flat, num_keypoints, alpha, has_history, kpts_is_none]
    Flat lists of [x, y, confidence] * num_keypoints.
    Returns flat list of smoothed keypoints.
    """
    var history_flat = args[0]
    var kpts_flat = args[1]
    var nk = Int(args[2])
    var a = Float64(args[3])
    var has_hist = Float64(args[4]) > 0.5
    var kpts_none = Float64(args[5]) > 0.5

    var py = Python.import_module("builtins")
    var result = py.list()

    if kpts_none and has_hist:
        for i in range(nk):
            var hx = Float64(history_flat[i * 3 + 0])
            var hy = Float64(history_flat[i * 3 + 1])
            var hc = Float64(history_flat[i * 3 + 2]) * 0.8
            result.append(hx)
            result.append(hy)
            result.append(hc)
        return result

    if kpts_none and not has_hist:
        return result

    if not has_hist:
        for i in range(nk * 3):
            result.append(Float64(kpts_flat[i]))
        return result

    for i in range(nk):
        var kx = Float64(kpts_flat[i * 3 + 0])
        var ky = Float64(kpts_flat[i * 3 + 1])
        var kc = Float64(kpts_flat[i * 3 + 2])
        var hx = Float64(history_flat[i * 3 + 0])
        var hy = Float64(history_flat[i * 3 + 1])
        var hc = Float64(history_flat[i * 3 + 2])

        if kc > 0.30 and hc > 0.30:
            var dx = kx - hx
            var dy = ky - hy
            var jump = sqrt(dx * dx + dy * dy)
            var dynamic_alpha = a if jump < 50.0 else 1.0
            result.append(dynamic_alpha * kx + (1.0 - dynamic_alpha) * hx)
            result.append(dynamic_alpha * ky + (1.0 - dynamic_alpha) * hy)
            result.append(kc)
        elif kc > 0.30:
            result.append(kx)
            result.append(ky)
            result.append(kc)
        else:
            result.append(hx)
            result.append(hy)
            result.append(hc * 0.8)

    return result


# ==============================================================================
# 7. COMPUTE_Z_DEPTH — Faux-3D Z-Depth Score from 2D Bounding Box
# ==============================================================================
fn compute_z_depth(args: PythonObject) raises -> PythonObject:
    """Compute faux-3D Z-depth score from a bounding box.
    
    Z-Depth = normalized_area * foot_y_normalized³
    
    args: [x1, y1, x2, y2, frame_height]
    Returns: Float64 score in [0, 1]. Higher = closer to camera.
    """
    var x1 = Float64(args[0])
    var y1 = Float64(args[1])
    var x2 = Float64(args[2])
    var y2 = Float64(args[3])
    var h = Float64(args[4])

    var bw = x2 - x1
    var bh = y2 - y1
    # Normalize area by frame_height² for scale invariance
    var area_norm = (bw * bh) / (h * h)
    # foot_y = bottom of box = where feet touch floor, normalized [0,1]
    var foot_y_norm = y2 / h
    # Cubic exponent aggressively weights proximity:
    # y=0.9 → 0.729, y=0.5 → 0.125 → 5.8x difference
    return area_norm * foot_y_norm * foot_y_norm * foot_y_norm


# ==============================================================================
# 8. CLASSIFY_REF_ARM_SIGNAL — Referee Arm Posture from COCO Keypoints
# ==============================================================================
fn classify_ref_arm_signal(args: PythonObject) raises -> PythonObject:
    """Classify referee arm signal from flat keypoint array.
    
    args: flat list of 17*3 = 51 floats [x0,y0,c0, x1,y1,c1, ...]
    Returns: [signal_code, confidence, l_angle, r_angle]
       signal_code: 0=none, 1=arms_up, 2=arm_raised, 3=arm_angled, 4=arm_horizontal, 5=arms_crossed.
    """
    var py = Python.import_module("builtins")
    var math_mod = Python.import_module("math")
    
    var n = Int(py.len(args))
    if n < 33:  # Need at least keypoints 0-10 (11 * 3 = 33)
        return py.list([0, 0.0, 0.0, 0.0])
    
    var CONF_MIN: Float64 = 0.3
    
    # Extract keypoints: nose(0), l_shoulder(5), r_shoulder(6), l_elbow(7), r_elbow(8), l_wrist(9), r_wrist(10)
    var nose_x = Float64(args[0]); var nose_y = Float64(args[1]); var nose_c = Float64(args[2])
    var ls_x = Float64(args[15]); var ls_y = Float64(args[16]); var ls_c = Float64(args[17])
    var rs_x = Float64(args[18]); var rs_y = Float64(args[19]); var rs_c = Float64(args[20])
    var le_x = Float64(args[21]); var le_y = Float64(args[22]); var le_c = Float64(args[23])
    var re_x = Float64(args[24]); var re_y = Float64(args[25]); var re_c = Float64(args[26])
    var lw_x = Float64(args[27]); var lw_y = Float64(args[28]); var lw_c = Float64(args[29])
    var rw_x = Float64(args[30]); var rw_y = Float64(args[31]); var rw_c = Float64(args[32])
    
    var has_left = ls_c > CONF_MIN and le_c > CONF_MIN and lw_c > CONF_MIN
    var has_right = rs_c > CONF_MIN and re_c > CONF_MIN and rw_c > CONF_MIN
    
    if not has_left and not has_right:
        return py.list([0, 0.0, 0.0, 0.0])
    
    # Arm angle from horizontal (0=horizontal, 90=vertical up)
    var l_angle: Float64 = -90.0
    var r_angle: Float64 = -90.0
    var l_ext: Float64 = 0.0
    var r_ext: Float64 = 0.0
    
    if has_left:
        var dx = lw_x - ls_x
        var dy = ls_y - lw_y  # inverted Y
        var abs_dx = dx if dx > 0 else -dx
        l_angle = Float64(math_mod.degrees(math_mod.atan2(dy, abs_dx + 1e-5)))
        var full_len = sqrt((lw_x-ls_x)*(lw_x-ls_x) + (lw_y-ls_y)*(lw_y-ls_y))
        var seg_len = sqrt((le_x-ls_x)*(le_x-ls_x) + (le_y-ls_y)*(le_y-ls_y)) + sqrt((lw_x-le_x)*(lw_x-le_x) + (lw_y-le_y)*(lw_y-le_y))
        l_ext = full_len / (seg_len + 1e-5)
    
    if has_right:
        var dx = rw_x - rs_x
        var dy = rs_y - rw_y
        var abs_dx = dx if dx > 0 else -dx
        r_angle = Float64(math_mod.degrees(math_mod.atan2(dy, abs_dx + 1e-5)))
        var full_len = sqrt((rw_x-rs_x)*(rw_x-rs_x) + (rw_y-rs_y)*(rw_y-rs_y))
        var seg_len = sqrt((re_x-rs_x)*(re_x-rs_x) + (re_y-rs_y)*(re_y-rs_y)) + sqrt((rw_x-re_x)*(rw_x-re_x) + (rw_y-re_y)*(rw_y-re_y))
        r_ext = full_len / (seg_len + 1e-5)
    
    var l_above_head = has_left and nose_c > CONF_MIN and lw_y < nose_y - 20
    var r_above_head = has_right and nose_c > CONF_MIN and rw_y < nose_y - 20
    
    var max_angle = l_angle if l_angle > r_angle else r_angle
    var max_ext = l_ext if l_ext > r_ext else r_ext
    
    var signal_code: Int = 0
    var confidence: Float64 = 0.0
    
    var both_high = l_angle > 70 and r_angle > 70
    var one_high = l_angle > 70 or r_angle > 70
    
    if both_high and l_above_head and r_above_head:
        signal_code = 1  # ARMS_UP
        confidence = 0.7 + max_ext * 0.25
        if confidence > 0.95: confidence = 0.95
    elif one_high and (l_above_head or r_above_head) and max_ext > 0.75:
        signal_code = 2  # ARM_RAISED
        confidence = 0.6 + max_ext * 0.2
        if confidence > 0.85: confidence = 0.85
    elif max_angle > 30 and max_angle < 70 and max_ext > 0.6:
        signal_code = 3  # ARM_ANGLED
        confidence = 0.4 + max_ext * 0.2
        if confidence > 0.7: confidence = 0.7
    elif max_angle >= -10 and max_angle <= 30 and max_ext > 0.7:
        signal_code = 4  # ARM_HORIZONTAL
        confidence = 0.4 + max_ext * 0.15
        if confidence > 0.65: confidence = 0.65
    
    return py.list([signal_code, confidence, l_angle, r_angle])


# ==============================================================================
# 9. PIXEL_TO_MAT — Apply 3x3 Homography Transform
# ==============================================================================
fn pixel_to_mat(args: PythonObject) raises -> PythonObject:
    """Transform pixel coords to mat-space via 3x3 homography.
    
    args: [px, py, h00, h01, h02, h10, h11, h12, h20, h21, h22]
    Returns: [mat_x, mat_y]
    """
    var py = Python.import_module("builtins")
    
    var px = Float64(args[0])
    var py_coord = Float64(args[1])
    var h00 = Float64(args[2]); var h01 = Float64(args[3]); var h02 = Float64(args[4])
    var h10 = Float64(args[5]); var h11 = Float64(args[6]); var h12 = Float64(args[7])
    var h20 = Float64(args[8]); var h21 = Float64(args[9]); var h22 = Float64(args[10])
    
    var denom = h20 * px + h21 * py_coord + h22
    if denom == 0.0:
        denom = 1e-10
    
    var mat_x = (h00 * px + h01 * py_coord + h02) / denom
    var mat_y = (h10 * px + h11 * py_coord + h12) / denom
    
    return py.list([mat_x, mat_y])


# ==============================================================================
# 10. IS_ON_MAT — Check if transformed point is within mat bounds
# ==============================================================================
fn is_on_mat(args: PythonObject) raises -> PythonObject:
    """Check if pixel coords map to a point on the mat.
    
    args: [px, py, h00..h22, mat_w, mat_h, margin]
    Returns: 1 if on mat, 0 if off mat
    """
    var py = Python.import_module("builtins")
    
    var px = Float64(args[0])
    var py_coord = Float64(args[1])
    var h00 = Float64(args[2]); var h01 = Float64(args[3]); var h02 = Float64(args[4])
    var h10 = Float64(args[5]); var h11 = Float64(args[6]); var h12 = Float64(args[7])
    var h20 = Float64(args[8]); var h21 = Float64(args[9]); var h22 = Float64(args[10])
    var mat_w = Float64(args[11])
    var mat_h = Float64(args[12])
    var margin = Float64(args[13])
    
    var denom = h20 * px + h21 * py_coord + h22
    if denom == 0.0:
        denom = 1e-10
    
    var mat_x = (h00 * px + h01 * py_coord + h02) / denom
    var mat_y = (h10 * px + h11 * py_coord + h12) / denom
    
    var on = mat_x >= -margin and mat_x <= mat_w + margin and mat_y >= -margin and mat_y <= mat_h + margin
    
    if on:
        return py.list([1])
    return py.list([0])


# ==============================================================================
# PyInit — Expose all functions to Python via PythonModuleBuilder
# ==============================================================================
@export
fn PyInit_mojo_analytics() -> PythonObject:
    try:
        var m = PythonModuleBuilder("mojo_analytics")
        m.def_function[bb_iou]("bb_iou", docstring="Compute IoU between two bounding boxes")
        m.def_function[calculate_fast_kuzushi]("calculate_fast_kuzushi", docstring="Biomechanics kuzushi detection")
        m.def_function[compute_cost_matrix]("compute_cost_matrix", docstring="Compute assignment cost matrix")
        m.def_function[score_foreground_pair]("score_foreground_pair", docstring="Score foreground pair quality")
        m.def_function[detect_kinematic_events]("detect_kinematic_events", docstring="Detect takedown events")
        m.def_function[update_skeleton_ema]("update_skeleton_ema", docstring="EMA update for skeleton keypoints")
        m.def_function[compute_z_depth]("compute_z_depth", docstring="Compute faux-3D Z-depth score")
        m.def_function[classify_ref_arm_signal]("classify_ref_arm_signal", docstring="Classify referee arm signal from keypoints")
        m.def_function[pixel_to_mat]("pixel_to_mat", docstring="Transform pixel coords to mat-space via homography")
        m.def_function[is_on_mat]("is_on_mat", docstring="Check if pixel coords are on the mat")
        return m.finalize()
    except e:
        abort(String("error creating mojo_analytics Python module:", e))
        return PythonObject()



