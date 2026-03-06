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
        return m.finalize()
    except e:
        abort(String("error creating mojo_analytics Python module:", e))
        return PythonObject()

