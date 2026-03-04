from math import sqrt
from time import perf_counter
from collections import List

fn calculate_fast_kuzushi(kpts: List[Float64]) -> Bool:
    """
    Fast 2D biomechanics projected directly onto the broadcast video using pure Mojo.
    Expects a flattened array of 51 Float64s (17 YOLO joints * 3 dimensions).
    """
    # Hips (11, 12)
    var p11x = kpts[11 * 3 + 0]
    var p11y = kpts[11 * 3 + 1]
    var p12x = kpts[12 * 3 + 0]
    var p12y = kpts[12 * 3 + 1]
    var pelvis_x = (p11x + p12x) / 2.0
    var pelvis_y = (p11y + p12y) / 2.0

    # Shoulders (5, 6)
    var p5x = kpts[5 * 3 + 0]
    var p5y = kpts[5 * 3 + 1]
    var p6x = kpts[6 * 3 + 0]
    var p6y = kpts[6 * 3 + 1]
    var neck_x = (p5x + p6x) / 2.0
    var neck_y = (p5y + p6y) / 2.0

    # Ankles (15, 16)
    var p15x = kpts[15 * 3 + 0]
    var p15y = kpts[15 * 3 + 1]
    var p16x = kpts[16 * 3 + 0]
    var p16y = kpts[16 * 3 + 1]

    # Center of Mass (Weighted 60% lower, 40% upper)
    var com_x = (pelvis_x * 0.6) + (neck_x * 0.4)
    var com_y = (pelvis_y * 0.6) + (neck_y * 0.4)

    # Stance Width and Vectors
    var dx = p15x - p16x
    var dy = p15y - p16y
    var stance_width = sqrt((dx * dx) + (dy * dy))

    var ab_x = dx
    var ab_y = dy
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

    # Distance to base
    var dist_x = com_x - cp_x
    var dist_y = com_y - cp_y
    var distance_to_base = sqrt((dist_x * dist_x) + (dist_y * dist_y))

    var dynamic_threshold = 15.0 + (stance_width * 0.30)
    
    if distance_to_base > dynamic_threshold:
        return True
    return False

fn main():
    var iterations = 100_000
    print("🚀 Racing Pure Mojo (LLVM/MLIR) over", iterations, "frames...")
    
    # 1. Generate a mock batch of frames simulating the YOLO (17, 3) arrays
    var kpts_batch = List[List[Float64]]()
    
    for i in range(iterations):
        var mock_frame = List[Float64]()
        # Force 51 elements to simulate 17 joints * 3 planes
        for j in range(51):
            mock_frame.append(100.0 + Float64(j)) 
        kpts_batch.append(mock_frame)

    # 2. Benchmark execution
    var start_time = perf_counter()
    
    for i in range(iterations):
        var result = calculate_fast_kuzushi(kpts_batch[i])
        
    var end_time = perf_counter()
    var elapsed_sec = end_time - start_time

    print("🏁 RESULTS:")
    print("   Modular Mojo Native (seconds):")
    print(elapsed_sec)
    print("   🔥 Compare this directly against the Golang and Python output!")
