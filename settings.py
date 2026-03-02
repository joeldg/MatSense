import os
import torch

# Hardware Acceleration
if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

# Default Model Paths
DEFAULT_POSE_MODEL = 'yolov8n-pose.pt'
PROPRIETARY_MODEL = 'yolov8s-pose.pt'

# Cognitive Engine Settings
DEFAULT_VIDEOMAE_MODEL = "MCG-NJU/videomae-base-finetuned-kinetics"
VIDEOMAE_NUM_FRAMES = 16

# 3D Cognitive Engine Settings
DEFAULT_WHAM_MODEL = "models/wham_vit_l.pth"
DEFAULT_SMPLX_MODEL = "models/smplx/SMPLX_NEUTRAL.npz"
USE_3D_ENGINE = True

# Analysis Settings
TARGET_FPS = 30.0
TRACKER_CONF = 0.25
PREDICT_CONF = 0.4

# UI / Rendering Settings
HUD_COLORS = {
    "STANDING": (100, 255, 100),       # Green
    "MATE": (0, 200, 255),             # Yellow
    "NE-WAZA": (255, 100, 100),        # Blue
    "KUZUSHI": (0, 0, 255),            # Red
    "TEXT": (255, 255, 255),           # White
    "TEXT_DIM": (150, 150, 150),       # Light Gray
    "SKELETON_0": (0, 255, 100),
    "SKELETON_1": (255, 50, 255),
    "BG": (100, 100, 100),
    "SPEC": (0, 100, 255),
    "REF": (0, 215, 255),
    "UNK": (200, 200, 200),
    "PLUMB_LINE": (0, 255, 255)
}

DASHBOARD_WIDTH = 500

# Temp Directory for Downloads
TEMP_DIR = os.path.join(os.getcwd(), 'tmp_downloads')
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)
