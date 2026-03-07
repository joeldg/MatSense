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
DEFAULT_WHAM_MODEL = "third_party/wham/checkpoints/wham_vit_w_3dpw.pth.tar"
DEFAULT_SMPLX_MODEL = "models/smplx/SMPLX_NEUTRAL.npz"
USE_3D_ENGINE = True

# Analysis Settings
TARGET_FPS = 30.0
TRACKER_CONF = 0.25
PREDICT_CONF = 0.4

# UI / Rendering Settings
HUD_COLORS = {
    # State indicators
    "STANDING": (100, 255, 100),       # Green
    "MATE": (0, 200, 255),             # Yellow
    "NE-WAZA": (255, 100, 100),        # Blue
    "KUZUSHI": (0, 0, 255),            # Red
    # Text
    "TEXT": (255, 255, 255),           # White
    "TEXT_DIM": (150, 150, 150),       # Light Gray
    # Entity colors — consistent identification
    "ATHLETE_1": (255, 255, 255),      # White — Athlete 1
    "ATHLETE_2": (255, 100, 0),        # Blue — Athlete 2
    "SKELETON_0": (255, 255, 255),     # White skeleton for Athlete 1
    "SKELETON_1": (255, 100, 0),       # Blue skeleton for Athlete 2
    "REF": (0, 0, 255),               # Red — Referee (must exist)
    "SPEC": (0, 165, 255),            # Orange — Spectators
    "BG": (128, 128, 128),            # Grey — Background
    "UNK": (200, 200, 200),           # Light grey — Unknown
    # Overlays
    "PERSPECTIVE": (0, 255, 0),        # Green — 3D depth grid lines
    "PLUMB_LINE": (0, 255, 255),       # Yellow — vertical reference
    "SIGNAL": (0, 255, 255),           # Yellow — ref arm signal text
    "SIGNAL_FLASH": (0, 0, 255),       # Red — ref signal flash
}

DASHBOARD_WIDTH = 500

# Temp Directory for Downloads
TEMP_DIR = os.path.join(os.getcwd(), 'tmp_downloads')
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)
