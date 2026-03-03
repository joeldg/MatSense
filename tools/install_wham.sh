#!/bin/bash

# Configuration
WHAM_DIR="third_party/wham"
WEIGHTS_DIR="$WHAM_DIR/checkpoints"

echo "🌍 Initiating WHAM 3D Volumetric Engine Setup..."

# 1. Ensure third_party directory exists
mkdir -p third_party

# 2. Clone the official WHAM repository if it doesn't exist
if [ ! -d "$WHAM_DIR" ]; then
    echo "⬇️ Cloning WHAM repository..."
    git clone https://github.com/yohanshin/WHAM.git $WHAM_DIR
else
    echo "✅ WHAM repository already exists at $WHAM_DIR"
fi

# 3. Create checkpoints directory
mkdir -p $WEIGHTS_DIR

# 4. Install Gdown if missing (Required for Google Drive WHAM weights)
if ! python3 -c "import gdown" &> /dev/null; then
    echo "⬇️ Installing gdown to fetch weights..."
    pip3 install gdown
fi

# 5. Download Official WHAM Checkpoints via Google Drive
echo "⬇️ Downloading authoritative WHAM model weights into $WEIGHTS_DIR..."

cd $WHAM_DIR

# Using the official Google Drive file IDs provided by the WHAM authors
if [ ! -f "checkpoints/wham_vit_w_3dpw.pth.tar" ]; then
    echo "   Downloading wham_vit_w_3dpw.pth.tar..."
    python3 -m gdown "https://drive.google.com/uc?id=1i7kt9RlCCCNEW2aYaDWVr-G778JkLNcB&export=download&confirm=t" -O 'checkpoints/wham_vit_w_3dpw.pth.tar'
else
    echo "✅ wham_vit_w_3dpw.pth.tar already exists."
fi

if [ ! -f "checkpoints/wham_vit_bedlam_w_3dpw.pth.tar" ]; then
    echo "   Downloading wham_vit_bedlam_w_3dpw.pth.tar..."
    python3 -m gdown "https://drive.google.com/uc?id=19qkI-a6xuwob9_RFNSPWf1yWErwVVlks&export=download&confirm=t" -O 'checkpoints/wham_vit_bedlam_w_3dpw.pth.tar'
else
    echo "✅ wham_vit_bedlam_w_3dpw.pth.tar already exists."
fi

if [ ! -f "checkpoints/hmr2a.ckpt" ]; then
    echo "   Downloading hmr2a.ckpt..."
    python3 -m gdown "https://drive.google.com/uc?id=1J6l8teyZrL0zFzHhzkC7efRhU0ZJ5G9Y&export=download&confirm=t" -O 'checkpoints/hmr2a.ckpt'
else
    echo "✅ hmr2a.ckpt already exists."
fi

if [ ! -f "checkpoints/dpvo.pth" ]; then
    echo "   Downloading dpvo.pth (SLAM)..."
    python3 -m gdown "https://drive.google.com/uc?id=1kXTV4EYb-BI3H7J-bkR3Bc4gT9zfnHGT&export=download&confirm=t" -O 'checkpoints/dpvo.pth'
else
    echo "✅ dpvo.pth already exists."
fi

if [ ! -f "checkpoints/vitpose-h-multi-coco.pth" ]; then
    echo "   Downloading vitpose-h-multi-coco.pth..."
    python3 -m gdown "https://drive.google.com/uc?id=1xyF7F3I7lWtdq82xmEPVQ5zl4HaasBso&export=download&confirm=t" -O 'checkpoints/vitpose-h-multi-coco.pth'
else
    echo "✅ vitpose-h-multi-coco.pth already exists."
fi

cd ../../

# 5. Install Python dependencies specific to WHAM if needed
# Note: Since the environment might be sensitive on Apple Silicon, 
# we will just instruct the user on what is missing or install 
# pure python components first.
echo ""
echo "🎉 WHAM Download Complete!"
echo "Make sure you have basic requirements (PyTorch, torchvision, etc.) installed."
echo "Note: WHAM may require additional libraries like 'yacs' or 'chumpy' which are easily pip installable if they throw an error."
