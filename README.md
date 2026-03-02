# 🥋 Grappling AI: Kinematic Tracker & Cognitive Engine

## 📖 Project Overview
This repository contains a state-of-the-art computer vision pipeline designed to autonomously track, analyze, and highlight grappling matches (Brazilian Jiu-Jitsu, Judo, Submission Grappling) from raw, single-angle smartphone footage.

Grappling is widely considered the "Final Boss" of computer vision due to severe body occlusion (the "human pretzel" effect), tangled limbs, identical uniforms, complex camera panning, and chaotic gym backgrounds. To solve this, we have engineered a **Kinematic State Engine** that acts as an advanced mathematical attention mechanism. It stabilizes the environment, isolates the true foreground athletes, maps their physics, completely ignores background interference, and generates broadcast-ready cinematic replays.

---

## 🚀 Goals & Architecture

### Phase 1: Kinematic Tracking (Completed)
We have iteratively transformed a basic pose tracker into a robust, context-aware physics pipeline heavily optimized for Apple Silicon (M4 Max / MPS).

* **Absolute Frame Synchronization:** Prevents Variable Frame Rate (VFR) desync, ensuring 3D skeletal wireframes stick to the fighters perfectly even during fast camera pans.
* **Foreground Supremacy Matrix:** Exponential depth-scaling matrix (depth³ * size²) pairs with median-average scoring to mathematically isolate the two primary grappling athletes, ignoring bystanders.
* **Centerline Orbit Matrix (Sentinel Ref Classifier):** Maps the first 8 seconds of the match to build an invisible geometric column, identifying the referee without misclassifying them as a grappler.
* **Immutable Biometric DNA (Anti-Swap Lock):** Extracts pure upper-torso color histograms at the start of the match and permanently locks identities, preventing ID swapping when athletes roll on the floor.
* **Kinematic Transition Engine:** Detects "Deep Pits" (sustained drops in head-altitude). Traces time backward frame-by-frame to find the exact moment the fighters left their feet.
* **Broadcast UI HUD:** Renders data-overlays, custom EMA-smoothed skeletons, and automatically generates dynamically shifted 14-second Director's Cut cinematic slow-motion replays.

### Phase 2: Cognitive Action Recognition (In Progress)
The Kinematic Engine understands *physics* (a takedown occurred). The Cognitive Engine understands *techniques* (the takedown was an *Osoto Gari*).

1. **Data Harvester:** Uses the Phase 1 engine to generate tight, 4-second, background-isolated video tensors of the impact.
2. **Zero-Shot VLM Auto-Labeling:** Passes the tensors to a Vision-Language Model to auto-label the techniques.
3. **End-Node Classifiers:**
   * **VideoMAE Cognitive Engine:** A Hugging Face Spatio-Temporal model trained to infer missing data, resilient to heavy occlusion.
   * **3D Volumetric Engine (WHAM / SMPL-X):** Resolves occlusion entirely by lifting 2D video into 3D interacting meshes, letting a lightweight classifier evaluate *only* the 3D joint coordinate trajectories.

---

## 📁 Repository Structure

```text
grappling-ai/
│
├── main.py                          # Unified CLI orchestrator (analyze, trim, classify, train, prep)
├── settings.py                      # Global configuration, model paths, hardware checks, thresholds
├── requirements.txt                 # Project dependencies (yt-dlp, transformers, smplx, etc.)
├── .gitignore                       # Custom ignore rules for video cache and massive model weights
│
├── src/                             # Core Application Source Code
│   ├── pipeline.py                  # Integration layer orchestrating tracking, analysis, and rendering
│   ├── media_handler.py             # VideoFetcher class: ingests local files, URLs, and YouTube links
│   │
│   └── core/                        # The internal pipeline engines
│       ├── tracker.py               # MatchTracker: Kinematic isolation, DNA locking, BoT-SORT
│       ├── analyzer.py              # MatchAnalyzer: Event detection (Drop severity) and Kuzushi states
│       ├── renderer.py              # BroadcastRenderer: Custom EMA wireframes and cinematic HUD output
│       ├── cognitive_engine.py      # VideoMAE Hugging Face image processor and classifier
│       ├── cognitive_engine_3d.py   # WHAM and SMPL-X mesh extraction lifting 2D to interacting 3D
│       └── volumetric_classifier.py # Lightweight LSTM to classify technique purely from 3D trajectories
│
└── tools/                           # Standalone Utility Scripts
    ├── auto_trimmer.py              # High-speed script highlighting takedowns without rendering HUDs
    ├── prep_data.py                 # Converts ViCoS annotations to YOLO format for dataset generation
    ├── train.py                     # Training orchestrator configured for Apple Silicon (MPS)
    └── resume.py                    # Helper to resume interrupted YOLO training checkpoints
```

---

## 💻 Usage

The entire pipeline is wrapped into a unified Command Line Interface (CLI) via `main.py`. 
You can pass in local video files, direct `.mp4` URLs, or standard YouTube links (using `yt-dlp`).

### View Available Commands
```bash
python3 main.py --help
```

### 1. Full Analysis & Cinematic Rendering
Run the Kinematic tracker to draw the HUD, identify takedowns, and generate broadcast replays.
```bash
# Analyze a local file
python3 main.py analyze -i /path/to/match.mp4

# Analyze a YouTube video
python3 main.py analyze -i "https://www.youtube.com/watch?v=..."
```

### 2. High-Speed Trimming
Scan a long video and quickly extract 14-second clips of every detected takedown (skips HUD rendering).
```bash
python3 main.py trim -i /path/to/long_event_video.mp4 -o ./highlights/
```

### 3. Cognitive Engine Classification
Once you have a cropped takedown clip, pass it through the Spatio-Temporal engines.
```bash
# 2D VideoMAE Classification
python3 main.py classify -i takedown_clip.mp4

# 3D Volumetric Classification (WHAM/SMPL-X)
python3 main.py classify-3d -i takedown_clip.mp4 --engine wham
```

### 4. Training Utilities
Prepare custom datasets and fine-tune models on Apple Silicon.
```bash
# Convert dataset annotations
python3 main.py prep

# Train new pose models natively on MPS
python3 main.py train
python3 main.py train --resume
```

---

## 🔮 Roadmap / Next Steps
- Implement the "Data Harvester" auto-cropper module.
- Plumb the Data Harvester outputs automatically into zero-shot VLM APIs (Gemini 1.5 Pro) for auto-labeling.
- Fine-tune the `MCG-NJU/videomae-base` with the newly generated Grappling dataset.
- Procure and integrate the full WHAM `.pth` weights to switch the 3D Cognitive Engine from simulated mock-mode to live production mode.
