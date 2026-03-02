import json
import os
from ultralytics import YOLO

from settings import PROPRIETARY_MODEL
from src.core.tracker import MatchTracker
from src.core.analyzer import MatchAnalyzer
from src.core.renderer import BroadcastRenderer

class GrapplingPipeline:
    def __init__(self, model_path=PROPRIETARY_MODEL):
        if not os.path.exists(model_path):
            print(f"⚠️ Model {model_path} not found locally! Ensure training is complete or check paths.")
            
        print("🤖 Loading YOLO inference engine...")
        self.model = YOLO(model_path)
        self.tracker = MatchTracker(self.model)

    def analyze_match(self, video_path, output_dir="."):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
            
        print(f"\n--- INITIATING GRAPPLING MASTER PIPELINE ---")
        raw_data, total_frames, fps, w, h = self.tracker.extract_raw_data(video_path)
        
        anchor = self.tracker.find_foreground_anchor(raw_data, w, h, fps)
        if not anchor: 
            print("❌ No valid foreground players found. Aborting.")
            return
            
        true_coach_id, spec_ids, bg_ids = self.tracker.build_global_blacklist(raw_data, anchor, w, h, fps)
        resolved_timeline = self.tracker.resolve_timeline(raw_data, total_frames, anchor, true_coach_id, spec_ids, bg_ids, w, h)
        
        analyzer = MatchAnalyzer(fps, h)
        events = analyzer.detect_events_from_timeline(resolved_timeline, total_frames)
            
        print(f"\n🎬 [PHASE 5] RENDERING 3D BROADCAST CLIPS...")
        master_report = {"source_video": video_path, "total_engagements": len(events), "highlights": []}
        
        renderer = BroadcastRenderer(fps, w, h)
        
        for i, event in enumerate(events, 1):
            print(f"\n   Rendering Match Event {i}/{len(events)}...")
            event_stats = renderer.render_event_clip(video_path, event, resolved_timeline, i, output_dir=output_dir)
            
            master_report["highlights"].append({
                "filename": event_stats["filename"],
                "slow_mo_filename": event_stats["slow_mo_file"],
                "phases_detected": event_stats["phases"],
                "match_timestamp_sec": round(float(event['start_frame'] / fps), 1)
            })
            
        report_path = os.path.join(output_dir, 'master_match_report.json')
        with open(report_path, 'w') as f: 
            json.dump(master_report, f, indent=4)
            
        print("\n🎉 MASTER PIPELINE IS COMPLETE. Foreground Median & DNA Lock Online! 🎉")
