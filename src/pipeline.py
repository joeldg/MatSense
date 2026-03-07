import json
import os
import cv2
from ultralytics import YOLO

from settings import PROPRIETARY_MODEL
from src.core.tracker import MatchTracker
from src.core.analyzer import MatchAnalyzer
from src.core.renderer import BroadcastRenderer
from src.core.harvester import DataHarvester
from src.core.mat_homography import MatHomography

class GrapplingPipeline:
    def __init__(self, model_path=PROPRIETARY_MODEL, use_mojo=False):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"⚠️ Model {model_path} not found locally! Ensure training is complete or check paths.")
            
        self.use_mojo = use_mojo
        print("🤖 Loading YOLO inference engine...")
        self.model = YOLO(model_path)
        self.tracker = MatchTracker(self.model, use_mojo=use_mojo)

    def analyze_match(self, video_path, output_dir="."):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Create per-video subdirectory
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(output_dir, video_basename)
        os.makedirs(video_output_dir, exist_ok=True)
            
        print(f"\n--- INITIATING GRAPPLING MASTER PIPELINE ---")
        print(f"   📁 Output: {video_output_dir}")
        raw_data, total_frames, fps, w, h = self.tracker.extract_raw_data(video_path)
        
        # For long videos (>5 min), use windowed analysis to find multiple matches.
        window_duration_sec = 120
        window_frames = int(fps * window_duration_sec)
        
        if total_frames <= window_frames * 2:
            all_events = self._analyze_single_window(raw_data, total_frames, fps, w, h, 0)
        else:
            all_events = self._analyze_multi_window(raw_data, total_frames, fps, w, h, window_frames)
        
        if not all_events:
            print("❌ No takedown events detected in any window.")
            return
            
        print(f"\n🎬 [PHASE 5] RENDERING {len(all_events)} BROADCAST CLIPS...")
        master_report = {"source_video": video_path, "video_name": video_basename, "total_matches": len(all_events), "highlights": []}
        
        renderer = BroadcastRenderer(fps, w, h, use_mojo=self.use_mojo)
        harvester = DataHarvester(output_dir=os.path.join(video_output_dir, "dataset", "raw_clips"))
        
        for match_num, (event, timeline) in enumerate(all_events, 1):
            print(f"\n   Rendering Match {match_num}/{len(all_events)}...")
            event_stats = renderer.render_event_clip(video_path, event, timeline, match_num, output_dir=video_output_dir)
            
            tensor_path = harvester.harvest_kinematic_tensor(video_path, event, timeline, fps, w, h, match_num)
            
            master_report["highlights"].append({
                "match_number": match_num,
                "filename": event_stats["filename"],
                "slow_mo_filename": event_stats["slow_mo_file"],
                "tensor_filename": tensor_path,
                "phases_detected": event_stats["phases"],
                "match_timestamp_sec": round(float(event['start_frame'] / fps), 1)
            })
            
        report_path = os.path.join(video_output_dir, 'master_match_report.json')
        with open(report_path, 'w') as f: 
            json.dump(master_report, f, indent=4)
            
        print(f"\n🎉 MASTER PIPELINE COMPLETE: {len(all_events)} matches found! Output: {video_output_dir} 🎉")

    def _compute_mat_homography(self, video_path, w, h):
        """Compute mat homography from the first usable frame of the video.
        
        Reads 5 evenly-spaced frames and picks the one with the best
        mat detection. This is computed once per segment.
        """
        mat_H = MatHomography()
        cap = cv2.VideoCapture(video_path) if isinstance(video_path, str) else None
        
        if cap is None or not cap.isOpened():
            return mat_H  # Unavailable
        
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Sample 5 evenly-spaced frames to find best mat detection
        for sample_idx in range(5):
            target_frame = int(total * (0.2 + sample_idx * 0.15))  # 20%, 35%, 50%, 65%, 80%
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if ret and mat_H.compute(frame, w, h):
                print(f"   📐 MAT HOMOGRAPHY: Computed from frame {target_frame} — mat-space coordinates available")
                cap.release()
                return mat_H
        
        cap.release()
        print("   ⚠️ MAT HOMOGRAPHY: Could not detect mat quad — using pixel-space only")
        return mat_H

    def _analyze_single_window(self, raw_data, total_frames, fps, w, h, frame_offset):
        """Run the full analysis pipeline on a single window of raw_data. Returns [(event, timeline), ...].
        
        MATCH TRIAD REQUIREMENT: A valid match requires all 3 entities:
        1. Athlete_1 (foreground, left)
        2. Athlete_2 (foreground, right)  
        3. Referee (upright, orbiting)
        
        If any are missing, the segment is skipped (it's likely a transition/replay/leaderboard).
        """
        # Compute mat homography (optional augmentation)
        mat_H = self._compute_mat_homography(self.tracker.last_video_path, w, h) if hasattr(self.tracker, 'last_video_path') else MatHomography()
        
        anchor = self.tracker.find_foreground_anchor(raw_data, w, h, fps, mat_H=mat_H)
        if not anchor:
            return []
            
        true_coach_id, spec_ids, bg_ids = self.tracker.build_global_blacklist(raw_data, anchor, w, h, fps, mat_H=mat_H)
        
        # TRIAD GATE: No ref found = not a valid match segment
        if true_coach_id is None:
            print("   ⛔ TRIAD GATE: No referee detected — skipping segment (not a valid match)")
            return []
        
        resolved_timeline = self.tracker.resolve_timeline(raw_data, total_frames, anchor, true_coach_id, spec_ids, bg_ids, w, h, mat_H=mat_H)
        
        # ENGAGEMENT GATE: Verify athletes are actually engaged, not just standing idle
        # Count frames where both athletes are detected (not melded from missing data)
        engaged_frames = sum(1 for f, data in resolved_timeline.items() 
                           if not data.get('melded', True) 
                           and data.get('p1') is not None 
                           and data.get('p2') is not None)
        engagement_ratio = engaged_frames / max(1, total_frames)
        
        if engagement_ratio < 0.15:
            # Less than 15% of frames have both athletes visible = probably not a real match
            print(f"   ⛔ ENGAGEMENT GATE: Only {engagement_ratio:.0%} of frames have both athletes — skipping")
            return []
        
        analyzer = MatchAnalyzer(fps, h, use_mojo=self.use_mojo)
        events = analyzer.detect_events_from_timeline(resolved_timeline, total_frames)
        
        return [(event, resolved_timeline) for event in events]

    def _analyze_multi_window(self, raw_data, total_frames, fps, w, h, window_frames):
        """Split raw_data at scene boundaries (detection gaps) and analyze each match segment."""
        # Step 1: Find scene boundaries by scanning for detection gaps
        # A gap of >3 seconds with 0-1 detections = match boundary (replay/scoreboard/transition)
        gap_threshold = int(fps * 3)
        max_window = int(fps * 180)  # 3-minute max window
        min_window = int(fps * 10)   # 10-second minimum to avoid tiny fragments
        
        boundaries = [0]  # Start of first segment
        consecutive_empty = 0
        
        print(f"\n🔎 Scanning for match boundaries (gap threshold: {gap_threshold/fps:.0f}s)...")
        
        for f in range(total_frames):
            dets = raw_data.get(f, [])
            if len(dets) <= 1:
                consecutive_empty += 1
            else:
                if consecutive_empty >= gap_threshold:
                    # Found a boundary — mark the start of the gap as segment end
                    gap_start = f - consecutive_empty
                    if gap_start - boundaries[-1] >= min_window:
                        boundaries.append(gap_start)
                        boundaries.append(f)  # New segment starts after the gap
                consecutive_empty = 0
        boundaries.append(total_frames)
        
        # Build segments from boundary pairs
        segments = []
        for i in range(0, len(boundaries) - 1, 2):
            seg_start = boundaries[i]
            seg_end = boundaries[i + 1] if i + 1 < len(boundaries) else total_frames
            seg_len = seg_end - seg_start
            
            if seg_len < min_window:
                continue  # Skip tiny fragments
            
            # If segment is too long, split it into chunks
            if seg_len > max_window:
                sub_start = seg_start
                while sub_start < seg_end:
                    sub_end = min(sub_start + max_window, seg_end)
                    if sub_end - sub_start >= min_window:
                        segments.append((sub_start, sub_end))
                    sub_start += int(max_window * 0.8)  # 20% overlap for long segments
            else:
                segments.append((seg_start, seg_end))
        
        if not segments:
            # Fallback: treat entire video as one segment
            segments = [(0, total_frames)]
        
        print(f"   📐 Found {len(segments)} match segments")
        
        all_events = []
        for seg_idx, (start, end) in enumerate(segments, 1):
            window_len = end - start
            start_sec = start / fps
            end_sec = end / fps
            print(f"\n{'='*60}")
            print(f"📺 SEGMENT {seg_idx}/{len(segments)}: {start_sec/60:.1f}min - {end_sec/60:.1f}min ({window_len} frames)")
            print(f"{'='*60}")
            
            # Slice raw_data for this window (re-key to 0-based for the tracker)
            window_data = {}
            for f in range(start, end):
                if f in raw_data:
                    window_data[f - start] = raw_data[f]
                else:
                    window_data[f - start] = []
            
            # Run full analysis on this segment
            window_events = self._analyze_single_window(window_data, window_len, fps, w, h, start)
            
            # Adjust ALL frame offsets back to global positions
            for event, timeline in window_events:
                event['start_frame'] += start
                event['end_frame'] += start
                event['impact_frame'] += start
                event['transition_frame'] += start
                
                # Re-key timeline to global frame indices
                global_timeline = {}
                for f, data in timeline.items():
                    global_timeline[f + start] = data
                
                all_events.append((event, global_timeline))
            
            if window_events:
                print(f"   🎯 Found {len(window_events)} events in this segment")
            else:
                print(f"   ⚫ No events in this segment")
        
        # Deduplicate events that fall within the overlap region
        all_events = self._deduplicate_events(all_events, fps)
        
        print(f"\n📊 Total events across all segments: {len(all_events)}")
        return all_events
    
    def _deduplicate_events(self, events, fps):
        """Remove duplicate events that were detected in overlapping windows."""
        if len(events) <= 1:
            return events
        
        # Sort by impact frame
        events.sort(key=lambda x: x[0]['impact_frame'])
        
        deduped = [events[0]]
        min_gap = fps * 5  # Events within 5 seconds are considered duplicates
        
        for event, timeline in events[1:]:
            prev_impact = deduped[-1][0]['impact_frame']
            curr_impact = event['impact_frame']
            
            if curr_impact - prev_impact > min_gap:
                deduped.append((event, timeline))
            # else: skip duplicate from overlap region
        
        return deduped
