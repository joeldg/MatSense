import pytest
import numpy as np

from src.core.analyzer import MatchAnalyzer
from src.core.tracker import MatchTracker

def test_analyzer_detect_event():
    """
    Tests the Kuzushi Action recognition drop matrix.
    Given a mock timeline where the upper-body Y-coordinate violently drops,
    the analyzer should successfully flag an 'event' phase.
    """
    analyzer = MatchAnalyzer(fps=30, height=1080)
    
    # Create a mock timeline:
    # 3 seconds of standing (Y=200), then a violent takedown (Y drops to 800) sustained 3 seconds
    timeline = {}
    for i in range(180): # total_frames = 180 (6 sec)
        if i < 90:
            timeline[i] = {
                'melded': False,
                'p1': {'box': [800, 200, 1100, 600]},
                'p2': {'box': [850, 220, 1150, 620]}
            }
        else:
            timeline[i] = {
                'melded': False,
                'p1': {'box': [800, 800, 1100, 1000]},
                'p2': {'box': [850, 820, 1150, 1020]}
            }

    events = analyzer.detect_events_from_timeline(timeline, total_frames=180)
    
    # We should have at least 1 event detected 
    assert len(events) >= 1, "Analyzer failed to detect the violent drop."
    
    event = events[0]
    # The start should be slightly before the drop happens (due to the pre-action buffer)
    assert event['start_frame'] < 91
    assert event['end_frame'] >= 150

def test_tracker_foreground_supremacy_matrix():
    """
    Tests the math in `MatchTracker` to identify the two main combatants.
    It should accurately penalize distance from the center and prioritize screen area, 
    so the fighters in the middle are chosen over the coaches on the sides.
    """
    class MockModel:
        def __init__(self):
            pass
            
    tracker = MatchTracker(MockModel())
    
    # Frame dimensions
    w, h, fps = 1920, 1080, 30.0
    
    # Actor 1: Large object right in the middle
    # Actor 2: Large object slightly offset
    # Actor 3: Coach/Ref standing near the edge of the screen
    raw_data = {
        1: [
            {'id': 101, 'box': [800, 400, 1100, 900]}, # Central Fighter 1
            {'id': 102, 'box': [850, 420, 1150, 950]}, # Central Fighter 2
            {'id': 999, 'box': [100, 400, 300, 900]}   # Edge bystander
        ]
    }
    
    # Generate 5 seconds of the same layout
    for i in range(2, 150):
        raw_data[i] = raw_data[1]
        
    anchor = tracker.find_foreground_anchor(raw_data, w, h, fps)
    
    # The anchor should be a dictionary holding p1 and p2 info
    assert anchor is not None, "Failed to find any foreground anchor."
    assert anchor['p1']['id'] in (101, 102)
    assert anchor['p2']['id'] in (101, 102)
    assert anchor['p1']['id'] != 999 and anchor['p2']['id'] != 999, "Bystander was incorrectly flagged as a combatant."
