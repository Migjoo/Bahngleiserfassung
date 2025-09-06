#!/usr/bin/env python3
"""
Comprehensive test of all videos in test folder to create best person-on-track implementation
"""
import sys
import os
from io import BytesIO
import glob

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_all_videos_person_on_track():
    """Test all videos in test folder for person-on-track scenarios"""
    print("COMPREHENSIVE PERSON-ON-TRACK DETECTION ANALYSIS")
    print("=" * 70)
    print("Testing all videos in test folder to find best implementation")
    print()
    
    try:
        from local_models import get_local_model_manager
        from app import extract_frames_from_video, process_image_locally
        print("+ Components loaded successfully")
    except ImportError as e:
        print(f"- Import error: {e}")
        return
    
    # Find all videos in test folder
    test_videos = glob.glob("test\\*.mp4")
    if not test_videos:
        print("- No MP4 files found in test folder")
        return
    
    print(f"+ Found {len(test_videos)} test videos: {[os.path.basename(v) for v in test_videos]}")
    
    # Initialize models
    try:
        local_manager = get_local_model_manager()
        print("+ All models ready for testing")
    except Exception as e:
        print(f"- Model initialization error: {e}")
        return
    
    # Test different approaches
    approaches = {
        "Approach 1 - People Counter": {
            "model": "People Counter",
            "prompt": "Track Safety Analysis"
        },
        "Approach 2 - Direct CNN": {
            "model": "CNN (BLIP)", 
            "prompt": "Is there a person standing on train tracks? Answer yes or no."
        },
        "Approach 3 - Detailed Transformer": {
            "model": "Transformer (ViT-GPT2)",
            "prompt": "Describe people and train tracks in this image"
        },
        "Approach 4 - Safety Focus": {
            "model": "CNN (BLIP)",
            "prompt": "Describe any safety concerns with people near train tracks"
        }
    }
    
    all_results = {}
    
    # Test each video with each approach
    for video_idx, video_path in enumerate(test_videos):
        video_name = os.path.basename(video_path)
        print(f"\n" + "=" * 70)
        print(f"TESTING VIDEO {video_idx + 1}: {video_name}")
        print("=" * 70)
        
        try:
            # Extract frames
            with open(video_path, 'rb') as f:
                video_data = f.read()
            
            video_file = BytesIO(video_data)
            frames = extract_frames_from_video(video_file, fps=0.5)  # Every 2 seconds
            
            if not frames:
                print(f"- No frames extracted from {video_name}")
                continue
            
            print(f"+ Extracted {len(frames)} frames from {video_name}")
            
            # Test 2-3 frames per video to get representative sample
            test_frames = frames[:min(3, len(frames))]
            video_results = {}
            
            # Test each approach on this video
            for approach_name, config in approaches.items():
                print(f"\n  Testing {approach_name}:")
                print(f"  {'-' * 40}")
                
                approach_results = []
                
                for frame_idx, frame_data in enumerate(test_frames):
                    frame_num = frame_idx + 1
                    timestamp = frame_data['timestamp']
                    
                    try:
                        result = process_image_locally(
                            frame_data['frame'],
                            config["prompt"],
                            config["model"],
                            local_manager
                        )
                        
                        # Analyze result for person-on-track
                        person_on_track_analysis = analyze_for_person_on_track(result, config["model"])
                        
                        approach_results.append({
                            'frame': frame_num,
                            'timestamp': timestamp,
                            'raw_result': result,
                            'person_on_track': person_on_track_analysis['on_track'],
                            'confidence': person_on_track_analysis['confidence'],
                            'reasoning': person_on_track_analysis['reasoning']
                        })
                        
                        status = "ON TRACK" if person_on_track_analysis['on_track'] else "SAFE"
                        print(f"    Frame {frame_num} ({timestamp:.1f}s): {status} - {person_on_track_analysis['confidence']:.0%} confidence")
                        print(f"      Reasoning: {person_on_track_analysis['reasoning'][:80]}...")
                        
                    except Exception as e:
                        approach_results.append({
                            'frame': frame_num,
                            'timestamp': timestamp,
                            'raw_result': {'error': str(e)},
                            'person_on_track': False,
                            'confidence': 0,
                            'reasoning': f"Error: {str(e)}"
                        })
                        print(f"    Frame {frame_num} ({timestamp:.1f}s): ERROR - {str(e)}")
                
                video_results[approach_name] = approach_results
            
            all_results[video_name] = video_results
            
        except Exception as e:
            print(f"- Failed to process {video_name}: {e}")
            continue
    
    # Comprehensive analysis and recommendation
    analyze_all_approaches(all_results, approaches)
    
    return all_results

def analyze_for_person_on_track(result, model_type):
    """Analyze model result to determine if person is on train tracks"""
    
    if 'error' in result:
        return {
            'on_track': False,
            'confidence': 0,
            'reasoning': f"Error in processing: {result['error']}"
        }
    
    # Handle different result types
    if 'people_analysis' in result:
        # People Counter result
        analysis = result['people_analysis']
        on_track = analysis.get('on_tracks', False) or analysis.get('safety_risk', False)
        confidence = analysis.get('confidence', 0)
        reasoning = analysis.get('analysis_summary', 'People Counter analysis')
        
        return {
            'on_track': on_track,
            'confidence': confidence,
            'reasoning': reasoning
        }
    
    elif 'yes_no_detection' in result:
        # Yes/No detector result
        detection = result['yes_no_detection']
        # For track detection, we need more than just person presence
        return {
            'on_track': False,  # Yes/No detector doesn't check tracks specifically
            'confidence': 0.3,
            'reasoning': "Yes/No detector not suitable for track-specific detection"
        }
    
    elif 'generated_text' in result:
        # Text analysis result
        text = result['generated_text'].lower()
        
        # Keywords for person on tracks
        person_keywords = ['person', 'people', 'man', 'woman', 'human', 'individual']
        track_keywords = ['track', 'tracks', 'rail', 'rails', 'railway']
        position_keywords = ['on', 'standing', 'walking', 'sitting', 'crossing']
        danger_keywords = ['danger', 'unsafe', 'risk', 'hazard', 'warning']
        
        # Strong indicators
        strong_patterns = [
            'person on track', 'man on track', 'woman on track',
            'standing on track', 'walking on track', 'person crossing',
            'on the tracks', 'on train tracks', 'on railway'
        ]
        
        # Count indicators
        person_mentions = sum(1 for kw in person_keywords if kw in text)
        track_mentions = sum(1 for kw in track_keywords if kw in text)
        position_mentions = sum(1 for kw in position_keywords if kw in text)
        danger_mentions = sum(1 for kw in danger_keywords if kw in text)
        strong_indicators = sum(1 for pattern in strong_patterns if pattern in text)
        
        # Decision logic
        if strong_indicators > 0:
            on_track = True
            confidence = min(0.8 + strong_indicators * 0.1, 1.0)
            reasoning = f"Strong indicators: {strong_indicators} pattern matches"
            
        elif person_mentions > 0 and track_mentions > 0 and position_mentions > 0:
            on_track = True
            confidence = 0.6 + min(person_mentions + track_mentions + position_mentions, 3) * 0.1
            reasoning = f"Person + track + position keywords: {person_mentions}+{track_mentions}+{position_mentions}"
            
        elif danger_mentions > 0 and (person_mentions > 0 or track_mentions > 0):
            on_track = True
            confidence = 0.5 + danger_mentions * 0.1
            reasoning = f"Safety concern mentioned with people/tracks: {danger_mentions} danger keywords"
            
        else:
            on_track = False
            confidence = 0.7 if person_mentions == 0 else 0.4
            reasoning = f"No clear person-on-track indicators. Person:{person_mentions}, Track:{track_mentions}"
        
        return {
            'on_track': on_track,
            'confidence': confidence,
            'reasoning': reasoning
        }
    
    else:
        return {
            'on_track': False,
            'confidence': 0,
            'reasoning': "Unknown result format"
        }

def analyze_all_approaches(all_results, approaches):
    """Analyze all approaches and provide recommendations"""
    
    print(f"\n" + "=" * 80)
    print("COMPREHENSIVE ANALYSIS OF ALL APPROACHES")
    print("=" * 80)
    
    # Calculate performance metrics for each approach
    approach_metrics = {}
    
    for approach_name in approaches.keys():
        total_frames = 0
        on_track_detections = 0
        avg_confidence = 0
        error_count = 0
        
        for video_name, video_results in all_results.items():
            if approach_name in video_results:
                for frame_result in video_results[approach_name]:
                    total_frames += 1
                    if frame_result['person_on_track']:
                        on_track_detections += 1
                    avg_confidence += frame_result['confidence']
                    if 'error' in frame_result.get('raw_result', {}):
                        error_count += 1
        
        if total_frames > 0:
            avg_confidence = avg_confidence / total_frames
            detection_rate = on_track_detections / total_frames * 100
            error_rate = error_count / total_frames * 100
        else:
            avg_confidence = 0
            detection_rate = 0
            error_rate = 100
        
        approach_metrics[approach_name] = {
            'total_frames': total_frames,
            'on_track_detections': on_track_detections,
            'detection_rate': detection_rate,
            'avg_confidence': avg_confidence,
            'error_rate': error_rate
        }
    
    # Display results table
    print(f"\nAPPROACH PERFORMANCE COMPARISON:")
    print("-" * 80)
    print(f"{'Approach':<25} {'Frames':<8} {'On-Track':<10} {'Rate':<8} {'Confidence':<12} {'Errors':<8}")
    print("-" * 80)
    
    for approach, metrics in approach_metrics.items():
        print(f"{approach:<25} {metrics['total_frames']:<8} {metrics['on_track_detections']:<10} "
              f"{metrics['detection_rate']:<8.1f}% {metrics['avg_confidence']:<12.0%} {metrics['error_rate']:<8.1f}%")
    
    # Find best approach
    best_approach = max(approach_metrics.items(), 
                       key=lambda x: x[1]['avg_confidence'] * (100 - x[1]['error_rate']) / 100)
    
    print(f"\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    print(f"BEST APPROACH: {best_approach[0]}")
    print(f"  - Average Confidence: {best_approach[1]['avg_confidence']:.0%}")
    print(f"  - Detection Rate: {best_approach[1]['detection_rate']:.1f}%")
    print(f"  - Error Rate: {best_approach[1]['error_rate']:.1f}%")
    print(f"  - Total Frames Tested: {best_approach[1]['total_frames']}")
    
    # Detailed recommendations
    print(f"\nDETAILED ANALYSIS:")
    
    if best_approach[0] == "Approach 1 - People Counter":
        print("+ People Counter is most effective for track safety")
        print("+ Uses specialized multi-prompt analysis")
        print("+ Provides detailed safety risk assessment")
        
    elif "CNN" in best_approach[0]:
        print("+ CNN model provides good balance of speed and accuracy")
        print("+ Direct prompting works well for specific scenarios")
        print("+ Consider using for real-time applications")
        
    elif "Transformer" in best_approach[0]:
        print("+ Transformer model provides detailed scene understanding")
        print("+ Better for complex scene analysis")
        print("+ Higher computational cost but more accurate descriptions")
        
    # Video-by-video breakdown
    print(f"\nPER-VIDEO ANALYSIS:")
    print("-" * 50)
    
    for video_name, video_results in all_results.items():
        print(f"\n{video_name}:")
        for approach_name, results in video_results.items():
            on_track_frames = sum(1 for r in results if r['person_on_track'])
            total_frames = len(results)
            print(f"  {approach_name}: {on_track_frames}/{total_frames} frames with person on track")

if __name__ == "__main__":
    test_all_videos_person_on_track()