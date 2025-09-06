#!/usr/bin/env python3
"""
Final test of the optimized Person on Track Detector on all test videos
"""
import sys
import os
from io import BytesIO
import glob

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_person_on_track_final():
    """Test the optimized Person on Track Detector on all test videos"""
    print("FINAL PERSON ON TRACK DETECTOR TEST")
    print("=" * 50)
    print("Testing optimized detector based on comprehensive analysis")
    print()
    
    try:
        from local_models import get_local_model_manager
        from app import extract_frames_from_video, process_image_locally
        print("+ Components loaded successfully")
    except ImportError as e:
        print(f"- Import error: {e}")
        return
    
    # Find all test videos
    test_videos = glob.glob("test\\*.mp4")
    if not test_videos:
        print("- No MP4 files found in test folder")
        return
    
    print(f"+ Found {len(test_videos)} test videos")
    
    # Initialize models
    try:
        local_manager = get_local_model_manager()
        available_models = local_manager.get_available_models()
        print(f"+ Available models: {available_models}")
        
        if "Person on Track Detector" not in available_models:
            print("- Person on Track Detector not found!")
            return
        
        print("+ Person on Track Detector ready")
    except Exception as e:
        print(f"- Model initialization error: {e}")
        return
    
    all_results = []
    
    # Test each video
    for video_idx, video_path in enumerate(test_videos):
        video_name = os.path.basename(video_path)
        print(f"\n" + "=" * 60)
        print(f"TESTING VIDEO {video_idx + 1}: {video_name}")
        print("=" * 60)
        
        try:
            # Extract frames
            with open(video_path, 'rb') as f:
                video_data = f.read()
            
            video_file = BytesIO(video_data)
            frames = extract_frames_from_video(video_file, fps=0.5)
            
            if not frames:
                print(f"- No frames extracted from {video_name}")
                continue
            
            print(f"+ Extracted {len(frames)} frames from {video_name}")
            
            # Test first 3 frames
            test_frames = frames[:3]
            video_results = []
            
            for frame_idx, frame_data in enumerate(test_frames):
                frame_num = frame_idx + 1
                timestamp = frame_data['timestamp']
                
                print(f"\n  Frame {frame_num} ({timestamp:.1f}s):")
                print(f"  {'-' * 40}")
                
                try:
                    result = process_image_locally(
                        frame_data['frame'],
                        "Track Safety Analysis",  # Prompt is ignored for this detector
                        'Person on Track Detector',
                        local_manager
                    )
                    
                    if 'error' in result:
                        print(f"  ERROR: {result['error']}")
                        video_results.append({
                            'video': video_name,
                            'frame': frame_num,
                            'timestamp': timestamp,
                            'on_track': False,
                            'answer': 'ERROR',
                            'confidence': 0,
                            'reasoning': result['error']
                        })
                    elif 'person_on_track_detection' in result:
                        detection = result['person_on_track_detection']
                        
                        on_track = detection.get('person_on_track', False)
                        answer = detection.get('answer', 'UNKNOWN')
                        confidence = detection.get('confidence', 0)
                        reasoning = detection.get('reasoning', 'No reasoning')
                        detailed = detection.get('detailed_analysis', {})
                        
                        # Display results
                        status = "ON TRACK" if on_track else "CLEAR"
                        print(f"  Result: {status} ({answer})")
                        print(f"  Confidence: {confidence:.0%}")
                        print(f"  Reasoning: {reasoning}")
                        
                        # Show detailed analysis
                        if detailed:
                            print(f"  Details: Person={detailed.get('person_keywords_found', 0)}, " +
                                  f"Track={detailed.get('track_keywords_found', 0)}, " +
                                  f"Danger={detailed.get('danger_position_keywords', 0)}, " +
                                  f"Safety={detailed.get('safety_concern_keywords', 0)}")
                        
                        video_results.append({
                            'video': video_name,
                            'frame': frame_num,
                            'timestamp': timestamp,
                            'on_track': on_track,
                            'answer': answer,
                            'confidence': confidence,
                            'reasoning': reasoning,
                            'detailed_analysis': detailed
                        })
                        
                    else:
                        print(f"  Unexpected result format: {result}")
                        video_results.append({
                            'video': video_name,
                            'frame': frame_num,
                            'timestamp': timestamp,
                            'on_track': False,
                            'answer': 'UNKNOWN',
                            'confidence': 0,
                            'reasoning': 'Unknown result format'
                        })
                        
                except Exception as e:
                    print(f"  ERROR: {e}")
                    video_results.append({
                        'video': video_name,
                        'frame': frame_num,
                        'timestamp': timestamp,
                        'on_track': False,
                        'answer': 'ERROR',
                        'confidence': 0,
                        'reasoning': str(e)
                    })
            
            all_results.extend(video_results)
            
        except Exception as e:
            print(f"- Failed to process {video_name}: {e}")
            continue
    
    # Comprehensive summary
    print(f"\n" + "=" * 80)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 80)
    
    # Results table
    print(f"\nDETAILED RESULTS:")
    print("-" * 90)
    print(f"{'Video':<10} {'Frame':<6} {'Time':<6} {'On Track':<9} {'Answer':<7} {'Confidence':<11} {'Reasoning':<30}")
    print("-" * 90)
    
    total_frames = len(all_results)
    on_track_count = 0
    error_count = 0
    total_confidence = 0
    
    for result in all_results:
        video = result['video'][:8]
        frame = result['frame']
        timestamp = result['timestamp']
        on_track = "YES" if result['on_track'] else "NO"
        answer = result['answer']
        confidence = result['confidence']
        reasoning = result['reasoning'][:25] + "..." if len(result['reasoning']) > 25 else result['reasoning']
        
        print(f"{video:<10} {frame:<6} {timestamp:<6.1f} {on_track:<9} {answer:<7} {confidence:<11.0%} {reasoning:<30}")
        
        if result['on_track']:
            on_track_count += 1
        if result['answer'] == 'ERROR':
            error_count += 1
        total_confidence += confidence
    
    # Overall statistics
    print(f"\n" + "=" * 80)
    print("OVERALL PERFORMANCE")
    print("=" * 80)
    
    print(f"Total frames tested: {total_frames}")
    print(f"Videos tested: {len(test_videos)}")
    print(f"Person on track detections: {on_track_count}")
    print(f"Clear/safe detections: {total_frames - on_track_count - error_count}")
    print(f"Error responses: {error_count}")
    
    if total_frames > 0:
        detection_rate = on_track_count / total_frames * 100
        avg_confidence = total_confidence / total_frames
        error_rate = error_count / total_frames * 100
        
        print(f"Detection rate: {detection_rate:.1f}%")
        print(f"Average confidence: {avg_confidence:.0%}")
        print(f"Error rate: {error_rate:.1f}%")
    
    # Per-video breakdown
    print(f"\nPER-VIDEO ANALYSIS:")
    print("-" * 50)
    
    for video_path in test_videos:
        video_name = os.path.basename(video_path)
        video_results = [r for r in all_results if r['video'] == video_name]
        
        if video_results:
            on_track_frames = sum(1 for r in video_results if r['on_track'])
            total_video_frames = len(video_results)
            avg_video_confidence = sum(r['confidence'] for r in video_results) / len(video_results)
            
            print(f"{video_name}: {on_track_frames}/{total_video_frames} frames with person on track "
                  f"(avg confidence: {avg_video_confidence:.0%})")
    
    print(f"\n" + "=" * 80)
    print("FINAL ASSESSMENT")
    print("=" * 80)
    
    if error_rate < 10:
        print("+ EXCELLENT: Low error rate, detector is working reliably")
    elif error_rate < 25:
        print("+ GOOD: Acceptable error rate")
    else:
        print("- HIGH ERROR RATE: Needs improvement")
    
    if avg_confidence > 70:
        print("+ HIGH CONFIDENCE: Detector provides confident results")
    elif avg_confidence > 50:
        print("+ MODERATE CONFIDENCE: Results are reasonably confident")
    else:
        print("- LOW CONFIDENCE: Results may be unreliable")
    
    print(f"\nRECOMMENDATION:")
    if error_rate < 10 and avg_confidence > 70:
        print("✅ READY FOR PRODUCTION: Person on Track Detector is highly reliable")
        print("   - Use in Streamlit app for real-time track safety monitoring")
        print("   - Suitable for automated safety systems")
    elif error_rate < 25 and avg_confidence > 50:
        print("⚠️ SUITABLE WITH CAUTION: Good performance but monitor results")
        print("   - Use for preliminary screening")
        print("   - Consider human verification for critical decisions")
    else:
        print("❌ NEEDS IMPROVEMENT: Not reliable enough for production use")
        print("   - Improve keyword detection")
        print("   - Adjust confidence thresholds")
        print("   - Test with more diverse video content")
    
    print(f"\nNext steps:")
    print("1. Open http://localhost:8502")
    print("2. Select 'Person on Track Detector' from model dropdown")
    print("3. Upload test videos from test/ folder")
    print("4. Compare results with this analysis")
    
    return all_results

if __name__ == "__main__":
    test_person_on_track_final()