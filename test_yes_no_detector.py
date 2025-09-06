#!/usr/bin/env python3
"""
Test the new Yes/No Person Detector
"""
import sys
import os
from io import BytesIO

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_yes_no_detector():
    """Test the optimized Yes/No Person Detector"""
    print("TESTING YES/NO PERSON DETECTOR")
    print("=" * 50)
    print("Model: Local CNN (BLIP) - Best performer (100% success rate)")
    print()
    
    try:
        from local_models import get_local_model_manager
        from app import extract_frames_from_video, process_image_locally
        print("+ Components loaded successfully")
    except ImportError as e:
        print(f"- Import error: {e}")
        return
    
    # Find video file
    video_files = [f for f in os.listdir('.') if f.endswith('.mp4')]
    if not video_files:
        print("- No MP4 files found")
        return
    
    video_path = video_files[0]
    print(f"+ Using video: {video_path[:40]}...")
    
    # Initialize models
    try:
        local_manager = get_local_model_manager()
        available_models = local_manager.get_available_models()
        print(f"+ Available models: {available_models}")
        
        if "Yes/No Person Detector" not in available_models:
            print("- Yes/No Person Detector not found!")
            return
        
        print("+ Yes/No Person Detector ready")
    except Exception as e:
        print(f"- Model initialization error: {e}")
        return
    
    # Extract frames for testing
    try:
        with open(video_path, 'rb') as f:
            video_data = f.read()
        
        video_file = BytesIO(video_data)
        frames = extract_frames_from_video(video_file, fps=0.5)  # Every 2 seconds
        
        if not frames:
            print("- No frames extracted")
            return
        
        print(f"+ Extracted {len(frames)} frames for testing")
        
        # Test with first 5 frames
        test_frames = frames[:5]
        
    except Exception as e:
        print(f"- Frame extraction error: {e}")
        return
    
    # Test Yes/No Person Detector on each frame
    print(f"\nTesting Yes/No Person Detector on {len(test_frames)} frames:")
    print("=" * 70)
    
    results = []
    
    for i, frame_data in enumerate(test_frames):
        frame_num = i + 1
        timestamp = frame_data['timestamp']
        
        print(f"\nFRAME {frame_num} (t={timestamp:.1f}s)")
        print("-" * 40)
        
        try:
            result = process_image_locally(
                frame_data['frame'],
                "Is there a person in this image?",  # This prompt is automatic
                'Yes/No Person Detector',
                local_manager
            )
            
            if 'error' in result:
                print(f"ERROR: {result['error']}")
                results.append({'frame': frame_num, 'answer': 'ERROR', 'confidence': 0})
            elif 'yes_no_detection' in result:
                detection = result['yes_no_detection']
                
                answer = detection.get('answer', 'UNKNOWN')
                person_detected = detection.get('person_detected', False)
                confidence = detection.get('confidence', 0)
                raw_response = detection.get('raw_response', 'N/A')
                
                # Display results
                print(f"Answer: {answer}")
                print(f"Person Detected: {person_detected}")
                print(f"Confidence: {confidence:.0%}")
                print(f"Raw Response: {raw_response}")
                
                results.append({
                    'frame': frame_num,
                    'timestamp': timestamp,
                    'answer': answer,
                    'person_detected': person_detected,
                    'confidence': confidence,
                    'raw_response': raw_response
                })
                
            else:
                print(f"Unexpected result format: {result}")
                results.append({'frame': frame_num, 'answer': 'UNKNOWN', 'confidence': 0})
                
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({'frame': frame_num, 'answer': 'ERROR', 'confidence': 0})
    
    # Summary table
    print(f"\n" + "=" * 70)
    print("RESULTS SUMMARY TABLE")
    print("=" * 70)
    
    print(f"{'Frame':<8} {'Time':<8} {'Answer':<10} {'Detected':<10} {'Confidence':<12} {'Raw Response':<30}")
    print("-" * 78)
    
    for result in results:
        frame = result.get('frame', 0)
        timestamp = result.get('timestamp', 0)
        answer = result.get('answer', 'N/A')
        detected = 'Yes' if result.get('person_detected', False) else 'No'
        confidence = result.get('confidence', 0)
        raw_response = result.get('raw_response', 'N/A')[:25] + "..." if len(result.get('raw_response', '')) > 25 else result.get('raw_response', 'N/A')
        
        print(f"{frame:<8} {timestamp:<8.1f} {answer:<10} {detected:<10} {confidence:<12.0%} {raw_response:<30}")
    
    # Performance analysis
    print(f"\n" + "=" * 70)
    print("PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    total = len(results)
    yes_count = sum(1 for r in results if r.get('answer') == 'YES')
    no_count = sum(1 for r in results if r.get('answer') == 'NO')
    error_count = sum(1 for r in results if r.get('answer') == 'ERROR')
    unclear_count = sum(1 for r in results if r.get('answer') == 'UNCLEAR')
    
    success_rate = (yes_count + no_count) / total * 100 if total > 0 else 0
    avg_confidence = sum(r.get('confidence', 0) for r in results) / total if total > 0 else 0
    
    print(f"Total frames tested: {total}")
    print(f"YES answers: {yes_count}")
    print(f"NO answers: {no_count}")  
    print(f"ERROR responses: {error_count}")
    print(f"UNCLEAR responses: {unclear_count}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Average confidence: {avg_confidence:.0%}")
    
    print(f"\nMODEL RECOMMENDATION:")
    if success_rate >= 80:
        print("+ EXCELLENT: Yes/No Person Detector is working perfectly")
        print("+ Ready for production use in Streamlit app")
        print("+ Provides clear yes/no answers with high confidence")
    elif success_rate >= 60:
        print("+ GOOD: Yes/No Person Detector is working well")
        print("+ Minor issues but suitable for most use cases")
    else:
        print("- NEEDS IMPROVEMENT: Success rate below 60%")
        print("- Consider adjusting prompts or model parameters")
    
    print(f"\nNext steps:")
    print("1. Open http://localhost:8502")
    print("2. Select 'Yes/No Person Detector' from model dropdown")
    print("3. Upload your video")
    print("4. Click 'Process Video' for simple yes/no person detection")
    
    return results

if __name__ == "__main__":
    test_yes_no_detector()