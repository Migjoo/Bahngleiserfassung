#!/usr/bin/env python3
"""
Debug why the person-on-track detector always gives false positives
"""
import sys
import os
from io import BytesIO
import glob

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def debug_false_positives():
    """Debug why detector always says YES"""
    print("DEBUGGING FALSE POSITIVES IN PERSON-ON-TRACK DETECTOR")
    print("=" * 60)
    
    try:
        from local_models import get_local_model_manager
        from app import extract_frames_from_video, process_image_locally
        print("+ Components loaded successfully")
    except ImportError as e:
        print(f"- Import error: {e}")
        return
    
    # Test with one video to see raw model responses
    test_videos = glob.glob("test\\*.mp4")
    if not test_videos:
        print("- No test videos found")
        return
    
    video_path = test_videos[0]  # Use first video
    video_name = os.path.basename(video_path)
    print(f"+ Debugging with: {video_name}")
    
    try:
        local_manager = get_local_model_manager()
        print("+ Models ready")
    except Exception as e:
        print(f"- Model error: {e}")
        return
    
    # Extract one frame for detailed analysis
    try:
        with open(video_path, 'rb') as f:
            video_data = f.read()
        
        video_file = BytesIO(video_data)
        frames = extract_frames_from_video(video_file, fps=0.5)
        
        if not frames:
            print("- No frames extracted")
            return
        
        frame_data = frames[0]  # Use first frame
        print(f"+ Using frame at {frame_data['timestamp']:.1f}s for detailed analysis")
        
    except Exception as e:
        print(f"- Frame extraction error: {e}")
        return
    
    # Test the three individual model responses that the detector uses
    print(f"\n" + "=" * 60)
    print("DETAILED MODEL RESPONSE ANALYSIS")
    print("=" * 60)
    
    # Test 1: CNN Safety prompt
    print(f"\n1. CNN SAFETY ANALYSIS:")
    print("-" * 30)
    try:
        safety_result = process_image_locally(
            frame_data['frame'],
            "Describe any safety concerns with people near train tracks",
            'CNN (BLIP)',
            local_manager
        )
        safety_response = safety_result.get('generated_text', 'No response')
        print(f"Raw Response: '{safety_response}'")
        
        # Manual keyword analysis
        safety_lower = safety_response.lower()
        person_keywords = ['person', 'people', 'man', 'woman', 'human']
        track_keywords = ['track', 'tracks', 'rail', 'railway']
        danger_keywords = ['on track', 'standing on', 'danger', 'unsafe']
        
        person_count = sum(1 for kw in person_keywords if kw in safety_lower)
        track_count = sum(1 for kw in track_keywords if kw in safety_lower)
        danger_count = sum(1 for kw in danger_keywords if kw in safety_lower)
        
        print(f"Keywords found - Person: {person_count}, Track: {track_count}, Danger: {danger_count}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: Transformer descriptive
    print(f"\n2. TRANSFORMER DESCRIPTIVE ANALYSIS:")
    print("-" * 30)
    try:
        desc_result = process_image_locally(
            frame_data['frame'],
            "Describe people and train tracks in this image",
            'Transformer (ViT-GPT2)',
            local_manager
        )
        desc_response = desc_result.get('generated_text', 'No response')
        print(f"Raw Response: '{desc_response}'")
        
        # Manual keyword analysis
        desc_lower = desc_response.lower()
        person_count = sum(1 for kw in person_keywords if kw in desc_lower)
        track_count = sum(1 for kw in track_keywords if kw in desc_lower)
        danger_count = sum(1 for kw in danger_keywords if kw in desc_lower)
        
        print(f"Keywords found - Person: {person_count}, Track: {track_count}, Danger: {danger_count}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 3: CNN Direct question
    print(f"\n3. CNN DIRECT QUESTION:")
    print("-" * 30)
    try:
        direct_result = process_image_locally(
            frame_data['frame'],
            "Is there a person standing on train tracks? Answer yes or no.",
            'CNN (BLIP)',
            local_manager
        )
        direct_response = direct_result.get('generated_text', 'No response')
        print(f"Raw Response: '{direct_response}'")
        
        # Check for yes/no
        direct_lower = direct_response.lower()
        has_yes = 'yes' in direct_lower
        has_no = 'no' in direct_lower
        print(f"Contains 'yes': {has_yes}, Contains 'no': {has_no}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 4: Full Person on Track Detector
    print(f"\n4. FULL PERSON-ON-TRACK DETECTOR:")
    print("-" * 30)
    try:
        full_result = process_image_locally(
            frame_data['frame'],
            "Track Safety Analysis",
            'Person on Track Detector',
            local_manager
        )
        
        if 'person_on_track_detection' in full_result:
            detection = full_result['person_on_track_detection']
            
            print(f"Final Result: {detection.get('answer', 'UNKNOWN')}")
            print(f"Person on Track: {detection.get('person_on_track', False)}")
            print(f"Confidence: {detection.get('confidence', 0):.0%}")
            print(f"Reasoning: {detection.get('reasoning', 'No reasoning')}")
            
            # Show detailed analysis
            detailed = detection.get('detailed_analysis', {})
            if detailed:
                print(f"\nDetailed Analysis:")
                print(f"  Person keywords found: {detailed.get('person_keywords_found', 0)}")
                print(f"  Track keywords found: {detailed.get('track_keywords_found', 0)}")
                print(f"  Danger position keywords: {detailed.get('danger_position_keywords', 0)}")
                print(f"  Safety concern keywords: {detailed.get('safety_concern_keywords', 0)}")
                print(f"  Direct YES indicators: {detailed.get('direct_yes_indicators', 0)}")
                print(f"  Direct NO indicators: {detailed.get('direct_no_indicators', 0)}")
        else:
            print(f"Unexpected result format: {full_result}")
            
    except Exception as e:
        print(f"Error: {e}")
    
    print(f"\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    
    print("POTENTIAL ISSUES:")
    print("1. Models might be describing the train station/platform scene generally")
    print("2. Keywords like 'track' and 'person' might appear even when person is NOT on track")
    print("3. CNN model might be giving the prompt back instead of actual analysis")
    print("4. Decision logic might be too aggressive in detecting positive cases")
    
    print(f"\nRECOMMENDATIONS:")
    print("1. Check if models are actually analyzing the specific scenario")
    print("2. Tighten keyword matching to require specific combinations")
    print("3. Add negative indicators (person NOT on track)")
    print("4. Test with images that clearly have no people")
    print("5. Require higher confidence thresholds for positive detection")

if __name__ == "__main__":
    debug_false_positives()