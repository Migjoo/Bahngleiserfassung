#!/usr/bin/env python3
"""
Improved Person on Track Detector using a completely different approach
Instead of relying on text descriptions, use multiple specific questions and cross-validation
"""
import sys
import os
from io import BytesIO
from PIL import Image

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class ImprovedPersonOnTrackDetector:
    """Much better person-on-track detector using multiple validation approaches"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.cnn_model = model_manager.cnn_model
        self.transformer_model = model_manager.transformer_model
    
    def detect_person_on_track(self, image: Image.Image) -> dict:
        """Improved detection using multiple specific questions and validation"""
        
        try:
            # APPROACH 1: Multiple specific questions to CNN model
            questions = [
                "Are there any people visible in this image?",
                "Is anyone standing on railway tracks?", 
                "Do you see a person on train tracks?",
                "Are the train tracks empty of people?",
                "Is this image showing people near trains?"
            ]
            
            cnn_responses = {}
            for i, question in enumerate(questions):
                response = self.cnn_model.generate_caption(image, question)
                cleaned_response = self._clean_response(response, question)
                cnn_responses[f"q{i+1}"] = {
                    "question": question,
                    "response": cleaned_response,
                    "analysis": self._analyze_yes_no_response(cleaned_response, question)
                }
            
            # APPROACH 2: Use Transformer for scene description
            scene_description = self.transformer_model.generate_caption(image, "Describe this scene in detail")
            
            # APPROACH 3: Use CNN for object detection
            objects_response = self.cnn_model.generate_caption(image, "What objects do you see in this image?")
            objects_cleaned = self._clean_response(objects_response, "What objects do you see in this image?")
            
            # COMBINE ALL APPROACHES
            final_analysis = self._combine_all_analyses(cnn_responses, scene_description, objects_cleaned)
            
            return final_analysis
            
        except Exception as e:
            return {
                "person_on_track": False,
                "people_count": 0,
                "confidence": 0.0,
                "analysis": f"Detection failed: {str(e)}",
                "detailed_analysis": {"error": str(e)}
            }
    
    def _clean_response(self, response, original_question):
        """Remove question repetition and extract meaningful response"""
        if not response:
            return ""
        
        response = response.strip()
        question_lower = original_question.lower()
        response_lower = response.lower()
        
        # If response is just the question, return empty
        if response_lower == question_lower:
            return ""
        
        # If response starts with the question, remove it
        if response_lower.startswith(question_lower):
            cleaned = response[len(original_question):].strip()
            return cleaned.lstrip('?.,!:') if cleaned else ""
        
        # If response contains too many words from the question, likely repetition
        question_words = set(question_lower.split())
        response_words = set(response_lower.split())
        overlap = len(question_words.intersection(response_words))
        
        if len(response_words) < 10 and overlap > len(question_words) * 0.6:
            return ""  # Likely question repetition
        
        return response
    
    def _analyze_yes_no_response(self, response, question):
        """Analyze response to extract yes/no meaning"""
        if not response:
            return {"answer": "UNCLEAR", "confidence": 0.1}
        
        response_lower = response.lower().strip()
        
        # Direct yes/no answers
        if response_lower in ["yes", "no"]:
            return {"answer": response_lower.upper(), "confidence": 0.9}
        
        # Check for yes indicators
        yes_indicators = ["yes", "there is", "there are", "i see", "visible", "present", "standing", "person"]
        no_indicators = ["no", "not", "none", "empty", "clear", "nobody", "no one", "absent"]
        
        yes_score = sum(1 for indicator in yes_indicators if indicator in response_lower)
        no_score = sum(1 for indicator in no_indicators if indicator in response_lower)
        
        if yes_score > no_score:
            confidence = min(0.7, 0.4 + yes_score * 0.1)
            return {"answer": "YES", "confidence": confidence}
        elif no_score > yes_score:
            confidence = min(0.7, 0.4 + no_score * 0.1)
            return {"answer": "NO", "confidence": confidence}
        else:
            return {"answer": "UNCLEAR", "confidence": 0.3}
    
    def _combine_all_analyses(self, cnn_responses, scene_description, objects_response):
        """Combine all analysis approaches to make final decision"""
        
        # Count YES/NO responses from CNN questions
        yes_count = 0
        no_count = 0
        unclear_count = 0
        total_confidence = 0
        
        question_results = []
        for key, response_data in cnn_responses.items():
            analysis = response_data["analysis"]
            answer = analysis["answer"]
            confidence = analysis["confidence"]
            
            if answer == "YES":
                yes_count += 1
            elif answer == "NO":
                no_count += 1
            else:
                unclear_count += 1
            
            total_confidence += confidence
            question_results.append({
                "question": response_data["question"],
                "response": response_data["response"],
                "answer": answer,
                "confidence": confidence
            })
        
        # Analyze scene description for people/track keywords
        scene_lower = scene_description.lower()
        people_keywords = ["person", "people", "man", "woman", "human", "individual"]
        track_keywords = ["track", "tracks", "rail", "railway", "train"]
        
        people_in_scene = any(keyword in scene_lower for keyword in people_keywords)
        tracks_in_scene = any(keyword in scene_lower for keyword in track_keywords)
        
        # Analyze objects response
        objects_lower = objects_response.lower() if objects_response else ""
        people_in_objects = any(keyword in objects_lower for keyword in people_keywords)
        
        # DECISION LOGIC - Much more sophisticated
        person_on_track = False
        people_count = 0
        confidence = 0.3
        
        # Method 1: Majority vote from specific questions
        total_responses = yes_count + no_count + unclear_count
        if total_responses > 0:
            yes_percentage = yes_count / total_responses
            no_percentage = no_count / total_responses
            
            if yes_percentage >= 0.6:  # 60% or more say YES
                person_on_track = True
                confidence = 0.6 + yes_percentage * 0.2
                analysis = f"Multiple questions confirm person presence ({yes_count}/{total_responses} positive)"
                people_count = min(yes_count, 3)  # Estimate based on positive responses
                
            elif no_percentage >= 0.6:  # 60% or more say NO
                person_on_track = False
                confidence = 0.6 + no_percentage * 0.2
                analysis = f"Multiple questions confirm no person on tracks ({no_count}/{total_responses} negative)"
                people_count = 0
                
            else:
                # Mixed responses - use secondary validation
                if people_in_scene and tracks_in_scene:
                    person_on_track = True
                    confidence = 0.5
                    analysis = f"Scene analysis suggests person near tracks (mixed question results)"
                    people_count = 1
                else:
                    person_on_track = False
                    confidence = 0.4
                    analysis = f"Unclear from questions, scene analysis suggests safe"
                    people_count = 0
        
        # Method 2: Cross-validation with scene description
        if people_in_scene and tracks_in_scene and not person_on_track:
            # Scene suggests people + tracks but questions said no - be conservative
            person_on_track = False
            analysis = f"Scene mentions people and tracks but specific questions indicate safe"
            confidence = max(confidence, 0.5)
        
        elif not people_in_scene and person_on_track:
            # Questions said yes but scene doesn't mention people - lower confidence
            confidence *= 0.7
            analysis = f"Questions suggest person present but scene unclear"
        
        # Method 3: Object detection validation
        if people_in_objects and not people_in_scene and not person_on_track:
            # Objects mention people but scene doesn't - possible person present
            person_on_track = True
            confidence = 0.4
            analysis = f"Object detection suggests person presence"
            people_count = 1
        
        # Final confidence adjustment
        avg_question_confidence = total_confidence / max(len(cnn_responses), 1)
        confidence = (confidence + avg_question_confidence) / 2
        
        return {
            "person_on_track": person_on_track,
            "people_count": people_count,
            "confidence": min(confidence, 1.0),
            "analysis": analysis,
            "detailed_analysis": {
                "question_results": question_results,
                "yes_responses": yes_count,
                "no_responses": no_count,
                "unclear_responses": unclear_count,
                "scene_description": scene_description,
                "people_in_scene": people_in_scene,
                "tracks_in_scene": tracks_in_scene,
                "objects_response": objects_response,
                "people_in_objects": people_in_objects
            }
        }


def test_improved_detector():
    """Test the improved detector approach"""
    print("TESTING IMPROVED PERSON ON TRACK DETECTOR")
    print("=" * 60)
    print("Using multiple questions + scene analysis + object detection")
    print()
    
    try:
        from local_models import get_local_model_manager
        from app import extract_frames_from_video
        
        local_manager = get_local_model_manager()
        improved_detector = ImprovedPersonOnTrackDetector(local_manager)
        print("+ Improved detector ready")
    except Exception as e:
        print(f"- Setup error: {e}")
        return
    
    # Test with first video
    video_path = "test\\1.mp4"
    if not os.path.exists(video_path):
        print(f"- Video not found: {video_path}")
        return
    
    try:
        with open(video_path, 'rb') as f:
            video_data = f.read()
        
        video_file = BytesIO(video_data)
        frames = extract_frames_from_video(video_file, fps=0.5)
        
        if not frames:
            print("- No frames extracted")
            return
        
        frame_data = frames[0]
        print(f"+ Testing frame at {frame_data['timestamp']:.1f}s")
        
        # Test improved detector
        result = improved_detector.detect_person_on_track(frame_data['frame'])
        
        print(f"\n" + "=" * 50)
        print("IMPROVED DETECTOR RESULTS")
        print("=" * 50)
        
        analysis = result.get('analysis', 'No analysis')
        people_count = result.get('people_count', 0)
        confidence = result.get('confidence', 0)
        person_on_track = result.get('person_on_track', False)
        
        if person_on_track:
            print(f"🚨 ALERT: {analysis}")
        else:
            print(f"✅ SAFE: {analysis}")
        
        print(f"👥 People Count: {people_count}")
        print(f"📊 Confidence: {confidence:.0%}")
        
        # Show detailed analysis
        detailed = result.get('detailed_analysis', {})
        if 'question_results' in detailed:
            print(f"\n📋 Question Analysis:")
            for q_result in detailed['question_results']:
                print(f"  Q: {q_result['question']}")
                print(f"  A: {q_result['answer']} ({q_result['confidence']:.0%}) - {q_result['response'][:50]}...")
        
        print(f"\n🎯 This approach should be much more accurate!")
        
    except Exception as e:
        print(f"- Test error: {e}")

if __name__ == "__main__":
    test_improved_detector()