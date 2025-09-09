#!/usr/bin/env python3
"""
Ontology integration module for scene analysis and risk assessment
"""
from ontology_eval import Observation, evaluate, Severity
from typing import Dict, Any, Optional


def analyze_scene_with_ontology(scene_description: str, use_ontology: bool = True) -> Dict[str, Any]:
    """
    Analyze scene description using ontology-based evaluation
    Returns classification and explanation
    """
    if not use_ontology:
        return {
            "severity": "NONE",
            "severity_icon": "✅",
            "score": 0,
            "explanation": "Ontology-based analysis skipped",
            "ontology_used": False,
            "raw_description": scene_description
        }
    
    # Extract relevant information from scene description for ontology
    scene_lower = scene_description.lower().strip() if scene_description else ""
    
    # Initialize observation based on scene analysis
    obs = _extract_ontology_features(scene_lower)
    
    # Evaluate using ontology
    decision = evaluate(obs)
    
    # Map severity to icons and colors
    severity_mapping = {
        Severity.NONE: {"icon": "✅", "color": "green"},
        Severity.LOW: {"icon": "🟢", "color": "lightgreen"}, 
        Severity.MEDIUM: {"icon": "🟠", "color": "orange"},
        Severity.HIGH: {"icon": "⚠️", "color": "red"},
        Severity.CRITICAL: {"icon": "🚨", "color": "darkred"}
    }
    
    severity_info = severity_mapping[decision.severity]
    
    return {
        "severity": decision.severity.name,
        "severity_icon": severity_info["icon"],
        "severity_color": severity_info["color"],
        "score": decision.score_0_100,
        "labels": [label.value for label in decision.labels],
        "explanations": decision.explanations,
        "fired_rules": decision.fired_rules,
        "ontology_used": True,
        "raw_description": scene_description,
        "observation": obs,
        "decision": decision
    }


def _extract_ontology_features(scene_lower: str) -> Observation:
    """
    Extract ontology-relevant features from scene description
    """
    # Initialize observation
    obs = Observation()
    
    # Define keyword categories
    person_words = ['person', 'people', 'man', 'woman', 'boy', 'girl', 'human', 'individual', 'someone']
    track_words = ['track', 'tracks', 'rail', 'rails', 'railway', 'railroad']
    platform_words = ['platform', 'station', 'bahnsteig']
    danger_words = ['fallen', 'lying', 'down', 'accident', 'emergency']
    fire_words = ['fire', 'smoke', 'flames', 'burning']
    crowd_words = ['crowd', 'many people', 'group', 'mehrere personen']
    safe_words = ['no people', 'empty', 'clear', 'safe', 'nobody', 'without people']
    
    # Count keyword mentions
    person_mentions = sum(1 for word in person_words if word in scene_lower)
    track_mentions = sum(1 for word in track_words if word in scene_lower)
    platform_mentions = sum(1 for word in platform_words if word in scene_lower)
    danger_mentions = sum(1 for word in danger_words if word in scene_lower)
    fire_mentions = sum(1 for word in fire_words if word in scene_lower)
    crowd_mentions = sum(1 for word in crowd_words if word in scene_lower)
    safe_mentions = sum(1 for word in safe_words if word in scene_lower)
    
    # Person on track detection (but not if explicitly safe)
    if person_mentions > 0 and track_mentions > 0 and safe_mentions == 0:
        obs.on_track_person = _calculate_person_on_track_confidence(scene_lower, person_mentions, track_mentions)
    
    # Fallen person detection
    if person_mentions > 0 and danger_mentions > 0:
        obs.fallen_person = min(0.7, 0.4 + danger_mentions * 0.1)
    
    # Fire/smoke detection
    if fire_mentions > 0:
        obs.smoke_or_fire = min(0.8, 0.5 + fire_mentions * 0.15)
    
    # Crowd detection
    if crowd_mentions > 0 and (track_mentions > 0 or platform_mentions > 0):
        obs.crowd_on_track = min(0.7, 0.4 + crowd_mentions * 0.1)
    
    # Generic object detection (if no person but something mentioned on tracks)
    if track_mentions > 0 and person_mentions == 0 and any(word in scene_lower for word in ['object', 'item', 'thing', 'debris']):
        obs.object_on_track = 0.6
    
    return obs


def _calculate_person_on_track_confidence(scene_lower: str, person_mentions: int, track_mentions: int) -> float:
    """
    Calculate confidence for person on track detection based on specific indicators
    """
    # Check for specific on-track indicators
    on_track_indicators = ['on track', 'on the track', 'on rails', 'on the rails', 'standing on', 'walking on']
    on_track_specific = sum(1 for phrase in on_track_indicators if phrase in scene_lower)
    
    if on_track_specific > 0:
        return min(0.8, 0.6 + on_track_specific * 0.1)
    else:
        # Check for proximity indicators
        near_indicators = ['near', 'close to', 'next to', 'beside', 'by the']
        near_mentions = sum(1 for phrase in near_indicators if phrase in scene_lower)
        
        if near_mentions > 0:
            # Person near tracks but not necessarily on them - lower confidence
            return min(0.4, 0.25 + near_mentions * 0.05)
        else:
            # Just mention of person and tracks together - very low confidence
            return min(0.3, 0.2 + (person_mentions + track_mentions) * 0.02)


def extract_scene_description(result: Dict[str, Any]) -> str:
    """
    Extract scene description from various model result formats
    """
    scene_description = ""
    
    if 'person_on_track_detection' in result:
        # For person detection results, use the analysis text
        scene_description = result['person_on_track_detection'].get('detailed_analysis', {}).get('scene_description', '')
    elif 'generated_text' in result:
        scene_description = result['generated_text']
    elif isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
        scene_description = result[0]['generated_text']
    
    return scene_description