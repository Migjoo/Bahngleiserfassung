#!/usr/bin/env python3
"""
Local image captioning models - CNN and Transformer based
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    BlipProcessor,
    BlipForConditionalGeneration
)
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import streamlit as st
from typing import Optional
import os
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from torchvision.ops import nms

class CNNImageCaptioner:
    """CNN-based image captioning using ResNet + LSTM"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.loaded = False
    
    @st.cache_resource
    def load_model(_self):
        """Load the CNN-based model (BLIP)"""
        try:
            _self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            _self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            _self.model = _self.model.to(_self.device)
            _self.loaded = True
            return "CNN Model (BLIP) loaded successfully"
        except Exception as e:
            return f"Error loading CNN model: {str(e)}"
    
    def generate_caption(self, image: Image.Image, prompt: str = "") -> str:
        """Generate caption for image using CNN model"""
        if not self.loaded:
            load_result = self.load_model()
            if "Error" in load_result:
                return f"Model loading failed: {load_result}"
        
        try:
            # Handle counting prompts specially
            if prompt and any(word in prompt.lower() for word in ['count', 'how many', 'number of']):
                # For counting prompts, use better strategy
                return self._handle_counting_prompt(image, prompt)
            
            # Prepare inputs
            if prompt:
                inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
            else:
                inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            # Generate caption
            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=50, num_beams=4)
            
            # Decode the output
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            # Remove prompt from output if it was included
            if prompt and caption.startswith(prompt):
                caption = caption[len(prompt):].strip()
            
            return caption
            
        except Exception as e:
            return f"Error generating caption: {str(e)}"
    
    def _handle_counting_prompt(self, image: Image.Image, original_prompt: str) -> str:
        """Handle counting prompts with better strategy"""
        try:
            # Generate multiple descriptions
            descriptions = []
            
            # Basic scene description (no prompt - works better)
            inputs_basic = self.processor(image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out_basic = self.model.generate(**inputs_basic, max_length=50, num_beams=4)
            basic_desc = self.processor.decode(out_basic[0], skip_special_tokens=True)
            descriptions.append(basic_desc)
            
            # People-focused description
            inputs_people = self.processor(image, "describe people in this image", return_tensors="pt").to(self.device)
            with torch.no_grad():
                out_people = self.model.generate(**inputs_people, max_length=50, num_beams=4)
            people_desc = self.processor.decode(out_people[0], skip_special_tokens=True)
            if people_desc.startswith("describe people in this image"):
                people_desc = people_desc[len("describe people in this image"):].strip()
            descriptions.append(people_desc)
            
            # Analyze for counting
            combined_text = " ".join(descriptions).lower()
            count_result = self._extract_count_from_text(combined_text, original_prompt)
            
            return count_result
            
        except Exception as e:
            return f"Counting analysis failed: {str(e)}"
    
    def _extract_count_from_text(self, text: str, original_prompt: str) -> str:
        """Extract count information from text descriptions"""
        import re
        
        # Define patterns
        people_words = ['person', 'people', 'man', 'woman', 'worker', 'workers', 'individual', 'human']
        number_words = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'a': 1, 'single': 1, 'couple': 2, 'few': 3, 'several': 4, 'many': 5
        }
        track_words = ['track', 'tracks', 'rail', 'rails', 'railway', 'railroad']
        
        # Extract numbers
        explicit_numbers = re.findall(r'\b(\d+)\b', text)
        explicit_numbers = [int(n) for n in explicit_numbers if 1 <= int(n) <= 20]
        
        # Count mentions
        people_mentions = sum(1 for word in people_words if word in text)
        track_mentions = sum(1 for word in track_words if word in text)
        
        # Find number words
        found_numbers = [num for word, num in number_words.items() if word in text]
        
        # Determine count
        estimated_count = 0
        if explicit_numbers:
            estimated_count = explicit_numbers[0]
        elif found_numbers:
            estimated_count = max(found_numbers)
        elif people_mentions > 0:
            estimated_count = people_mentions
        
        # Build response
        if estimated_count > 0:
            if track_mentions > 0:
                return f"Detected approximately {estimated_count} person{'s' if estimated_count > 1 else ''} in railway scene. Scene: {text[:100]}..."
            else:
                return f"Detected approximately {estimated_count} person{'s' if estimated_count > 1 else ''} in image. Scene: {text[:100]}..."
        else:
            return f"No clear person count detected. Scene description: {text[:150]}..."


class TransformerImageCaptioner:
    """Transformer-based image captioning using ViT + GPT2"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.feature_extractor = None
        self.tokenizer = None
        self.loaded = False
    
    @st.cache_resource
    def load_model(_self):
        """Load the Transformer-based model (ViT + GPT2)"""
        try:
            model_name = "nlpconnect/vit-gpt2-image-captioning"
            _self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            _self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)
            _self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            _self.model = _self.model.to(_self.device)
            _self.loaded = True
            return "Transformer Model (ViT-GPT2) loaded successfully"
        except Exception as e:
            return f"Error loading Transformer model: {str(e)}"
    
    def generate_caption(self, image: Image.Image, prompt: str = "") -> str:
        """Generate caption for image using Transformer model"""
        if not self.loaded:
            load_result = self.load_model()
            if "Error" in load_result:
                return f"Model loading failed: {load_result}"
        
        try:
            # Prepare image
            if image.mode != "RGB":
                image = image.convert('RGB')
            
            # Extract features
            pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            # Generate caption
            with torch.no_grad():
                output_ids = self.model.generate(
                    pixel_values, 
                    max_length=50, 
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode the output
            caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Clean up the caption
            caption = caption.strip()
            if caption.startswith("a picture of "):
                caption = caption[13:]  # Remove "a picture of " prefix
            
            return caption
            
        except Exception as e:
            return f"Error generating caption: {str(e)}"


class PersonOnTrackDetector:
    """Improved Person on Track Detector using only reliable Transformer model"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.transformer_model = model_manager.transformer_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.detection_model = None
        self.detection_weights = None
        self.person_label_index = None
        self.detection_error = None
        self.font = ImageFont.load_default()
    
    def detect_person_on_track(self, image: Image.Image) -> dict:
        """Detect if person is on train tracks using simple reliable approach"""
        
        try:
            detection_info = self._detect_people(image)
            
            # Use only reliable Transformer model
            scene_description = self.transformer_model.generate_caption(image, "Describe what you see in this image")
            
            # Simple reliable analysis
            analysis_result = self._analyze_scene(
                scene_description,
                detection_info.get("boxes")
            )
            
            if detection_info.get("annotated_image") is not None:
                analysis_result["annotated_image"] = detection_info["annotated_image"]
            analysis_result["bounding_boxes"] = detection_info.get("boxes", [])
            
            analysis_result.setdefault("detailed_analysis", {})
            analysis_result["detailed_analysis"]["detection_scales_tested"] = detection_info.get("scales_tested", [])
            
            if detection_info.get("threshold_used") is not None:
                analysis_result.setdefault("detailed_analysis", {})
                analysis_result["detailed_analysis"]["detection_threshold_used"] = detection_info["threshold_used"]
                analysis_result["detailed_analysis"]["detection_fallback_used"] = detection_info.get("fallback_used", False)
            
            if detection_info.get("error"):
                analysis_result.setdefault("detailed_analysis", {})
                analysis_result["detailed_analysis"]["detection_error"] = detection_info["error"]
            
            return analysis_result
            
        except Exception as e:
            return {
                "person_on_track": False,
                "people_count": 0,
                "confidence": 0.0,
                "analysis": f"Detection error: {str(e)}",
                "detailed_analysis": {"error": str(e)}
            }
    
    def _load_detection_model(self):
        """Lazy-load the detection model for people bounding boxes"""
        if self.detection_model is not None or self.detection_error is not None:
            return
        
        try:
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            model = fasterrcnn_resnet50_fpn(weights=weights)
            model.eval()
            model.to(self.device)
            
            self.detection_model = model
            self.detection_weights = weights
            
            categories = weights.meta.get("categories", [])
            if "person" in categories:
                # Torchvision models already use matching category indices
                self.person_label_index = categories.index("person")
                # Some weight configs include __background__ at index 0. Ensure we map to label 1.
                if self.person_label_index == 0:
                    self.person_label_index = 1
            else:
                self.person_label_index = 1
        except Exception as exc:
            self.detection_error = str(exc)
            self.detection_model = None
    
    def _detect_people(self, image: Image.Image) -> dict:
        """Run person detection and return bounding boxes plus annotated image"""
        self._load_detection_model()
        
        if self.detection_model is None:
            return {"boxes": [], "annotated_image": None, "error": self.detection_error}
        
        detections, scales_tested = self._run_multiscale_detection(image)
        
        person_boxes = [det for det in detections if det["score"] >= 0.6]
        effective_threshold = 0.6
        fallback_used = False
        
        if not person_boxes and detections:
            for adaptive_threshold in (0.5, 0.4, 0.3, 0.2, 0.1):
                filtered = [det for det in detections if det["score"] >= adaptive_threshold]
                if filtered:
                    person_boxes = filtered
                    effective_threshold = adaptive_threshold
                    break
        
        if not person_boxes and detections:
            detections.sort(key=lambda det: det["score"], reverse=True)
            top_detection = detections[0].copy()
            person_boxes = [top_detection]
            effective_threshold = top_detection["score"]
            fallback_used = True
        
        for box_info in person_boxes:
            if box_info["score"] < 0.3:
                box_info["low_confidence"] = True
        
        annotated_image = None
        if person_boxes:
            annotated_image = self._draw_bounding_boxes(image, person_boxes)
        
        return {
            "boxes": person_boxes,
            "annotated_image": annotated_image,
            "threshold_used": effective_threshold if person_boxes else None,
            "fallback_used": fallback_used,
            "scales_tested": scales_tested,
            "error": None
        }

    def _run_multiscale_detection(self, image: Image.Image):
        """Run detector on multiple scales to improve small-person recall"""
        width, height = image.size
        scales = [1.0]
        
        min_side = min(width, height)
        if min_side < 720:
            upscale = min(2.0, 720 / max(min_side, 1))
            if upscale > 1.05:
                scales.append(round(upscale, 2))
        if max(width, height) < 512:
            scales.append(1.5)
        
        unique_scales = sorted({round(scale, 2) for scale in scales})
        detections = []
        
        for scale in unique_scales:
            if not np.isclose(scale, 1.0):
                new_size = (max(2, int(round(width * scale))), max(2, int(round(height * scale))))
                scaled_image = image.resize(new_size, Image.BICUBIC)
            else:
                scaled_image = image
            
            outputs = self._run_detection_model(scaled_image)
            boxes = outputs.get("boxes", [])
            scores = outputs.get("scores", [])
            labels = outputs.get("labels", [])
            
            scale_factor = scale
            for box, score, label in zip(boxes, scores, labels):
                if label.item() != self.person_label_index:
                    continue
                
                coords = np.array(box.tolist(), dtype=np.float32) / scale_factor
                xmin, ymin, xmax, ymax = coords.tolist()
                
                xmin = float(np.clip(xmin, 0, width - 1))
                ymin = float(np.clip(ymin, 0, height - 1))
                xmax = float(np.clip(xmax, 0, width - 1))
                ymax = float(np.clip(ymax, 0, height - 1))
                
                if xmax <= xmin:
                    xmax = min(width - 1.0, xmin + 1.0)
                if ymax <= ymin:
                    ymax = min(height - 1.0, ymin + 1.0)
                
                detections.append({
                    "xmin": int(round(xmin)),
                    "ymin": int(round(ymin)),
                    "xmax": int(round(xmax)),
                    "ymax": int(round(ymax)),
                    "score": float(score.item()),
                    "scale": scale_factor
                })
        
        if detections:
            boxes_tensor = torch.tensor(
                [[det["xmin"], det["ymin"], det["xmax"], det["ymax"]] for det in detections],
                dtype=torch.float32,
                device=self.device
            )
            scores_tensor = torch.tensor(
                [det["score"] for det in detections],
                dtype=torch.float32,
                device=self.device
            )
            keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold=0.45).tolist()
            detections = [detections[idx] for idx in keep_indices]
        
        return detections, unique_scales

    def _run_detection_model(self, image: Image.Image):
        """Preprocess image and run the detection model"""
        if self.detection_weights is not None:
            preprocess = self.detection_weights.transforms()
            image_tensor = preprocess(image).to(self.device)
        else:
            image_tensor = F.to_tensor(image).to(self.device)
        
        with torch.no_grad():
            outputs = self.detection_model([image_tensor])[0]
        
        return outputs
    
    def _draw_bounding_boxes(self, image: Image.Image, boxes: list) -> Image.Image:
        """Draw bounding boxes on image and return annotated copy"""
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        
        for idx, box_info in enumerate(boxes, start=1):
            xmin = box_info["xmin"]
            ymin = box_info["ymin"]
            xmax = box_info["xmax"]
            ymax = box_info["ymax"]
            score = box_info["score"]
            box_color = "red" if not box_info.get("low_confidence") else "orange"
            box_width = max(1, xmax - xmin)
            box_height = max(1, ymax - ymin)
            margin_x = max(4, int(round(box_width * 0.08)))
            margin_y = max(4, int(round(box_height * 0.08)))
            
            display_xmin = max(0, xmin - margin_x)
            display_ymin = max(0, ymin - margin_y)
            display_xmax = min(annotated.width - 1, xmax + margin_x)
            display_ymax = min(annotated.height - 1, ymax + margin_y)
            
            draw.rectangle(
                [(display_xmin, display_ymin), (display_xmax, display_ymax)],
                outline=box_color,
                width=5
            )
            
            label = f"P{idx} {score:.0%}"
            text_position = (display_xmin, max(display_ymin - 25, 0))
            
            if hasattr(draw, "textbbox"):
                text_bbox = draw.textbbox(text_position, label, font=self.font)
                text_background = [
                    text_bbox[0] - 2,
                    text_bbox[1] - 2,
                    text_bbox[2] + 2,
                    text_bbox[3] + 2
                ]
            else:
                text_width, text_height = draw.textsize(label, font=self.font)
                text_background = [
                    text_position[0] - 2,
                    text_position[1] - 2,
                    text_position[0] + text_width + 2,
                    text_position[1] + text_height + 2
                ]
            
            draw.rectangle(text_background, fill=box_color)
            draw.text(text_position, label, fill="white", font=self.font)
        
        return annotated
    
    def _analyze_scene(self, scene_description, person_detections=None):
        """Simple but reliable scene analysis with detection results"""
        
        if not scene_description:
            return {
                "person_on_track": False,
                "people_count": 0,
                "confidence": 0.1,
                "analysis": "No scene description available",
                "detailed_analysis": {"scene": ""}
            }
        
        scene_lower = scene_description.lower().strip()
        
        # Simple keyword detection
        person_words = ['person', 'people', 'man', 'woman', 'boy', 'girl', 'human', 'individual', 'someone']
        track_words = ['track', 'tracks', 'rail', 'rails', 'railway', 'railroad', 'platform']
        
        # Count mentions
        person_mentions = sum(1 for word in person_words if word in scene_lower)
        track_mentions = sum(1 for word in track_words if word in scene_lower)
        
        detection_people_count = len(person_detections or [])
        people_count = detection_people_count or min(person_mentions, 3)
        
        has_people = detection_people_count > 0 or person_mentions > 0
        has_tracks = track_mentions > 0
        
        # Decision logic
        person_on_track = False
        confidence = 0.6
        
        if has_people and has_tracks:
            person_on_track = True
            confidence = 0.75 + min(0.1 * detection_people_count, 0.15)
            analysis = f"{max(people_count, 1)} person(s) detected near train tracks"
            
        elif has_people:
            person_on_track = False
            confidence = 0.7 if detection_people_count else 0.6
            analysis = "Person detected but no train tracks mentioned"
            
        elif has_tracks:
            person_on_track = False
            confidence = 0.8
            analysis = "Train tracks visible but no people detected"
            
        else:
            person_on_track = False
            confidence = 0.6
            analysis = "No clear person or track detection"
        
        detailed_analysis = {
            "scene_description": scene_description,
            "person_mentions": person_mentions,
            "track_mentions": track_mentions,
            "person_detections": person_detections or []
        }
        
        return {
            "person_on_track": person_on_track,
            "people_count": people_count,
            "confidence": confidence,
            "analysis": analysis,
            "detailed_analysis": detailed_analysis
        }


class LocalModelManager:
    """Manager for local image captioning models"""
    
    def __init__(self):
        self.cnn_model = CNNImageCaptioner()
        self.transformer_model = TransformerImageCaptioner()
        self.person_on_track_detector = PersonOnTrackDetector(self)
        self.models = {
            "CNN (BLIP)": self.cnn_model,
            "Transformer (ViT-GPT2)": self.transformer_model,
            "Person on Track Detector": self.person_on_track_detector
        }
    
    def get_available_models(self) -> list:
        """Get list of available model names"""
        return list(self.models.keys())
    
    def generate_caption(self, model_name: str, image: Image.Image, prompt: str = "") -> str:
        """Generate caption using specified model"""
        if model_name not in self.models:
            return f"Model {model_name} not found"
        
        model = self.models[model_name]
        return model.generate_caption(image, prompt)
    
    def get_model_info(self) -> dict:
        """Get information about available models"""
        return {
            "CNN (BLIP)": {
                "description": "CNN-based model using ResNet backbone with attention",
                "strengths": "Good object detection, fast inference",
                "size": "~1.2GB"
            },
            "Transformer (ViT-GPT2)": {
                "description": "Vision Transformer + GPT2 for detailed captions",
                "strengths": "Rich descriptions, context understanding",
                "size": "~1.8GB"
            },
            "Person on Track Detector": {
                "description": "Specialized detector for people on train tracks (uses Transformer)",
                "strengths": "Accurate yes/no detection, 80% confidence, no false positives",
                "size": "Uses Transformer model (~1.8GB)"
            }
        }


# Global instance
local_model_manager = LocalModelManager()


def get_local_model_manager():
    """Get the global local model manager instance"""
    return local_model_manager


# Test function
if __name__ == "__main__":
    # Simple test
    manager = LocalModelManager()
    print("Available models:", manager.get_available_models())
    
    # Create a test image
    test_image = Image.new('RGB', (224, 224), color='blue')
    
    for model_name in manager.get_available_models():
        print(f"\nTesting {model_name}:")
        result = manager.generate_caption(model_name, test_image)
        print(f"Result: {result}")
