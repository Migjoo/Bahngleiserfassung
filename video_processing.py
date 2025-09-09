#!/usr/bin/env python3
"""
Video processing utilities for frame extraction and repair
"""
import cv2
import os
import tempfile
import subprocess
import streamlit as st
from PIL import Image
from typing import List, Dict


def repair_video_with_ffmpeg(input_path: str, output_path: str) -> bool:
    """
    Repair corrupted video by moving moov atom to the beginning
    """
    try:
        # Try to fix the video using FFmpeg
        cmd = [
            'ffmpeg', 
            '-i', input_path,
            '-c', 'copy',
            '-movflags', 'faststart',
            '-avoid_negative_ts', 'make_zero',
            '-y',  # Overwrite output file
            output_path
        ]
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=300  # 5 minute timeout
        )
        
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def extract_frames_from_video(video_file, fps: float = 1) -> List[Dict]:
    """
    Extract frames from video at specified FPS (default 1 frame per second)
    Automatically handles corrupted videos by attempting repair with FFmpeg
    """
    frames = []
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.read())
        tmp_file_path = tmp_file.name
    
    repaired_path = None
    
    try:
        # First attempt: try to open video directly
        cap = cv2.VideoCapture(tmp_file_path)
        
        # Check if video opened successfully and has frames
        if not cap.isOpened() or cap.get(cv2.CAP_PROP_FRAME_COUNT) == 0:
            cap.release()
            
            # Second attempt: try to repair the video with FFmpeg
            st.warning("Video appears corrupted (moov atom issue). Attempting repair...")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='_repaired.mp4') as repaired_file:
                repaired_path = repaired_file.name
            
            if repair_video_with_ffmpeg(tmp_file_path, repaired_path):
                st.success("Video repair successful! Processing frames...")
                cap = cv2.VideoCapture(repaired_path)
            else:
                st.error("Failed to repair video. FFmpeg may not be installed or video is severely corrupted.")
                return frames
        
        # Extract video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30  # Default fallback FPS
            
        frame_interval = int(video_fps / fps) if video_fps > fps else 1
        
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append({
                    'frame': pil_image,
                    'timestamp': frame_count / video_fps,
                    'frame_number': extracted_count
                })
                extracted_count += 1
            
            frame_count += 1
        
        cap.release()
        
    finally:
        # Clean up temporary files
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        if repaired_path and os.path.exists(repaired_path):
            os.unlink(repaired_path)
    
    return frames