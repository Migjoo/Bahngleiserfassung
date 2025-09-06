#!/usr/bin/env python3
"""
Simple test without downloading models
"""
import sys
import os
from PIL import Image

def test_basic_functionality():
    """Test basic imports and functionality"""
    print("Testing basic functionality...")
    
    # Test PIL
    try:
        test_image = Image.new('RGB', (224, 224), color='blue')
        print("+ PIL Image creation works")
    except Exception as e:
        print(f"- PIL Error: {e}")
        return False
    
    # Test file operations
    try:
        with open('test_file.txt', 'w') as f:
            f.write('test')
        os.remove('test_file.txt')
        print("+ File operations work")
    except Exception as e:
        print(f"- File operation error: {e}")
        return False
    
    # Test video file detection
    video_files = [f for f in os.listdir('.') if f.endswith('.mp4')]
    if video_files:
        print(f"+ Found video file: {video_files[0]}")
    else:
        print("! No video files found")
    
    # Test settings file
    if os.path.exists('settings.json'):
        print("+ Settings file exists")
    else:
        print("! Settings file not found")
    
    return True

def test_app_imports():
    """Test if app components can be imported"""
    print("\nTesting app imports...")
    
    try:
        # Test basic app imports without torch dependencies
        import json
        import tempfile
        import subprocess
        print("+ Basic Python modules import correctly")
    except Exception as e:
        print(f"- Basic import error: {e}")
        return False
    
    try:
        import streamlit as st
        print("+ Streamlit imports correctly")
    except Exception as e:
        print(f"- Streamlit import error: {e}")
        return False
    
    try:
        import cv2
        print("+ OpenCV imports correctly")
    except Exception as e:
        print(f"- OpenCV import error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Simple Test Suite")
    print("=" * 30)
    
    basic_ok = test_basic_functionality()
    imports_ok = test_app_imports()
    
    print("\n" + "=" * 30)
    if basic_ok and imports_ok:
        print("+ Basic functionality tests PASSED")
        print("Ready to install AI models!")
    else:
        print("- Some tests FAILED")
        print("Fix issues before proceeding")
    
    print("\nNext Steps:")
    print("1. Install AI packages: pip install torch torchvision transformers accelerate sentencepiece")
    print("2. Run: streamlit run app.py")
    print("3. Upload your video and test local AI models")