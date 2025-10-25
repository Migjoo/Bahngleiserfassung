from pathlib import Path
from local_models import get_local_model_manager
from video_processing import extract_frames_from_video
import torch
class BytesFile:
    def __init__(self, data):
        self._data = data
    def read(self):
        return self._data
manager = get_local_model_manager()
detector = manager.person_on_track_detector
with open('2.mp4', 'rb') as fh:
    frames = extract_frames_from_video(BytesFile(fh.read()), fps=1.0)
frame7 = frames[7]['frame']
detector._load_detection_model()
outputs = detector._run_detection_model(frame7)
print('labels', outputs['labels'])
print('scores', outputs['scores'])
print('person index', detector.person_label_index)
