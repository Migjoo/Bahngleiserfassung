import torch
from pathlib import Path
from local_models import get_local_model_manager
from video_processing import extract_frames_from_video

class BytesFile:
    def __init__(self, data):
        self._data = data
    def read(self):
        return self._data

video_path = Path('2.mp4')
manager = get_local_model_manager()
with video_path.open('rb') as fh:
    frames = extract_frames_from_video(BytesFile(fh.read()), fps=1.0)

frame = next(f for f in frames if f['frame_number'] == 12)
image = frame['frame']

# Run detection model manually
manager.person_on_track_detector._load_detection_model()
outputs = manager.person_on_track_detector._run_detection_model(image)
boxes = outputs['boxes']
scores = outputs['scores']
labels = outputs['labels']
print('total outputs', len(boxes))
pairs = [(score.item(), label.item()) for score, label in zip(scores, labels)]
print('top10', pairs[:10])
