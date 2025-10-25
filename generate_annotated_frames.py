import torch
from pathlib import Path
from local_models import get_local_model_manager
from video_processing import extract_frames_from_video

class BytesFile:
    def __init__(self, data):
        self._data = data
    def read(self):
        return self._data

def save_annotations(video_path: Path, output_root: Path, fps: float = 1.0):
    manager = get_local_model_manager()

    with video_path.open('rb') as fh:
        frames = extract_frames_from_video(BytesFile(fh.read()), fps=fps)

    print(f"Processing {video_path.name}: extracted {len(frames)} frames")

    output_root.mkdir(parents=True, exist_ok=True)

    saved = []
    for frame_info in frames:
        frame_number = frame_info['frame_number']
        image = frame_info['frame']

        detection = manager.person_on_track_detector.detect_person_on_track(image)
        annotated = detection.get('annotated_image')
        boxes = detection.get('bounding_boxes', [])

        print(f"Frame {frame_number}: {len(boxes)} boxes")

        if annotated is not None and boxes:
            out_path = output_root / f"frame_{frame_number:03d}_annotated.png"
            annotated.save(out_path)
            saved.append(out_path)

    print(f"Saved {len(saved)} annotated frames under {output_root}")
    for path in saved:
        print(f"  - {path}")

if __name__ == "__main__":
    video_file = Path('2.mp4')
    if not video_file.exists():
        raise SystemExit("2.mp4 not found in repository root")

    output_dir = Path('test') / 'annotated_2mp4'
    save_annotations(video_file, output_dir, fps=1.0)
