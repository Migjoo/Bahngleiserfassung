from local_models import get_local_model_manager
manager = get_local_model_manager()
detector = manager.person_on_track_detector
print('error', detector.detection_error)
