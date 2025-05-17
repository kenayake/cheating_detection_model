import cv2
from ultralytics import YOLO


class CustomYolo:
    def __init__(self, video_path, model_path):
        self.video_path = video_path
        self.model_path = model_path
    
    def load_model(self):
        self.model = YOLO(self.model_path)
    
    def detect_video(self):
        pass
    
    def detect_image(self):
        pass
    
    def save_video(self, output_path):
        pass
    
    