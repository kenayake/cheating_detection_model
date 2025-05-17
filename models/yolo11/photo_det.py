import cv2
import torch
from PIL import Image
import numpy as np
from ultralytics import YOLO

def detect_objects(image_path, model_path):
    # Load YOLO model
    model = YOLO(model_path)
    
    # Read and process image
    img = Image.open(image_path)
    
    # Perform detection
    results = model(img, iou=0.3, conf=0.1, classes=[0])
    
    # Convert PIL image to OpenCV format for visualization
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    for result in results:
        boxes = result.boxes

        # Draw bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = box.cls[0]

            # Convert to integer coordinates
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0) if int(cls) == 0 else (0, 0, 255),
                2,
            )

            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0) if int(cls) == 0 else (0, 0, 255),
                2,
            ) 
            
            # Extract and save ROI
            roi = frame[y1+1:y2-1, x1:x2]
            roi_filename = f"dataset/temp/extract/roi_{int(cls)}_{int(conf*100)}.jpg"
            cv2.imwrite(roi_filename, roi)
    
    return frame

if __name__ == "__main__":
    # Example usage
    image_path = "C:/Users/kenny/OneDrive/Documents/Kuliah/SKRIPSI/frames/around2.png"
    model_paths = [
        "C:/Users/kenny/Coding/SKRIPSI/cheating_detection_model/saved_checkpoints/yolo/yolo11s_dataset_final_sitting_standing/weights/best.pt",
        "C:/Users/kenny/Coding/SKRIPSI/cheating_detection_model/saved_checkpoints/yolo/yolo11s_dataset_final_sitting_only/weights/best.pt",
        "C:/Users/kenny/Coding/SKRIPSI/cheating_detection_model/saved_checkpoints/yolo/yolo11s_dataset_alt_v5/weights/best.pt",
        "C:/Users/kenny/Coding/SKRIPSI/cheating_detection_model/saved_checkpoints/yolo/yolo11s_datasetv9/weights/best.pt",
    ]
    model_path = model_paths[3]
    
    result_image = detect_objects(image_path, model_path)
    
    # Display result
    cv2.namedWindow('Detection Result', cv2.WINDOW_NORMAL)
    cv2.imshow('Detection Result', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()