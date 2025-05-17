from calendar import c
import io
from ultralytics import YOLO
import cv2
import numpy as np

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def detect_video(video_path, model_path, view_only):
    model = YOLO(model_path)
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    # output_path = "detection_output.mp4"
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    # out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Keep track of processed frames
    frame_count = 0
    n = 18  # Process only first 100 frames - adjust this number as needed
    framed = 0

    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        
        if not success:
            break
        
        # Skip processing if we've reached n frames
        if frame_count >= n:
            break
        frame_count += 1
        
        print(frame_count)

        # Perform prediction on the frame
        results = model(frame, conf=0.6)

        # Process results
        for result in results:
            boxes = result.boxes
            
            box_num = 0

            # Draw bounding boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls = box.cls[0]

                # Convert to integer coordinates
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                if view_only:
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
                
                framed = frame
                
                if not view_only:
                    roi = frame[y1+1:y2-1, x1:x2]
                    roi_filename = f"dataset/temp/queue/roi_{box_num}_d.jpg"
                    cv2.imwrite(roi_filename, roi)
                box_num += 1

        # Write the frame to the output video
        # out.write(frame)

        # Show the frame
    # cv2.imwrite("contoh_model.jpg", framed)
    if view_only:
        cv2.imshow("Detection Result", framed)
    
    # Break the loop if 'q' is pressed
    cv2.waitKey(0)

    # Release resources
    cap.release()
    # out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # video_path = 0
    view_only = True
    # view_only = False
    video_path = "test_data/student detection/per_classroom/room4_sitting.mp4"
    model_path = "saved_checkpoints/yolo/yolo11s_dataset_final_sitting_standing/weights/best.pt"
    detect_video(video_path, model_path, view_only)
