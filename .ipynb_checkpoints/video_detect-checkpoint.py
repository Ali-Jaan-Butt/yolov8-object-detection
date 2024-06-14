# Import Dependencies
from ultralytics import YOLO
import cv2
import math

# Path to the video file
video_path = 'video.mp4'  # Change this to the path of your video file

# Start video capture
cap = cv2.VideoCapture(video_path)

# YOLO Model
model = YOLO("yolo-Weights/yolov8n.pt")

# Object Classes
classNames = ["Person"]

# Initialize person count
total_person_count = 0

while cap.isOpened():
    try:
        success, img = cap.read()
        if not success:
            break
        
        results = model(img, stream=True)
        
        frame_person_count = 0  # Count of persons in the current frame
        for r in results:
            boxes = r.boxes
            frame_person_count += len(boxes)  # Count the number of boxes (persons detected)
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw bounding box
                cv2.line(img, (x1, y1), (x1 + 15, y1), (0, 255, 0), 2)                      # Top line
                cv2.line(img, (x1, y1), (x1, y1 + 15), (0, 255, 0), 2)                      # Left line
                cv2.line(img, (x2, y1), (x2 - 15, y1), (0, 255, 0), 2)                      # Right line
                cv2.line(img, (x1, y2), (x1 + 15, y2), (0, 255, 0), 2)                      # Bottom line
                cv2.line(img, (x1, y2), (x1, y2 - 15), (0, 255, 0), 2)                      # Bottom left line
                cv2.line(img, (x2, y2), (x2 - 15, y2), (0, 255, 0), 2)                      # Bottom right line
                cv2.line(img, (x2, y1), (x2, y1 + 15), (0, 255, 0), 2)                      # Right top line
                cv2.line(img, (x2, y2), (x2, y2 - 15), (0, 255, 0), 2)                      # Right bottom line
                
                # Draw confidence and class name
                confidence = round(float(box.conf[0]) * 100, 2)
                class_index = int(box.cls[0])
                class_name = classNames[class_index]
                text = f"{class_name}: {confidence}%"
                org = (x1, y1 - 10)  # Place text slightly above the bounding box
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.5
                color = (0, 255, 0)
                thickness = 1
                cv2.putText(img, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
        
        total_person_count += frame_person_count  # Add current frame's person count to total count
    
    except Exception as e:
        print(f"Error: {e}")
        break

# Release video capture object
cap.release()

# Save the total person count to a file
with open('person_count.txt', 'w') as f:
    f.write(f"Total person count: {total_person_count}\n")

print(f"Total person count: {total_person_count}")