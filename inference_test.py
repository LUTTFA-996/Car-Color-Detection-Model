# inference_test.py
# Simple inference script for testing

import cv2
from ultralytics import YOLO
import numpy as np

def test_inference(model_path, image_path):
    """Test inference on a single image"""
    try:
        # Load model
        model = YOLO(model_path)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return
        
        # Run inference
        results = model(image_path, conf=0.5)
        
        # Process results
        car_count = 0
        person_count = 0
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    # Get coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Count objects
                    if 'car' in class_name:
                        car_count += 1
                        # Red box for blue cars, blue box for others
                        color = (0, 0, 255) if class_name == 'blue_car' else (255, 0, 0)
                    elif class_name == 'person':
                        person_count += 1
                        color = (0, 255, 255)
                    else:
                        color = (255, 0, 0)
                    
                    # Draw box and label
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(image, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add count text
        count_text = f"Cars: {car_count}, People: {person_count}"
        cv2.putText(image, count_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save result
        output_path = "inference_result.jpg"
        cv2.imwrite(output_path, image)
        print(f"Result saved to: {output_path}")
        print(f"Detection: {count_text}")
        
    except Exception as e:
        print(f"Inference error: {e}")

if __name__ == "__main__":
    # Example usage
    model_path = "yolo11n.pt"  # or path to your trained model
    image_path = "path/to/test/image.jpg"  # replace with actual image path
    
    test_inference(model_path, image_path)
