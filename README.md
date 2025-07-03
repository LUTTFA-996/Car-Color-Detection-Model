# Car-Color-Detection-Model
This project uses a YOLOv8 model to detect cars, classify their colors (highlighting blue cars with red boxes), count people at traffic signals, and display the results in a GUI built with Tkinter.
# Features
- Detects cars, people, and traffic lights
- Classifies blue-colored cars
- Red rectangle for blue cars, blue for others
- Counts number of people at traffic signal
- GUI for easy image upload and preview
# Requirements
- Python 3.8+
- OpenCV
- Tkinter
- Pillow
- ultralytics
- numpy
# Dataset
- Dataset contains images labeled for 3 classes: car, person, and traffic_light in YOLO format.
# How to Run
```bash
python gui_detector.py

## **Structure **
car-color-detection-model/
│
├── train_model.py              
│ inference_test.py
├──C:\Users\Lutifah\Desktop\INTERSHIP\Car colour detection Model\car_color_detection\training_run\weights\best.pt                 
├── dataset/             
│   ├── train/
│   │   ├── images/               
│   │   └── labels/              
│   └── valid/
│       ├── images/               
│       └── labels/              
├── car_color_detection.py
├── data.yaml               
├── requirements.txt             
