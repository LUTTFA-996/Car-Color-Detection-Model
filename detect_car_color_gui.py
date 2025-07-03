# Car Color Detection Model with YOLOv11
# Complete implementation with training, inference, and GUI

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
from ultralytics import YOLO
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
import threading

class CarColorDetectionModel:
    def __init__(self):
        self.model = None
        self.class_names = []
        self.color_classes = {}
        self.setup_classes()
    
    def setup_classes(self):
        """Define color classes and their mappings"""
        self.color_classes = {
            'blue_car': {'color': (0, 0, 255), 'box_color': (0, 0, 255)},  # Red box for blue cars
            'red_car': {'color': (255, 0, 0), 'box_color': (255, 0, 0)},   # Blue box for other cars
            'white_car': {'color': (255, 255, 255), 'box_color': (255, 0, 0)},
            'black_car': {'color': (0, 0, 0), 'box_color': (255, 0, 0)},
            'silver_car': {'color': (192, 192, 192), 'box_color': (255, 0, 0)},
            'gray_car': {'color': (128, 128, 128), 'box_color': (255, 0, 0)},
            'green_car': {'color': (0, 255, 0), 'box_color': (255, 0, 0)},
            'yellow_car': {'color': (255, 255, 0), 'box_color': (255, 0, 0)},
            'person': {'color': (0, 255, 255), 'box_color': (0, 255, 255)}
        }
    
    def create_dataset_config(self, dataset_path):
        """Create YAML configuration file for the dataset"""
        config = {
            'path': dataset_path,
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': len(self.color_classes),
            'names': list(self.color_classes.keys())
        }
        
        config_path = os.path.join(dataset_path, 'C:\\Users\\Lutifah\\Desktop\\INTERSHIP\\Car colour detection Model\\data.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config_path
    
    def train_model(self, dataset_path, epochs=5, img_size=640):
        """Train the YOLOv11 model"""
        try:
            # Create dataset configuration
            config_path = self.create_dataset_config(dataset_path)
            
            # Initialize YOLOv11 model
            self.model = YOLO('yolo11n.pt')  # or yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
            
            # Train the model
            results = self.model.train(
                data=config_path,
                epochs=epochs,
                imgsz=img_size,
                device='auto',
                workers=8,
                batch=16,
                patience=50,
                save=True,
                save_period=10,
                cache=True,
                amp=True,
                project='car_color_detection',
                name='yolo11_car_color'
            )
            
            return results
            
        except Exception as e:
            print(f"Training error: {e}")
            return None
    
    def load_model(self, model_path):
        """Load trained model"""
        try:
            self.model = YOLO(model_path)
            print(f"Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_image(self, image_path, conf_threshold=0.5):
        """Predict on a single image"""
        if not self.model:
            return None, "Model not loaded"
        
        try:
            # Run inference
            results = self.model(image_path, conf=conf_threshold)
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return None, "Could not load image"
            
            # Process results
            car_count = 0
            person_count = 0
            color_counts = Counter()
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class name
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        confidence = float(box.conf[0])
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Count objects
                        if 'car' in class_name:
                            car_count += 1
                            color_counts[class_name] += 1
                            
                            # Determine box color based on car color
                            if class_name == 'blue_car':
                                box_color = (0, 0, 255)  # Red box for blue cars
                            else:
                                box_color = (255, 0, 0)  # Blue box for other cars
                        elif class_name == 'person':
                            person_count += 1
                            box_color = (0, 255, 255)  # Yellow box for persons
                        else:
                            box_color = (255, 0, 0)  # Default blue box
                        
                        # Draw bounding box
                        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)
                        
                        # Draw label
                        label = f"{class_name}: {confidence:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), box_color, -1)
                        cv2.putText(image, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Add count information
            count_text = f"Cars: {car_count}, People: {person_count}"
            cv2.putText(image, count_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add color breakdown
            y_offset = 70
            for color, count in color_counts.items():
                color_text = f"{color}: {count}"
                cv2.putText(image, color_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 30
            
            return image, f"Cars: {car_count}, People: {person_count}"
            
        except Exception as e:
            return None, f"Prediction error: {e}"

class CarColorDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Car Color Detection Model")
        self.root.geometry("1200x800")
        
        self.model = CarColorDetectionModel()
        self.current_image = None
        self.result_image = None
        
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the GUI components"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Buttons
        ttk.Button(control_frame, text="Load Model", command=self.load_model).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="Train Model", command=self.train_model).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Load Image", command=self.load_image).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="Predict", command=self.predict_image).grid(row=0, column=3, padx=5)
        ttk.Button(control_frame, text="Save Result", command=self.save_result).grid(row=0, column=4, padx=5)
        
        # Confidence threshold
        ttk.Label(control_frame, text="Confidence:").grid(row=0, column=5, padx=5)
        self.conf_var = tk.DoubleVar(value=0.5)
        ttk.Scale(control_frame, from_=0.1, to=0.9, variable=self.conf_var, 
                 orient=tk.HORIZONTAL, length=100).grid(row=0, column=6, padx=5)
        
        # Image display frames
        image_frame = ttk.Frame(main_frame)
        image_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        image_frame.columnconfigure(0, weight=1)
        image_frame.columnconfigure(1, weight=1)
        image_frame.rowconfigure(0, weight=1)
        
        # Original image
        self.original_frame = ttk.LabelFrame(image_frame, text="Original Image", padding="10")
        self.original_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        self.original_label = ttk.Label(self.original_frame, text="No image loaded")
        self.original_label.pack(expand=True)
        
        # Result image
        self.result_frame = ttk.LabelFrame(image_frame, text="Detection Result", padding="10")
        self.result_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        self.result_label = ttk.Label(self.result_frame, text="No prediction yet")
        self.result_label.pack(expand=True)
        
        # Status and info
        self.status_frame = ttk.Frame(main_frame)
        self.status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.status_frame, textvariable=self.status_var).pack(side=tk.LEFT)
        
        self.info_var = tk.StringVar(value="Load a model and image to start detection")
        ttk.Label(self.status_frame, textvariable=self.info_var).pack(side=tk.RIGHT)
    
    def load_model(self):
        """Load a trained model"""
        model_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        
        if model_path:
            if self.model.load_model(model_path):
                self.status_var.set("Model loaded successfully")
                self.info_var.set(f"Model: {os.path.basename(model_path)}")
            else:
                messagebox.showerror("Error", "Failed to load model")
    
    def train_model(self):
        """Train a new model"""
        dataset_path = filedialog.askdirectory(title="Select Dataset Directory")
        
        if dataset_path:
            # Training parameters dialog
            train_dialog = TrainingDialog(self.root)
            if train_dialog.result:
                epochs = train_dialog.epochs
                img_size = train_dialog.img_size
                
                self.status_var.set("Training in progress...")
                self.info_var.set("This may take a while...")
                
                # Run training in a separate thread
                threading.Thread(target=self._train_model_thread, 
                               args=(dataset_path, epochs, img_size), daemon=True).start()
    
    def _train_model_thread(self, dataset_path, epochs, img_size):
        """Training thread"""
        try:
            results = self.model.train_model(dataset_path, epochs, img_size)
            if results:
                self.root.after(0, lambda: self.status_var.set("Training completed successfully"))
                self.root.after(0, lambda: self.info_var.set("Model saved in runs/detect/"))
            else:
                self.root.after(0, lambda: self.status_var.set("Training failed"))
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Training error: {e}"))
    
    def load_image(self):
        """Load an image for prediction"""
        image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if image_path:
            try:
                # Load and display original image
                image = Image.open(image_path)
                image.thumbnail((400, 400), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                
                self.original_label.configure(image=photo, text="")
                self.original_label.image = photo
                
                self.current_image = image_path
                self.status_var.set("Image loaded")
                self.info_var.set(f"Image: {os.path.basename(image_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")
    
    def predict_image(self):
        """Run prediction on loaded image"""
        if not self.current_image:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        if not self.model.model:
            messagebox.showwarning("Warning", "Please load a model first")
            return
        
        self.status_var.set("Predicting...")
        
        # Run prediction
        result_image, info = self.model.predict_image(self.current_image, self.conf_var.get())
        
        if result_image is not None:
            # Convert BGR to RGB
            result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL and display
            pil_image = Image.fromarray(result_image_rgb)
            pil_image.thumbnail((400, 400), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil_image)
            
            self.result_label.configure(image=photo, text="")
            self.result_label.image = photo
            
            self.result_image = result_image
            self.status_var.set("Prediction completed")
            self.info_var.set(info)
        else:
            messagebox.showerror("Error", f"Prediction failed: {info}")
            self.status_var.set("Prediction failed")
    
    def save_result(self):
        """Save the result image"""
        if self.result_image is None:
            messagebox.showwarning("Warning", "No result to save")
            return
        
        save_path = filedialog.asksaveasfilename(
            title="Save Result",
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if save_path:
            cv2.imwrite(save_path, self.result_image)
            self.status_var.set("Result saved")
            messagebox.showinfo("Success", f"Result saved to {save_path}")

class TrainingDialog:
    def __init__(self, parent):
        self.result = False
        self.epochs = 100
        self.img_size = 640
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Training Parameters")
        self.dialog.geometry("300x200")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (300 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (200 // 2)
        self.dialog.geometry(f"300x200+{x}+{y}")
        
        self.setup_dialog()
        
        # Wait for dialog to close
        self.dialog.wait_window()
    
    def setup_dialog(self):
        frame = ttk.Frame(self.dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Epochs
        ttk.Label(frame, text="Epochs:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.epochs_var = tk.IntVar(value=100)
        ttk.Entry(frame, textvariable=self.epochs_var, width=10).grid(row=0, column=1, pady=5)
        
        # Image size
        ttk.Label(frame, text="Image Size:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.img_size_var = tk.IntVar(value=640)
        ttk.Entry(frame, textvariable=self.img_size_var, width=10).grid(row=1, column=1, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="OK", command=self.ok_clicked).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.cancel_clicked).pack(side=tk.LEFT, padx=5)
    
    def ok_clicked(self):
        self.epochs = self.epochs_var.get()
        self.img_size = self.img_size_var.get()
        self.result = True
        self.dialog.destroy()
    
    def cancel_clicked(self):
        self.result = False
        self.dialog.destroy()

def main():
    root = tk.Tk()
    app = CarColorDetectionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()