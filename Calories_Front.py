import numpy as np
import tflite_runtime.interpreter as tflite  # Changed from tensorflow
import cv2
from PIL import Image
from picamera2 import Picamera2
from libcamera import controls
import time
import datetime
import tkinter as tk
from tkinter import ttk, messagebox

class FoodDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🍏 Food Calorie Calculator")
        self.root.geometry("800x600")
        
        # Configure style
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f5f5f5')
        self.style.configure('TLabel', background='#f5f5f5', font=('Helvetica', 10))
        self.style.configure('TButton', font=('Helvetica', 10, 'bold'))
        self.style.configure('Header.TLabel', font=('Helvetica', 14, 'bold'))
        self.style.configure('Result.TLabel', font=('Helvetica', 12))
        
        # Load model and labels
        self.model = self.load_model()
        self.labels = self.load_labels()
        self.food_db = self.load_food_database()
        
        # Initialize camera
        self.cam = Picamera2()
        self.cam_config = self.cam.create_still_configuration()
        self.cam.configure(self.cam_config)
        
        # Create GUI
        self.setup_gui()
        
    def load_model(self):
        """Load the TFLite model using tflite_runtime"""
        model_filepath = "/home/pi/Downloads/model.tflite"
        interpreter = tflite.Interpreter(model_path=model_filepath)  # Changed from tf.lite
        interpreter.allocate_tensors()
        return interpreter
    
    def load_labels(self):
        """Load food labels from file"""
        try:
            with open("labels.txt", "r") as f:
                return [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            messagebox.showerror("Error", "labels.txt file not found!")
            return []
    
    def load_food_database(self):
        """Load food calorie database"""
        return {
            "Apple": {"calories": 52, "healthy": True},
            "Banana": {"calories": 89, "healthy": True},
            "Burger": {"calories": 313, "healthy": False},
            "Chocolate": {"calories": 535, "healthy": False},
            "Chocolate Donut": {"calories": 452, "healthy": False},
            "French Fries": {"calories": 312, "healthy": False},
            "Fruit Oatmeal": {"calories": 68, "healthy": True},
            "Pear": {"calories": 57, "healthy": True},
            "Potato Chips": {"calories": 536, "healthy": False},
            "Rice": {"calories": 130, "healthy": True}
        }
    
    # [Rest of your methods remain exactly the same until detect_food()]

    def detect_food(self, image_path):
        """Detect food from image using the model"""
        try:
            input_details = self.model.get_input_details()
            output_details = self.model.get_output_details()
            input_shape = input_details[0]['shape'][1:3]

            # Improved image loading and preprocessing
            image = Image.open(image_path).convert('RGB').resize(input_shape)
            input_array = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
            
            # Normalize if your model expects values between 0-1
            input_array = input_array / 255.0
            
            # Remove BGR conversion unless your model specifically needs it
            # input_array = input_array[:, :, :, (2, 1, 0)]  # Only use if model expects BGR

            self.model.set_tensor(input_details[0]['index'], input_array)
            self.model.invoke()

            outputs = self.model.get_tensor(output_details[0]['index'])
            max_index = np.argmax(outputs[0])
            tag = self.labels[max_index] if self.labels else f"Class {max_index}"
            probability = outputs[0][max_index]

            print(f"Detected: {tag} with probability {probability:.2f}")

            if probability < 0.5:  # 50% confidence threshold
                return None, 0.0
            
            return tag, probability
            
        except Exception as e:
            print(f"Detection error: {str(e)}")
            messagebox.showerror("Detection Error", f"Failed to detect food: {str(e)}")
            return None, 0.0

    # [All other methods remain exactly the same]

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = FoodDetectionApp(root)
        root.protocol("WM_DELETE_WINDOW", lambda: (app.cleanup(), root.destroy()))
        root.mainloop()
    except Exception as e:
        print(f"Application error: {str(e)}")
        messagebox.showerror("Fatal Error", f"The application crashed: {str(e)}")
