import numpy as np
import tensorflow as tf
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
        """Load the TFLite model"""
        model_filepath = "/home/pi/Downloads/model.tflite"
        interpreter = tf.lite.Interpreter(model_path=model_filepath)
        interpreter.allocate_tensors()
        return interpreter
    
    def load_labels(self):
        """Load food labels from file"""
        with open("labels.txt", "r") as f:
            return [line.strip() for line in f.readlines()]
    
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
    
    def setup_gui(self):
        """Set up the graphical user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        ttk.Label(header_frame, text="🍏 Food Calorie Calculator", style='Header.TLabel').pack()
        
        # Camera section
        camera_frame = ttk.LabelFrame(main_frame, text="📷 Food Detection", padding="15")
        camera_frame.pack(fill=tk.X, pady=5)
        
        self.camera_label = ttk.Label(camera_frame, text="Camera ready - press 'Capture' to take photo", style='Result.TLabel')
        self.camera_label.pack(pady=10)
        
        btn_frame = ttk.Frame(camera_frame)
        btn_frame.pack()
        
        self.capture_btn = ttk.Button(btn_frame, text="Capture Image", command=self.capture_and_detect, style='TButton')
        self.capture_btn.pack(side=tk.LEFT, padx=5)
        
        self.result_label = ttk.Label(camera_frame, text="Detected food: None", style='Result.TLabel')
        self.result_label.pack(pady=5)
        
        self.confidence_label = ttk.Label(camera_frame, text="Confidence: 0%", style='Result.TLabel')
        self.confidence_label.pack()
        
        # Portion section
        portion_frame = ttk.LabelFrame(main_frame, text="⚖️ Portion Details", padding="15")
        portion_frame.pack(fill=tk.X, pady=5)
        
        portion_grid = ttk.Frame(portion_frame)
        portion_grid.pack()
        
        ttk.Label(portion_grid, text="Standard portions:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.portion_combo = ttk.Combobox(portion_grid, values=["Custom", "Small (50g)", "Medium (100g)", "Large (150g)"])
        self.portion_combo.current(1)
        self.portion_combo.grid(row=0, column=1, padx=5, pady=5)
        self.portion_combo.bind("<<ComboboxSelected>>", self.update_weight_from_portion)
        
        ttk.Label(portion_grid, text="Weight (grams):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.weight_entry = ttk.Entry(portion_grid)
        self.weight_entry.insert(0, "100")
        self.weight_entry.grid(row=1, column=1, padx=5, pady=5)
        
        self.calculate_btn = ttk.Button(portion_frame, text="Calculate Calories", 
                                      command=self.calculate_calories, state=tk.DISABLED)
        self.calculate_btn.pack(pady=10)
        
        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="📊 Nutritional Information", padding="15")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        results_grid = ttk.Frame(results_frame)
        results_grid.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(results_grid, text="Food:", style='Result.TLabel').grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.result_food = ttk.Label(results_grid, text="-", style='Result.TLabel')
        self.result_food.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(results_grid, text="Weight:", style='Result.TLabel').grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.result_weight = ttk.Label(results_grid, text="- g", style='Result.TLabel')
        self.result_weight.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(results_grid, text="Calories:", style='Result.TLabel').grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.result_calories = ttk.Label(results_grid, text="-", style='Result.TLabel')
        self.result_calories.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(results_grid, text="Health Status:", style='Result.TLabel').grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.result_health = ttk.Label(results_grid, text="-", style='Result.TLabel')
        self.result_health.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        self.start_camera_preview()
    
    def start_camera_preview(self):
        """Start the camera preview"""
        try:
            self.cam.start()
        except Exception as e:
            messagebox.showerror("Camera Error", f"Failed to start camera: {str(e)}")
    
    def capture_and_detect(self):
        """Capture image and detect food"""
        try:
            self.cam.set_controls({"AfMode": controls.AfModeEnum.Continuous})
            time.sleep(2)  # Allow for autofocus

            # Show live camera feed
            self.show_camera_feed()

            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            image_path = f'capture-{timestamp}.jpg'
            self.cam.capture_file(image_path)

            self.camera_label.config(text=f"Image captured: {image_path}")

            detected_food, confidence = self.detect_food(image_path)

            if detected_food:
                self.result_label.config(text=f"Detected food: {detected_food}")
                self.confidence_label.config(text=f"Confidence: {confidence:.1%}")
                self.calculate_btn.config(state=tk.NORMAL)
                self.current_food = detected_food
            else:
                self.result_label.config(text="No food detected")
                self.confidence_label.config(text="")
                self.calculate_btn.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to capture/detect: {str(e)}")
    
    def show_camera_feed(self):
        """Show the live camera feed in a pop-up window."""
        cv2.namedWindow("Camera Feed - Press 'q' to capture", cv2.WINDOW_NORMAL)
        self.cam.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        start_time = time.time()

        while True:
            frame = self.cam.capture_array()
            cv2.imshow("Camera Feed - Press 'q' to capture", frame)

            # Break the loop after a few seconds or when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q') or time.time() - start_time > 10:
                break

        cv2.destroyAllWindows()
    
    def detect_food(self, image_path):
        """Detect food from image using the model"""
        try:
            input_details = self.model.get_input_details()
            output_details = self.model.get_output_details()
            input_shape = input_details[0]['shape'][1:3]

            image = Image.open(image_path).resize(input_shape)
            input_array = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
            input_array = input_array[:, :, :, (2, 1, 0)]  # Convert to BGR

            self.model.set_tensor(input_details[0]['index'], input_array)
            self.model.invoke()

            outputs = self.model.get_tensor(output_details[0]['index'])
            max_index = np.argmax(outputs[0])
            tag = self.labels[max_index]
            probability = outputs[0][max_index]

            print(f"Detected: {tag} with probability {probability:.2f}")

            if probability < 0.5:  # 50% confidence threshold
                return None, 0.0
            
            return tag, probability
            
        except Exception as e:
            print(f"Detection error: {str(e)}")
            return None, 0.0
    
    def update_weight_from_portion(self, event=None):
        """Update weight entry based on portion selection"""
        portion = self.portion_combo.get()
        if portion == "Small (50g)":
            self.weight_entry.delete(0, tk.END)
            self.weight_entry.insert(0, "50")
        elif portion == "Medium (100g)":
            self.weight_entry.delete(0, tk.END)
            self.weight_entry.insert(0, "100")
        elif portion == "Large (150g)":
            self.weight_entry.delete(0, tk.END)
            self.weight_entry.insert(0, "150")
    
    def calculate_calories(self):
        """Calculate calories based on detected food and portion"""
        try:
            if not hasattr(self, 'current_food'):
                messagebox.showerror("Error", "No food detected yet")
                return
            
            # Normalize the detected food name
            normalized_food = self.current_food.strip().title()
            food_data = self.food_db.get(normalized_food)
            if not food_data:
                messagebox.showerror("Error", f"No data available for {self.current_food}")
                return
            
            try:
                weight = float(self.weight_entry.get())
                if weight <= 0:
                    raise ValueError("Weight must be positive")
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid weight in grams")
                return
            
            # Calculate calories
            calories = (food_data['calories'] * weight) / 100
            
            self.result_food.config(text=self.current_food)
            self.result_weight.config(text=f"{weight} g")
            self.result_calories.config(text=f"{calories:.1f} kcal")
            
            if food_data['healthy']:
                self.result_health.config(text="HEALTHY ✅", foreground="green")
            else:
                self.result_health.config(text="NOT RECOMMENDED ❌", foreground="red")
                
        except Exception as e:
            messagebox.showerror("Error", f"Calculation failed: {str(e)}")
    
    def cleanup(self):
        """Clean up resources"""
        self.cam.stop()
        self.cam.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = FoodDetectionApp(root)
    
    root.protocol("WM_DELETE_WINDOW", lambda: (app.cleanup(), root.destroy()))
    
    root.mainloop()
