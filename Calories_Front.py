import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from picamera2 import Picamera2
from libcamera import controls
import tkinter as tk
from tkinter import ttk, messagebox, Toplevel
import time
import datetime
import uuid
import threading

class FoodDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Food Calorie Calculator")
        
        # Load model and labels
        self.model = self.load_model()
        self.labels = self.load_labels()
        self.food_db = self.load_food_database()
        
        # Initialize camera with smaller resolution
        self.cam = Picamera2()
        self.cam_config = self.cam.create_still_configuration(
            main={"size": (640, 480)}  # Smaller resolution for preview
        )
        self.cam.configure(self.cam_config)
        
        # Camera preview variables
        self.preview_window = None
        self.preview_running = False
        self.capture_requested = False
        
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
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Camera section
        camera_frame = ttk.LabelFrame(main_frame, text="Food Detection", padding="10")
        camera_frame.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.camera_btn = ttk.Button(
            camera_frame, 
            text="Open Camera Preview", 
            command=self.start_camera_preview
        )
        self.camera_btn.grid(row=0, column=0, pady=5)
        
        # Detection results
        self.result_label = ttk.Label(camera_frame, text="Detected food: None")
        self.result_label.grid(row=1, column=0, pady=5)
        
        self.confidence_label = ttk.Label(camera_frame, text="Confidence: 0%")
        self.confidence_label.grid(row=2, column=0)
        
        # Portion input section
        portion_frame = ttk.LabelFrame(main_frame, text="Portion Details", padding="10")
        portion_frame.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(portion_frame, text="Weight (grams):").grid(row=0, column=0, sticky=tk.W)
        self.weight_entry = ttk.Entry(portion_frame)
        self.weight_entry.insert(0, "100")  # Default value
        self.weight_entry.grid(row=0, column=1, padx=5)
        
        ttk.Label(portion_frame, text="Standard portions:").grid(row=1, column=0, sticky=tk.W)
        self.portion_combo = ttk.Combobox(portion_frame, values=[
            "Custom", "Small (50g)", "Medium (100g)", "Large (150g)"])
        self.portion_combo.current(1)  # Default to Medium
        self.portion_combo.grid(row=1, column=1, padx=5)
        self.portion_combo.bind("<<ComboboxSelected>>", self.update_weight_from_portion)
        
        # Calculate button
        self.calculate_btn = ttk.Button(
            portion_frame, 
            text="Calculate Calories", 
            command=self.calculate_calories, 
            state=tk.DISABLED
        )
        self.calculate_btn.grid(row=2, column=0, columnspan=2, pady=5)
        
        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="Nutritional Information", padding="10")
        results_frame.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(results_frame, text="Food:").grid(row=0, column=0, sticky=tk.W)
        self.result_food = ttk.Label(results_frame, text="-")
        self.result_food.grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(results_frame, text="Weight:").grid(row=1, column=0, sticky=tk.W)
        self.result_weight = ttk.Label(results_frame, text="- g")
        self.result_weight.grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(results_frame, text="Calories:").grid(row=2, column=0, sticky=tk.W)
        self.result_calories = ttk.Label(results_frame, text="-")
        self.result_calories.grid(row=2, column=1, sticky=tk.W)
        
        ttk.Label(results_frame, text="Health Status:").grid(row=3, column=0, sticky=tk.W)
        self.result_health = ttk.Label(results_frame, text="-")
        self.result_health.grid(row=3, column=1, sticky=tk.W)
    
    def start_camera_preview(self):
        """Start the camera preview in a smaller popup window"""
        if self.preview_running:
            return
            
        self.preview_window = Toplevel(self.root)
        self.preview_window.title("Camera Preview")
        self.preview_window.geometry("400x300")  # Smaller window size
        self.preview_window.protocol("WM_DELETE_WINDOW", self.stop_camera_preview)
        
        # Add preview label
        self.preview_label = ttk.Label(self.preview_window)
        self.preview_label.pack(padx=10, pady=10)
        
        # Add capture button
        capture_btn = ttk.Button(
            self.preview_window, 
            text="Capture Image", 
            command=self.request_capture
        )
        capture_btn.pack(pady=5)
        
        # Start camera
        self.cam.start()
        self.preview_running = True
        
        # Start preview thread
        self.preview_thread = threading.Thread(target=self.update_preview, daemon=True)
        self.preview_thread.start()
    
    def stop_camera_preview(self):
        """Stop the camera preview"""
        if self.preview_running:
            self.preview_running = False
            self.preview_thread.join()
            self.cam.stop()
            self.preview_window.destroy()
            self.preview_window = None
    
    def request_capture(self):
        """Request an image capture from the preview"""
        self.capture_requested = True
    
    def update_preview(self):
        """Update the camera preview continuously"""
        while self.preview_running:
            # Get camera image
            rgb_array = self.cam.capture_array()
            
            # Resize to fit in our smaller window
            rgb_array = cv2.resize(rgb_array, (320, 240))
            
            # Convert to PIL Image
            image = Image.fromarray(rgb_array)
            
            # Convert to Tkinter PhotoImage
            photo = self.convert_to_tkimage(image)
            
            # Update preview label
            self.preview_label.configure(image=photo)
            self.preview_label.image = photo
            
            # Check if capture requested
            if self.capture_requested:
                self.capture_requested = False
                self.capture_image(image)
                
            time.sleep(0.05)  # Control frame rate
    
    def convert_to_tkimage(self, pil_image):
        """Convert PIL Image to Tkinter PhotoImage"""
        import io
        bio = io.BytesIO()
        pil_image.save(bio, format="PNG")
        from PIL import ImageTk
        return ImageTk.PhotoImage(data=bio.getvalue())
    
    def capture_image(self, image):
        """Capture and process the current image"""
        try:
            # Save the captured image
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            image_path = f'capture-{timestamp}.jpg'
            image.save(image_path)
            
            # Detect food
            detected_food, confidence = self.detect_food(image_path)
            
            if detected_food:
                self.result_label.config(text=f"Detected food: {detected_food}")
                self.confidence_label.config(text=f"Confidence: {confidence:.1%}")
                self.calculate_btn.config(state=tk.NORMAL)
                self.current_food = detected_food
                messagebox.showinfo("Success", f"Detected: {detected_food} ({confidence:.1%} confidence)")
            else:
                self.result_label.config(text="No food detected")
                self.confidence_label.config(text="")
                self.calculate_btn.config(state=tk.DISABLED)
                messagebox.showwarning("Warning", "No food item detected with sufficient confidence")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to capture/detect: {str(e)}")
    
    def detect_food(self, image_path):
        """Detect food from image using the model"""
        try:
            # Get model input details
            input_details = self.model.get_input_details()
            output_details = self.model.get_output_details()
            input_shape = input_details[0]['shape'][1:3]
            
            # Preprocess image
            image = Image.open(image_path).resize(input_shape)
            input_array = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
            input_array = input_array[:, :, :, (2, 1, 0)]  # Convert to BGR
            
            # Run inference
            self.model.set_tensor(input_details[0]['index'], input_array)
            self.model.invoke()
            
            # Get results
            outputs = self.model.get_tensor(output_details[0]['index'])
            max_index = np.argmax(outputs[0])
            tag = self.labels[max_index]
            probability = outputs[0][max_index]
            
            # Apply confidence threshold
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
                
            # Get weight input
            try:
                weight = float(self.weight_entry.get())
                if weight <= 0:
                    raise ValueError("Weight must be positive")
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid weight in grams")
                return
                
            # Get food data
            food_data = self.food_db.get(self.current_food)
            if not food_data:
                messagebox.showerror("Error", f"No data available for {self.current_food}")
                return
                
            # Calculate calories
            calories = (food_data['calories'] * weight) / 100
            
            # Update results
            self.result_food.config(text=self.current_food)
            self.result_weight.config(text=f"{weight} g")
            self.result_calories.config(text=f"{calories:.1f} kcal")
            
            # Set health status with color
            if food_data['healthy']:
                self.result_health.config(text="HEALTHY", foreground="green")
            else:
                self.result_health.config(text="NOT RECOMMENDED", foreground="red")
                
        except Exception as e:
            messagebox.showerror("Error", f"Calculation failed: {str(e)}")
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_camera_preview()
        if hasattr(self, 'cam'):
            self.cam.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = FoodDetectionApp(root)
    
    # Handle window close
    root.protocol("WM_DELETE_WINDOW", lambda: (app.cleanup(), root.destroy()))
    
    root.mainloop()
