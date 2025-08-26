import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import transforms, models
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time

class RatingPredictionModel(nn.Module):
    """Neural network model for rating prediction - same as training script"""
    
    def __init__(self, backbone='resnet50', pretrained=True, num_classes=1, dropout=0.5):
        super(RatingPredictionModel, self).__init__()
        
        self.backbone_name = backbone
        
        # Load backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'efficientnet':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output.squeeze()

class ModelTesterUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Product Rating Prediction - Model Tester")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Model variables
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = None
        self.model_info = None
        self.prediction_history = []
        
        # Setup UI
        self.setup_ui()
        
        # Status
        self.update_status("Ready - Load a model to start testing")
    
    def setup_ui(self):
        """Setup the user interface"""
        # Create main frames
        self.create_header_frame()
        self.create_main_content_frame()
        self.create_status_frame()
        
        # Setup transforms
        self.setup_transforms()
    
    def create_header_frame(self):
        """Create header with model loading controls"""
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        header_frame.pack(fill='x', padx=5, pady=5)
        header_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(header_frame, text="🤖 Product Rating Prediction Tester", 
                              font=('Arial', 18, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(side='left', padx=20, pady=20)
        
        # Model loading controls
        controls_frame = tk.Frame(header_frame, bg='#2c3e50')
        controls_frame.pack(side='right', padx=20, pady=10)
        
        tk.Button(controls_frame, text="Load Model", command=self.load_model,
                 bg='#3498db', fg='white', font=('Arial', 12, 'bold'),
                 padx=20, pady=5, relief='flat').pack(side='left', padx=5)
        
        tk.Button(controls_frame, text="Model Info", command=self.show_model_info,
                 bg='#95a5a6', fg='white', font=('Arial', 12),
                 padx=15, pady=5, relief='flat').pack(side='left', padx=5)
    
    def create_main_content_frame(self):
        """Create main content area"""
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left panel - Image upload and display
        self.create_image_panel(main_frame)
        
        # Right panel - Results and history
        self.create_results_panel(main_frame)
    
    def create_image_panel(self, parent):
        """Create left panel for image upload and display"""
        left_frame = tk.Frame(parent, bg='white', relief='ridge', bd=2)
        left_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        # Upload section
        upload_frame = tk.Frame(left_frame, bg='white')
        upload_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(upload_frame, text="📸 Upload Product Image", 
                font=('Arial', 14, 'bold'), bg='white').pack(anchor='w')
        
        button_frame = tk.Frame(upload_frame, bg='white')
        button_frame.pack(fill='x', pady=10)
        
        tk.Button(button_frame, text="Choose Image File", command=self.upload_image,
                 bg='#27ae60', fg='white', font=('Arial', 12, 'bold'),
                 padx=20, pady=8, relief='flat').pack(side='left')
        
        tk.Button(button_frame, text="Use Sample Image", command=self.use_sample_image,
                 bg='#f39c12', fg='white', font=('Arial', 12),
                 padx=15, pady=8, relief='flat').pack(side='left', padx=(10, 0))
        
        # Image display area
        self.image_frame = tk.Frame(left_frame, bg='#ecf0f1', relief='sunken', bd=2)
        self.image_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.image_label = tk.Label(self.image_frame, text="No image loaded\n\nClick 'Choose Image File' to upload a product image",
                                   font=('Arial', 12), bg='#ecf0f1', fg='#7f8c8d')
        self.image_label.pack(expand=True)
        
        # Prediction button
        self.predict_button = tk.Button(left_frame, text="🔮 Predict Rating", 
                                       command=self.predict_rating,
                                       bg='#e74c3c', fg='white', font=('Arial', 14, 'bold'),
                                       padx=30, pady=10, relief='flat', state='disabled')
        self.predict_button.pack(pady=10)
    
    def create_results_panel(self, parent):
        """Create right panel for results and history"""
        right_frame = tk.Frame(parent, bg='white', relief='ridge', bd=2)
        right_frame.pack(side='right', fill='both', expand=True, padx=5)
        
        # Current prediction section
        pred_frame = tk.Frame(right_frame, bg='white')
        pred_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(pred_frame, text="⭐ Current Prediction", 
                font=('Arial', 14, 'bold'), bg='white').pack(anchor='w')
        
        # Prediction display
        self.prediction_frame = tk.Frame(pred_frame, bg='#f8f9fa', relief='ridge', bd=1)
        self.prediction_frame.pack(fill='x', pady=10)
        
        self.prediction_label = tk.Label(self.prediction_frame, text="No prediction yet", 
                                        font=('Arial', 16), bg='#f8f9fa', pady=20)
        self.prediction_label.pack()
        
        # Confidence/Details frame
        self.details_frame = tk.Frame(pred_frame, bg='white')
        self.details_frame.pack(fill='x', pady=5)
        
        # History section
        history_frame = tk.Frame(right_frame, bg='white')
        history_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        tk.Label(history_frame, text="📊 Prediction History", 
                font=('Arial', 14, 'bold'), bg='white').pack(anchor='w')
        
        # History list with scrollbar
        list_frame = tk.Frame(history_frame, bg='white')
        list_frame.pack(fill='both', expand=True, pady=5)
        
        # Scrollable listbox
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side='right', fill='y')
        
        self.history_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set,
                                         font=('Arial', 10), selectmode='single')
        self.history_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.history_listbox.yview)
        
        # History controls
        history_controls = tk.Frame(history_frame, bg='white')
        history_controls.pack(fill='x', pady=5)
        
        tk.Button(history_controls, text="Clear History", command=self.clear_history,
                 bg='#95a5a6', fg='white', font=('Arial', 10),
                 padx=10, pady=5, relief='flat').pack(side='left')
        
        tk.Button(history_controls, text="Export Results", command=self.export_results,
                 bg='#3498db', fg='white', font=('Arial', 10),
                 padx=10, pady=5, relief='flat').pack(side='left', padx=(10, 0))
    
    def create_status_frame(self):
        """Create status bar at bottom"""
        status_frame = tk.Frame(self.root, bg='#34495e', height=30)
        status_frame.pack(fill='x', side='bottom')
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(status_frame, text="Ready", 
                                    font=('Arial', 10), bg='#34495e', fg='white')
        self.status_label.pack(side='left', padx=10, pady=5)
        
        # Device info
        device_text = f"Device: {self.device.type.upper()}"
        if self.device.type == 'cuda':
            device_text += f" ({torch.cuda.get_device_name()})"
        
        tk.Label(status_frame, text=device_text, 
                font=('Arial', 10), bg='#34495e', fg='#bdc3c7').pack(side='right', padx=10, pady=5)
    
    def setup_transforms(self):
        """Setup image transforms for inference"""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def update_status(self, message):
        """Update status bar message"""
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def load_model(self):
        """Load trained model from file"""
        model_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Models", "*.pth"), ("All Files", "*.*")]
        )
        
        if not model_path:
            return
        
        try:
            self.update_status("Loading model...")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Try to load model info from training results
            model_dir = os.path.dirname(model_path)
            results_path = os.path.join(model_dir, 'training_results.json')
            
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    self.model_info = json.load(f)
                backbone = self.model_info['model_config']['backbone']
            else:
                # Default backbone if no info available
                backbone = 'resnet50'
                self.model_info = {'model_config': {'backbone': backbone}}
            
            # Create model
            self.model = RatingPredictionModel(backbone=backbone, pretrained=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Enable prediction button
            self.predict_button.config(state='normal')
            
            self.update_status(f"Model loaded successfully - {backbone.upper()} architecture")
            messagebox.showinfo("Success", f"Model loaded successfully!\nArchitecture: {backbone}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            self.update_status("Failed to load model")
    
    def show_model_info(self):
        """Show model information dialog"""
        if self.model_info is None:
            messagebox.showwarning("No Model", "Please load a model first")
            return
        
        info_window = tk.Toplevel(self.root)
        info_window.title("Model Information")
        info_window.geometry("500x400")
        info_window.configure(bg='white')
        
        # Create scrollable text widget
        text_frame = tk.Frame(info_window, bg='white')
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side='right', fill='y')
        
        text_widget = tk.Text(text_frame, wrap='word', yscrollcommand=scrollbar.set,
                             font=('Courier', 10), bg='#f8f9fa')
        text_widget.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=text_widget.yview)
        
        # Display model info
        info_text = json.dumps(self.model_info, indent=2)
        text_widget.insert('1.0', info_text)
        text_widget.config(state='disabled')
    
    def upload_image(self):
        """Upload and display image"""
        file_path = filedialog.askopenfilename(
            title="Select Product Image",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                ("All Files", "*.*")
            ]
        )
        
        if file_path:
            self.load_and_display_image(file_path)
    
    def use_sample_image(self):
        """Use a sample image for testing"""
        # Create a simple sample image
        sample_image = Image.new('RGB', (300, 300), color='lightblue')
        
        # Save temporarily
        temp_path = 'temp_sample.jpg'
        sample_image.save(temp_path)
        
        self.load_and_display_image(temp_path)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    def load_and_display_image(self, image_path):
        """Load and display image in the UI"""
        try:
            # Load image
            self.current_image = Image.open(image_path)
            self.current_image_path = image_path
            
            # Create display version
            display_image = self.current_image.copy()
            
            # Resize for display while maintaining aspect ratio
            display_size = (400, 400)
            display_image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(display_image)
            
            # Update display
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
            
            self.update_status(f"Image loaded: {os.path.basename(image_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
            self.update_status("Failed to load image")
    
    def predict_rating(self):
        """Predict rating for current image"""
        if self.model is None:
            messagebox.showwarning("No Model", "Please load a model first")
            return
        
        if not hasattr(self, 'current_image'):
            messagebox.showwarning("No Image", "Please upload an image first")
            return
        
        try:
            self.update_status("Making prediction...")
            self.predict_button.config(state='disabled', text="Predicting...")
            
            # Process image in separate thread to prevent UI freezing
            threading.Thread(target=self._predict_thread, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed:\n{str(e)}")
            self.update_status("Prediction failed")
            self.predict_button.config(state='normal', text="🔮 Predict Rating")
    
    def _predict_thread(self):
        """Prediction in separate thread"""
        try:
            # Preprocess image
            image_rgb = self.current_image.convert('RGB')
            input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                prediction = output.item()
            
            # Clamp to valid range
            clamped_prediction = max(1.0, min(5.0, prediction))
            
            # Schedule UI update in main thread
            self.root.after(0, self._update_prediction_ui, prediction, clamped_prediction)
            
        except Exception as e:
            self.root.after(0, self._prediction_error, str(e))
    
    def _update_prediction_ui(self, raw_prediction, clamped_prediction):
        """Update UI with prediction results"""
        # Update prediction display
        stars = "⭐" * int(round(clamped_prediction))
        self.prediction_label.config(
            text=f"{stars}\n{clamped_prediction:.2f} / 5.0",
            font=('Arial', 18, 'bold'),
            fg='#e67e22' if clamped_prediction >= 3.5 else '#e74c3c'
        )
        
        # Update details
        for widget in self.details_frame.winfo_children():
            widget.destroy()
        
        tk.Label(self.details_frame, text=f"Raw prediction: {raw_prediction:.4f}", 
                font=('Arial', 10), bg='white', fg='#7f8c8d').pack(anchor='w')
        tk.Label(self.details_frame, text=f"Clamped (1-5): {clamped_prediction:.4f}", 
                font=('Arial', 10), bg='white', fg='#7f8c8d').pack(anchor='w')
        
        confidence = "High" if abs(clamped_prediction - round(clamped_prediction)) < 0.3 else "Medium"
        tk.Label(self.details_frame, text=f"Confidence: {confidence}", 
                font=('Arial', 10), bg='white', fg='#27ae60' if confidence == "High" else '#f39c12').pack(anchor='w')
        
        # Add to history
        timestamp = time.strftime("%H:%M:%S")
        image_name = os.path.basename(getattr(self, 'current_image_path', 'Unknown'))
        history_entry = f"{timestamp} | {image_name} | Rating: {clamped_prediction:.2f}"
        
        self.history_listbox.insert(0, history_entry)
        self.prediction_history.append({
            'timestamp': timestamp,
            'image_name': image_name,
            'raw_prediction': raw_prediction,
            'clamped_prediction': clamped_prediction
        })
        
        # Re-enable button
        self.predict_button.config(state='normal', text="🔮 Predict Rating")
        self.update_status("Prediction completed")
    
    def _prediction_error(self, error_message):
        """Handle prediction error"""
        messagebox.showerror("Prediction Error", f"Failed to predict rating:\n{error_message}")
        self.predict_button.config(state='normal', text="🔮 Predict Rating")
        self.update_status("Prediction failed")
    
    def clear_history(self):
        """Clear prediction history"""
        if messagebox.askyesno("Clear History", "Are you sure you want to clear all prediction history?"):
            self.history_listbox.delete(0, 'end')
            self.prediction_history.clear()
            self.update_status("History cleared")
    
    def export_results(self):
        """Export prediction results to file"""
        if not self.prediction_history:
            messagebox.showwarning("No Data", "No prediction history to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.prediction_history, f, indent=2)
                messagebox.showinfo("Success", f"Results exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results:\n{str(e)}")

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = ModelTesterUI(root)
    
    # Set window icon (if available)
    try:
        root.iconbitmap('icon.ico')  # Add an icon file if you have one
    except:
        pass
    
    # Start the application
    root.mainloop()

if __name__ == "__main__":
    main()