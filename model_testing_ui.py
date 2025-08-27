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

class CaptionGenerationModel(nn.Module):
    """Neural network model for product caption generation"""
    
    def __init__(self, vocab_size, embed_dim=512, hidden_dim=512, num_layers=2, backbone='resnet50', pretrained=True):
        super(CaptionGenerationModel, self).__init__()
        
        self.backbone_name = backbone
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Image encoder (CNN backbone)
        if backbone == 'resnet50':
            self.encoder = models.resnet50(pretrained=pretrained)
            encoder_dim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        elif backbone == 'resnet18':
            self.encoder = models.resnet18(pretrained=pretrained)
            encoder_dim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        elif backbone == 'efficientnet':
            self.encoder = models.efficientnet_b0(pretrained=pretrained)
            encoder_dim = self.encoder.classifier[1].in_features
            self.encoder.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Project image features to decoder dimension
        self.feature_projection = nn.Linear(encoder_dim, hidden_dim)
        
        # Text decoder (LSTM)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Attention mechanism (optional)
        self.attention = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
    
    def encode_image(self, images):
        """Encode images to feature vectors"""
        features = self.encoder(images)
        projected_features = self.feature_projection(features)
        return projected_features
    
    def forward(self, images, captions=None, max_length=50):
        """Forward pass for training or inference"""
        batch_size = images.size(0)
        
        # Encode images
        image_features = self.encode_image(images)  # [batch_size, hidden_dim]
        
        if captions is not None:
            # Training mode
            # Embed captions
            embedded_captions = self.embedding(captions)  # [batch_size, seq_len, embed_dim]
            
            # Initialize LSTM hidden state with image features
            h0 = image_features.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)  # [num_layers, batch_size, hidden_dim]
            c0 = torch.zeros_like(h0)
            
            # Pass through LSTM
            lstm_output, _ = self.lstm(embedded_captions, (h0, c0))
            
            # Apply attention with image features
            attended_output, _ = self.attention(lstm_output, 
                                              image_features.unsqueeze(1).repeat(1, lstm_output.size(1), 1),
                                              image_features.unsqueeze(1).repeat(1, lstm_output.size(1), 1))
            
            # Output projection
            output = self.output_projection(attended_output)
            return output
        else:
            # Inference mode - generate caption
            return self.generate_caption(image_features, max_length)
    
    def generate_caption(self, image_features, max_length=50, start_token=1, end_token=2):
        """Generate caption for given image features"""
        batch_size = image_features.size(0)
        device = image_features.device
        
        # Initialize
        captions = torch.zeros(batch_size, max_length, dtype=torch.long, device=device)
        captions[:, 0] = start_token
        
        # Initialize LSTM hidden state
        h = image_features.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        c = torch.zeros_like(h)
        
        for t in range(1, max_length):
            # Get current word embedding
            embedded_word = self.embedding(captions[:, t-1:t])
            
            # LSTM step
            lstm_output, (h, c) = self.lstm(embedded_word, (h, c))
            
            # Apply attention
            attended_output, _ = self.attention(lstm_output, 
                                              image_features.unsqueeze(1),
                                              image_features.unsqueeze(1))
            
            # Predict next word
            output = self.output_projection(attended_output.squeeze(1))
            predicted_word = output.argmax(dim=-1)
            
            captions[:, t] = predicted_word
            
            # Stop if end token is generated
            if (predicted_word == end_token).all():
                break
        
        return captions

class ModelTesterUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Product Analysis - Rating & Caption Prediction")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Model variables
        self.rating_model = None
        self.caption_model = None
        self.caption_vocab = None  # Will store vocabulary for caption model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = None
        self.rating_model_info = None
        self.caption_model_info = None
        self.prediction_history = []
        
        # Setup UI
        self.setup_ui()
        
        # Status
        self.update_status("Ready - Load models to start analysis")
    
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
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=100)
        header_frame.pack(fill='x', padx=5, pady=5)
        header_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(header_frame, text="🤖 Product Analysis Suite", 
                              font=('Arial', 20, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(side='left', padx=20, pady=15)
        
        subtitle_label = tk.Label(header_frame, text="Rating Prediction & Caption Generation", 
                                 font=('Arial', 12), fg='#bdc3c7', bg='#2c3e50')
        subtitle_label.place(x=20, y=50)
        
        # Model loading controls
        controls_frame = tk.Frame(header_frame, bg='#2c3e50')
        controls_frame.pack(side='right', padx=20, pady=10)
        
        # Rating model controls
        rating_frame = tk.LabelFrame(controls_frame, text="Rating Model", fg='white', bg='#2c3e50',
                                   font=('Arial', 10, 'bold'))
        rating_frame.pack(side='left', padx=5, pady=5, fill='y')
        
        tk.Button(rating_frame, text="Load Rating Model", command=self.load_rating_model,
                 bg='#3498db', fg='white', font=('Arial', 10, 'bold'),
                 padx=15, pady=3, relief='flat').pack(padx=5, pady=2)
        
        self.rating_status_label = tk.Label(rating_frame, text="Not loaded", 
                                          font=('Arial', 9), fg='#e74c3c', bg='#2c3e50')
        self.rating_status_label.pack(padx=5, pady=2)
        
        # Caption model controls
        caption_frame = tk.LabelFrame(controls_frame, text="Caption Model", fg='white', bg='#2c3e50',
                                    font=('Arial', 10, 'bold'))
        caption_frame.pack(side='left', padx=5, pady=5, fill='y')
        
        tk.Button(caption_frame, text="Load Caption Model", command=self.load_caption_model,
                 bg='#9b59b6', fg='white', font=('Arial', 10, 'bold'),
                 padx=15, pady=3, relief='flat').pack(padx=5, pady=2)
        
        self.caption_status_label = tk.Label(caption_frame, text="Not loaded", 
                                           font=('Arial', 9), fg='#e74c3c', bg='#2c3e50')
        self.caption_status_label.pack(padx=5, pady=2)
        
        # Info button
        tk.Button(controls_frame, text="Models Info", command=self.show_models_info,
                 bg='#95a5a6', fg='white', font=('Arial', 10),
                 padx=15, pady=5, relief='flat').pack(side='left', padx=5)
    
    def create_main_content_frame(self):
        """Create main content area"""
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left panel - Image upload and display
        self.create_image_panel(main_frame)
        
        # Middle panel - Predictions
        self.create_predictions_panel(main_frame)
        
        # Right panel - History
        self.create_history_panel(main_frame)
    
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
        
        # Analysis button
        self.analyze_button = tk.Button(left_frame, text="🔍 Analyze Product", 
                                       command=self.analyze_product,
                                       bg='#e74c3c', fg='white', font=('Arial', 14, 'bold'),
                                       padx=30, pady=10, relief='flat', state='disabled')
        self.analyze_button.pack(pady=10)
    
    def create_predictions_panel(self, parent):
        """Create middle panel for predictions display"""
        middle_frame = tk.Frame(parent, bg='white', relief='ridge', bd=2)
        middle_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        tk.Label(middle_frame, text="📊 Analysis Results", 
                font=('Arial', 14, 'bold'), bg='white').pack(anchor='w', padx=10, pady=10)
        
        # Rating prediction section
        rating_section = tk.LabelFrame(middle_frame, text="⭐ Rating Prediction", 
                                     font=('Arial', 12, 'bold'), bg='white', fg='#3498db')
        rating_section.pack(fill='x', padx=10, pady=5)
        
        self.rating_display_frame = tk.Frame(rating_section, bg='#f8f9fa', relief='ridge', bd=1)
        self.rating_display_frame.pack(fill='x', padx=5, pady=5)
        
        self.rating_label = tk.Label(self.rating_display_frame, text="No rating prediction yet", 
                                    font=('Arial', 14), bg='#f8f9fa', pady=15)
        self.rating_label.pack()
        
        self.rating_details_frame = tk.Frame(rating_section, bg='white')
        self.rating_details_frame.pack(fill='x', padx=5, pady=5)
        
        # Caption generation section
        caption_section = tk.LabelFrame(middle_frame, text="📝 Caption Generation", 
                                      font=('Arial', 12, 'bold'), bg='white', fg='#9b59b6')
        caption_section.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.caption_display_frame = tk.Frame(caption_section, bg='#f8f9fa', relief='ridge', bd=1)
        self.caption_display_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Scrollable text for caption
        caption_scroll_frame = tk.Frame(self.caption_display_frame, bg='#f8f9fa')
        caption_scroll_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        caption_scrollbar = ttk.Scrollbar(caption_scroll_frame)
        caption_scrollbar.pack(side='right', fill='y')
        
        self.caption_text = tk.Text(caption_scroll_frame, height=4, wrap='word', 
                                   yscrollcommand=caption_scrollbar.set,
                                   font=('Arial', 11), bg='#f8f9fa', relief='flat',
                                   state='disabled')
        self.caption_text.pack(side='left', fill='both', expand=True)
        caption_scrollbar.config(command=self.caption_text.yview)
        
        # Copy caption button
        tk.Button(caption_section, text="📋 Copy Caption", command=self.copy_caption,
                 bg='#95a5a6', fg='white', font=('Arial', 10),
                 padx=10, pady=3, relief='flat').pack(pady=5)
        
        # Initialize with placeholder text
        self.caption_text.config(state='normal')
        self.caption_text.insert('1.0', "No caption generated yet")
        self.caption_text.config(state='disabled')
    
    def create_history_panel(self, parent):
        """Create right panel for history"""
        right_frame = tk.Frame(parent, bg='white', relief='ridge', bd=2)
        right_frame.pack(side='right', fill='both', expand=True, padx=5)
        
        # History section
        tk.Label(right_frame, text="📚 Analysis History", 
                font=('Arial', 14, 'bold'), bg='white').pack(anchor='w', padx=10, pady=10)
        
        # History list with scrollbar
        list_frame = tk.Frame(right_frame, bg='white')
        list_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Scrollable listbox
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side='right', fill='y')
        
        self.history_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set,
                                         font=('Arial', 9), selectmode='single')
        self.history_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.history_listbox.yview)
        
        # Bind selection event
        self.history_listbox.bind('<<ListboxSelect>>', self.on_history_select)
        
        # History controls
        history_controls = tk.Frame(right_frame, bg='white')
        history_controls.pack(fill='x', padx=10, pady=10)
        
        tk.Button(history_controls, text="Clear History", command=self.clear_history,
                 bg='#95a5a6', fg='white', font=('Arial', 10),
                 padx=10, pady=5, relief='flat').pack(side='left')
        
        tk.Button(history_controls, text="Export Results", command=self.export_results,
                 bg='#3498db', fg='white', font=('Arial', 10),
                 padx=10, pady=5, relief='flat').pack(side='left', padx=(10, 0))
        
        tk.Button(history_controls, text="View Details", command=self.view_history_details,
                 bg='#27ae60', fg='white', font=('Arial', 10),
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
    
    def load_rating_model(self):
        """Load rating prediction model"""
        model_path = filedialog.askopenfilename(
            title="Select Rating Model File",
            filetypes=[("PyTorch Models", "*.pth"), ("All Files", "*.*")]
        )
        
        if not model_path:
            return
        
        try:
            self.update_status("Loading rating model...")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Try to load model info
            model_dir = os.path.dirname(model_path)
            results_path = os.path.join(model_dir, 'training_results.json')
            
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    self.rating_model_info = json.load(f)
                backbone = self.rating_model_info['model_config']['backbone']
            else:
                backbone = 'resnet50'
                self.rating_model_info = {'model_config': {'backbone': backbone}}
            
            # Create and load model
            self.rating_model = RatingPredictionModel(backbone=backbone, pretrained=False)
            self.rating_model.load_state_dict(checkpoint['model_state_dict'])
            self.rating_model.to(self.device)
            self.rating_model.eval()
            
            # Update UI
            self.rating_status_label.config(text="✓ Loaded", fg='#27ae60')
            self.update_analyze_button()
            
            self.update_status(f"Rating model loaded - {backbone.upper()}")
            messagebox.showinfo("Success", f"Rating model loaded!\nArchitecture: {backbone}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load rating model:\n{str(e)}")
            self.update_status("Failed to load rating model")
    
    def load_caption_model(self):
        """Load caption generation model"""
        model_path = filedialog.askopenfilename(
            title="Select Caption Model File",
            filetypes=[("PyTorch Models", "*.pth"), ("All Files", "*.*")]
        )
        
        if not model_path:
            return
        
        # Also ask for vocabulary file
        vocab_path = filedialog.askopenfilename(
            title="Select Vocabulary File",
            filetypes=[("JSON Files", "*.json"), ("Pickle Files", "*.pkl"), ("All Files", "*.*")]
        )
        
        if not vocab_path:
            messagebox.showwarning("Warning", "Vocabulary file is required for caption generation")
            return
        
        try:
            self.update_status("Loading caption model...")
            
            # Load vocabulary
            if vocab_path.endswith('.json'):
                with open(vocab_path, 'r') as f:
                    self.caption_vocab = json.load(f)
            else:
                import pickle
                with open(vocab_path, 'rb') as f:
                    self.caption_vocab = pickle.load(f)
            
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Try to load model info
            model_dir = os.path.dirname(model_path)
            results_path = os.path.join(model_dir, 'caption_training_results.json')
            
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    self.caption_model_info = json.load(f)
                config = self.caption_model_info['model_config']
            else:
                # Default configuration
                config = {
                    'backbone': 'resnet50',
                    'vocab_size': len(self.caption_vocab),
                    'embed_dim': 512,
                    'hidden_dim': 512,
                    'num_layers': 2
                }
                self.caption_model_info = {'model_config': config}
            
            # Create and load model
            self.caption_model = CaptionGenerationModel(
                vocab_size=config['vocab_size'],
                embed_dim=config.get('embed_dim', 512),
                hidden_dim=config.get('hidden_dim', 512),
                num_layers=config.get('num_layers', 2),
                backbone=config.get('backbone', 'resnet50'),
                pretrained=False
            )
            
            self.caption_model.load_state_dict(checkpoint['model_state_dict'])
            self.caption_model.to(self.device)
            self.caption_model.eval()
            
            # Update UI
            self.caption_status_label.config(text="✓ Loaded", fg='#27ae60')
            self.update_analyze_button()
            
            self.update_status(f"Caption model loaded - {config.get('backbone', 'resnet50').upper()}")
            messagebox.showinfo("Success", f"Caption model loaded!\nVocabulary size: {len(self.caption_vocab)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load caption model:\n{str(e)}")
            self.update_status("Failed to load caption model")
    
    def update_analyze_button(self):
        """Update analyze button state based on loaded models"""
        if self.rating_model is not None or self.caption_model is not None:
            self.analyze_button.config(state='normal')
        else:
            self.analyze_button.config(state='disabled')
    
    def show_models_info(self):
        """Show models information dialog"""
        if self.rating_model_info is None and self.caption_model_info is None:
            messagebox.showwarning("No Models", "Please load models first")
            return
        
        info_window = tk.Toplevel(self.root)
        info_window.title("Models Information")
        info_window.geometry("600x500")
        info_window.configure(bg='white')
        
        # Create notebook for tabs
        notebook = ttk.Notebook(info_window)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Rating model tab
        if self.rating_model_info:
            rating_frame = ttk.Frame(notebook)
            notebook.add(rating_frame, text="Rating Model")
            
            rating_text = tk.Text(rating_frame, wrap='word', font=('Courier', 10))
            rating_text.pack(fill='both', expand=True, padx=5, pady=5)
            rating_text.insert('1.0', json.dumps(self.rating_model_info, indent=2))
            rating_text.config(state='disabled')
        
        # Caption model tab
        if self.caption_model_info:
            caption_frame = ttk.Frame(notebook)
            notebook.add(caption_frame, text="Caption Model")
            
            caption_text = tk.Text(caption_frame, wrap='word', font=('Courier', 10))
            caption_text.pack(fill='both', expand=True, padx=5, pady=5)
            caption_text.insert('1.0', json.dumps(self.caption_model_info, indent=2))
            caption_text.config(state='disabled')
    
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
            display_size = (350, 350)
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
    
    def analyze_product(self):
        """Analyze product image with both models"""
        if self.rating_model is None and self.caption_model is None:
            messagebox.showwarning("No Models", "Please load at least one model first")
            return
        
        if not hasattr(self, 'current_image'):
            messagebox.showwarning("No Image", "Please upload an image first")
            return
        
        try:
            self.update_status("Analyzing product...")
            self.analyze_button.config(state='disabled', text="Analyzing...")
            
            # Process image in separate thread to prevent UI freezing
            threading.Thread(target=self._analyze_thread, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed:\n{str(e)}")
            self.update_status("Analysis failed")
            self.analyze_button.config(state='normal', text="🔍 Analyze Product")
    
    def _analyze_thread(self):
        """Analysis in separate thread"""
        results = {}
        
        try:
            # Preprocess image
            image_rgb = self.current_image.convert('RGB')
            input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
            
            # Rating prediction
            if self.rating_model is not None:
                with torch.no_grad():
                    rating_output = self.rating_model(input_tensor)
                    rating_prediction = rating_output.item()
                    clamped_rating = max(1.0, min(5.0, rating_prediction))
                    
                results['rating'] = {
                    'raw': rating_prediction,
                    'clamped': clamped_rating
                }
            
            # Caption generation
            if self.caption_model is not None:
                with torch.no_grad():
                    caption_tokens = self.caption_model(input_tensor)
                    caption_text = self.decode_caption(caption_tokens[0])
                    
                results['caption'] = caption_text
            
            # Schedule UI update in main thread
            self.root.after(0, self._update_analysis_ui, results)
            
        except Exception as e:
            self.root.after(0, self._analysis_error, str(e))
    
    def decode_caption(self, token_sequence):
        """Decode token sequence to text caption"""
        if self.caption_vocab is None:
            return "Error: No vocabulary loaded"
        
        # Create reverse vocabulary mapping
        if isinstance(self.caption_vocab, dict):
            if 'idx_to_word' in self.caption_vocab:
                idx_to_word = self.caption_vocab['idx_to_word']
            elif 'word_to_idx' in self.caption_vocab:
                word_to_idx = self.caption_vocab['word_to_idx']
                idx_to_word = {v: k for k, v in word_to_idx.items()}
            else:
                # Assume direct mapping
                idx_to_word = {v: k for k, v in self.caption_vocab.items()}
        else:
            # Assume it's a list
            idx_to_word = {i: word for i, word in enumerate(self.caption_vocab)}
        
        # Decode tokens
        words = []
        for token in token_sequence:
            token_id = token.item() if hasattr(token, 'item') else int(token)
            
            if token_id in idx_to_word:
                word = idx_to_word[token_id]
                if word in ['<start>', '<end>', '<pad>']:
                    if word == '<end>':
                        break
                    continue
                words.append(word)
            else:
                words.append(f'<UNK_{token_id}>')
        
        caption = ' '.join(words)
        
        # Clean up caption
        caption = caption.replace(' ,', ',').replace(' .', '.')
        caption = caption.replace(' \'', '\'').replace(' "', '"')
        
        return caption.strip() if caption.strip() else "Unable to generate caption"
    
    def _update_analysis_ui(self, results):
        """Update UI with analysis results"""
        timestamp = time.strftime("%H:%M:%S")
        image_name = os.path.basename(getattr(self, 'current_image_path', 'Unknown'))
        
        # Update rating display
        if 'rating' in results:
            rating = results['rating']
            stars = "⭐" * int(round(rating['clamped']))
            
            self.rating_label.config(
                text=f"{stars}\n{rating['clamped']:.2f} / 5.0",
                font=('Arial', 16, 'bold'),
                fg='#e67e22' if rating['clamped'] >= 3.5 else '#e74c3c'
            )
            
            # Update rating details
            for widget in self.rating_details_frame.winfo_children():
                widget.destroy()
            
            tk.Label(self.rating_details_frame, text=f"Raw: {rating['raw']:.4f}", 
                    font=('Arial', 9), bg='white', fg='#7f8c8d').pack(anchor='w', padx=5)
            tk.Label(self.rating_details_frame, text=f"Clamped: {rating['clamped']:.4f}", 
                    font=('Arial', 9), bg='white', fg='#7f8c8d').pack(anchor='w', padx=5)
            
            confidence = "High" if abs(rating['clamped'] - round(rating['clamped'])) < 0.3 else "Medium"
            tk.Label(self.rating_details_frame, text=f"Confidence: {confidence}", 
                    font=('Arial', 9), bg='white', fg='#27ae60' if confidence == "High" else '#f39c12').pack(anchor='w', padx=5)
        else:
            self.rating_label.config(text="Rating model not loaded", font=('Arial', 12), fg='#95a5a6')
        
        # Update caption display
        if 'caption' in results:
            self.caption_text.config(state='normal')
            self.caption_text.delete('1.0', 'end')
            self.caption_text.insert('1.0', results['caption'])
            self.caption_text.config(state='disabled')
            self.current_caption = results['caption']
        else:
            self.caption_text.config(state='normal')
            self.caption_text.delete('1.0', 'end')
            self.caption_text.insert('1.0', "Caption model not loaded")
            self.caption_text.config(state='disabled')
            self.current_caption = None
        
        # Add to history
        history_entry_text = f"{timestamp} | {image_name}"
        if 'rating' in results:
            history_entry_text += f" | Rating: {results['rating']['clamped']:.2f}"
        if 'caption' in results:
            # Truncate caption for history display
            caption_preview = results['caption'][:50] + "..." if len(results['caption']) > 50 else results['caption']
            history_entry_text += f" | Caption: {caption_preview}"
        
        self.history_listbox.insert(0, history_entry_text)
        
        # Store full results in history
        history_data = {
            'timestamp': timestamp,
            'image_name': image_name,
            'image_path': getattr(self, 'current_image_path', ''),
        }
        
        if 'rating' in results:
            history_data.update({
                'rating_raw': results['rating']['raw'],
                'rating_clamped': results['rating']['clamped']
            })
        
        if 'caption' in results:
            history_data['caption'] = results['caption']
        
        self.prediction_history.append(history_data)
        
        # Re-enable button
        self.analyze_button.config(state='normal', text="🔍 Analyze Product")
        self.update_status("Analysis completed")
    
    def _analysis_error(self, error_message):
        """Handle analysis error"""
        messagebox.showerror("Analysis Error", f"Failed to analyze product:\n{error_message}")
        self.analyze_button.config(state='normal', text="🔍 Analyze Product")
        self.update_status("Analysis failed")
    
    def copy_caption(self):
        """Copy current caption to clipboard"""
        if hasattr(self, 'current_caption') and self.current_caption:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.current_caption)
            messagebox.showinfo("Copied", "Caption copied to clipboard!")
        else:
            messagebox.showwarning("No Caption", "No caption to copy")
    
    def on_history_select(self, event):
        """Handle history selection"""
        selection = self.history_listbox.curselection()
        if selection:
            index = selection[0]
            if index < len(self.prediction_history):
                # Could implement preview of selected history item here
                pass
    
    def view_history_details(self):
        """View detailed history information"""
        selection = self.history_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a history item first")
            return
        
        index = selection[0]
        if index >= len(self.prediction_history):
            return
        
        data = self.prediction_history[index]
        
        # Create details window
        details_window = tk.Toplevel(self.root)
        details_window.title("Analysis Details")
        details_window.geometry("500x400")
        details_window.configure(bg='white')
        
        # Create scrollable text widget
        text_frame = tk.Frame(details_window, bg='white')
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side='right', fill='y')
        
        text_widget = tk.Text(text_frame, wrap='word', yscrollcommand=scrollbar.set,
                             font=('Arial', 11), bg='#f8f9fa')
        text_widget.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=text_widget.yview)
        
        # Format details
        details_text = f"""Analysis Details
{'='*50}

Timestamp: {data['timestamp']}
Image: {data['image_name']}
Image Path: {data.get('image_path', 'N/A')}

"""
        
        if 'rating_clamped' in data:
            details_text += f"""Rating Analysis:
  • Final Rating: {data['rating_clamped']:.2f} / 5.0
  • Raw Prediction: {data['rating_raw']:.4f}
  • Stars: {'⭐' * int(round(data['rating_clamped']))}

"""
        
        if 'caption' in data:
            details_text += f"""Caption Generation:
  • Generated Caption:
    "{data['caption']}"

"""
        
        text_widget.insert('1.0', details_text)
        text_widget.config(state='disabled')
        
        # Copy buttons
        button_frame = tk.Frame(details_window, bg='white')
        button_frame.pack(fill='x', padx=10, pady=5)
        
        if 'caption' in data:
            tk.Button(button_frame, text="Copy Caption", 
                     command=lambda: self._copy_to_clipboard(data['caption']),
                     bg='#3498db', fg='white', font=('Arial', 10),
                     padx=10, pady=5, relief='flat').pack(side='left', padx=5)
        
        tk.Button(button_frame, text="Copy All Details", 
                 command=lambda: self._copy_to_clipboard(details_text),
                 bg='#95a5a6', fg='white', font=('Arial', 10),
                 padx=10, pady=5, relief='flat').pack(side='left', padx=5)
    
    def _copy_to_clipboard(self, text):
        """Copy text to clipboard"""
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        messagebox.showinfo("Copied", "Content copied to clipboard!")
    
    def clear_history(self):
        """Clear prediction history"""
        if messagebox.askyesno("Clear History", "Are you sure you want to clear all analysis history?"):
            self.history_listbox.delete(0, 'end')
            self.prediction_history.clear()
            self.update_status("History cleared")
    
    def export_results(self):
        """Export analysis results to file"""
        if not self.prediction_history:
            messagebox.showwarning("No Data", "No analysis history to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("CSV Files", "*.csv"), ("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                if file_path.endswith('.json'):
                    # Export as JSON
                    export_data = {
                        'export_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                        'models_info': {
                            'rating_model': self.rating_model_info,
                            'caption_model': self.caption_model_info
                        },
                        'analysis_results': self.prediction_history
                    }
                    
                    with open(file_path, 'w') as f:
                        json.dump(export_data, f, indent=2)
                
                elif file_path.endswith('.csv'):
                    # Export as CSV
                    import csv
                    with open(file_path, 'w', newline='', encoding='utf-8') as f:
                        fieldnames = ['timestamp', 'image_name', 'rating_clamped', 'rating_raw', 'caption']
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        
                        writer.writeheader()
                        for item in self.prediction_history:
                            row = {
                                'timestamp': item['timestamp'],
                                'image_name': item['image_name'],
                                'rating_clamped': item.get('rating_clamped', ''),
                                'rating_raw': item.get('rating_raw', ''),
                                'caption': item.get('caption', '')
                            }
                            writer.writerow(row)
                
                else:
                    # Export as text
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write("Product Analysis Results\n")
                        f.write("="*50 + "\n")
                        f.write(f"Export Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Total Analyses: {len(self.prediction_history)}\n\n")
                        
                        for i, item in enumerate(self.prediction_history, 1):
                            f.write(f"Analysis #{i}\n")
                            f.write(f"  Time: {item['timestamp']}\n")
                            f.write(f"  Image: {item['image_name']}\n")
                            
                            if 'rating_clamped' in item:
                                f.write(f"  Rating: {item['rating_clamped']:.2f}/5.0 (raw: {item['rating_raw']:.4f})\n")
                            
                            if 'caption' in item:
                                f.write(f"  Caption: {item['caption']}\n")
                            
                            f.write("\n")
                
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
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    # Start the application
    root.mainloop()

if __name__ == "__main__":
    main()