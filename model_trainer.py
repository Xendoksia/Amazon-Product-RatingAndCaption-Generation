import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class ProductImageDataset(Dataset):
    """Dataset class for product images and ratings"""
    
    def __init__(self, metadata_csv, images_dir, transform=None, main_images_only=True):
        self.metadata = pd.read_csv(metadata_csv)
        self.images_dir = images_dir
        self.transform = transform
        
        initial_count = len(self.metadata)
        
        # Filter for main images only if specified
        if main_images_only:
            self.metadata = self.metadata[self.metadata['is_main'] == True]
        
        # Remove any rows with missing ratings
        self.metadata = self.metadata.dropna(subset=['rating'])
        
        # Filter out missing images upfront
        self.metadata = self._filter_existing_images()
        
        print(f"Dataset initialized with {len(self.metadata)} samples (from {initial_count} total)")
    
    def _filter_existing_images(self):
        """Filter out rows where image files don't exist"""
        existing_mask = []
        missing_count = 0
        
        for idx, row in self.metadata.iterrows():
            image_path = os.path.join(self.images_dir, row['filename'])
            if os.path.exists(image_path):
                existing_mask.append(True)
            else:
                existing_mask.append(False)
                missing_count += 1
        
        if missing_count > 0:
            print(f"Warning: Skipped {missing_count} samples with missing image files")
        
        return self.metadata[existing_mask].reset_index(drop=True)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load image
        image_path = os.path.join(self.images_dir, row['filename'])
        
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            # This should not happen after filtering, but just in case
            print(f"Error: Image file missing during training: {image_path}")
            # Create a neutral gray image as fallback
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        except Exception as e:
            # Handle corrupted images or other PIL errors
            print(f"Warning: Could not load image {image_path}: {e}")
            # Create a neutral gray image as fallback
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get rating (ensure it's float32 for PyTorch)
        rating = float(row['rating'])
        
        return image, rating, row['product_id']

# Original simple CNN model for rating prediction
class RatingPredictionModel(nn.Module):
    """Neural network model for rating prediction"""
    
    def __init__(self, backbone='resnet50', pretrained=True, num_classes=1, dropout=0.5):
        super(RatingPredictionModel, self).__init__()
        
        self.backbone_name = backbone
        
        # Load backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final layer
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

class RatingPredictor:
    """Main class for training and evaluating rating prediction models"""
    
    def __init__(self, dataset_dir='product_dataset', model_save_dir='models'):
        self.dataset_dir = dataset_dir
        self.model_save_dir = model_save_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model save directory
        os.makedirs(model_save_dir, exist_ok=True)
        
        print(f"Using device: {self.device}")
    
    def get_transforms(self, augment=True):
        """Get data transforms for training and validation"""
        if augment:
            # Training transforms with augmentation
            train_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Validation/test transforms (no augmentation)
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
    
    def create_data_loaders(self, batch_size=16, num_workers=4, augment=True):
        """Create data loaders for training, validation, and testing"""
        train_transform, val_transform = self.get_transforms(augment)
        
        # Create datasets
        train_dataset = ProductImageDataset(
            os.path.join(self.dataset_dir, 'train_metadata.csv'),
            os.path.join(self.dataset_dir, 'images'),
            transform=train_transform
        )
        
        val_dataset = ProductImageDataset(
            os.path.join(self.dataset_dir, 'val_metadata.csv'),
            os.path.join(self.dataset_dir, 'images'),
            transform=val_transform
        )
        
        test_dataset = ProductImageDataset(
            os.path.join(self.dataset_dir, 'test_metadata.csv'),
            os.path.join(self.dataset_dir, 'images'),
            transform=val_transform
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=True
        )
        
        return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
    
    def train_model(self, model, train_loader, val_loader, num_epochs=50, 
                   learning_rate=0.001, weight_decay=1e-4, patience=10):
        """Train the rating prediction model"""
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience//2, verbose=True
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'learning_rate': []
        }
        
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        model.to(self.device)
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_predictions = []
            train_targets = []
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
            for batch_idx, batch_data in enumerate(train_pbar):
                if len(batch_data) == 3:  # Original format: image, rating, product_id
                    images, ratings, _ = batch_data
                    images, ratings = images.to(self.device), ratings.to(self.device, dtype=torch.float32)
                else:  # Extended format with metadata
                    images, metadata, ratings, _ = batch_data
                    images, ratings = images.to(self.device), ratings.to(self.device, dtype=torch.float32)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, ratings)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_predictions.extend(outputs.detach().cpu().numpy().flatten())
                train_targets.extend(ratings.detach().cpu().numpy().flatten())
                
                # Update progress bar
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{train_loss/(batch_idx+1):.4f}'
                })
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
                for batch_idx, batch_data in enumerate(val_pbar):
                    if len(batch_data) == 3:  # Original format
                        images, ratings, _ = batch_data
                        images, ratings = images.to(self.device), ratings.to(self.device, dtype=torch.float32)
                    else:  # Extended format with metadata
                        images, metadata, ratings, _ = batch_data
                        images, ratings = images.to(self.device), ratings.to(self.device, dtype=torch.float32)
                    
                    outputs = model(images)
                    loss = criterion(outputs, ratings)
                    
                    val_loss += loss.item()
                    val_predictions.extend(outputs.cpu().numpy().flatten())
                    val_targets.extend(ratings.cpu().numpy().flatten())
                    
                    val_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{val_loss/(batch_idx+1):.4f}'
                    })
            
            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_mae = mean_absolute_error(train_targets, train_predictions)
            val_mae = mean_absolute_error(val_targets, val_predictions)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_mae'].append(train_mae)
            history['val_mae'].append(val_mae)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # Learning rate scheduler
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'history': history
                }, os.path.join(self.model_save_dir, 'best_model.pth'))
            else:
                epochs_without_improvement += 1
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}')
            print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            print(f'  Best Val Loss: {best_val_loss:.4f}')
            print('-' * 50)
            
            # Early stopping check
            if epochs_without_improvement >= patience:
                print(f'Early stopping after {epoch+1} epochs')
                break
        
        return history
    
    def evaluate_model(self, model, test_loader):
        """Evaluate model performance on test set"""
        model.eval()
        model.to(self.device)
        predictions = []
        targets = []
        product_ids = []
        
        with torch.no_grad():
            for batch_data in tqdm(test_loader, desc='Evaluating'):
                if len(batch_data) == 3:  # Original format: image, rating, product_id
                    images, ratings, p_ids = batch_data
                    images, ratings = images.to(self.device), ratings.to(self.device, dtype=torch.float32)
                else:  # Extended format with metadata
                    images, metadata, ratings, p_ids = batch_data
                    images, ratings = images.to(self.device), ratings.to(self.device, dtype=torch.float32)
                
                outputs = model(images)
                
                predictions.extend(outputs.cpu().numpy().flatten())
                targets.extend(ratings.cpu().numpy().flatten())
                product_ids.extend(p_ids.numpy().flatten())
        
        # Calculate metrics
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, predictions)
        
        # Clamp predictions to valid rating range (1-5)
        clamped_predictions = np.clip(predictions, 1.0, 5.0)
        clamped_mae = mean_absolute_error(targets, clamped_predictions)
        clamped_rmse = np.sqrt(mean_squared_error(targets, clamped_predictions))
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'clamped_mae': clamped_mae,
            'clamped_rmse': clamped_rmse
        }
        
        print("Test Set Evaluation Results:")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"Clamped MAE (1-5 range): {clamped_mae:.4f}")
        print(f"Clamped RMSE (1-5 range): {clamped_rmse:.4f}")
        
        return metrics, predictions, targets, product_ids
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(history['train_loss'], label='Train Loss')
        axes[0, 0].plot(history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('MSE Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE plot
        axes[0, 1].plot(history['train_mae'], label='Train MAE')
        axes[0, 1].plot(history['val_mae'], label='Val MAE')
        axes[0, 1].set_title('Training and Validation MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Mean Absolute Error')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        axes[1, 0].plot(history['learning_rate'])
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Loss comparison
        axes[1, 1].plot(np.array(history['train_loss']) - np.array(history['val_loss']))
        axes[1, 1].set_title('Training vs Validation Loss Difference')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Train Loss - Val Loss')
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_save_dir, 'training_history.png'), dpi=300)
        plt.show()
    
    def plot_predictions(self, predictions, targets, save_name='test_predictions.png'):
        """Plot prediction vs actual ratings"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        axes[0].scatter(targets, predictions, alpha=0.6)
        axes[0].plot([1, 5], [1, 5], 'r--', label='Perfect Prediction')
        axes[0].set_xlabel('Actual Rating')
        axes[0].set_ylabel('Predicted Rating')
        axes[0].set_title('Predicted vs Actual Ratings')
        axes[0].legend()
        axes[0].grid(True)
        
        # Error histogram
        errors = np.array(predictions) - np.array(targets)
        axes[1].hist(errors, bins=30, alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', label='No Error')
        axes[1].set_xlabel('Prediction Error')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Prediction Errors')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_save_dir, save_name), dpi=300)
        plt.show()
    
    def train_and_evaluate(self, backbone='resnet50', batch_size=16, num_epochs=50, 
                          learning_rate=0.001, augment=True):
        """Complete training and evaluation pipeline"""
        print("Starting Rating Prediction Training Pipeline")
        print("=" * 60)
        
        # Create data loaders
        print("Creating data loaders...")
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = \
            self.create_data_loaders(batch_size, augment=augment)
        
        # Create model
        print(f"Creating {backbone} model...")
        model = RatingPredictionModel(backbone=backbone, pretrained=True, dropout=0.3)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Train model
        print("Starting training...")
        history = self.train_model(model, train_loader, val_loader, 
                                 num_epochs=num_epochs, learning_rate=learning_rate)
        
        # Load best model for evaluation
        checkpoint = torch.load(os.path.join(self.model_save_dir, 'best_model.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate on test set
        print("Evaluating on test set...")
        metrics, predictions, targets, product_ids = self.evaluate_model(model, test_loader)
        
        # Save results
        results = {
            'model_config': {
                'backbone': backbone,
                'batch_size': batch_size,
                'num_epochs': num_epochs,
                'learning_rate': learning_rate,
                'augment': augment
            },
            'dataset_info': {
                'train_samples': len(train_dataset),
                'val_samples': len(val_dataset),
                'test_samples': len(test_dataset)
            },
            'metrics': metrics,
            'history': history
        }
        
        with open(os.path.join(self.model_save_dir, 'training_results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Plot results
        self.plot_training_history(history)
        self.plot_predictions(predictions, targets)
        
        print("=" * 60)
        print("Training completed successfully!")
        print(f"Results saved to {self.model_save_dir}")
        
        return model, results

# Usage example
if __name__ == "__main__":
    # Initialize trainer
    trainer = RatingPredictor(dataset_dir='product_dataset', model_save_dir='models')
    
    # Train and evaluate model
    model, results = trainer.train_and_evaluate(
        backbone='efficientnet',
        batch_size=32,
        num_epochs=40, 
        learning_rate=0.0003,
        augment=True
    )