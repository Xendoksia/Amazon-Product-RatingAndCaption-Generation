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
    """Dataset class for product images and ratings with metadata features"""
    
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
        
        # Prepare categorical encoders
        self._prepare_categorical_features()
        
        print(f"Dataset initialized with {len(self.metadata)} samples (from {initial_count} total)")
    
    def _prepare_categorical_features(self):
        """Prepare categorical feature encodings"""
        from sklearn.preprocessing import LabelEncoder
        
        # Initialize encoders
        self.category_encoder = LabelEncoder()
        self.store_encoder = LabelEncoder()
        self.variant_encoder = LabelEncoder()
        
        # Handle missing values and encode
        self.metadata['category_clean'] = self.metadata['category'].fillna('unknown')
        self.metadata['store_clean'] = self.metadata['store'].fillna('unknown')
        self.metadata['variant_clean'] = self.metadata['variant'].fillna('MAIN')
        
        # Fit encoders
        self.category_encoder.fit(self.metadata['category_clean'])
        self.store_encoder.fit(self.metadata['store_clean'])
        self.variant_encoder.fit(self.metadata['variant_clean'])
        
        # Encode categories
        self.metadata['category_encoded'] = self.category_encoder.transform(self.metadata['category_clean'])
        self.metadata['store_encoded'] = self.store_encoder.transform(self.metadata['store_clean'])
        self.metadata['variant_encoded'] = self.variant_encoder.transform(self.metadata['variant_clean'])
        
        # Normalize rating_count
        max_rating_count = self.metadata['rating_count'].max()
        self.metadata['rating_count_norm'] = self.metadata['rating_count'] / max_rating_count
        
        print(f"Categorical features prepared:")
        print(f"  - Categories: {len(self.category_encoder.classes_)}")
        print(f"  - Stores: {len(self.store_encoder.classes_)}")
        print(f"  - Variants: {len(self.variant_encoder.classes_)}")
    
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
        
        # Prepare metadata features
        metadata_features = torch.tensor([
            row['category_encoded'],
            row['store_encoded'], 
            row['variant_encoded'],
            row['rating_count_norm']
        ], dtype=torch.float32)
        
        return image, metadata_features, rating, row['product_id']

class TwoTowerRatingModel(nn.Module):
    """Two Tower Architecture for Rating Prediction
    
    Tower 1: Visual features from images (CNN)
    Tower 2: Metadata features (categorical + numerical)
    Final: Combined prediction head
    """
    
    def __init__(self, backbone='efficientnet', pretrained=True, 
                 num_categories=50, num_stores=100, num_variants=10, dropout=0.3):
        super(TwoTowerRatingModel, self).__init__()
        
        self.backbone_name = backbone
        
        # TOWER 1: Visual Feature Extractor
        if backbone == 'resnet50':
            self.visual_backbone = models.resnet50(pretrained=pretrained)
            visual_features = self.visual_backbone.fc.in_features
            self.visual_backbone.fc = nn.Identity()
        elif backbone == 'resnet18':
            self.visual_backbone = models.resnet18(pretrained=pretrained)
            visual_features = self.visual_backbone.fc.in_features
            self.visual_backbone.fc = nn.Identity()
        elif backbone == 'efficientnet':
            self.visual_backbone = models.efficientnet_b0(pretrained=pretrained)
            visual_features = self.visual_backbone.classifier[1].in_features
            self.visual_backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Visual tower head
        self.visual_tower = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(visual_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # TOWER 2: Metadata Feature Processor
        # Embedding layers for categorical features
        self.category_embedding = nn.Embedding(num_categories, 32)
        self.store_embedding = nn.Embedding(num_stores, 16) 
        self.variant_embedding = nn.Embedding(num_variants, 8)
        
        # Metadata tower (categorical embeddings + numerical features)
        metadata_input_size = 32 + 16 + 8 + 1  # embeddings + rating_count_norm
        self.metadata_tower = nn.Sequential(
            nn.Linear(metadata_input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # FUSION: Combined prediction head
        combined_input_size = 128 + 32  # visual_tower + metadata_tower
        self.fusion_head = nn.Sequential(
            nn.Linear(combined_input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, images, metadata_features):
        # Tower 1: Visual features
        visual_features = self.visual_backbone(images)
        visual_output = self.visual_tower(visual_features)
        
        # Tower 2: Metadata features
        # Extract and embed categorical features
        category_idx = metadata_features[:, 0].long()
        store_idx = metadata_features[:, 1].long() 
        variant_idx = metadata_features[:, 2].long()
        rating_count_norm = metadata_features[:, 3:4]  # Keep as 2D
        
        category_emb = self.category_embedding(category_idx)
        store_emb = self.store_embedding(store_idx)
        variant_emb = self.variant_embedding(variant_idx)
        
        # Combine metadata features
        metadata_combined = torch.cat([
            category_emb, store_emb, variant_emb, rating_count_norm
        ], dim=1)
        
        metadata_output = self.metadata_tower(metadata_combined)
        
        # Fusion: Combine both towers
        combined_features = torch.cat([visual_output, metadata_output], dim=1)
        final_output = self.fusion_head(combined_features)
        
        return final_output.squeeze()

# Keep the original model for backward compatibility
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
            for batch_idx, (images, metadata, ratings, _) in enumerate(train_pbar):
                images, metadata, ratings = images.to(self.device), metadata.to(self.device), ratings.to(self.device, dtype=torch.float32)
                
                optimizer.zero_grad()
                
                # Check if model is Two Tower
                if hasattr(model, 'fusion_head'):  # Two Tower model
                    outputs = model(images, metadata)
                else:  # Original single tower model
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
                for batch_idx, (images, metadata, ratings, _) in enumerate(val_pbar):
                    images, metadata, ratings = images.to(self.device), metadata.to(self.device), ratings.to(self.device, dtype=torch.float32)
                    
                    # Check if model is Two Tower
                    if hasattr(model, 'fusion_head'):  # Two Tower model
                        outputs = model(images, metadata)
                    else:  # Original single tower model
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
            for images, metadata, ratings, p_ids in tqdm(test_loader, desc='Evaluating'):
                images, metadata, ratings = images.to(self.device), metadata.to(self.device), ratings.to(self.device, dtype=torch.float32)
                
                # Check if model is Two Tower
                if hasattr(model, 'fusion_head'):  # Two Tower model
                    outputs = model(images, metadata)
                else:  # Original single tower model
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
                          learning_rate=0.001, augment=True, use_two_tower=True):
        """Complete training and evaluation pipeline"""
        print("Starting Rating Prediction Training Pipeline")
        print("=" * 60)
        
        # Create data loaders
        print("Creating data loaders...")
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = \
            self.create_data_loaders(batch_size, augment=augment)
        
        # Create model
        if use_two_tower:
            print(f"Creating Two Tower {backbone} model...")
            
            # Get dataset info for embedding dimensions
            sample_dataset = train_dataset.metadata
            num_categories = len(train_dataset.category_encoder.classes_)
            num_stores = len(train_dataset.store_encoder.classes_)
            num_variants = len(train_dataset.variant_encoder.classes_)
            
            model = TwoTowerRatingModel(
                backbone=backbone, 
                pretrained=True, 
                num_categories=num_categories,
                num_stores=num_stores,
                num_variants=num_variants,
                dropout=0.3
            )
            print(f"Two Tower Architecture:")
            print(f"  - Visual Tower: {backbone}")
            print(f"  - Metadata Tower: {num_categories} categories, {num_stores} stores, {num_variants} variants")
        else:
            print(f"Creating single tower {backbone} model...")
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
    
    # Train and evaluate model with Two Tower Architecture
    model, results = trainer.train_and_evaluate(
        backbone='efficientnet',
        batch_size=32,
        num_epochs=40, 
        learning_rate=0.0003,
        augment=True,
        use_two_tower=True  # Enable Two Tower Architecture
    )