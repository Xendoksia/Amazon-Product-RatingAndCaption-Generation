import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import json
from tqdm import tqdm
import warnings
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ProductCaptionDataset(Dataset):
    """Dataset class for product images and captions"""
    
    def __init__(self, metadata_csv, images_dir, processor, max_length=128):
        self.metadata = pd.read_csv(metadata_csv)
        self.images_dir = images_dir
        self.processor = processor
        self.max_length = max_length
        
        # Remove any rows with missing captions (product_title)
        self.metadata = self.metadata.dropna(subset=['product_title'])
        
        # Filter out rows where image files don't exist
        before_count = len(self.metadata)
        existing_files = []
        
        for idx, row in self.metadata.iterrows():
            image_path = os.path.join(self.images_dir, row['filename'])
            if os.path.exists(image_path):
                existing_files.append(idx)
        
        self.metadata = self.metadata.loc[existing_files].reset_index(drop=True)
        after_count = len(self.metadata)
        
        if before_count != after_count:
            print(f"Filtered dataset: {before_count} -> {after_count} samples (removed {before_count - after_count} missing files)")
        else:
            print(f"Dataset initialized with {after_count} samples")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load image
        image_path = os.path.join(self.images_dir, row['filename'])
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            # Return a black image if file is corrupted
            print(f"Warning: Could not load image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        # Get caption (product_title)
        caption = str(row['product_title'])
        
        # Process image and text
        inputs = self.processor(
            images=image,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(),
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': inputs['input_ids'].squeeze(),
            'product_id': row['product_id'],
            'original_caption': caption
        }

class CaptionTrainer:
    """Main class for training and evaluating caption generation models"""
    
    def __init__(self, dataset_dir='product_dataset', model_save_dir='caption_models', model_name='Salesforce/blip-image-captioning-base'):
        self.dataset_dir = dataset_dir
        self.model_save_dir = model_save_dir
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model save directory
        os.makedirs(model_save_dir, exist_ok=True)
        
        print(f"Using device: {self.device}")
        print(f"Model: {model_name}")
        
        # Initialize model and processor
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        
        # ROUGE scorer for evaluation
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def create_data_loaders(self, batch_size=8, num_workers=4, max_length=128):
        """Create data loaders for training, validation, and testing"""
        
        # Create datasets
        train_dataset = ProductCaptionDataset(
            os.path.join(self.dataset_dir, 'train_metadata.csv'),
            os.path.join(self.dataset_dir, 'images'),
            self.processor,
            max_length=max_length
        )
        
        val_dataset = ProductCaptionDataset(
            os.path.join(self.dataset_dir, 'val_metadata.csv'),
            os.path.join(self.dataset_dir, 'images'),
            self.processor,
            max_length=max_length
        )
        
        test_dataset = ProductCaptionDataset(
            os.path.join(self.dataset_dir, 'test_metadata.csv'),
            os.path.join(self.dataset_dir, 'images'),
            self.processor,
            max_length=max_length
        )
        
        # Create data loaders with custom collate function
        def collate_fn(batch):
            return {
                'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
                'input_ids': torch.stack([item['input_ids'] for item in batch]),
                'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
                'labels': torch.stack([item['labels'] for item in batch]),
                'product_ids': [item['product_id'] for item in batch],
                'original_captions': [item['original_caption'] for item in batch]
            }
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, pin_memory=True, collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=True, collate_fn=collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=True, collate_fn=collate_fn
        )
        
        return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
    
    def calculate_metrics(self, predictions, targets):
        """Calculate BLEU and ROUGE scores"""
        bleu_scores = []
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        # BLEU smoothing function
        smoothing = SmoothingFunction().method4
        
        for pred, target in zip(predictions, targets):
            # Tokenize for BLEU
            pred_tokens = pred.lower().split()
            target_tokens = target.lower().split()
            
            # Calculate BLEU score
            bleu = sentence_bleu([target_tokens], pred_tokens, smoothing_function=smoothing)
            bleu_scores.append(bleu)
            
            # Calculate ROUGE scores
            rouge = self.rouge_scorer.score(target.lower(), pred.lower())
            for key in rouge_scores:
                rouge_scores[key].append(rouge[key].fmeasure)
        
        return {
            'bleu': np.mean(bleu_scores),
            'rouge1': np.mean(rouge_scores['rouge1']),
            'rouge2': np.mean(rouge_scores['rouge2']),
            'rougeL': np.mean(rouge_scores['rougeL'])
        }
    
    def train_model(self, model, train_loader, val_loader, num_epochs=20, 
                   learning_rate=5e-5, weight_decay=1e-4, patience=5):
        """Train the caption generation model"""
        
        # Optimizer
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience//2, verbose=True
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_bleu': [],
            'val_rouge1': [],
            'val_rouge2': [],
            'val_rougeL': [],
            'learning_rate': []
        }
        
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        model.to(self.device)
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
            for batch_idx, batch in enumerate(train_pbar):
                # Move batch to device
                pixel_values = batch['pixel_values'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
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
                for batch_idx, batch in enumerate(val_pbar):
                    pixel_values = batch['pixel_values'].to(self.device)
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Calculate loss
                    outputs = model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    val_loss += outputs.loss.item()
                    
                    # Generate captions for metrics
                    generated_ids = model.generate(
                        pixel_values=pixel_values,
                        max_length=50,
                        num_beams=3,
                        early_stopping=True
                    )
                    
                    generated_captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                    original_captions = batch['original_captions']
                    
                    val_predictions.extend(generated_captions)
                    val_targets.extend(original_captions)
                    
                    val_pbar.set_postfix({
                        'loss': f'{outputs.loss.item():.4f}',
                        'avg_loss': f'{val_loss/(batch_idx+1):.4f}'
                    })
            
            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_metrics = self.calculate_metrics(val_predictions, val_targets)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_bleu'].append(val_metrics['bleu'])
            history['val_rouge1'].append(val_metrics['rouge1'])
            history['val_rouge2'].append(val_metrics['rouge2'])
            history['val_rougeL'].append(val_metrics['rougeL'])
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # Learning rate scheduler
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                # Save best model
                model.save_pretrained(os.path.join(self.model_save_dir, 'best_model'))
                self.processor.save_pretrained(os.path.join(self.model_save_dir, 'best_model'))
                
                # Save training state
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'history': history
                }, os.path.join(self.model_save_dir, 'training_state.pth'))
            else:
                epochs_without_improvement += 1
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  Val BLEU: {val_metrics["bleu"]:.4f}')
            print(f'  Val ROUGE-1: {val_metrics["rouge1"]:.4f}')
            print(f'  Val ROUGE-L: {val_metrics["rougeL"]:.4f}')
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
            for batch in tqdm(test_loader, desc='Evaluating'):
                pixel_values = batch['pixel_values'].to(self.device)
                
                # Generate captions
                generated_ids = model.generate(
                    pixel_values=pixel_values,
                    max_length=50,
                    num_beams=5,
                    early_stopping=True,
                    do_sample=True,
                    temperature=0.7
                )
                
                generated_captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                original_captions = batch['original_captions']
                
                predictions.extend(generated_captions)
                targets.extend(original_captions)
                product_ids.extend(batch['product_ids'])
        
        # Calculate metrics
        metrics = self.calculate_metrics(predictions, targets)
        
        print("Test Set Evaluation Results:")
        print(f"BLEU Score: {metrics['bleu']:.4f}")
        print(f"ROUGE-1: {metrics['rouge1']:.4f}")
        print(f"ROUGE-2: {metrics['rouge2']:.4f}")
        print(f"ROUGE-L: {metrics['rougeL']:.4f}")
        
        return metrics, predictions, targets, product_ids
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(history['train_loss'], label='Train Loss')
        axes[0, 0].plot(history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # BLEU score plot
        axes[0, 1].plot(history['val_bleu'], label='BLEU', color='green')
        axes[0, 1].set_title('Validation BLEU Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('BLEU Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # ROUGE scores plot
        axes[1, 0].plot(history['val_rouge1'], label='ROUGE-1')
        axes[1, 0].plot(history['val_rouge2'], label='ROUGE-2')
        axes[1, 0].plot(history['val_rougeL'], label='ROUGE-L')
        axes[1, 0].set_title('Validation ROUGE Scores')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('ROUGE Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate plot
        axes[1, 1].plot(history['learning_rate'])
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_save_dir, 'training_history.png'), dpi=300)
        plt.show()
    
    def show_sample_predictions(self, predictions, targets, product_ids, num_samples=10):
        """Show sample predictions vs actual captions"""
        print("=" * 80)
        print("SAMPLE PREDICTIONS")
        print("=" * 80)
        
        indices = np.random.choice(len(predictions), min(num_samples, len(predictions)), replace=False)
        
        for i, idx in enumerate(indices):
            print(f"\nSample {i+1} (Product ID: {product_ids[idx]}):")
            print(f"Actual:    {targets[idx]}")
            print(f"Generated: {predictions[idx]}")
            print("-" * 80)
    
    def train_and_evaluate(self, batch_size=8, num_epochs=20, learning_rate=5e-5, max_length=128):
        """Complete training and evaluation pipeline"""
        print("🚀 Starting Caption Generation Training Pipeline")
        print("=" * 60)
        print(f"📦 Model: {self.model_name}")
        print(f"🔧 Config: {batch_size} batch, {num_epochs} epochs, LR {learning_rate}")
        print(f"📝 Max Length: {max_length}")
        print()
        
        # Create data loaders
        print("📁 Creating data loaders...")
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = \
            self.create_data_loaders(batch_size, max_length=max_length)
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        # Train model
        print("🚀 Starting training...")
        history = self.train_model(self.model, train_loader, val_loader, 
                                 num_epochs=num_epochs, learning_rate=learning_rate)
        
        # Load best model for evaluation
        best_model = BlipForConditionalGeneration.from_pretrained(
            os.path.join(self.model_save_dir, 'best_model')
        )
        
        # Evaluate on test set
        print("📊 Evaluating on test set...")
        metrics, predictions, targets, product_ids = self.evaluate_model(best_model, test_loader)
        
        # Save results
        results = {
            'model_config': {
                'model_name': self.model_name,
                'batch_size': batch_size,
                'num_epochs': num_epochs,
                'learning_rate': learning_rate,
                'max_length': max_length
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
        
        # Show results
        self.plot_training_history(history)
        self.show_sample_predictions(predictions, targets, product_ids)
        
        print("=" * 60)
        print("✅ Training completed successfully!")
        print(f"📁 Results saved to {self.model_save_dir}")
        
        return best_model, results

# ============================================================================
# CENTRALIZED CONFIGURATION
# ============================================================================

# Model Configuration
CAPTION_MODEL_CONFIG = {
    'model_name': 'Salesforce/blip-image-captioning-base',  # Ana model
    'batch_size': 8,                # Caption generation için daha küçük batch
    'num_epochs': 20,               # Caption için yeterli
    'learning_rate': 5e-5,          # Transformer için uygun LR
    'max_length': 128,              # Caption uzunluğu
}

# Alternative models (kullanmak istersen)
ALTERNATIVE_MODELS = {
    'blip_large': 'Salesforce/blip-image-captioning-large',
    'blip2': 'Salesforce/blip2-opt-2.7b',
    'git_base': 'microsoft/git-base',
    'vit_gpt2': 'nlpconnect/vit-gpt2-image-captioning'
}

# Dataset Configuration  
DATASET_CONFIG = {
    'dataset_dir': 'product_dataset',
    'model_save_dir': 'caption_models'
}

# Usage example
if __name__ == "__main__":
    print("🚀 Amazon Product Caption Generation Training")
    print("=" * 50)
    print(f"📋 Configuration:")
    for key, value in CAPTION_MODEL_CONFIG.items():
        print(f"   {key}: {value}")
    print()
    
    print("🎯 Available Alternative Models:")
    for name, model_path in ALTERNATIVE_MODELS.items():
        print(f"   {name}: {model_path}")
    print()
    
    # Initialize trainer
    trainer = CaptionTrainer(
        dataset_dir=DATASET_CONFIG['dataset_dir'], 
        model_save_dir=DATASET_CONFIG['model_save_dir'],
        model_name=CAPTION_MODEL_CONFIG['model_name']
    )
    
    # Train and evaluate model
    model, results = trainer.train_and_evaluate(
        batch_size=CAPTION_MODEL_CONFIG['batch_size'],
        num_epochs=CAPTION_MODEL_CONFIG['num_epochs'],
        learning_rate=CAPTION_MODEL_CONFIG['learning_rate'],
        max_length=CAPTION_MODEL_CONFIG['max_length']
    )
    
    print(f"✅ Caption training completed! Results saved to {DATASET_CONFIG['model_save_dir']}")