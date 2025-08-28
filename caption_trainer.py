"""
Amazon Product Caption Generation with ViT-Base + LSTM
Optimized for ~44M parameters
"""

import os
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import ViTModel, ViTImageProcessor
import warnings
warnings.filterwarnings('ignore')

# We use simple custom metrics instead of external dependencies
# This avoids import errors and keeps the code self-contained

class SimpleViTCaptionModel(nn.Module):
    """ViT-Base + LSTM Caption Model (~44M parameters)"""
    
    def __init__(self, vocab_size, embed_dim=512, hidden_dim=256, num_layers=1, dropout=0.2):
        super().__init__()
        
        # Vision encoder - ViT-Small (completely frozen)
        self.vision_encoder = ViTModel.from_pretrained('WinKawaks/vit-small-patch16-224')
        
        # Freeze ALL parameters of ViT
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        # Text generation components
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Vision to text projection (reduce dimension)
        self.vision_proj = nn.Linear(384, hidden_dim)  # ViT-Small output is 384
        
        # Smaller LSTM decoder
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, pixel_values, input_ids=None):
        batch_size = pixel_values.size(0)
        
        # Encode image
        vision_outputs = self.vision_encoder(pixel_values)
        image_features = vision_outputs.last_hidden_state[:, 0]  # CLS token
        
        # Project to hidden space
        hidden_state = self.vision_proj(image_features).unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        cell_state = torch.zeros_like(hidden_state)
        
        if input_ids is not None:
            # Training mode with teacher forcing
            seq_len = input_ids.size(1)
            embedded = self.embedding(input_ids)
            
            # LSTM forward pass
            lstm_out, _ = self.lstm(embedded, (hidden_state, cell_state))
            
            # Output projection
            outputs = self.output_proj(self.dropout(lstm_out))
            return outputs
        else:
            # Inference mode - generate sequence
            return self.generate(hidden_state, cell_state, max_length=50)
    
    def generate(self, hidden_state, cell_state, max_length=50, temperature=0.7):
        """Generate caption sequence"""
        batch_size = hidden_state.size(1)
        
        # Start with <START> token (assuming index 1)
        input_token = torch.ones(batch_size, 1, dtype=torch.long, device=hidden_state.device)
        
        outputs = []
        
        for _ in range(max_length):
            # Embed current token
            embedded = self.embedding(input_token)
            
            # LSTM step
            lstm_out, (hidden_state, cell_state) = self.lstm(embedded, (hidden_state, cell_state))
            
            # Get next token probabilities
            logits = self.output_proj(lstm_out[:, -1, :]) / temperature
            probs = torch.softmax(logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, 1)
            outputs.append(next_token)
            
            # Check for <END> token (assuming index 2)
            if (next_token == 2).all():
                break
                
            input_token = next_token
        
        return torch.cat(outputs, dim=1)

class ProductCaptionDataset(Dataset):
    """Dataset for ViT + LSTM training"""
    
    def __init__(self, csv_path, images_dir, image_processor, caption_trainer, max_length=64):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.image_processor = image_processor
        self.caption_trainer = caption_trainer
        self.max_length = max_length
        
        # Filter valid samples
        valid_samples = []
        for _, row in self.df.iterrows():
            # Use filename from CSV instead of constructing from product_id
            image_path = os.path.join(images_dir, row['filename']) 
            if os.path.exists(image_path) and pd.notna(row['product_title']):
                valid_samples.append(row)
        
        self.df = pd.DataFrame(valid_samples).reset_index(drop=True)
        print(f"Dataset size: {len(self.df)} samples")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load and process image
        image_path = os.path.join(self.images_dir, row['filename'])
        image = Image.open(image_path).convert('RGB')
        pixel_values = self.image_processor(image, return_tensors="pt")['pixel_values'].squeeze(0)
        
        # Process caption
        caption = str(row['product_title'])
        input_ids = self.caption_trainer.text_to_indices(caption, self.max_length)
        
        # Create labels (shifted by 1 for teacher forcing)
        labels = input_ids[1:] + [self.caption_trainer.vocab.get('<PAD>', 0)]
        labels = labels[:self.max_length-1]  # Ensure correct length
        
        return {
            'pixel_values': pixel_values,
            'input_ids': torch.tensor(input_ids[:-1], dtype=torch.long),  # Remove last token
            'labels': torch.tensor(labels, dtype=torch.long),
            'product_id': row['product_id'],
            'original_caption': caption
        }

class CaptionTrainer:
    """ViT-Base + LSTM Caption Trainer (~44M params)"""
    
    def __init__(self, dataset_dir='product_dataset', model_save_dir='vit_caption_models'):
        self.dataset_dir = dataset_dir
        self.model_save_dir = model_save_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        os.makedirs(model_save_dir, exist_ok=True)
        
        print(f"🔧 ViT-Base + LSTM Caption Training")
        print(f"Device: {self.device}")
        
        # Build vocabulary from dataset first
        self.vocab = self.build_vocabulary()
        vocab_size = len(self.vocab)
        
        # Initialize image processor
        self.image_processor = ViTImageProcessor.from_pretrained('WinKawaks/vit-small-patch16-224')
        
        # Create reverse vocab for decoding
        self.idx_to_word = {idx: word for word, idx in self.vocab.items()}
        
        print(f"Vocabulary size: {vocab_size}")
        
        # We use simple custom metrics instead of external dependencies
    
    def build_vocabulary(self):
        """Build vocabulary from all captions"""
        print("📚 Building vocabulary from dataset...")
        
        vocab = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
        word_counts = {}
        
        # Read all metadata files
        for split in ['train', 'val', 'test']:
            csv_path = os.path.join(self.dataset_dir, f'{split}_metadata.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                for caption in df['product_title'].fillna(''):
                    words = str(caption).lower().split()
                    for word in words:
                        # Simple cleaning
                        word = ''.join(c for c in word if c.isalnum() or c in ['-', "'"])
                        if len(word) > 0:
                            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Keep only words with count >= 2 and limit vocab size
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        max_vocab = 8000  # Smaller vocab for efficiency
        
        for word, count in sorted_words[:max_vocab-4]:  # -4 for special tokens
            if count >= 2:
                vocab[word] = len(vocab)
        
        print(f"Vocabulary size: {len(vocab)}")
        return vocab
    
    def text_to_indices(self, text, max_length=64):
        """Convert text to indices using vocabulary"""
        words = str(text).lower().split()
        indices = [self.vocab['<START>']]
        
        for word in words:
            word = ''.join(c for c in word if c.isalnum() or c in ['-', "'"])
            if word:
                indices.append(self.vocab.get(word, self.vocab['<UNK>']))
        
        indices.append(self.vocab['<END>'])
        
        # Pad or truncate
        if len(indices) < max_length:
            indices.extend([self.vocab['<PAD>']] * (max_length - len(indices)))
        else:
            indices = indices[:max_length-1] + [self.vocab['<END>']]
        
        return indices
    
    def indices_to_text(self, indices):
        """Convert indices back to text"""
        words = []
        for idx in indices:
            word = self.idx_to_word.get(idx, '<UNK>')
            if word == '<END>':
                break
            if word not in ['<START>', '<PAD>', '<UNK>']:
                words.append(word)
        return ' '.join(words)
    
    def collate_fn(self, batch):
        """Custom collate function for padding"""
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        input_ids = torch.stack([item['input_ids'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'labels': labels,
            'product_ids': [item['product_id'] for item in batch],
            'original_captions': [item['original_caption'] for item in batch]
        }
    
    def create_data_loaders(self, batch_size=16, max_length=64):
        """Create data loaders for ViT + LSTM training"""
        
        datasets = {}
        loaders = {}
        
        for split in ['train', 'val', 'test']:
            csv_path = os.path.join(self.dataset_dir, f'{split}_metadata.csv')
            if os.path.exists(csv_path):
                dataset = ProductCaptionDataset(
                    csv_path,
                    os.path.join(self.dataset_dir, 'images'),
                    self.image_processor,
                    self,  # Pass trainer for vocab access
                    max_length=max_length
                )
                datasets[split] = dataset
                
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=(split == 'train'),
                    num_workers=0,  # Windows compatibility
                    collate_fn=self.collate_fn
                )
                loaders[split] = loader
        
        return loaders['train'], loaders['val'], loaders['test']
    
    def train_model(self, num_epochs=5, learning_rate=1e-4, batch_size=16):
        """Train the ViT + LSTM caption model"""
        
        train_loader, val_loader, test_loader = self.create_data_loaders(batch_size=batch_size)
        
        model = SimpleViTCaptionModel(
            vocab_size=len(self.vocab),
            embed_dim=512,  # Reduced embedding size
            hidden_dim=256,  # Reduced hidden size
            num_layers=1,    # Single layer LSTM
            dropout=0.2
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
        
        # Quick model test with sample batch
        print("🧪 Testing model with sample batch...")
        sample_batch = next(iter(train_loader))
        with torch.no_grad():
            test_output = model(sample_batch['pixel_values'][:2].to(self.device), 
                              sample_batch['input_ids'][:2].to(self.device))
            print(f"  Output shape: {test_output.shape}")
            
            # Test generation
            gen_output = model(sample_batch['pixel_values'][:1].to(self.device))
            sample_caption = self.indices_to_text(gen_output[0].cpu().numpy())
            print(f"  Sample generation: {sample_caption[:50]}...")
        print("✅ Model test passed!")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.vocab.get('<PAD>', 0))
        
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
            for batch_idx, batch in enumerate(progress_bar):
                pixel_values = batch['pixel_values'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass with teacher forcing
                outputs = model(pixel_values, input_ids)
                
                # Reshape for loss calculation
                outputs = outputs.reshape(-1, len(self.vocab))
                labels = labels.reshape(-1)
                
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                
                # Update progress bar with additional info
                if batch_idx % 100 == 0:  # Every 100 batches
                    current_avg_loss = train_loss / (batch_idx + 1)
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{current_avg_loss:.4f}',
                        'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                    })
                else:
                    progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation", leave=False):
                    pixel_values = batch['pixel_values'].to(self.device)
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    original_captions = batch['original_captions']
                    
                    # Calculate loss
                    outputs = model(pixel_values, input_ids)
                    outputs_flat = outputs.reshape(-1, len(self.vocab))
                    labels_flat = labels.reshape(-1)
                    
                    loss = criterion(outputs_flat, labels_flat)
                    val_loss += loss.item()
                    
                    # Generate predictions for metrics (every 5 batches to save time)
                    if len(val_predictions) < 100:  # Limit to 100 samples for speed
                        generated_indices = model(pixel_values)  # Generate without teacher forcing
                        
                        for i in range(min(generated_indices.size(0), len(original_captions))):
                            pred_caption = self.indices_to_text(generated_indices[i].cpu().numpy())
                            val_predictions.append(pred_caption)
                            val_targets.append(original_captions[i])
            
            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)
            
            # Calculate validation metrics
            if val_predictions:
                val_metrics = self.calculate_simple_metrics(val_predictions, val_targets)
                print(f"Epoch {epoch+1}/{num_epochs}:")
                print(f"  Train Loss: {avg_train_loss:.4f}")
                print(f"  Val Loss: {avg_val_loss:.4f}")
                print(f"  Word Overlap: {val_metrics['word_overlap']:.3f}")
                print(f"  Jaccard Sim: {val_metrics['jaccard_similarity']:.3f}")
                print(f"  Length Sim: {val_metrics['length_similarity']:.3f}")
                
                # Show sample predictions every epoch
                if len(val_predictions) >= 3:
                    print("  Sample Predictions:")
                    for i in range(min(3, len(val_predictions))):
                        print(f"    Original:  {val_targets[i][:60]}...")
                        print(f"    Generated: {val_predictions[i][:60]}...")
                        print()
            else:
                print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'vocab': self.vocab,
                    'idx_to_word': self.idx_to_word,
                    'epoch': epoch,
                    'val_loss': avg_val_loss
                }, os.path.join(self.model_save_dir, 'best_vit_caption_model.pth'))
                print("✅ Saved new best model")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
        
        return model
    
    def evaluate_model(self, test_loader=None):
        """Evaluate model performance"""
        if test_loader is None:
            _, _, test_loader = self.create_data_loaders()
        
        # Load best model
        checkpoint = torch.load(os.path.join(self.model_save_dir, 'best_vit_caption_model.pth'), 
                               map_location=self.device)
        
        model = SimpleViTCaptionModel(
            vocab_size=len(self.vocab),
            embed_dim=512,
            hidden_dim=256,
            num_layers=1,
            dropout=0.2
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        predictions = []
        targets = []
        
        print("🔍 Evaluating model...")
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Generating captions'):
                pixel_values = batch['pixel_values'].to(self.device)
                original_captions = batch['original_captions']
                
                # Generate captions
                generated_indices = model(pixel_values)
                
                for i in range(generated_indices.size(0)):
                    pred_caption = self.indices_to_text(generated_indices[i].cpu().numpy())
                    predictions.append(pred_caption)
                    targets.append(original_captions[i])
        
        # Calculate metrics
        metrics = self.calculate_simple_metrics(predictions, targets)
        
        print("📊 ViT + LSTM Test Results:")
        print(f"Exact Match: {metrics['exact_match']:.4f}")
        print(f"Word Overlap: {metrics['word_overlap']:.4f}")
        print(f"Jaccard Similarity: {metrics['jaccard_similarity']:.4f}")
        print(f"Length Similarity: {metrics['length_similarity']:.4f}")
        print(f"Simple BLEU: {metrics['simple_bleu']:.4f}")
        
        return metrics, predictions, targets
    
    def calculate_simple_metrics(self, predictions, targets):
        """Calculate simple evaluation metrics without external dependencies"""
        if len(predictions) != len(targets):
            return {'error': 'Prediction and target lengths do not match'}
        
        metrics = {
            'exact_match': 0.0,
            'word_overlap': 0.0,
            'length_similarity': 0.0,
            'jaccard_similarity': 0.0,
            'simple_bleu': 0.0
        }
        
        total_samples = len(predictions)
        if total_samples == 0:
            return metrics
        
        for pred, target in zip(predictions, targets):
            pred_lower = pred.lower().strip()
            target_lower = target.lower().strip()
            
            # Exact match
            if pred_lower == target_lower:
                metrics['exact_match'] += 1
            
            # Word-level metrics
            pred_words = set(pred_lower.split())
            target_words = set(target_lower.split())
            
            # Word overlap (precision-like)
            if len(pred_words) > 0:
                overlap = len(pred_words.intersection(target_words))
                metrics['word_overlap'] += overlap / len(pred_words)
            
            # Length similarity
            if len(target_lower) > 0:
                length_ratio = min(len(pred_lower), len(target_lower)) / max(len(pred_lower), len(target_lower))
                metrics['length_similarity'] += length_ratio
            
            # Jaccard similarity
            union = pred_words.union(target_words)
            if len(union) > 0:
                jaccard = len(pred_words.intersection(target_words)) / len(union)
                metrics['jaccard_similarity'] += jaccard
            
            # Simple BLEU-like score
            if len(target_words) > 0:
                matches = len(pred_words.intersection(target_words))
                simple_bleu = matches / len(target_words)
                metrics['simple_bleu'] += simple_bleu
        
        # Average all metrics
        for key in metrics:
            metrics[key] /= total_samples
        
        return metrics
    
    def generate_sample_captions(self, num_samples=5):
        """Generate and display sample captions"""
        _, _, test_loader = self.create_data_loaders(batch_size=1)
        
        # Load best model
        checkpoint = torch.load(os.path.join(self.model_save_dir, 'best_vit_caption_model.pth'), 
                               map_location=self.device)
        
        model = SimpleViTCaptionModel(
            vocab_size=len(self.vocab),
            embed_dim=512,
            hidden_dim=256,
            num_layers=1,
            dropout=0.2
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"\n📝 Sample Caption Generation:")
        print("="*60)
        
        samples_shown = 0
        with torch.no_grad():
            for batch in test_loader:
                if samples_shown >= num_samples:
                    break
                
                pixel_values = batch['pixel_values'].to(self.device)
                original_caption = batch['original_captions'][0]
                
                generated_indices = model(pixel_values)
                generated_caption = self.indices_to_text(generated_indices[0].cpu().numpy())
                
                print(f"\nSample {samples_shown + 1}:")
                print(f"Original:  {original_caption}")
                print(f"Generated: {generated_caption}")
                
                samples_shown += 1
    
    def train_and_evaluate(self, **kwargs):
        """Complete training and evaluation pipeline"""
        print("🚀 Starting ViT-Base + LSTM training...")
        
        # Train model
        model = self.train_model(**kwargs)
        
        # Evaluate model
        metrics, predictions, targets = self.evaluate_model()
        
        # Show sample captions
        self.generate_sample_captions()
        
        return {
            'model': model,
            'metrics': metrics,
            'predictions': predictions,
            'targets': targets
        }

# Configuration
VIT_CAPTION_CONFIG = {
    'model_type': 'ViT-Small + LSTM',
    'vision_model': 'WinKawaks/vit-small-patch16-224',
    'batch_size': 15,
    'num_epochs': 50,
    'learning_rate': 5e-5,
    'max_length': 32,

}

if __name__ == "__main__":
    print("🚀 Amazon Product Caption Generation - ViT-Base + LSTM")
    print("=" * 70)
    print(f"📋 Configuration:")
    for key, value in VIT_CAPTION_CONFIG.items():
        print(f"   {key}: {value}")
    print()
    
    # Initialize trainer
    trainer = CaptionTrainer()
    
    # Train and evaluate model
    results = trainer.train_and_evaluate(
        batch_size=VIT_CAPTION_CONFIG['batch_size'],
        num_epochs=VIT_CAPTION_CONFIG['num_epochs'],
        learning_rate=VIT_CAPTION_CONFIG['learning_rate']
    )
    
    print(f"✅ ViT + LSTM caption training completed!")
