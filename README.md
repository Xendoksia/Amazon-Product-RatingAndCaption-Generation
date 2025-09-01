# Amazon Product Rating & Caption Generation

This project contains deep learning models for rating prediction and caption generation from Amazon product images.



<img width="1387" height="922" alt="Ekran görüntüsü 2025-08-31 142619" src="https://github.com/user-attachments/assets/68770eba-e12b-4042-b06b-6a28e3363e52" />






<img width="1153" height="877" alt="Ekran görüntüsü 2025-08-31 142517" src="https://github.com/user-attachments/assets/518c5a18-22f6-4c08-a867-480483866371" />







## Project Overview

- **Rating Prediction**: Predicts 1-5 star ratings from product images
- **Caption Generation**: Generates descriptive text from product images
- **Dual Architecture**: Two specialized model architectures
- **Large Dataset**: 63K+ Amazon Fashion product images

## Project Structure

```
amazon/
├── data_analysis.py           # Data analysis and preprocessing
├── dataset_downloader.py      # Product image downloader
├── model_trainer.py          # Rating prediction model
├── caption_trainer.py        # Caption generation model
├── model_testing_ui.py       # Testing GUI interface
├── product_dataset/          # Main data directory
│   ├── images/              # Product images (63K+)
│   ├── train_metadata.csv   # Training data
│   ├── val_metadata.csv     # Validation data
│   └── test_metadata.csv    # Test data
└── models/                   # Trained models
    ├── best_rating_model.pth
    └── best_vit_caption_model.pth
```

## Model Architectures

### Rating Prediction Model (ResNet50)

- **Backbone**: ResNet50 (pretrained)
- **Parameters**: ~23M
- **Output**: Single rating value (1-5)
- **Loss**: Huber Loss (robust regression)
- **Performance**: MAE ~0.65-0.70

```python
MODEL_CONFIG = {
    'backbone': 'resnet50',      # resnet18/50/101, efficientnet
    'batch_size': 16,
    'num_epochs': 50,
    'learning_rate': 0.0001,
    'dropout': 0.3,
    'augment': True
}
```

### Caption Generation Model (ViT-Small + LSTM)

- **Vision Encoder**: ViT-Small (frozen, 21.8M params)
- **Text Decoder**: LSTM (trainable, 7M params)
- **Total Parameters**: ~29M
- **Vocabulary**: 8,000 most common words
- **Max Caption Length**: 32 tokens

```python
VIT_CAPTION_CONFIG = {
    'batch_size': 16,
    'num_epochs': 50,
    'learning_rate': 5e-5,
    'max_length': 32,
    'vocab_size': 8000
}
```

## Performance Metrics

### Rating Prediction

- **Mean Absolute Error (MAE)**: 0.65-0.70
- **Root Mean Square Error (RMSE)**: 0.85-0.95
- **R² Score**: 0.15-0.25

### Caption Generation

- **Word Overlap**: 25-35%
- **Jaccard Similarity**: 15-20%
- **BLEU Score**: 20-25%
- **Length Similarity**: 60-70%

## Features

### Data Pipeline

- **Automatic Data Filtering**: Automatically removes missing files
- **Train/Val/Test Split**: 70/20/10 split ratio
- **Data Augmentation**: Visual augmentation techniques
- **Batch Processing**: Memory-efficient batch loading

### Model Features

- **Transfer Learning**: Pretrained models (ResNet, ViT)
- **Mixed Precision**: CUDA optimization
- **Early Stopping**: Overfitting prevention
- **Learning Rate Scheduling**: Adaptive learning rate
- **Gradient Clipping**: Training stability

### GUI Test Interface

- **Interactive UI**: Tkinter-based test interface
- **Real-time Predictions**: Instant model testing
- **Visual Results**: Prediction visualization
- **Model Comparison**: Compare different models

## Model Testing UI

Test models easily with GUI interface:

1. **Rating Model Test**: Upload image and get rating prediction
2. **Caption Model Test**: Upload image and generate caption
3. **Batch Processing**: Test multiple images
4. **Results Visualization**: Visualize results with graphs

### Dataset Structure

```csv
product_id,filename,rating,caption
B00001,product_1_img_00.jpg,4.5,"Men's casual shirt..."
B00002,product_2_img_00.jpg,3.2,"Women's summer dress..."
```

## Results Visualization

Training progress and model performance visualizations:

- Loss curves (training vs validation)
- Metric tracking over epochs
- Sample predictions display
- Confusion matrices for rating model
