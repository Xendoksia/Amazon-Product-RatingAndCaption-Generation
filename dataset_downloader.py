import json
import requests
import os
from PIL import Image
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
import pandas as pd
from tqdm import tqdm

class ProductImageDownloader:
    """Download and organize product images from JSONL dataset
    
    Recommended to use shrunk dataset created by data_analysis.py
    instead of the full dataset (~3.7M images) for faster processing.
    """
    
    def __init__(self, jsonl_path, base_dir='dataset'):
        self.jsonl_path = jsonl_path
        self.base_dir = base_dir
        self.images_dir = os.path.join(base_dir, 'images')
        self.metadata_file = os.path.join(base_dir, 'image_metadata.csv')
        self.failed_downloads = []
        
        # Create directories
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(base_dir, exist_ok=True)
        
    def load_data(self):
        """Load and filter valid data from JSONL"""
        data = []
        print("Loading JSONL data...")
        
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    # Only include items with images and ratings
                    if (item.get('images') and 
                        item.get('average_rating') is not None and
                        len(item['images']) > 0):
                        data.append(item)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
        
        print(f"Found {len(data)} valid products with images and ratings")
        return data
    
    def create_image_metadata(self, data):
        """Create metadata CSV for all images with duplicate removal"""
        metadata = []
        seen_urls = set()  # Track duplicate image URLs across all products
        seen_titles = set()  # Track duplicate product titles
        removed_products = 0
        removed_images = 0
        sequential_product_id = 0  # Use sequential IDs for non-skipped products
        
        for original_idx, product in enumerate(data):
            product_title = product.get('title', '').strip().lower()
            
            # Skip if we've seen this exact title before
            if product_title and product_title in seen_titles:
                removed_products += 1
                if removed_products <= 5:  # Only show first few
                    print(f"Skipping duplicate product title: {product.get('title', '')[:50]}...")
                continue
            
            # Add title to seen set if it's not empty
            if product_title:
                seen_titles.add(product_title)
            
            # Process images for this product
            product_urls = set()  # Track URLs within this product
            img_idx = 0
            product_has_images = False
            
            for image_info in product['images']:
                # Get the best available URL
                url = image_info.get('large', image_info.get('thumb', ''))
                
                # Skip if URL is empty or already seen (globally or within this product)
                if not url or url in seen_urls or url in product_urls:
                    if url and url in seen_urls:
                        removed_images += 1
                    continue
                
                # Add URL to both global and product-specific sets
                seen_urls.add(url)
                product_urls.add(url)
                
                file_ext = self.get_file_extension(url)
                filename = f"product_{sequential_product_id:05d}_img_{img_idx:02d}{file_ext}"
                
                metadata.append({
                    'product_id': sequential_product_id,  # Use sequential ID
                    'image_id': img_idx,
                    'filename': filename,
                    'original_url': url,
                    'variant': image_info.get('variant', 'UNKNOWN'),
                    'is_main': image_info.get('variant') == 'MAIN',
                    'product_title': product.get('title', ''),
                    'category': product.get('main_category', ''),
                    'rating': product.get('average_rating'),
                    'rating_count': product.get('rating_number', 0),
                    'store': product.get('store', ''),
                    'parent_asin': product.get('parent_asin', '')
                })
                img_idx += 1
                product_has_images = True
            
            # Only increment sequential ID if product has valid images
            if product_has_images:
                sequential_product_id += 1
        
        print(f"Deduplication results:")
        print(f"  - Removed {removed_products} duplicate products by title")
        print(f"  - Removed {removed_images} duplicate images by URL")
        print(f"  - Final dataset: {len(metadata)} unique images from {sequential_product_id} products")
        
        df = pd.DataFrame(metadata)
        
        # Final safety check - Remove any remaining duplicates based on URL
        initial_count = len(df)
        df = df.drop_duplicates(subset=['original_url'], keep='first')
        removed_count = initial_count - len(df)
        
        if removed_count > 0:
            print(f"  - Removed {removed_count} additional duplicate URLs in final cleanup")
        
        # Verify no duplicate titles remain
        title_duplicates = df.duplicated(subset=['product_title']).sum()
        if title_duplicates > 0:
            print(f"  - Warning: {title_duplicates} duplicate titles still present!")
            # Remove duplicate titles, keeping first occurrence
            df = df.drop_duplicates(subset=['product_title'], keep='first')
            print(f"  - Removed duplicate titles, final count: {len(df)} images")
        
        df.to_csv(self.metadata_file, index=False)
        print(f"Image metadata saved to {self.metadata_file}")
        print(f"Final verification: {len(df)} images from {df['product_id'].nunique()} unique products")
        return df
    
    def get_file_extension(self, url):
        """Extract file extension from URL"""
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        if '.jpg' in path or '.jpeg' in path:
            return '.jpg'
        elif '.png' in path:
            return '.png'
        elif '.webp' in path:
            return '.webp'
        else:
            return '.jpg'  # Default to jpg
    
    def download_single_image(self, url, filepath, timeout=30, max_retries=3):
        """Download a single image with retry logic"""
        for attempt in range(max_retries):
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                response = requests.get(url, headers=headers, timeout=timeout, stream=True)
                response.raise_for_status()
                
                # Verify it's an image
                content_type = response.headers.get('content-type', '')
                if not content_type.startswith('image/'):
                    raise ValueError(f"Invalid content type: {content_type}")
                
                # Save image
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Verify image can be opened
                try:
                    with Image.open(filepath) as img:
                        img.verify()
                except Exception:
                    os.remove(filepath)
                    raise ValueError("Downloaded file is not a valid image")
                
                return True
                
            except Exception as e:
                if attempt == max_retries - 1:
                    return False, str(e)
                time.sleep(1)  # Wait before retry
        
        return False, "Max retries exceeded"
    
    def download_images_sequential(self, metadata_df, max_images=None):
        """Download images sequentially (slower but more reliable)"""
        if max_images:
            metadata_df = metadata_df.head(max_images)
        
        successful = 0
        failed = 0
        
        print(f"Downloading {len(metadata_df)} images sequentially...")
        
        for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
            filepath = os.path.join(self.images_dir, row['filename'])
            
            # Skip if already exists
            if os.path.exists(filepath):
                successful += 1
                continue
            
            result = self.download_single_image(row['original_url'], filepath)
            
            if isinstance(result, tuple):  # Failed
                failed += 1
                self.failed_downloads.append({
                    'url': row['original_url'],
                    'filename': row['filename'],
                    'error': result[1]
                })
            else:  # Success
                successful += 1
            
            # Small delay to be respectful
            time.sleep(0.1)
        
        print(f"Download complete: {successful} successful, {failed} failed")
        return successful, failed
    
    def download_images_parallel(self, metadata_df, max_workers=5, max_images=None):
        """Download images in parallel (faster but more resource intensive)"""
        if max_images:
            metadata_df = metadata_df.head(max_images)
        
        successful = 0
        failed = 0
        
        print(f"Downloading {len(metadata_df)} images with {max_workers} parallel workers...")
        
        def download_task(row):
            filepath = os.path.join(self.images_dir, row['filename'])
            
            # Skip if already exists
            if os.path.exists(filepath):
                return True, row['filename']
            
            result = self.download_single_image(row['original_url'], filepath)
            return result, row
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_row = {executor.submit(download_task, row): row 
                           for _, row in metadata_df.iterrows()}
            
            for future in tqdm(as_completed(future_to_row), total=len(metadata_df)):
                result, row_data = future.result()
                
                if isinstance(result, tuple):  # Failed
                    failed += 1
                    if isinstance(row_data, pd.Series):
                        self.failed_downloads.append({
                            'url': row_data['original_url'],
                            'filename': row_data['filename'],
                            'error': result[1]
                        })
                else:  # Success
                    successful += 1
        
        print(f"Parallel download complete: {successful} successful, {failed} failed")
        return successful, failed
    
    def download_main_images_only(self, metadata_df, parallel=True, max_workers=5):
        """Download only main product images (variant='MAIN')"""
        main_images = metadata_df[metadata_df['is_main'] == True]
        print(f"Downloading {len(main_images)} main product images only...")
        
        if parallel:
            return self.download_images_parallel(main_images, max_workers)
        else:
            return self.download_images_sequential(main_images)
    
    def create_dataset_splits(self, metadata_df, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """Create train/val/test splits and save them"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        # Get unique products (group by product_id)
        products = metadata_df['product_id'].unique()
        n_products = len(products)
        
        # Shuffle products
        import numpy as np
        np.random.seed(42)
        shuffled_products = np.random.permutation(products)
        
        # Calculate split indices
        train_end = int(n_products * train_ratio)
        val_end = int(n_products * (train_ratio + val_ratio))
        
        # Split product IDs
        train_products = shuffled_products[:train_end]
        val_products = shuffled_products[train_end:val_end]
        test_products = shuffled_products[val_end:]
        
        # Create splits dataframes
        train_df = metadata_df[metadata_df['product_id'].isin(train_products)]
        val_df = metadata_df[metadata_df['product_id'].isin(val_products)]
        test_df = metadata_df[metadata_df['product_id'].isin(test_products)]
        
        # Save splits
        train_df.to_csv(os.path.join(self.base_dir, 'train_metadata.csv'), index=False)
        val_df.to_csv(os.path.join(self.base_dir, 'val_metadata.csv'), index=False)
        test_df.to_csv(os.path.join(self.base_dir, 'test_metadata.csv'), index=False)
        
        print(f"Dataset splits created:")
        print(f"  Train: {len(train_df)} images from {len(train_products)} products")
        print(f"  Val:   {len(val_df)} images from {len(val_products)} products")
        print(f"  Test:  {len(test_df)} images from {len(test_products)} products")
        
        return train_df, val_df, test_df
    
    def save_failed_downloads(self):
        """Save failed download information"""
        if self.failed_downloads:
            failed_df = pd.DataFrame(self.failed_downloads)
            failed_df.to_csv(os.path.join(self.base_dir, 'failed_downloads.csv'), index=False)
            print(f"Failed downloads saved to {os.path.join(self.base_dir, 'failed_downloads.csv')}")
    
    def get_download_stats(self):
        """Get statistics about downloaded images"""
        if not os.path.exists(self.images_dir):
            return None
        
        image_files = [f for f in os.listdir(self.images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        
        total_size = 0
        for filename in image_files:
            filepath = os.path.join(self.images_dir, filename)
            total_size += os.path.getsize(filepath)
        
        stats = {
            'total_images': len(image_files),
            'total_size_mb': total_size / (1024 * 1024),
            'avg_size_kb': (total_size / len(image_files)) / 1024 if image_files else 0
        }
        
        return stats
    
    def create_complete_dataset(self, parallel=True, max_workers=5, main_images_only=False, max_images=None):
        """Complete dataset creation pipeline"""
        print("Starting complete dataset creation pipeline...")
        print("-" * 50)
        
        # Step 1: Load data
        data = self.load_data()
        
        # Step 2: Create metadata
        metadata_df = self.create_image_metadata(data)
        
        # Step 3: Download images
        if main_images_only:
            successful, failed = self.download_main_images_only(metadata_df, parallel, max_workers)
        else:
            if parallel:
                successful, failed = self.download_images_parallel(metadata_df, max_workers, max_images)
            else:
                successful, failed = self.download_images_sequential(metadata_df, max_images)
        
        # Step 4: Save failed downloads
        self.save_failed_downloads()
        
        # Step 5: Create train/val/test splits
        train_df, val_df, test_df = self.create_dataset_splits(metadata_df)
        
        # Step 6: Generate statistics
        stats = self.get_download_stats()
        
        print("-" * 50)
        print("DATASET CREATION COMPLETE!")
        print(f"Images downloaded: {successful}")
        print(f"Failed downloads: {failed}")
        if stats:
            print(f"Total dataset size: {stats['total_size_mb']:.2f} MB")
            print(f"Average image size: {stats['avg_size_kb']:.2f} KB")
        
        return {
            'metadata': metadata_df,
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'stats': stats,
            'successful_downloads': successful,
            'failed_downloads': failed
        }

# Usage example
if __name__ == "__main__":
    # Check if shrunk dataset exists, otherwise use original
    import os
    shrunk_file = 'src/meta_Amazon_Fashion_shrunk_300k.jsonl'
    original_file = 'src/meta_Amazon_Fashion.jsonl'
    
    if os.path.exists(shrunk_file):
        print(f"Using shrunk dataset: {shrunk_file}")
        dataset_file = shrunk_file
    elif os.path.exists(original_file):
        print(f"Shrunk dataset not found, using original: {original_file}")
        
        dataset_file = original_file
    else:
        print("Error: No dataset file found!")
        exit(1)
    
    # Initialize downloader with appropriate dataset
    downloader = ProductImageDownloader(dataset_file, base_dir='product_dataset')
   
    result = downloader.create_complete_dataset(
        parallel=True, 
        max_workers=3, 
        main_images_only=True
    )
    
