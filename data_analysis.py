import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

class JSONLAnalyzer:
    """Analyze JSONL dataset for product rating prediction"""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = []
        self.df = None
        
    def load_data(self):
        """Load data from JSONL file"""
        print("Loading JSONL data...")
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    item = json.loads(line.strip())
                    self.data.append(item)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {i+1}: {e}")
        
        print(f"Loaded {len(self.data)} total records")
        return self.data
    
    def filter_valid_data(self):
        """Filter data that has images and ratings"""
        valid_data = []
        for item in self.data:
            if (item.get('images') and 
                item.get('average_rating') is not None and
                len(item['images']) > 0):
                valid_data.append(item)
        
        print(f"Found {len(valid_data)} records with both images and ratings")
        return valid_data
    
    def create_dataframe(self):
        """Convert data to pandas DataFrame for easier analysis"""
        valid_data = self.filter_valid_data()
        
        records = []
        for item in valid_data:
            record = {
                'title': item.get('title', ''),
                'main_category': item.get('main_category', ''),
                'average_rating': item.get('average_rating'),
                'rating_number': item.get('rating_number', 0),
                'store': item.get('store', ''),
                'num_images': len(item.get('images', [])),
                'has_videos': len(item.get('videos', [])) > 0,
                'num_features': len(item.get('features', [])),
                'has_description': len(item.get('description', [])) > 0,
                'price': item.get('price'),
                'parent_asin': item.get('parent_asin', '')
            }
            records.append(record)
        
        self.df = pd.DataFrame(records)
        return self.df
    
    def basic_statistics(self):
        """Generate basic statistics about the dataset"""
        if self.df is None:
            self.create_dataframe()
        
        print("=== BASIC DATASET STATISTICS ===")
        print(f"Total valid records: {len(self.df)}")
        print(f"Categories: {self.df['main_category'].nunique()}")
        print(f"Unique stores: {self.df['store'].nunique()}")
        print()
        
        print("=== RATING STATISTICS ===")
        print(f"Average rating range: {self.df['average_rating'].min():.1f} - {self.df['average_rating'].max():.1f}")
        print(f"Mean rating: {self.df['average_rating'].mean():.2f}")
        print(f"Median rating: {self.df['average_rating'].median():.2f}")
        print(f"Standard deviation: {self.df['average_rating'].std():.2f}")
        print()
        
        print("=== RATING NUMBER STATISTICS ===")
        print(f"Rating count range: {self.df['rating_number'].min()} - {self.df['rating_number'].max()}")
        print(f"Mean rating count: {self.df['rating_number'].mean():.1f}")
        print(f"Median rating count: {self.df['rating_number'].median():.1f}")
        print()
        
        print("=== IMAGE STATISTICS ===")
        print(f"Images per product range: {self.df['num_images'].min()} - {self.df['num_images'].max()}")
        print(f"Average images per product: {self.df['num_images'].mean():.1f}")
        print()
        
        return self.df.describe()
    
    def rating_distribution_analysis(self):
        """Analyze rating distribution"""
        if self.df is None:
            self.create_dataframe()
        
        # Create rating distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Rating distribution histogram
        axes[0,0].hist(self.df['average_rating'], bins=20, edgecolor='black', alpha=0.7)
        axes[0,0].set_title('Average Rating Distribution')
        axes[0,0].set_xlabel('Average Rating')
        axes[0,0].set_ylabel('Frequency')
        
        # Rating count distribution
        axes[0,1].hist(self.df['rating_number'], bins=30, edgecolor='black', alpha=0.7)
        axes[0,1].set_title('Rating Count Distribution')
        axes[0,1].set_xlabel('Number of Ratings')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_yscale('log')  # Log scale due to potential wide range
        
        # Rating vs Rating Count scatter
        axes[1,0].scatter(self.df['rating_number'], self.df['average_rating'], alpha=0.6)
        axes[1,0].set_title('Average Rating vs Rating Count')
        axes[1,0].set_xlabel('Number of Ratings')
        axes[1,0].set_ylabel('Average Rating')
        
        # Category distribution
        category_counts = self.df['main_category'].value_counts()
        axes[1,1].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
        axes[1,1].set_title('Product Category Distribution')
        
        plt.tight_layout()
        
        # Create results directory if it doesn't exist
        import os
        results_dir = 'analysis_results'
        os.makedirs(results_dir, exist_ok=True)
        
        # Save plot in results folder
        plot_path = os.path.join(results_dir, 'rating_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def category_analysis(self):
        """Analyze ratings by category"""
        if self.df is None:
            self.create_dataframe()
        
        print("=== CATEGORY ANALYSIS ===")
        category_stats = self.df.groupby('main_category').agg({
            'average_rating': ['mean', 'std', 'count'],
            'rating_number': ['mean', 'median'],
            'num_images': 'mean'
        }).round(2)
        
        print(category_stats)
        
        # Box plot of ratings by category
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.df, x='main_category', y='average_rating')
        plt.xticks(rotation=45)
        plt.title('Rating Distribution by Category')
        plt.tight_layout()
        
        # Create results directory if it doesn't exist
        import os
        results_dir = 'analysis_results'
        os.makedirs(results_dir, exist_ok=True)
        
        # Save plot in results folder
        plot_path = os.path.join(results_dir, 'category_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return category_stats
    
    def data_quality_check(self):
        """Check data quality issues"""
        if self.df is None:
            self.create_dataframe()
        
        print("=== DATA QUALITY CHECK ===")
        
        # Missing values
        print("Missing values:")
        print(self.df.isnull().sum())
        print()
        
        # Duplicate check
        duplicates = self.df['parent_asin'].duplicated().sum()
        print(f"Potential duplicates (same parent_asin): {duplicates}")
        print()
        
        # Outliers in ratings
        rating_outliers = self.df[(self.df['average_rating'] < 1) | (self.df['average_rating'] > 5)]
        print(f"Rating outliers (< 1 or > 5): {len(rating_outliers)}")
        
        # Products with very few ratings
        few_ratings = self.df[self.df['rating_number'] < 5]
        print(f"Products with < 5 ratings: {len(few_ratings)} ({len(few_ratings)/len(self.df)*100:.1f}%)")
        
        return {
            'missing_values': self.df.isnull().sum(),
            'duplicates': duplicates,
            'rating_outliers': rating_outliers,
            'few_ratings': few_ratings
        }
    
    def machine_learning_recommendations(self):
        """Provide ML recommendations based on data analysis"""
        if self.df is None:
            self.create_dataframe()
        
        print("=== MACHINE LEARNING RECOMMENDATIONS ===")
        
        dataset_size = len(self.df)
        print(f"Dataset size: {dataset_size}")
        
        if dataset_size < 100:
            print("SMALL DATASET WARNING:")
            print("- Consider data augmentation heavily")
            print("- Use transfer learning with frozen early layers")
            print("- Consider few-shot learning techniques")
            print("- Implement cross-validation")
        elif dataset_size < 1000:
            print("MEDIUM DATASET:")
            print("- Use transfer learning")
            print("- Apply data augmentation")
            print("- Consider k-fold cross-validation")
        else:
            print("LARGE DATASET:")
            print("- Can train from scratch or use transfer learning")
            print("- Standard train/validation/test split recommended")
        
        print()
        
        # Rating distribution recommendations
        rating_std = self.df['average_rating'].std()
        if rating_std < 0.5:
            print("📈 RATING DISTRIBUTION:")
            print("- Low rating variance detected")
            print("- Consider regression with MSE loss")
            print("- May need to collect more diverse products")
        else:
            print("📈 RATING DISTRIBUTION:")
            print("- Good rating variance")
            print("- Both classification and regression viable")
        
        print()
        
        # Image recommendations
        avg_images = self.df['num_images'].mean()
        print(f"🖼️  IMAGE ANALYSIS:")
        print(f"- Average {avg_images:.1f} images per product")
        if avg_images > 3:
            print("- Consider multi-image ensemble learning")
            print("- Can use image attention mechanisms")
        else:
            print("- Single image learning recommended")
            print("- Focus on data augmentation")
    
    def generate_summary_report(self):
        """Generate complete analysis report"""
        # Import os for directory operations
        import os
        
        # Create results directory early
        results_dir = 'analysis_results'
        os.makedirs(results_dir, exist_ok=True)
        
        # Capture all output to save to file
        results = []
        
        def add_to_results(text):
            results.append(text)
            print(text)
        
        add_to_results("=" * 60)
        add_to_results("JSONL DATASET ANALYSIS REPORT")
        add_to_results("=" * 60)
        
        self.load_data()
        
        # Basic Statistics
        add_to_results(f"Found {len([item for item in self.data if 'images' in item and 'average_rating' in item])} records with both images and ratings")
        df = self.create_dataframe()
        
        # Check total number of images
        total_images = df['num_images'].sum()
        add_to_results(f"Total images in dataset: {total_images:,}")

        # If more than 300k images, ask user if they want to shrink
        if total_images > 300000:
            print(f"\n⚠️  WARNING: Dataset contains {total_images:,} lines, which exceeds 150,000!")
            print("This may lead to very long processing times and high storage requirements.")
            user_input = input("Do you want to shrink the dataset to approximately 150,000 lines? (y/n): ")
            
            if user_input.strip().lower() in ['y', 'yes']:   
                import random
                random.seed(42)  # For reproducible results
                
                # Step 1: Remove duplicates based on parent_asin first
                df_unique_asin = df.drop_duplicates(subset=['parent_asin'], keep='first')
                add_to_results(f"Removed {len(df) - len(df_unique_asin)} duplicate products (same parent_asin)")
                
                # Step 2: Remove duplicates based on title (case-insensitive)
                print("Removing duplicate products by title...")
                seen_titles = set()
                unique_products = []
                removed_title_duplicates = 0
                
                for idx, row in df_unique_asin.iterrows():
                    title_clean = str(row['title']).strip().lower()
                    if title_clean and title_clean not in seen_titles:
                        seen_titles.add(title_clean)
                        unique_products.append(row)
                    else:
                        removed_title_duplicates += 1
                
                df_unique = pd.DataFrame(unique_products)
                add_to_results(f"Removed {removed_title_duplicates} additional duplicate products (same title)")
                add_to_results(f"Total duplicates removed: {len(df) - len(df_unique)}")
                add_to_results(f"Unique products after deduplication: {len(df_unique)}")
                
                # Shuffle the unique dataframe
                df_shuffled = df_unique.sample(frac=1, random_state=42).reset_index(drop=True)
                
                cumulative_lines = df_shuffled['num_images'].cumsum()
                cutoff_index = (cumulative_lines <= 300000).sum()

                if cutoff_index > 0:
                    df = df_shuffled.iloc[:cutoff_index].copy()
                    final_image_count = df['num_images'].sum()
                    add_to_results(f"Dataset shrunk to {len(df):,} products with {final_image_count:,} images")
                    add_to_results(f"Total reduction: {len(df_unique) - len(df):,} products removed")
                    
                    # Save shrunk dataset as new JSONL file
                    shrunk_file_path = 'src/meta_Amazon_Fashion_shrunk_300k.jsonl'
                    selected_parent_asins = set(df['parent_asin'].values)
                    
                    # Create new JSONL file with only selected products (remove duplicates)
                    seen_asins = set()
                    with open(shrunk_file_path, 'w', encoding='utf-8') as outfile:
                        for item in self.data:
                            parent_asin = item.get('parent_asin')
                            if (parent_asin in selected_parent_asins and
                                parent_asin not in seen_asins and  # Avoid duplicates
                                item.get('images') and 
                                item.get('average_rating') is not None):
                                outfile.write(json.dumps(item) + '\n')
                                seen_asins.add(parent_asin)
                    
                    add_to_results(f"Shrunk dataset saved to: {shrunk_file_path}")
                    
                    # Also save the dataframe as CSV for easy access
                    csv_file_path = os.path.join(results_dir, 'shrunk_dataset_metadata.csv')
                    df.to_csv(csv_file_path, index=False)
                    add_to_results(f"Shrunk dataset metadata saved to: {csv_file_path}")
                    
                    # Create comprehensive CSV with all original columns
                    comprehensive_data = []
                    
                    for item in self.data:
                        parent_asin = item.get('parent_asin')
                        if (parent_asin in selected_parent_asins and
                            item.get('images') and 
                            item.get('average_rating') is not None):
                            
                            # Flatten the complex data structure
                            flat_item = {
                                'parent_asin': item.get('parent_asin', ''),
                                'title': item.get('title', ''),
                                'main_category': item.get('main_category', ''),
                                'average_rating': item.get('average_rating'),
                                'rating_number': item.get('rating_number', 0),
                                'store': item.get('store', ''),
                                'price': item.get('price'),
                                'num_images': len(item.get('images', [])),
                                'image_urls': '|'.join([img.get('large', img.get('thumb', '')) for img in item.get('images', [])]),
                                'num_videos': len(item.get('videos', [])),
                                'video_urls': '|'.join([str(vid) if isinstance(vid, dict) else str(vid) for vid in item.get('videos', [])]),
                                'num_features': len(item.get('features', [])),
                                'features': '|'.join([str(f) for f in item.get('features', [])]),
                                'description': '|'.join([str(d) for d in item.get('description', [])]),
                                'also_buy': '|'.join([str(ab) for ab in item.get('also_buy', [])]),
                                'also_view': '|'.join([str(av) for av in item.get('also_view', [])]),
                                'brand': item.get('brand', ''),
                                'rank': item.get('rank', ''),
                                'similar_item': item.get('similar_item', ''),
                                'tech1': item.get('tech1', ''),
                                'tech2': item.get('tech2', ''),
                                'fit': item.get('fit', ''),
                                'date': item.get('date', '')
                            }
                            comprehensive_data.append(flat_item)
                    
                    # Remove duplicates from comprehensive data as well
                    comprehensive_df = pd.DataFrame(comprehensive_data)
                    comprehensive_df = comprehensive_df.drop_duplicates(subset=['parent_asin'], keep='first')
                    comprehensive_csv_path = os.path.join(results_dir, 'shrunk_dataset_comprehensive.csv')
                    comprehensive_df.to_csv(comprehensive_csv_path, index=False)
                    add_to_results(f"Comprehensive dataset saved to: {comprehensive_csv_path}")
                    add_to_results(f"Comprehensive CSV includes: {list(comprehensive_df.columns)}")
                else:
                    add_to_results("Could not shrink dataset effectively - continuing with full dataset")
            else:
                add_to_results("User chose to keep full dataset - continuing with all images")
        
        add_to_results("=== BASIC DATASET STATISTICS ===")
        add_to_results(f"Total valid records: {len(df)}")
        add_to_results(f"Categories: {df['main_category'].nunique()}")
        add_to_results(f"Unique stores: {df['store'].nunique()}")
        
        add_to_results("\n=== RATING STATISTICS ===")
        add_to_results(f"Average rating range: {df['average_rating'].min()} - {df['average_rating'].max()}")
        add_to_results(f"Mean rating: {df['average_rating'].mean():.2f}")
        add_to_results(f"Median rating: {df['average_rating'].median():.2f}")
        add_to_results(f"Standard deviation: {df['average_rating'].std():.2f}")
        
        add_to_results("\n=== RATING NUMBER STATISTICS ===")
        add_to_results(f"Rating count range: {df['rating_number'].min()} - {df['rating_number'].max()}")
        add_to_results(f"Mean rating count: {df['rating_number'].mean():.1f}")
        add_to_results(f"Median rating count: {df['rating_number'].median():.1f}")
        
        add_to_results("\n=== IMAGE STATISTICS ===")
        add_to_results(f"Images per product range: {df['num_images'].min()} - {df['num_images'].max()}")
        add_to_results(f"Average images per product: {df['num_images'].mean():.1f}")
        
        add_to_results("\n" + "-" * 40)
        
        # Category Analysis
        add_to_results("\n=== CATEGORY ANALYSIS ===")
        category_stats = df.groupby('main_category').agg({
            'average_rating': ['mean', 'std', 'count'],
            'rating_number': ['mean', 'median'],
            'num_images': 'mean'
        }).round(2)
        add_to_results(str(category_stats))
        
        add_to_results("\n" + "-" * 40)
        
        # Data Quality Check
        add_to_results("\n=== DATA QUALITY CHECK ===")
        add_to_results("Missing values:")
        missing_values = df.isnull().sum()
        for col, count in missing_values.items():
            add_to_results(f"{col:<20} {count}")
        
        duplicates = df.duplicated(subset=['parent_asin']).sum()
        add_to_results(f"\nPotential duplicates (same parent_asin): {duplicates}")
        
        rating_outliers = df[(df['average_rating'] < 1) | (df['average_rating'] > 5)].shape[0]
        add_to_results(f"\nRating outliers (< 1 or > 5): {rating_outliers}")
        
        low_rating_products = df[df['rating_number'] < 5].shape[0]
        low_rating_pct = (low_rating_products / len(df)) * 100
        add_to_results(f"Products with < 5 ratings: {low_rating_products} ({low_rating_pct:.1f}%)")
        
        add_to_results("\n" + "-" * 40)
        
        # ML Recommendations
        add_to_results("\n=== MACHINE LEARNING RECOMMENDATIONS ===")
        add_to_results(f"Dataset size: {len(df)}")
        if len(df) > 100000:
            add_to_results("✅ LARGE DATASET:")
            add_to_results("- Can train from scratch or use transfer learning")
            add_to_results("- Standard train/validation/test split recommended")
        elif len(df) > 10000:
            add_to_results("✅ MEDIUM DATASET:")
            add_to_results("- Transfer learning recommended")
            add_to_results("- Consider data augmentation")
        else:
            add_to_results("⚠️  SMALL DATASET:")
            add_to_results("- Transfer learning essential")
            add_to_results("- Heavy data augmentation needed")
        
        add_to_results("\n📈 RATING DISTRIBUTION:")
        rating_std = df['average_rating'].std()
        if rating_std > 0.5:
            add_to_results("- Good rating variance")
            add_to_results("- Both classification and regression viable")
        else:
            add_to_results("- Low rating variance")
            add_to_results("- Classification may be challenging")
        
        add_to_results("\n🖼️  IMAGE ANALYSIS:")
        avg_images = df['num_images'].mean()
        add_to_results(f"- Average {avg_images:.1f} images per product")
        if avg_images > 3:
            add_to_results("- Consider multi-image ensemble learning")
            add_to_results("- Can use image attention mechanisms")
        else:
            add_to_results("- Single image learning recommended")
            add_to_results("- Focus on data augmentation")
        
        add_to_results("\n" + "-" * 40)
        
        self.rating_distribution_analysis()
        
        # Save complete results to file
        results_file_path = os.path.join(results_dir, 'analysisResults.txt')
        with open(results_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(results))
            f.write('\n\nDetailed analysis completed on: ' + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Save summary JSON
        summary_stats = {
            'total_records': len(self.data),
            'valid_records': len(df),
            'total_images': int(df['num_images'].sum()),
            'rating_stats': df['average_rating'].describe().to_dict(),
            'categories': list(df['main_category'].unique()),
            'avg_images_per_product': df['num_images'].mean(),
            'dataset_shrunk': int(df['num_images'].sum()) <= 300000 and len(self.data) > len(df),
            'analysis_date': pd.Timestamp.now().isoformat()
        }
        
        summary_file_path = os.path.join(results_dir, 'dataset_summary.json')
        with open(summary_file_path, 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        
        print(f"\nResults saved to {results_file_path}")
        print(f"Summary saved to {summary_file_path}")
        print(f"Visualizations will be saved in {results_dir} folder")

# Usage example
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = JSONLAnalyzer('src/meta_Amazon_Fashion.jsonl')
    
    # Run complete analysis
    analyzer.generate_summary_report()
    
    # Or run individual analyses
    # analyzer.load_data()
    # df = analyzer.create_dataframe()
    # analyzer.basic_statistics()
    # analyzer.rating_distribution_analysis()