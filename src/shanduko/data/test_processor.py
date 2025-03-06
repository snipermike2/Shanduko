"""
src/shanduko/data/test_processor.py
Test script for data processing
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from src.shanduko.data.data_processor import WaterQualityDataProcessor

def generate_sample_data(num_days: int = 30) -> pd.DataFrame:
    """Generate sample data for testing"""
    # Generate timestamps
    base_date = datetime.now() - timedelta(days=num_days)
    timestamps = [base_date + timedelta(hours=i) for i in range(num_days * 24)]
    
    # Generate data with realistic patterns
    data = {
        'timestamp': timestamps,
        'temperature': [25 + 5 * np.sin(i/24 * 2*np.pi) + np.random.normal(0, 0.5) 
                       for i in range(len(timestamps))],
        'ph': [7 + 0.5 * np.sin(i/24 * 2*np.pi) + np.random.normal(0, 0.1) 
               for i in range(len(timestamps))],
        'dissolved_oxygen': [8 + 2 * np.sin(i/24 * 2*np.pi) + np.random.normal(0, 0.3) 
                           for i in range(len(timestamps))],
        'turbidity': [3 + np.sin(i/24 * 2*np.pi) + np.random.normal(0, 0.2) 
                     for i in range(len(timestamps))]
    }
    
    return pd.DataFrame(data)

def main():
    """Test data processing pipeline"""
    # Create data processor
    processor = WaterQualityDataProcessor()
    
    # Generate and save sample data
    print("Generating sample data...")
    sample_data = generate_sample_data()
    
    # Create data directory if it doesn't exist
    data_dir = Path('data/raw')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sample data
    sample_data_path = data_dir / 'sample_data.csv'
    sample_data.to_csv(sample_data_path, index=False)
    print(f"Saved sample data to {sample_data_path}")
    
    # Test data loading and processing
    print("\nTesting data processing pipeline...")
    
    # 1. Load and validate data
    df = processor.load_csv_data(sample_data_path)
    print("\nData loaded successfully")
    print(f"Shape: {df.shape}")
    
    # 2. Check data quality
    quality_metrics = processor.check_data_quality(df)
    print("\nData Quality Metrics:")
    print(f"Total rows: {quality_metrics['total_rows']}")
    print(f"Missing values: {quality_metrics['missing_values']}")
    print(f"Outliers: {quality_metrics['outliers']}")
    print(f"Rapid changes: {quality_metrics['rapid_changes']}")
    
    # 3. Generate synthetic data
    synthetic_data = processor.generate_synthetic_data(num_days=30)
    print("\nGenerated synthetic data")
    print(f"Synthetic data shape: {synthetic_data.shape}")
    
    # 4. Augment data
    augmented_data = processor.augment_data(df, noise_level=0.1)
    print("\nData augmented with noise")
    print(f"Augmented data shape: {augmented_data.shape}")
    
    # 5. Handle missing values
    processed_data = processor.handle_missing_values(augmented_data)
    print("\nMissing values handled")
    print(f"Missing values remaining: {processed_data.isnull().sum().sum()}")
    
    # 6. Normalize data
    normalized_data, norm_params = processor.normalize_data(processed_data)
    print("\nData normalized")
    print("Normalization parameters:", norm_params)
    
    # 7. Prepare sequences
    X, y = processor.prepare_sequences(normalized_data)
    print("\nSequences prepared")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Save processed data
    processed_dir = Path('data/processed')
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    processed_path = processed_dir / 'processed_data.csv'
    processor.save_processed_data(normalized_data, processed_path, norm_params)
    
    print("\nProcessing pipeline test completed successfully!")

if __name__ == "__main__":
    main()