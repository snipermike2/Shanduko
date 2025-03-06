"""
src/shanduko/data/analyze_data.py
Main script for data analysis and quality assessment
"""
from pathlib import Path
import pandas as pd
from datetime import datetime

from src.shanduko.data.data_processor import WaterQualityDataProcessor
from src.shanduko.data.quality_assessment import DataQualityAssessor
from src.shanduko.visualization.data_viz import WaterQualityVisualizer

def process_and_analyze_data(data_path: str):
    """
    Process and analyze water quality data
    
    Parameters:
        data_path: Path to raw data file
    """
    # Initialize components
    processor = WaterQualityDataProcessor()
    assessor = DataQualityAssessor()
    visualizer = WaterQualityVisualizer()
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = Path("reports")
    quality_dir = reports_dir / "quality" / timestamp
    viz_dir = reports_dir / "visualizations" / timestamp
    
    for dir_path in [quality_dir, viz_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Load and process data
        print("Loading and processing data...")
        df = processor.load_csv_data(data_path)
        df = processor.handle_missing_values(df)
        
        # 2. Assess data quality
        print("\nAssessing data quality...")
        quality_report = assessor.generate_quality_report(df)
        report_path = quality_dir / "quality_report.xlsx"
        assessor.save_report(quality_report, str(report_path))
        print(f"Quality report saved to {report_path}")
        
        # 3. Generate visualizations
        print("\nGenerating visualizations...")
        visualizer.create_dashboard(df, 
                                  output_dir=str(viz_dir),
                                  prefix="water_quality")
        print(f"Visualizations saved to {viz_dir}")
        
        # 4. Prepare data for model training
        print("\nPreparing sequences for model training...")
        df_norm, norm_params = processor.normalize_data(df)
        X, y = processor.prepare_sequences(df_norm)
        
        # Save processed data
        processed_dir = Path("data/processed") / timestamp
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        processed_path = processed_dir / "processed_data.csv"
        processor.save_processed_data(df_norm, 
                                   str(processed_path),
                                   norm_params)
        print(f"Processed data saved to {processed_path}")
        
        return {
            'raw_data': df,
            'normalized_data': df_norm,
            'sequences': (X, y),
            'quality_report': quality_report,
            'normalization_params': norm_params
        }
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    data_file = "data/raw/water_quality_data.csv"
    results = process_and_analyze_data(data_file)