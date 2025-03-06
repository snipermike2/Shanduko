"""
src/shanduko/analysis/standalone_analysis.py

Standalone analysis script for water quality data
"""
import sys
from pathlib import Path
import pandas as pd
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.shanduko.data.data_processor import WaterQualityDataProcessor
from src.shanduko.data.quality_assessment import DataQualityAssessor
from src.shanduko.visualization.data_viz import WaterQualityVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WaterQualityAnalyzer:
    def __init__(self):
        """Initialize analyzer components"""
        self.processor = WaterQualityDataProcessor()
        self.assessor = DataQualityAssessor()
        self.visualizer = WaterQualityVisualizer()
        
    def analyze_file(self, file_path: str):
        """
        Analyze a single data file
        
        Parameters:
            file_path: Path to the CSV file
        """
        try:
            # Create timestamp for output directories
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = Path(file_path).stem
            
            # Create output directories
            reports_base = project_root / "reports" / "standalone" / file_name / timestamp
            quality_dir = reports_base / "quality"
            viz_dir = reports_base / "visualizations"
            processed_dir = reports_base / "processed"
            
            for dir_path in [quality_dir, viz_dir, processed_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Analyzing file: {file_path}")
            
            # 1. Load and process data
            logger.info("Loading and processing data...")
            df = self.processor.load_csv_data(file_path)
            df = self.processor.handle_missing_values(df)
            
            # 2. Generate quality report
            logger.info("Generating quality report...")
            quality_report = self.assessor.generate_quality_report(df)
            report_path = quality_dir / "quality_report.xlsx"
            self.assessor.save_report(quality_report, str(report_path))
            
            # 3. Create visualizations
            logger.info("Creating visualizations...")
            self.visualizer.create_dashboard(
                df,
                output_dir=str(viz_dir),
                prefix="water_quality"
            )
            
            # 4. Process data for modeling
            logger.info("Processing data for modeling...")
            df_norm, norm_params = self.processor.normalize_data(df)
            sequences, targets = self.processor.prepare_sequences(df_norm)
            
            # Save processed data
            processed_path = processed_dir / "processed_data.csv"
            self.processor.save_processed_data(
                df_norm,
                str(processed_path),
                norm_params
            )
            
            # Print summary
            self._print_analysis_summary(df, quality_report, reports_base)
            
            return {
                'raw_data': df,
                'normalized_data': df_norm,
                'sequences': (sequences, targets),
                'quality_report': quality_report,
                'norm_params': norm_params,
                'output_dir': reports_base
            }
            
        except Exception as e:
            logger.error(f"Error analyzing file: {e}")
            raise
            
    def analyze_directory(self, dir_path: str):
        """
        Analyze all CSV files in a directory
        
        Parameters:
            dir_path: Path to directory containing CSV files
        """
        dir_path = Path(dir_path)
        results = {}
        
        for csv_file in dir_path.glob("*.csv"):
            try:
                logger.info(f"\nProcessing {csv_file.name}...")
                results[csv_file.name] = self.analyze_file(str(csv_file))
            except Exception as e:
                logger.error(f"Error processing {csv_file.name}: {e}")
                
        return results
    
    def _print_analysis_summary(self, df: pd.DataFrame, 
                              quality_report: dict,
                              output_dir: Path):
        """Print summary of analysis results"""
        print("\n" + "="*50)
        print("Analysis Summary")
        print("="*50)
        
        print("\nData Overview:")
        print(f"Total records: {len(df)}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        print("\nQuality Scores:")
        scores = quality_report['quality_scores']
        print(f"Overall Score: {scores['overall']:.1f}%")
        print(f"Completeness: {scores['completeness']:.1f}%")
        print(f"Consistency: {scores['consistency']:.1f}%")
        print(f"Timeliness: {scores['timeliness']:.1f}%")
        
        print("\nOutput Locations:")
        print(f"Quality Report: {output_dir/'quality'/'quality_report.xlsx'}")
        print(f"Visualizations: {output_dir/'visualizations'}")
        print(f"Processed Data: {output_dir/'processed'/'processed_data.csv'}")
        print("\n" + "="*50)

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze water quality data files"
    )
    parser.add_argument(
        "path",
        help="Path to CSV file or directory containing CSV files"
    )
    args = parser.parse_args()
    
    analyzer = WaterQualityAnalyzer()
    path = Path(args.path)
    
    try:
        if path.is_file():
            analyzer.analyze_file(str(path))
        elif path.is_dir():
            analyzer.analyze_directory(str(path))
        else:
            logger.error(f"Invalid path: {path}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()