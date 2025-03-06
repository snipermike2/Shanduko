"""
src/shanduko/visualization/test_viz.py
Test script for visualization tools
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent.parent)
sys.path.append(project_root)

from shanduko.data.data_processor import WaterQualityDataProcessor
from shanduko.visualization.data_viz import WaterQualityVisualizer

def main():
    """Test visualization module with sample data"""
    # Create data processor and generate sample data
    processor = WaterQualityDataProcessor()
    
    # Generate 30 days of sample data
    base_date = datetime.now() - timedelta(days=30)
    timestamps = [base_date + timedelta(hours=i) for i in range(30 * 24)]
    
    # Generate data with patterns and some anomalies
    np.random.seed(42)  # For reproducibility
    
    data = {
        'timestamp': timestamps,
        'temperature': [
            25 + 5 * np.sin(i/24 * 2*np.pi) + np.random.normal(0, 0.5) +
            (10 if i == 360 else 0)  # Add anomaly
            for i in range(len(timestamps))
        ],
        'ph': [
            7 + 0.5 * np.sin(i/24 * 2*np.pi) + np.random.normal(0, 0.1)
            for i in range(len(timestamps))
        ],
        'dissolved_oxygen': [
            8 + 2 * np.sin(i/24 * 2*np.pi) + np.random.normal(0, 0.3) -
            (4 if i == 480 else 0)  # Add anomaly
            for i in range(len(timestamps))
        ],
        'turbidity': [
            3 + np.sin(i/24 * 2*np.pi) + np.random.normal(0, 0.2) +
            (5 if i == 200 else 0)  # Add anomaly
            for i in range(len(timestamps))
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Create visualizer
    visualizer = WaterQualityVisualizer()
    
    # Create reports directory
    reports_dir = Path(project_root) / "reports" / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating visualizations...")
    
    # Generate individual plots
    visualizer.plot_time_series(df, 
                              save_path=str(reports_dir / "time_series.png"))
    
    visualizer.plot_correlation_heatmap(df,
                                      save_path=str(reports_dir / "correlations.png"))
    
    visualizer.plot_daily_patterns(df,
                                 save_path=str(reports_dir / "daily_patterns.png"))
    
    visualizer.plot_parameter_distributions(df,
                                         save_path=str(reports_dir / "distributions.png"))
    
    visualizer.plot_boxplots(df, by='hour',
                           save_path=str(reports_dir / "hourly_variations.png"))
    
    visualizer.plot_anomalies(df,
                            save_path=str(reports_dir / "anomalies.png"))
    
    # Generate complete dashboard
    print("\nGenerating dashboard...")
    visualizer.create_dashboard(df, output_dir=str(reports_dir / "dashboard"))
    
    print("\nVisualization test completed!")
    print(f"Reports saved to: {reports_dir}")

if __name__ == "__main__":
    main()