# src/shanduko/visualization/metrics_dashboard.py

from pathlib import Path
import json
import numpy as np
from datetime import datetime

def format_metrics_for_dashboard(evaluation_results):
    """Format evaluation metrics for dashboard visualization"""
    return {
        'basic_metrics': evaluation_results['basic_metrics'],
        'water_quality_standards': evaluation_results['water_quality_standards'],
        'ecological_impact': evaluation_results['ecological_impact'],
        'treatment_metrics': evaluation_results['treatment_requirements']
    }

def save_metrics_for_visualization(evaluation_results, output_dir='reports/metrics'):
    """Save metrics in JSON format for dashboard"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Format metrics
    dashboard_data = format_metrics_for_dashboard(evaluation_results)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"metrics_{timestamp}.json"
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(dashboard_data, f, indent=2)
    
    return output_file