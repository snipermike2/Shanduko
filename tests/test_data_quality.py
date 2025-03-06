"""
tests/test_data_quality.py
Tests for data quality assessment module
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from src.shanduko.data.quality_assessment import DataQualityAssessor

@pytest.fixture
def sample_data():
    """Generate sample data for testing"""
    # Generate timestamps for 5 days with hourly measurements
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(days=5),
        end=datetime.now(),
        freq='1H'
    )
    
    # Generate normal data
    data = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': np.random.normal(25, 2, len(timestamps)),
        'ph': np.random.normal(7, 0.5, len(timestamps)),
        'dissolved_oxygen': np.random.normal(8, 1, len(timestamps)),
        'turbidity': np.random.normal(3, 0.5, len(timestamps))
    })
    
    # Add some anomalies
    data.loc[10:15, 'temperature'] = 45  # Above max
    data.loc[20:25, 'ph'] = np.nan  # Missing values
    data.loc[30, 'dissolved_oxygen'] = -1  # Below min
    
    # Add time gap
    data = data.drop(index=range(40, 50))
    
    return data

def test_completeness_check(sample_data):
    """Test completeness checking"""
    assessor = DataQualityAssessor()
    results = assessor.check_completeness(sample_data)
    
    assert 'total_records' in results
    assert 'missing_counts' in results
    assert 'completeness_scores' in results
    
    # Check that pH has missing values
    assert results['missing_counts']['ph'] > 0

def test_consistency_check(sample_data):
    """Test consistency checking"""
    assessor = DataQualityAssessor()
    results = assessor.check_consistency(sample_data)
    
    # Check temperature violations
    assert results['temperature']['above_max'] > 0
    
    # Check dissolved oxygen violations
    assert results['dissolved_oxygen']['below_min'] > 0

def test_timeliness_check(sample_data):
    """Test timeliness checking"""
    assessor = DataQualityAssessor()
    results = assessor.check_timeliness(sample_data)
    
    assert results['total_gaps'] > 0
    assert isinstance(results['max_gap'], pd.Timedelta)
    assert isinstance(results['avg_gap'], pd.Timedelta)

def test_quality_report(sample_data):
    """Test quality report generation"""
    assessor = DataQualityAssessor()
    report = assessor.generate_quality_report(sample_data)
    
    assert 'completeness' in report
    assert 'consistency' in report
    assert 'timeliness' in report
    assert 'quality_scores' in report
    
    # Check quality scores
    scores = report['quality_scores']
    assert 0 <= scores['completeness'] <= 100
    assert 0 <= scores['consistency'] <= 100
    assert 0 <= scores['timeliness'] <= 100
    assert 0 <= scores['overall'] <= 100

def test_report_saving(sample_data, tmp_path):
    """Test report saving functionality"""
    assessor = DataQualityAssessor()
    report = assessor.generate_quality_report(sample_data)
    
    output_path = tmp_path / "quality_report.xlsx"
    assessor.save_report(report, str(output_path))
    
    assert output_path.exists()
    # Check that Excel file has all expected sheets
    xl = pd.ExcelFile(output_path)
    assert all(sheet in xl.sheet_names for sheet in 
              ['Completeness', 'Consistency', 'Timeliness', 'Quality Scores'])

if __name__ == "__main__":
    pytest.main([__file__])