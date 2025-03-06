"""
src/shanduko/data/quality_assessment.py
Data quality assessment module for water quality monitoring
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import timedelta
import logging


logger = logging.getLogger(__name__)

class DataQualityAssessor:
    def __init__(self, expected_frequency: str = '1H'):
        """
        Initialize data quality assessor
        
        Parameters:
            expected_frequency: Expected frequency of measurements (default: hourly)
        """
        self.expected_frequency = expected_frequency
        self.parameter_thresholds = {
            'temperature': {'min': 0, 'max': 40, 'rate_change': 5},  # °C per hour
            'ph': {'min': 0, 'max': 14, 'rate_change': 1},  # pH units per hour
            'dissolved_oxygen': {'min': 0, 'max': 20, 'rate_change': 2},  # mg/L per hour
            'turbidity': {'min': 0, 'max': 50, 'rate_change': 10}  # NTU per hour
        }
    
    def check_completeness(self, df: pd.DataFrame) -> Dict:
        """Check data completeness"""
        total_records = len(df)
        missing_counts = df.isnull().sum()
        completeness_scores = (1 - missing_counts / total_records) * 100
        
        return {
            'total_records': total_records,
            'missing_counts': missing_counts.to_dict(),
            'completeness_scores': completeness_scores.to_dict()
        }
    
    def check_consistency(self, df: pd.DataFrame) -> Dict:
        """Check data consistency and identify anomalies"""
        results = {}
        
        for param, thresholds in self.parameter_thresholds.items():
            if param not in df.columns:
                continue
                
            # Check range violations
            below_min = df[df[param] < thresholds['min']][param].count()
            above_max = df[df[param] > thresholds['max']][param].count()
            
            # Check rate of change violations
            rate_change = df[param].diff().abs()
            rapid_changes = rate_change[rate_change > thresholds['rate_change']].count()
            
            results[param] = {
                'below_min': below_min,
                'above_max': above_max,
                'rapid_changes': rapid_changes
            }
        
        return results
    
    def check_timeliness(self, df: pd.DataFrame) -> Dict:
        """Check measurement timeliness and identify gaps"""
        df = df.sort_values('timestamp')
        time_diffs = df['timestamp'].diff()
        
        expected_diff = pd.Timedelta(self.expected_frequency)
        gaps = time_diffs[time_diffs > expected_diff]
        
        return {
            'total_gaps': len(gaps),
            'max_gap': gaps.max() if len(gaps) > 0 else pd.Timedelta('0H'),
            'avg_gap': gaps.mean() if len(gaps) > 0 else pd.Timedelta('0H'),
            'gap_periods': [(start.strftime('%Y-%m-%d %H:%M'), 
                           (start + diff).strftime('%Y-%m-%d %H:%M'))
                          for start, diff in zip(gaps.index, gaps)]
        }
    
    def generate_quality_report(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive data quality report"""
        report = {
            'completeness': self.check_completeness(df),
            'consistency': self.check_consistency(df),
            'timeliness': self.check_timeliness(df)
        }
        
        # Add quality score
        completeness_score = np.mean(list(report['completeness']['completeness_scores'].values()))
        consistency_score = self._calculate_consistency_score(report['consistency'])
        timeliness_score = self._calculate_timeliness_score(report['timeliness'])
        
        report['quality_scores'] = {
            'completeness': completeness_score,
            'consistency': consistency_score,
            'timeliness': timeliness_score,
            'overall': np.mean([completeness_score, consistency_score, timeliness_score])
        }
        
        return report
    
    def _calculate_consistency_score(self, consistency_results: Dict) -> float:
        """Calculate consistency score based on violations"""
        total_score = 0
        num_params = len(consistency_results)
        
        for param_results in consistency_results.values():
            violations = sum(param_results.values())
            param_score = max(0, 100 - violations)  # Deduct points for each violation
            total_score += param_score
        
        return total_score / num_params if num_params > 0 else 0
    
    def _calculate_timeliness_score(self, timeliness_results: Dict) -> float:
        """Calculate timeliness score based on gaps"""
        expected_records = pd.Timedelta('30D') / pd.Timedelta(self.expected_frequency)
        gap_penalty = timeliness_results['total_gaps'] / expected_records * 100
        
        return max(0, 100 - gap_penalty)
    
    def save_report(self, report: Dict, output_path: str):
        """Save quality report to file"""
        try:
            # Convert report to DataFrame for better formatting
            report_df = pd.DataFrame()
            
            # Completeness section
            completeness_df = pd.DataFrame(report['completeness']['completeness_scores'].items(),
                                         columns=['Parameter', 'Completeness Score'])
            
            # Consistency section
            consistency_data = []
            for param, results in report['consistency'].items():
                consistency_data.append({
                    'Parameter': param,
                    'Below Min': results['below_min'],
                    'Above Max': results['above_max'],
                    'Rapid Changes': results['rapid_changes']
                })
            consistency_df = pd.DataFrame(consistency_data)
            
            # Timeliness section
            timeliness_df = pd.DataFrame([{
                'Total Gaps': report['timeliness']['total_gaps'],
                'Max Gap': report['timeliness']['max_gap'],
                'Average Gap': report['timeliness']['avg_gap']
            }])
            
            # Quality scores
            scores_df = pd.DataFrame([report['quality_scores']])
            
            # Save to Excel with multiple sheets
            with pd.ExcelWriter(output_path) as writer:
                completeness_df.to_excel(writer, sheet_name='Completeness', index=False)
                consistency_df.to_excel(writer, sheet_name='Consistency', index=False)
                timeliness_df.to_excel(writer, sheet_name='Timeliness', index=False)
                scores_df.to_excel(writer, sheet_name='Quality Scores', index=False)
            
            logger.info(f"Quality report saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving quality report: {e}")
            raise