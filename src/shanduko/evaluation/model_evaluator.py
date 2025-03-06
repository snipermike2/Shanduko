"""
src/shanduko/evaluation/model_evaluator.py
Enhanced evaluation metrics for water quality prediction model
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple
import logging
from scipy import stats

class WaterQualityEvaluator:
    def __init__(self):
        """Initialize evaluator with parameter thresholds and quality standards"""
        self.parameter_ranges = {
            'temperature': {
                'min': 0, 'max': 40, 
                'critical_threshold': 35,
                'optimal_range': (20, 30),
                'warning_range': (15, 35)
            },
            'ph': {
                'min': 0, 'max': 14, 
                'critical_threshold': 9,
                'optimal_range': (6.5, 8.5),
                'warning_range': (6.0, 9.0)
            },
            'dissolved_oxygen': {
                'min': 0, 'max': 20, 
                'critical_threshold': 4,
                'optimal_range': (6.5, 8.0),
                'warning_range': (5.0, 9.0)
            },
            'turbidity': {
                'min': 0, 'max': 50, 
                'critical_threshold': 30,
                'optimal_range': (0, 5),
                'warning_range': (0, 10)
            }
        }
        
        # WHO and EPA water quality standards
        self.quality_standards = {
            'drinking_water': {
                'ph': (6.5, 8.5),
                'turbidity': (0, 1),
                'temperature': (10, 25),
                'dissolved_oxygen': (6, float('inf'))
            },
            'aquatic_life': {
                'ph': (6.5, 9.0),
                'turbidity': (0, 5),
                'temperature': (18, 32),
                'dissolved_oxygen': (5, float('inf'))
            },
            'irrigation': {
                'ph': (6.0, 8.5),
                'turbidity': (0, 10),
                'temperature': (15, 35),
                'dissolved_oxygen': (3, float('inf'))
            }
        }
        
        self.logger = logging.getLogger(__name__)
    
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate basic regression metrics
        
        Parameters:
            y_true: Ground truth values
            y_pred: Model predictions
        
        Returns:
            Dictionary of metrics for each parameter
        """
        metrics = {}
        parameters = ['temperature', 'ph', 'dissolved_oxygen', 'turbidity']
        
        for i, param in enumerate(parameters):
            metrics[param] = {
                'mse': mean_squared_error(y_true[:, i], y_pred[:, i]),
                'rmse': np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])),
                'mae': mean_absolute_error(y_true[:, i], y_pred[:, i]),
                'r2': r2_score(y_true[:, i], y_pred[:, i]),
                'mape': self._calculate_mape(y_true[:, i], y_pred[:, i])
            }
            
        return metrics
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error"""
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def evaluate_critical_predictions(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Evaluate predictions for critical values
        
        Parameters:
            y_true: Ground truth values
            y_pred: Model predictions
            
        Returns:
            Dictionary containing critical prediction metrics
        """
        critical_metrics = {}
        parameters = ['temperature', 'ph', 'dissolved_oxygen', 'turbidity']
        
        for i, param in enumerate(parameters):
            threshold = self.parameter_ranges[param]['critical_threshold']
            
            # Identify critical situations
            true_critical = y_true[:, i] > threshold
            pred_critical = y_pred[:, i] > threshold
            
            # Calculate metrics
            true_positives = np.sum((true_critical) & (pred_critical))
            false_positives = np.sum((~true_critical) & (pred_critical))
            false_negatives = np.sum((true_critical) & (~pred_critical))
            true_negatives = np.sum((~true_critical) & (~pred_critical))
            
            # Calculate rates
            sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            
            critical_metrics[param] = {
                'sensitivity': sensitivity,  # True Positive Rate
                'specificity': specificity,  # True Negative Rate
                'precision': precision,      # Positive Predictive Value
                'critical_f1': 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            }
            
        return critical_metrics
    
    def analyze_temporal_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   sequence_length: int) -> Dict:
        """
        Analyze prediction performance over different time horizons
        
        Parameters:
            y_true: Ground truth values
            y_pred: Model predictions
            sequence_length: Length of input sequences
            
        Returns:
            Dictionary containing temporal performance metrics
        """
        temporal_metrics = {}
        parameters = ['temperature', 'ph', 'dissolved_oxygen', 'turbidity']
        
        # Analyze predictions at different horizons
        horizons = [1, 6, 12, 24]  # hours ahead
        
        for horizon in horizons:
            if horizon >= len(y_true):
                continue
                
            horizon_metrics = {}
            for i, param in enumerate(parameters):
                rmse = np.sqrt(mean_squared_error(y_true[horizon:, i], y_pred[:-horizon, i]))
                mae = mean_absolute_error(y_true[horizon:, i], y_pred[:-horizon, i])
                
                horizon_metrics[param] = {
                    'rmse': rmse,
                    'mae': mae
                }
            
            temporal_metrics[f'{horizon}h'] = horizon_metrics
            
        return temporal_metrics
    
    def analyze_prediction_stability(self, y_pred: np.ndarray) -> Dict:
        """
        Analyze stability and consistency of predictions
        
        Parameters:
            y_pred: Model predictions
            
        Returns:
            Dictionary containing stability metrics
        """
        stability_metrics = {}
        parameters = ['temperature', 'ph', 'dissolved_oxygen', 'turbidity']
        
        for i, param in enumerate(parameters):
            # Calculate prediction volatility
            volatility = np.std(np.diff(y_pred[:, i]))
            
            # Calculate extreme predictions
            param_range = self.parameter_ranges[param]
            extreme_pred_ratio = np.mean(
                (y_pred[:, i] < param_range['min']) | 
                (y_pred[:, i] > param_range['max'])
            )
            
            # Analyze prediction distribution
            pred_skew = stats.skew(y_pred[:, i])
            pred_kurtosis = stats.kurtosis(y_pred[:, i])
            
            stability_metrics[param] = {
                'volatility': volatility,
                'extreme_prediction_ratio': extreme_pred_ratio,
                'distribution_skew': pred_skew,
                'distribution_kurtosis': pred_kurtosis
            }
            
        return stability_metrics
    
    def evaluate_water_quality_standards(self, y_pred: np.ndarray) -> Dict:
        """
        Evaluate predictions against established water quality standards
        
        Parameters:
            y_pred: Model predictions
            
        Returns:
            Dictionary containing compliance metrics for different water uses
        """
        standards_compliance = {}
        parameters = ['temperature', 'ph', 'dissolved_oxygen', 'turbidity']
        
        for use_case, standards in self.quality_standards.items():
            compliance_metrics = {}
            
            for i, param in enumerate(parameters):
                if param in standards:
                    min_val, max_val = standards[param]
                    values_in_range = np.logical_and(
                        y_pred[:, i] >= min_val,
                        y_pred[:, i] <= max_val
                    )
                    compliance_rate = np.mean(values_in_range)
                    
                    compliance_metrics[param] = {
                        'compliance_rate': compliance_rate,
                        'mean_deviation': np.mean(np.abs(y_pred[:, i] - np.clip(
                            y_pred[:, i], min_val, max_val
                        )))
                    }
            
            standards_compliance[use_case] = compliance_metrics
            
        return standards_compliance
    
    def evaluate_ecological_impact(self, y_pred: np.ndarray) -> Dict:
        """
        Evaluate predictions for ecological impact assessment
        
        Parameters:
            y_pred: Model predictions
            
        Returns:
            Dictionary containing ecological impact metrics
        """
        ecological_metrics = {}
        parameters = ['temperature', 'ph', 'dissolved_oxygen', 'turbidity']
        
        for i, param in enumerate(parameters):
            param_range = self.parameter_ranges[param]
            optimal_min, optimal_max = param_range['optimal_range']
            warning_min, warning_max = param_range['warning_range']
            
            # Calculate time in different zones
            optimal_time = np.mean(np.logical_and(
                y_pred[:, i] >= optimal_min,
                y_pred[:, i] <= optimal_max
            ))
            
            warning_time = np.mean(np.logical_and(
                y_pred[:, i] >= warning_min,
                y_pred[:, i] <= warning_max
            )) - optimal_time
            
            critical_time = 1 - (optimal_time + warning_time)
            
            # Calculate stress indicators
            stress_score = np.mean(np.clip(
                np.abs(y_pred[:, i] - np.mean(param_range['optimal_range'])) /
                (param_range['max'] - param_range['min']),
                0, 1
            ))
            
            ecological_metrics[param] = {
                'optimal_time_ratio': optimal_time,
                'warning_time_ratio': warning_time,
                'critical_time_ratio': critical_time,
                'stress_score': stress_score
            }
            
        return ecological_metrics
    
    def evaluate_treatment_requirements(self, y_pred: np.ndarray) -> Dict:
        """
        Evaluate predictions for water treatment requirements
        
        Parameters:
            y_pred: Model predictions
            
        Returns:
            Dictionary containing treatment requirement metrics
        """
        treatment_metrics = {}
        parameters = ['temperature', 'ph', 'dissolved_oxygen', 'turbidity']
        
        for i, param in enumerate(parameters):
            # Calculate treatment intensity requirements
            param_range = self.parameter_ranges[param]
            optimal_min, optimal_max = param_range['optimal_range']
            
            # Calculate deviation from optimal range
            deviations = np.maximum(
                0,
                np.maximum(
                    optimal_min - y_pred[:, i],
                    y_pred[:, i] - optimal_max
                )
            )
            
            # Calculate treatment intensity scores
            treatment_intensity = np.mean(deviations / (param_range['max'] - param_range['min']))
            
            # Calculate treatment frequency requirement
            treatment_frequency = np.mean(deviations > 0)
            
            treatment_metrics[param] = {
                'treatment_intensity': treatment_intensity,
                'treatment_frequency': treatment_frequency,
                'max_deviation': np.max(deviations),
                'sustained_deviation': np.mean(
                    np.convolve(deviations > 0, np.ones(24)/24, mode='valid')
                )
            }
            
        return treatment_metrics

    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      sequence_length: int) -> Dict:
        """
        Perform comprehensive model evaluation
        
        Parameters:
            y_true: Ground truth values
            y_pred: Model predictions
            sequence_length: Length of input sequences
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        try:
            evaluation_results = {
                'basic_metrics': self.calculate_basic_metrics(y_true, y_pred),
                'critical_metrics': self.evaluate_critical_predictions(y_true, y_pred),
                'temporal_metrics': self.analyze_temporal_performance(y_true, y_pred, sequence_length),
                'stability_metrics': self.analyze_prediction_stability(y_pred),
                'water_quality_standards': self.evaluate_water_quality_standards(y_pred),
                'ecological_impact': self.evaluate_ecological_impact(y_pred),
                'treatment_requirements': self.evaluate_treatment_requirements(y_pred)
            }
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error during model evaluation: {str(e)}")
            raise
    
    def generate_evaluation_summary(self, evaluation_results: Dict) -> str:
        """
        Generate human-readable summary of evaluation results
        
        Parameters:
            evaluation_results: Dictionary of evaluation metrics
            
        Returns:
            Formatted string containing evaluation summary
        """
        summary = []
        summary.append("Model Evaluation Summary")
        summary.append("=" * 50)
        
        # Basic metrics summary
        summary.append("\nBasic Performance Metrics:")
        for param, metrics in evaluation_results['basic_metrics'].items():
            summary.append(f"\n{param.title()}:")
            summary.append(f"  RMSE: {metrics['rmse']:.4f}")
            summary.append(f"  R²: {metrics['r2']:.4f}")
            summary.append(f"  MAPE: {metrics['mape']:.2f}%")
        
        # Critical predictions summary
        summary.append("\nCritical Event Detection:")
        for param, metrics in evaluation_results['critical_metrics'].items():
            summary.append(f"\n{param.title()}:")
            summary.append(f"  Sensitivity: {metrics['sensitivity']:.2%}")
            summary.append(f"  Precision: {metrics['precision']:.2%}")
            summary.append(f"  F1 Score: {metrics['critical_f1']:.2%}")
        
        # Temporal performance summary
        summary.append("\nTemporal Performance:")
        for horizon, metrics in evaluation_results['temporal_metrics'].items():
            summary.append(f"\n{horizon} Prediction:")
            for param, values in metrics.items():
                summary.append(f"  {param.title()} - RMSE: {values['rmse']:.4f}")
        
        # Stability metrics summary
        summary.append("\nPrediction Stability:")
        for param, metrics in evaluation_results['stability_metrics'].items():
            summary.append(f"\n{param.title()}:")
            summary.append(f"  Volatility: {metrics['volatility']:.4f}")
            summary.append(f"  Extreme Predictions: {metrics['extreme_prediction_ratio']:.2%}")
        
        return "\n".join(summary)

# Example usage
if __name__ == "__main__":
    # Create sample data
    y_true = np.random.normal(size=(100, 4))
    y_pred = y_true + np.random.normal(0, 0.1, size=(100, 4))
    
    # Initialize evaluator
    evaluator = WaterQualityEvaluator()
    
    # Run evaluation
    results = evaluator.evaluate_model(y_true, y_pred, sequence_length=24)
    
    # Print summary
    print(evaluator.generate_evaluation_summary(results))