#src/models/prediction_system.py
import sys
import os
from pathlib import Path
import torch
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from collections import deque

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from src.shanduko.models.water_quality_lstm import WaterQualityLSTM

class WaterQualityPredictor:
    def __init__(self, model_path, sequence_length=24):
        """
        Initialize the prediction system
        Args:
            model_path: Path to saved model checkpoint
            sequence_length: Length of input sequence for prediction
        """
        self.sequence_length = sequence_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize logger
        self.setup_logger()
        
        # Load model
        self.load_model(model_path)
        
        # Initialize data buffer for real-time prediction
        self.data_buffer = deque(maxlen=sequence_length)
        
        # Parameter ranges for validation
        self.param_ranges = {
            'temperature': (0, 40),     # Celsius
            'ph': (0, 14),             # pH scale
            'dissolved_oxygen': (0, 20), # mg/L
            'turbidity': (0, 50)        # NTU
        }
        
    def setup_logger(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger('WaterQualityPredictor')
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler('prediction_log.log')
        
        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)
    
    def load_model(self, model_path):
        """Load the trained model with improved error handling"""
        try:
            # Check if file exists
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Check for required keys in checkpoint
            required_keys = ['model_state_dict']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            if missing_keys:
                raise KeyError(f"Checkpoint is missing required keys: {missing_keys}")
            
            # Get model configuration
            if 'config' in checkpoint:
                config = checkpoint['config']
                self.logger.info(f"Loaded model config: {config}")
            else:
                # Use default configuration if not found
                self.logger.warning("No config found in checkpoint, using default values")
                config = {
                    'input_size': 4,
                    'hidden_size': 64,
                    'num_layers': 2,
                    'output_size': 4
                }
            
            # Extract configuration safely
            input_size = config.get('input_size', 4)
            hidden_size = config.get('hidden_size', 64)
            num_layers = config.get('num_layers', 2)
            output_size = config.get('output_size', 4)
            
            # Initialize model
            self.model = WaterQualityLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=output_size
            ).to(self.device)
            
            # Load weights
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            except Exception as e:
                self.logger.error(f"Error loading model weights: {e}")
                raise ValueError(f"Failed to load model weights: {e}")
            
            # Set to evaluation mode
            self.model.eval()
            
            self.logger.info(f"Model loaded successfully from {model_path}")
            
        except FileNotFoundError as e:
            self.logger.error(f"Model file not found: {e}")
            raise
        except KeyError as e:
            self.logger.error(f"Invalid checkpoint format: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def validate_input(self, data):
        """Validate input data ranges"""
        if len(data) != 4:
            raise ValueError(f"Expected 4 parameters, got {len(data)}")
        
        params = ['temperature', 'ph', 'dissolved_oxygen', 'turbidity']
        for value, param in zip(data, params):
            min_val, max_val = self.param_ranges[param]
            if not min_val <= value <= max_val:
                raise ValueError(f"{param} value {value} outside valid range [{min_val}, {max_val}]")
    
    def add_reading(self, temperature, ph, dissolved_oxygen, turbidity):
        """
        Add a new sensor reading to the buffer
        Returns True if buffer is full and ready for prediction
        """
        try:
            data = [temperature, ph, dissolved_oxygen, turbidity]
            self.validate_input(data)
            
            self.data_buffer.append(data)
            self.logger.debug(f"Added reading: {data}")
            
            return len(self.data_buffer) == self.sequence_length
        except Exception as e:
            self.logger.error(f"Error adding reading: {str(e)}")
            raise
    
    def make_prediction(self):
        """
        Make prediction using current buffer
        Returns: Dictionary with predictions and confidence scores
        """
        try:
            if len(self.data_buffer) < self.sequence_length:
                raise ValueError(f"Not enough data. Need {self.sequence_length} readings, have {len(self.data_buffer)}")
            
            # Prepare input sequence
            sequence = torch.FloatTensor(list(self.data_buffer)).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                prediction = self.model(sequence).cpu().numpy()[0]
            
            # Format prediction results
            params = ['temperature', 'ph', 'dissolved_oxygen', 'turbidity']
            predictions = {}
            for param, value in zip(params, prediction):
                predictions[param] = {
                    'value': float(value),
                    'timestamp': datetime.now().isoformat(),
                    'is_anomaly': not (self.param_ranges[param][0] <= value <= self.param_ranges[param][1])
                }
            
            self.logger.info(f"Prediction made: {predictions}")
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def get_anomaly_status(self, predictions):
        """Check if any predicted values are anomalous"""
        anomalies = {}
        for param, pred in predictions.items():
            if pred['is_anomaly']:
                anomalies[param] = pred['value']
        return anomalies if anomalies else None

def main():
    # Example usage
    model_path = 'checkpoints/best_model.pth'
    predictor = WaterQualityPredictor(model_path)
    
    # Simulate real-time data
    for _ in range(24):  # Add 24 readings
        temperature = np.random.normal(25, 2)
        ph = np.random.normal(7, 0.5)
        dissolved_oxygen = np.random.normal(8, 1)
        turbidity = np.random.normal(3, 0.5)
        
        is_ready = predictor.add_reading(temperature, ph, dissolved_oxygen, turbidity)
        
        if is_ready:
            # Make prediction
            predictions = predictor.make_prediction()
            
            # Check for anomalies
            anomalies = predictor.get_anomaly_status(predictions)
            if anomalies:
                print(f"ALERT: Anomalies detected: {anomalies}")
            
            print("\nPredictions:")
            for param, pred in predictions.items():
                print(f"{param}: {pred['value']:.2f}")

if __name__ == "__main__":
    main()