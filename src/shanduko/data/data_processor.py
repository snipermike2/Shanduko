"""
src/shanduko/data/data_processor.py
Data processing module for water quality monitoring
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Optional, Dict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class WaterQualityDataProcessor:
    def __init__(self):
        """Initialize data processor with parameter ranges for validation"""
        self.parameter_ranges = {
            'temperature': (0, 40),     # Celsius
            'ph': (0, 14),             # pH scale
            'dissolved_oxygen': (0, 20), # mg/L
            'turbidity': (0, 50)        # NTU
        }
        
         # Add rate of change thresholds
        self.rate_thresholds = {
            'temperature': 5.0,      # °C per hour
            'ph': 1.0,              # pH units per hour
            'dissolved_oxygen': 2.0, # mg/L per hour
            'turbidity': 10.0       # NTU per hour
        }
        
        # Configure logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        
    def load_csv_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and validate CSV data
        
        Parameters:
            file_path: Path to CSV file
            
        Returns:
            Processed DataFrame
        """
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Check required columns
            required_columns = ['timestamp', 'temperature', 'ph', 'dissolved_oxygen', 'turbidity']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['timestamp'])
            
            # Validate data ranges
            self._validate_data_ranges(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
    
    def _validate_data_ranges(self, df: pd.DataFrame) -> None:
        """
        Validate that all parameters are within expected ranges
        """
        for param, (min_val, max_val) in self.parameter_ranges.items():
            invalid_mask = (df[param] < min_val) | (df[param] > max_val)
            if invalid_mask.any():
                invalid_count = invalid_mask.sum()
                logger.warning(f"Found {invalid_count} invalid values for {param}")
                
                # Replace invalid values with NaN
                df.loc[invalid_mask, param] = np.nan
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Parameters:
            df: Input DataFrame
            method: Method to handle missing values ('interpolate' or 'drop')
            
        Returns:
            DataFrame with handled missing values
        """
        if method == 'interpolate':
            # Interpolate missing values using time-aware interpolation
            df = df.set_index('timestamp')
            df = df.interpolate(method='time')
            df = df.reset_index()
        elif method == 'drop':
            # Drop rows with any missing values
            df = df.dropna()
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        return df
    
    def normalize_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        Normalize data to range [0, 1]
        
        Returns:
            Tuple of (normalized DataFrame, normalization parameters)
        """
        norm_params = {}
        df_norm = df.copy()
        
        for param in self.parameter_ranges.keys():
            min_val = df[param].min()
            max_val = df[param].max()
            
            # Handle case where min and max are equal (constant values)
            if min_val == max_val:
                self.logger.warning(f"Parameter {param} has constant value {min_val}. Setting normalized value to 0.5.")
                df_norm[param] = 0.5  # Set to middle of normalized range
                # Store parameters for denormalization
                norm_params[param] = {
                    'min': min_val,
                    'max': min_val + 1.0,  # Add a small range to avoid division by zero
                    'is_constant': True
                }
            else:
                # Store normalization parameters
                norm_params[param] = {
                    'min': min_val,
                    'max': max_val,
                    'is_constant': False
                }
                
                # Normalize data
                df_norm[param] = (df[param] - min_val) / (max_val - min_val)
        
        return df_norm, norm_params
    
    def denormalize_data(self, df_norm: pd.DataFrame, norm_params: dict) -> pd.DataFrame:
        """
        Denormalize data back to original range
        
        Parameters:
            df_norm: Normalized DataFrame
            norm_params: Normalization parameters from normalize_data
            
        Returns:
            Denormalized DataFrame
        """
        df_denorm = df_norm.copy()
        
        for param, params in norm_params.items():
            min_val = params['min']
            max_val = params['max']
            is_constant = params.get('is_constant', False)
            
            if is_constant:
                # For constant parameters, just set back to the original value
                df_denorm[param] = min_val
            else:
                # Denormalize back to original range
                df_denorm[param] = df_norm[param] * (max_val - min_val) + min_val
        
        return df_denorm
    
    def prepare_sequences(
        self, 
        df: pd.DataFrame, 
        sequence_length: int = 24,
        target_columns: Optional[list] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training
        
        Parameters:
            df: Input DataFrame
            sequence_length: Length of input sequences
            target_columns: List of columns to predict (defaults to all parameter columns)
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        if target_columns is None:
            target_columns = list(self.parameter_ranges.keys())
        
        # Extract features
        feature_columns = list(self.parameter_ranges.keys())
        data = df[feature_columns].values
        
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        
        return np.array(X), np.array(y)
    
    def save_processed_data(
        self, 
        df: pd.DataFrame, 
        output_path: str,
        norm_params: Optional[dict] = None
    ) -> None:
        """Save processed data and normalization parameters"""
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save processed data
        df.to_csv(output_path, index=False)
        
        # Save normalization parameters if provided
        if norm_params:
            norm_path = output_dir / 'normalization_params.csv'
            pd.DataFrame(norm_params).to_csv(norm_path)
            
        logger.info(f"Saved processed data to {output_path}")
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Perform quality checks on the data
        
        Returns:
            Dictionary containing quality metrics
        """
        quality_metrics = {
            'total_rows': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'outliers': self._detect_outliers(df),
            'rapid_changes': self._detect_rapid_changes(df)
        }
        
        return quality_metrics
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict:
        """Detect outliers using IQR method"""
        outliers = {}
        for param in self.parameter_ranges.keys():
            q1 = df[param].quantile(0.25)
            q3 = df[param].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers[param] = len(df[(df[param] < lower_bound) | (df[param] > upper_bound)])
        
        return outliers
    
    def _detect_rapid_changes(self, df: pd.DataFrame) -> Dict:
        """Detect rapid changes that exceed thresholds"""
        rapid_changes = {}
        for param, threshold in self.rate_thresholds.items():
            changes = df[param].diff().abs()
            rapid_changes[param] = len(changes[changes > threshold])
        
        return rapid_changes
    
    def generate_synthetic_data(
        self, 
        num_days: int = 30, 
        sampling_rate: str = '1H'
    ) -> pd.DataFrame:
        """
        Generate synthetic data for testing or augmentation
        
        Parameters:
            num_days: Number of days of data to generate
            sampling_rate: Data sampling frequency
        """
        # Generate timestamps
        timestamps = pd.date_range(
            start=datetime.now() - pd.Timedelta(days=num_days),
            end=datetime.now(),
            freq=sampling_rate
        )
        
        # Generate synthetic measurements with realistic patterns
        data = {
            'timestamp': timestamps,
            'temperature': [
                25 + 5 * np.sin(i/24 * 2*np.pi) + np.random.normal(0, 0.5)
                for i in range(len(timestamps))
            ],
            'ph': [
                7 + 0.5 * np.sin(i/24 * 2*np.pi) + np.random.normal(0, 0.1)
                for i in range(len(timestamps))
            ],
            'dissolved_oxygen': [
                8 + 2 * np.sin(i/24 * 2*np.pi) + np.random.normal(0, 0.3)
                for i in range(len(timestamps))
            ],
            'turbidity': [
                3 + np.sin(i/24 * 2*np.pi) + np.random.normal(0, 0.2)
                for i in range(len(timestamps))
            ]
        }
        
        return pd.DataFrame(data)
    
    def augment_data(
        self, 
        df: pd.DataFrame, 
        noise_level: float = 0.1
    ) -> pd.DataFrame:
        """
        Augment data by adding noise and variations
        
        Parameters:
            df: Input DataFrame
            noise_level: Level of noise to add (0-1)
        """
        df_augmented = df.copy()
        
        for param in self.parameter_ranges.keys():
            # Add random noise
            noise = np.random.normal(
                0, 
                noise_level * (self.parameter_ranges[param][1] - self.parameter_ranges[param][0]),
                len(df)
            )
            df_augmented[param] += noise
            
            # Ensure values stay within valid ranges
            df_augmented[param] = df_augmented[param].clip(
                self.parameter_ranges[param][0],
                self.parameter_ranges[param][1]
            )
        
        return df_augmented