# src/shanduko/data/data_collector.py

import serial
import time
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, Optional, List
import json

class WaterQualitySensor:
    def __init__(self, port: str, parameter: str, sampling_rate: int = 60):
        """
        Initialize sensor connection
        
        Args:
            port: Serial port (e.g., 'COM1', '/dev/ttyUSB0')
            parameter: Parameter being measured (e.g., 'temperature')
            sampling_rate: Sampling rate in seconds
        """
        self.port = port
        self.parameter = parameter
        self.sampling_rate = sampling_rate
        self.connection = None
        self.logger = logging.getLogger(f"{parameter}_sensor")
        
    def connect(self) -> bool:
        """Establish connection with sensor"""
        try:
            self.connection = serial.Serial(
                port=self.port,
                baudrate=9600,
                timeout=1.0
            )
            self.logger.info(f"Connected to {self.parameter} sensor on {self.port}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to sensor: {e}")
            return False
            
    def read_data(self) -> Optional[float]:
        """Read data from sensor"""
        if not self.connection:
            return None
            
        try:
            # Read raw data from sensor
            raw_data = self.connection.readline().decode().strip()
            # Convert to float and validate
            value = float(raw_data)
            return value
        except Exception as e:
            self.logger.error(f"Error reading sensor data: {e}")
            return None

class WaterQualityDataCollector:
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize data collector
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger("DataCollector")
        self.sensors: Dict[str, WaterQualitySensor] = {}
        self.data_buffer: List[Dict] = []
        self.config = self._load_config(config_path)
        self.setup_sensors()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            'sensors': {
                'temperature': {'port': 'COM1', 'sampling_rate': 60},
                'ph': {'port': 'COM2', 'sampling_rate': 60},
                'dissolved_oxygen': {'port': 'COM3', 'sampling_rate': 60},
                'turbidity': {'port': 'COM4', 'sampling_rate': 60}
            },
            'data_dir': 'data/raw',
            'buffer_size': 1000
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return default_config
        
    def setup_sensors(self):
        """Initialize all sensors"""
        for param, settings in self.config['sensors'].items():
            sensor = WaterQualitySensor(
                port=settings['port'],
                parameter=param,
                sampling_rate=settings['sampling_rate']
            )
            self.sensors[param] = sensor
            
    def connect_sensors(self) -> bool:
        """Connect to all sensors"""
        success = True
        for param, sensor in self.sensors.items():
            if not sensor.connect():
                self.logger.error(f"Failed to connect to {param} sensor")
                success = False
        return success
        
    def collect_data(self) -> Dict[str, float]:
        """Collect one reading from all sensors"""
        reading = {
            'timestamp': datetime.now().isoformat()
        }
        
        for param, sensor in self.sensors.items():
            value = sensor.read_data()
            if value is not None:
                reading[param] = value
            else:
                self.logger.warning(f"Failed to read {param}")
                reading[param] = np.nan
                
        return reading
        
    def start_collection(self, duration: Optional[int] = None):
        """
        Start continuous data collection
        
        Args:
            duration: Optional duration in seconds (None for infinite)
        """
        try:
            self.logger.info("Starting data collection...")
            start_time = time.time()
            
            while True:
                # Check duration
                if duration and (time.time() - start_time) > duration:
                    break
                    
                # Collect data
                reading = self.collect_data()
                self.data_buffer.append(reading)
                
                # Save buffer if full
                if len(self.data_buffer) >= self.config['buffer_size']:
                    self.save_data()
                    
                # Wait for next collection
                time.sleep(min(s.sampling_rate for s in self.sensors.values()))
                
        except KeyboardInterrupt:
            self.logger.info("Data collection stopped by user")
        except Exception as e:
            self.logger.error(f"Error during data collection: {e}")
        finally:
            self.save_data()  # Save any remaining data
            
    def save_data(self):
        """Save collected data to CSV"""
        if not self.data_buffer:
            return
            
        try:
            # Create data directory if needed
            data_dir = Path(self.config['data_dir'])
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert buffer to DataFrame
            df = pd.DataFrame(self.data_buffer)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = data_dir / f"water_quality_data_{timestamp}.csv"
            
            # Save to CSV
            df.to_csv(filename, index=False)
            self.logger.info(f"Data saved to {filename}")
            
            # Clear buffer
            self.data_buffer = []
            
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize collector
    collector = WaterQualityDataCollector()
    
    # Connect to sensors
    if collector.connect_sensors():
        # Start collection (run for 1 hour as example)
        collector.start_collection(duration=3600)
    else:
        logging.error("Failed to connect to sensors. Exiting.")

if __name__ == "__main__":
    main()