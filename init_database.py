# init_database.py
"""
Initialize the database for Shanduko water quality monitoring system.
This script ensures the database is properly set up before running the application.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

try:
    from src.shanduko.database.database import init_db, get_db, SensorReading
    from datetime import datetime, timedelta
    import random
except ImportError as e:
    print(f"Error importing database modules: {e}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    sys.exit(1)

def initialize_database():
    """Initialize the database and add some sample data"""
    try:
        print("Initializing database...")
        init_db()
        print("Database initialized successfully!")
        
        # Add some sample data
        add_sample_data()
        
        print("Database setup complete!")
        return True
    except Exception as e:
        print(f"Error initializing database: {e}")
        return False

def add_sample_data(num_samples=24):
    """Add sample data to the database for testing"""
    try:
        print(f"Adding {num_samples} sample readings to database...")
        
        # Generate sample data
        current_time = datetime.now()
        sample_readings = []
        
        for i in range(num_samples):
            # Create data point with realistic values and some variation
            timestamp = current_time - timedelta(hours=num_samples-i)
            
            # Base values with daily cycle patterns
            hour_of_day = timestamp.hour
            base_temp = 25 + 2 * (hour_of_day / 12 - 1)**2  # Peak at noon
            base_ph = 7.0 + 0.3 * (hour_of_day / 12 - 1)
            base_do = 8.0 - 0.1 * base_temp  # Inverse relationship with temperature
            base_turb = 3.0 + (hour_of_day % 6) / 10  # Small variations
            
            # Add some noise
            reading = SensorReading(
                temperature=base_temp + random.uniform(-0.5, 0.5),
                ph_level=base_ph + random.uniform(-0.1, 0.1),
                dissolved_oxygen=base_do + random.uniform(-0.2, 0.2),
                turbidity=base_turb + random.uniform(-0.3, 0.3),
                timestamp=timestamp
            )
            sample_readings.append(reading)
        
        # Add to database
        with get_db() as db:
            for reading in sample_readings:
                db.add(reading)
            db.commit()
            
        print(f"Added {num_samples} sample readings to database!")
        return True
    except Exception as e:
        print(f"Error adding sample data: {e}")
        return False

if __name__ == "__main__":
    initialize_database()