#src/shanduko/models/explore_data.py
import sys
import os
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

import numpy as np
import matplotlib.pyplot as plt
from src.shanduko.models.train_water_quality import generate_synthetic_data, prepare_sequences
from src.shanduko.models.water_quality_lstm import WaterQualityLSTM

def explore_synthetic_data():
    """
    Explore and visualize the synthetic water quality data
    """
    # Generate a small sample of data
    print("Generating synthetic data...")
    synthetic_data = generate_synthetic_data(num_days=10)
    
    # Print basic information
    print("\nData Information:")
    print("Data shape:", synthetic_data.shape)
    print("Sample data point (Temperature, pH, DO, Turbidity):", synthetic_data[0])
    
    # Visualize 24 hours of data
    plt.figure(figsize=(12, 8))
    
    # Plot each parameter
    parameters = ['Temperature', 'pH', 'Dissolved Oxygen', 'Turbidity']
    for i, param in enumerate(parameters):
        plt.subplot(2, 2, i+1)
        plt.plot(synthetic_data[:24, i], label=param)
        plt.title(f'24 Hours of {param}')
        plt.xlabel('Hour')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Show all parameters on one plot
    plt.figure(figsize=(12, 6))
    for i, param in enumerate(parameters):
        plt.plot(synthetic_data[:24, i], label=param)
    plt.title('24 Hours of Water Quality Parameters')
    plt.xlabel('Hour')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return synthetic_data

def test_data_preparation():
    """
    Test the sequence preparation for LSTM
    """
    # Generate data and prepare sequences
    synthetic_data = generate_synthetic_data(num_days=5)
    X, y = prepare_sequences(synthetic_data, sequence_length=24)
    
    print("\nSequence Information:")
    print("Input sequence shape:", X.shape)
    print("Target shape:", y.shape)
    print("\nExample sequence and target:")
    print("Sequence first timepoint:", X[0][0])
    print("Corresponding target:", y[0])

if __name__ == "__main__":
    print("Starting data exploration...")
    
    try:
        # Explore synthetic data
        synthetic_data = explore_synthetic_data()
        
        # Test sequence preparation
        test_data_preparation()
        
        print("\nData exploration completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()