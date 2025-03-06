# init_model.py
"""
Initialize a baseline model for the Shanduko water quality monitoring system.
This script creates a simple LSTM model and saves it to the expected path.
"""

import os
import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Import model classes
try:
    from src.shanduko.models.water_quality_lstm import WaterQualityLSTM
except ImportError:
    print("Error importing WaterQualityLSTM. Checking paths...")
    print(f"Current directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    sys.exit(1)

def init_model():
    """Create and save a baseline water quality prediction model"""
    print("Initializing baseline water quality prediction model...")
    
    # Create checkpoints directory
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model with default parameters
    model = WaterQualityLSTM(
        input_size=4,      # temperature, pH, dissolved oxygen, turbidity
        hidden_size=64,    # size of LSTM hidden layer
        num_layers=2,      # number of LSTM layers
        output_size=4      # prediction for each parameter
    )
    
    # Create config dictionary for the checkpoint
    config = {
        'input_size': 4,
        'hidden_size': 64,
        'num_layers': 2,
        'output_size': 4
    }
    
    # Save model to checkpoint file
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'timestamp': '2025-02-24'  # Current date
    }
    
    checkpoint_path = checkpoint_dir / "best_model.pth"
    torch.save(checkpoint, checkpoint_path)
    
    print(f"Model saved successfully to {checkpoint_path}")
    return checkpoint_path

if __name__ == "__main__":
    checkpoint_path = init_model()
    print(f"Baseline model created at: {checkpoint_path}")
    print("Now you can run the application without model loading errors.")