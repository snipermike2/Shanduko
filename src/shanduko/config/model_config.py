# Create a file named src/shanduko/config/model_config.py

class ModelConfig:
    """Configuration parameters for water quality prediction model"""
    
    def __init__(self):
        # Model architecture
        self.input_size = 4          # Number of input features
        self.hidden_size = 64        # Size of LSTM hidden layers
        self.num_layers = 2          # Number of LSTM layers
        self.output_size = 4         # Number of output features
        
        # Training parameters
        self.learning_rate = 0.001
        self.batch_size = 32
        self.num_epochs = 100
        self.sequence_length = 24    # 24 hours of data for predictions
        
        # Data splitting
        self.train_split = 0.7
        self.validation_split = 0.15
        self.test_split = 0.15
        
        # Early stopping
        self.patience = 10
        self.min_delta = 1e-4