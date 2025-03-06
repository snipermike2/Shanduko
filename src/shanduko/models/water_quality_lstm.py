"""
src/shanduko/models/water_quality_lstm.py
- Contains the WaterQualityLSTM class, which is an LSTM model for water quality prediction.
LSTM model for water quality prediction
"""
import torch
import torch.nn as nn

class WaterQualityLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, output_size=4):
        """
        Initialize the LSTM model for water quality prediction.
        
        Parameters:
        - input_size (int): Number of input features (temp, pH, dissolved oxygen, turbidity)
        - hidden_size (int): Number of features in the hidden state
        - num_layers (int): Number of LSTM layers
        - output_size (int): Number of output features (predictions)
        """
        super(WaterQualityLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        """Forward pass"""
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get output from last time step
        out = out[:, -1, :]
        
        # Pass through fully connected layers
        out = self.fc(out)
        return out

    def predict(self, x):
        """
        Make predictions with the model.
        
        Parameters:
        - x: Input tensor of shape (batch_size, sequence_length, input_size)
        
        Returns:
        - Predictions as numpy array
        """
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            # Ensure input is a tensor and on the correct device
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float)
            
            # Move input to the same device as the model
            x = x.to(next(self.parameters()).device)
            
            # Forward pass
            predictions = self.forward(x)
            
            # Move predictions to CPU before converting to numpy
            return predictions.cpu().numpy()