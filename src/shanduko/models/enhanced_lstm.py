# src/shanduko/models/enhanced_lstm.py

import torch
import torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger(__name__)

class EnhancedWaterQualityLSTM(nn.Module):
    def __init__(self, 
                 input_size=4,          # Default for water quality parameters
                 hidden_size=128,       # Increased from base model
                 num_layers=3,          # Deeper network
                 dropout=0.3,           # Increased dropout for regularization
                 bidirectional=True):   # Bidirectional for better context
        """
        Enhanced LSTM model for water quality prediction with attention mechanism
        and residual connections.
        
        Args:
            input_size (int): Number of input features (temp, pH, DO, turbidity)
            hidden_size (int): Number of hidden units in LSTM
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate for regularization
            bidirectional (bool): Whether to use bidirectional LSTM
        """
        super(EnhancedWaterQualityLSTM, self).__init__()
        
        # Save parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        
        # Batch normalization for input
        self.input_bn = nn.BatchNorm1d(input_size)
        
        # Main LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        attention_size = hidden_size * self.num_directions
        self.attention = nn.Sequential(
            nn.Linear(attention_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Fully connected layers with residual connections
        fc_input_size = hidden_size * self.num_directions
        self.fc_layers = nn.ModuleList([
            nn.Linear(fc_input_size, hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Linear(hidden_size // 2, hidden_size // 4)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size // 4, input_size)
        
        # Additional components
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.relu = nn.ReLU()
        
        logger.info(f"Initialized EnhancedWaterQualityLSTM with {num_layers} layers")
        
    def attention_net(self, lstm_output):
        """
        Apply attention mechanism to LSTM output
        
        Args:
            lstm_output: Output from LSTM layer
            
        Returns:
            context: Weighted sum of LSTM outputs
        """
        attention_weights = self.attention(lstm_output)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output predictions for water quality parameters
        """
        # Apply batch normalization to input
        batch_size, seq_len, features = x.size()
        x = x.view(-1, features)
        x = self.input_bn(x)
        x = x.view(batch_size, seq_len, features)
        
        # Initialize LSTM hidden state
        h0 = torch.zeros(self.num_layers * self.num_directions, 
                        x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, 
                        x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Apply attention
        context = self.attention_net(lstm_out)
        
        # Process through FC layers with residual connections
        out = context
        residual = out
        
        for i, fc_layer in enumerate(self.fc_layers):
            out = fc_layer(out)
            out = self.relu(out)
            out = self.dropout(out)
            
            # Add residual connection if shapes match
            if i == 0 and out.size() == residual.size():
                out = out + residual
                out = self.layer_norm(out)
        
        # Final output layer
        predictions = self.output_layer(out)
        
        return predictions
    
    def predict(self, x):
        """
        Make predictions on new data
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Predictions for water quality parameters
        """
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            predictions = self.forward(x)
            return predictions.cpu().numpy()
    
    def configure_optimizers(self, learning_rate=0.001):
        """
        Configure optimizers for training
        
        Args:
            learning_rate: Learning rate for optimizer
            
        Returns:
            Optimizer and learning rate scheduler
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        return optimizer, scheduler