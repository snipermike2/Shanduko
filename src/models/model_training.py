#src/models/model_training.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging

from src.shanduko.models.water_quality_lstm import WaterQualityLSTM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WaterQualityTrainer:
    def __init__(self, model=None, sequence_length=24, learning_rate=0.001):
        """
        Initialize the trainer for water quality prediction model.
        
        Parameters:
        - model: Optional pre-initialized model
        - sequence_length: Length of input sequences
        - learning_rate: Learning rate for optimization
        """
        self.sequence_length = sequence_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model if not provided
        self.model = model if model else WaterQualityLSTM().to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
    def prepare_sequences(self, data):
        """
        Convert time series data into sequences for training.
        
        Parameters:
        - data: numpy array of shape (n_samples, n_features)
        
        Returns:
        - Tuple of (sequences, targets)
        """
        sequences = []
        targets = []
        
        for i in range(len(data) - self.sequence_length):
            seq = data[i:i + self.sequence_length]
            target = data[i + self.sequence_length]
            sequences.append(seq)
            targets.append(target)
            
        return torch.FloatTensor(sequences), torch.FloatTensor(targets)
    
    def train_epoch(self, train_loader, val_loader=None):
        """
        Train the model for one epoch.
        
        Parameters:
        - train_loader: DataLoader for training data
        - val_loader: Optional DataLoader for validation data
        
        Returns:
        - Dictionary containing training and validation losses
        """
        self.model.train()
        total_loss = 0
        
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_train_loss)
        
        # Validation
        val_loss = None
        if val_loader:
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
        
        return {
            'train_loss': avg_train_loss,
            'val_loss': val_loss
        }
    
    def validate(self, val_loader):
        """
        Validate the model on validation data.
        
        Parameters:
        - val_loader: DataLoader for validation data
        
        Returns:
        - Average validation loss
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(sequences)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def save_model(self, path):
        """Save model and training history"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model and training history"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        logger.info(f"Model loaded from {path}") 