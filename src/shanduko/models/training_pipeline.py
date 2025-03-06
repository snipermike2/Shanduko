"""
src/shanduko/models/training_pipeline.py
Training module for water quality prediction model
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Optional, Dict, Tuple

from src.shanduko.models.water_quality_lstm import WaterQualityLSTM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WaterQualityTrainer:
    def __init__(
        self, 
        model: Optional[WaterQualityLSTM] = None,
        sequence_length: int = 24,
        learning_rate: float = 0.001,
        device: Optional[str] = None
    ):
        """Initialize the trainer for water quality prediction model."""
        self.sequence_length = sequence_length
        self.device = (device if device else 
                      torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize model if not provided
        self.model = model if model else WaterQualityLSTM()
        self.model = self.model.to(self.device)
        
        # Initialize optimizer and loss function later
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def initialize_optimizer(self) -> None:
        """Initialize or reset the optimizer."""
        if self.optimizer is None:
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.learning_rate
            )
    
    def prepare_sequences(
        self, 
        data: np.ndarray,
        validation_split: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert time series data into sequences for training.
        
        Parameters:
        - data: numpy array of shape (n_samples, n_features)
        - validation_split: fraction of data to use for validation
        
        Returns:
        - Tuple of (train_sequences, train_targets, val_sequences, val_targets)
        """
        sequences = []
        targets = []
        
        try:
            for i in range(len(data) - self.sequence_length):
                seq = data[i:i + self.sequence_length]
                target = data[i + self.sequence_length]
                sequences.append(seq)
                targets.append(target)
            
            sequences = torch.FloatTensor(sequences)
            targets = torch.FloatTensor(targets)
            
            # Split into train and validation
            split_idx = int(len(sequences) * (1 - validation_split))
            train_sequences = sequences[:split_idx]
            train_targets = targets[:split_idx]
            val_sequences = sequences[split_idx:]
            val_targets = targets[split_idx:]
            
            return train_sequences, train_targets, val_sequences, val_targets
            
        except Exception as e:
            logger.error(f"Error preparing sequences: {e}")
            raise
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 100,
        patience: int = 10,
        min_delta: float = 1e-4
    ) -> Dict:
        """
        Train the model with early stopping.
        
        Parameters:
        - train_loader: DataLoader for training data
        - val_loader: Optional DataLoader for validation data
        - epochs: Maximum number of epochs to train
        - patience: Number of epochs to wait for improvement before early stopping
        - min_delta: Minimum change in validation loss to qualify as an improvement
        
        Returns:
        - Dictionary containing training history
        """
        self.initialize_optimizer()
        patience_counter = 0
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}
        
        try:
            for epoch in range(epochs):
                # Training phase
                train_loss = self.train_epoch(train_loader)
                history['train_loss'].append(train_loss)
                
                # Validation phase
                val_loss = None
                if val_loader:
                    val_loss = self.validate(val_loader)
                    history['val_loss'].append(val_loss)
                    
                    # Early stopping check
                    if val_loss < (best_val_loss - min_delta):
                        best_val_loss = val_loss
                        patience_counter = 0
                        self.save_model('best_model.pth')
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                        break
                
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_loss:.6f}"
                    + (f" - Val Loss: {val_loss:.6f}" if val_loss else "")
                )
            
            return history
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        total_loss = 0
        
        try:
            for sequences, targets in train_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = self.criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
            
            return total_loss / len(train_loader)
            
        except Exception as e:
            logger.error(f"Error in training epoch: {e}")
            raise
    
    def validate(self, val_loader: torch.utils.data.DataLoader) -> float:
        """Validate the model and return average loss."""
        self.model.eval()
        total_loss = 0
        
        try:
            with torch.no_grad():
                for sequences, targets in val_loader:
                    sequences = sequences.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = self.model(sequences)
                    loss = self.criterion(outputs, targets)
                    total_loss += loss.item()
            
            return total_loss / len(val_loader)
            
        except Exception as e:
            logger.error(f"Error in validation: {e}")
            raise
    
    def predict(self, sequences: torch.Tensor) -> np.ndarray:
        """Make predictions with the model."""
        self.model.eval()
        try:
            with torch.no_grad():
                sequences = sequences.to(self.device)
                predictions = self.model(sequences)
                return predictions.cpu().numpy()
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def save_model(self, path: str) -> None:
        """Save model checkpoint with all necessary information."""
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'best_val_loss': self.best_val_loss,
                'sequence_length': self.sequence_length,
                'learning_rate': self.learning_rate,
                'timestamp': datetime.now().isoformat()
            }
            
            # Create directory if it doesn't exist
            save_dir = Path(path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save(checkpoint, path)
            logger.info(f"Model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, path: str) -> None:
        """Load model checkpoint and restore all states."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state if it exists
            if checkpoint['optimizer_state_dict']:
                self.initialize_optimizer()
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load training history
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.sequence_length = checkpoint.get('sequence_length', self.sequence_length)
            self.learning_rate = checkpoint.get('learning_rate', self.learning_rate)
            
            logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise