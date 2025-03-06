# src/shanduko/models/enhanced_training.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.shanduko.models.water_quality_lstm import WaterQualityLSTM

class EnhancedWaterQualityTrainer:
    def __init__(self, config=None):
        """
        Initialize trainer with configuration
        """
        self.config = config or {
            'sequence_length': 24,
            'batch_size': 32,
            'learning_rate': 0.001,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'epochs': 100
        }
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = StandardScaler()
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Setup logging configuration"""
        logger = logging.getLogger('ModelTrainer')
        logger.setLevel(logging.INFO)
        
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler('training.log')
        
        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        
        return logger
        
    def prepare_data(self, data):
        """
        Prepare data for training
        
        Args:
            data: DataFrame with columns ['timestamp', 'temperature', 'ph', 'dissolved_oxygen', 'turbidity']
        """
        # Extract features
        features = ['temperature', 'ph', 'dissolved_oxygen', 'turbidity']
        X = data[features].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(len(X_scaled) - self.config['sequence_length']):
            seq = X_scaled[i:i + self.config['sequence_length']]
            target = X_scaled[i + self.config['sequence_length']]
            sequences.append(seq)
            targets.append(target)
            
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            sequences, targets, test_size=0.2, random_state=42
        )
        
        # Create dataloaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), 
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size']
        )
        
        return train_loader, val_loader
        
    def train(self, train_loader, val_loader, early_stopping_patience=10):
        """
        Train the model with early stopping
        """
        self.model = WaterQualityLSTM(
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers']
        ).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config['learning_rate']
        )
        
        # Training tracking
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        self.logger.info("Starting training...")
        
        for epoch in range(self.config['epochs']):
            # Training phase
            self.model.train()
            train_loss = 0
            
            for batch_idx, (sequences, targets) in enumerate(train_loader):
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = criterion(outputs, targets)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for sequences, targets in val_loader:
                    sequences = sequences.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = self.model(sequences)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.save_model('best_model.pth')
            else:
                patience_counter += 1
            
            self.logger.info(
                f"Epoch {epoch+1}/{self.config['epochs']} - "
                f"Train Loss: {avg_train_loss:.6f} - "
                f"Val Loss: {avg_val_loss:.6f}"
            )
            
            if patience_counter >= early_stopping_patience:
                self.logger.info("Early stopping triggered!")
                break
        
        # Plot training history
        self.plot_training_history(train_losses, val_losses)
        
        return train_losses, val_losses
    
    def plot_training_history(self, train_losses, val_losses):
        """Plot training and validation loss history"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_history.png')
        plt.close()
    
    def save_model(self, path):
        """Save model and training configuration"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'scaler': self.scaler,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(save_dict, path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load saved model and configuration"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.config = checkpoint['config']
        self.scaler = checkpoint['scaler']
        
        self.model = WaterQualityLSTM(
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"Model loaded from {path}")

# Example usage
if __name__ == "__main__":
    import pandas as pd
    
    # Load your data
    data = pd.read_csv('water_quality_data.csv')
    
    # Initialize trainer
    trainer = EnhancedWaterQualityTrainer()
    
    # Prepare data
    train_loader, val_loader = trainer.prepare_data(data)
    
    # Train model
    train_losses, val_losses = trainer.train(train_loader, val_loader)