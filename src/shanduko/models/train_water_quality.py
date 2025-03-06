# src/shanduko/models/train_water_quality.py
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from src.shanduko.models.water_quality_lstm import WaterQualityLSTM
from src.shanduko.config.model_config import ModelConfig
from src.shanduko.utils.logger import setup_logger

class WaterQualityTrainer:
    def __init__(self, config: ModelConfig, model_dir: Path):
        self.config = config
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.logger = setup_logger(
            'water_quality_trainer',
            model_dir / f'training_{datetime.now():%Y%m%d_%H%M%S}.log'
        )
        
        # Initialize model
        self.model = WaterQualityLSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            output_size=config.output_size
        ).to(self.device)
        
        # Setup optimization
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        self.logger.info(f"Trainer initialized with device: {self.device}")
        
    def prepare_data(self, data: np.ndarray):
        """Prepare data for training"""
        sequences = []
        targets = []
        
        for i in range(len(data) - self.config.sequence_length):
            seq = data[i:i + self.config.sequence_length]
            target = data[i + self.config.sequence_length]
            sequences.append(seq)
            targets.append(target)
            
        # Convert to tensors
        sequences = torch.FloatTensor(sequences)
        targets = torch.FloatTensor(targets)
        
        # Split data
        train_size = int(len(sequences) * self.config.train_split)
        val_size = int(len(sequences) * self.config.validation_split)
        
        # Create data loaders
        train_data = TensorDataset(sequences[:train_size], targets[:train_size])
        val_data = TensorDataset(
            sequences[train_size:train_size+val_size],
            targets[train_size:train_size+val_size]
        )
        
        train_loader = DataLoader(
            train_data,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_data,
            batch_size=self.config.batch_size
        )
        
        return train_loader, val_loader
    
    def train(self, train_loader, val_loader):
        """Train the model"""
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            for batch_idx, (sequences, targets) in enumerate(train_loader):
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            val_loss = self.validate(val_loader)
            
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} - "
                f"Train Loss: {avg_train_loss:.6f} - "
                f"Val Loss: {val_loss:.6f}"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('best_model.pth')
    
    def validate(self, val_loader):
        """Validate the model"""
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
    
    def save_model(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__
        }
        
        save_path = self.model_dir / filename
        torch.save(checkpoint, save_path)
        self.logger.info(f"Model saved to {save_path}")

def main():
    # Load configuration
    config = ModelConfig()
    model_dir = Path('data/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create trainer
    trainer = WaterQualityTrainer(config, model_dir)
    
    # Generate or load your data here
    # train_loader, val_loader = trainer.prepare_data(data)
    
    # Train model
    # trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()