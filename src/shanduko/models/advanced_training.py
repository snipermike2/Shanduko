#src/shanduko/models/advanced_training.py

import sys
import os
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from src.shanduko.models.train_water_quality import generate_synthetic_data, prepare_sequences
from src.shanduko.models.water_quality_lstm import WaterQualityLSTM

class ModelEvaluator:
    def __init__(self, model, test_data):
        self.model = model
        self.test_data = test_data
        self.parameters = ['Temperature', 'pH', 'Dissolved Oxygen', 'Turbidity']
        
    def evaluate_predictions(self, X_test, y_test):
        """Make predictions and calculate metrics"""
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test)
            predictions = self.model(X_test_tensor).numpy()
            
        metrics = {}
        for i, param in enumerate(self.parameters):
            mse = mean_squared_error(y_test[:, i], predictions[:, i])
            r2 = r2_score(y_test[:, i], predictions[:, i])
            metrics[param] = {
                'MSE': mse,
                'RMSE': np.sqrt(mse),
                'R2': r2
            }
        
        return predictions, metrics
    
    def plot_predictions(self, y_true, predictions, param_index):
        """Plot actual vs predicted values for a parameter"""
        plt.figure(figsize=(12, 6))
        plt.plot(y_true[:, param_index], label='Actual', alpha=0.7)
        plt.plot(predictions[:, param_index], label='Predicted', alpha=0.7)
        plt.title(f'{self.parameters[param_index]} - Actual vs Predicted')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_prediction_scatter(self, y_true, predictions, param_index):
        """Create scatter plot of predicted vs actual values"""
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true[:, param_index], predictions[:, param_index], alpha=0.5)
        plt.plot([y_true[:, param_index].min(), y_true[:, param_index].max()], 
                 [y_true[:, param_index].min(), y_true[:, param_index].max()], 
                 'r--', label='Perfect Prediction')
        plt.title(f'{self.parameters[param_index]} - Predicted vs Actual')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def run_evaluation(self, X_test, y_test):
        """Run complete evaluation with all plots and metrics"""
        print("Starting model evaluation...")
        
        # Make predictions and get metrics
        predictions, metrics = self.evaluate_predictions(X_test, y_test)
        
        # Print metrics for each parameter
        print("\nModel Performance Metrics:")
        print("-" * 50)
        for param, metric in metrics.items():
            print(f"\n{param}:")
            print(f"RMSE: {metric['RMSE']:.4f}")
            print(f"R² Score: {metric['R2']:.4f}")
        
        # Plot predictions for each parameter
        for i in range(len(self.parameters)):
            print(f"\nGenerating plots for {self.parameters[i]}...")
            self.plot_predictions(y_test, predictions, i)
            self.plot_prediction_scatter(y_test, predictions, i)

def main():
    # Generate test data
    print("Generating test data...")
    test_data = generate_synthetic_data(num_days=30)
    X, y = prepare_sequences(test_data)
    
    # Split into train/test
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create and train model
    print("Training model...")
    model = WaterQualityLSTM(input_size=4, hidden_size=64, num_layers=2, output_size=4)
    
    # Train model (simplified for demonstration)
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    epochs = 50
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    # Evaluate model
    evaluator = ModelEvaluator(model, test_data)
    evaluator.run_evaluation(X_test, y_test)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()