#src/shanduko/models/test_model.py
# Test the WaterQualityLSTM model and WaterQualityTrainer class
import torch
import numpy as np
from src.shanduko.models.water_quality_lstm import WaterQualityLSTM
from src.shanduko.models.model_training import WaterQualityTrainer
from shanduko.evaluation.model_evaluator import WaterQualityEvaluator
from src.shanduko.visualization.metrics_dashboard import save_metrics_for_visualization

def generate_sample_data(num_samples=1000):
    """Generate synthetic water quality data for testing"""
    time = np.linspace(0, 100, num_samples)
    
    # Generate synthetic parameters with some realistic patterns
    temperature = 25 + 5 * np.sin(time / 10) + np.random.normal(0, 0.5, num_samples)
    ph = 7 + 0.5 * np.sin(time / 15) + np.random.normal(0, 0.1, num_samples)
    dissolved_oxygen = 8 + 2 * np.sin(time / 12) + np.random.normal(0, 0.3, num_samples)
    turbidity = 3 + np.sin(time / 20) + np.random.normal(0, 0.2, num_samples)
    
    # Combine into features array
    features = np.column_stack([temperature, ph, dissolved_oxygen, turbidity])
    return features

def test_model():
    # Generate sample data
    print("Generating sample data...")
    data = generate_sample_data()
    
    # Create trainer instance
    trainer = WaterQualityTrainer(sequence_length=24)
    
    # Prepare sequences
    print("Preparing sequences...")
    sequences, targets = trainer.prepare_sequences(data)
    
    # Split into train and validation sets
    split_idx = int(len(sequences) * 0.8)
    train_sequences = sequences[:split_idx]
    train_targets = targets[:split_idx]
    val_sequences = sequences[split_idx:]
    val_targets = targets[split_idx:]
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(train_sequences, train_targets)
    val_dataset = torch.utils.data.TensorDataset(val_sequences, val_targets)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
    
    # Train model
    print("Training model...")
    num_epochs = 10
    for epoch in range(num_epochs):
        results = trainer.train_epoch(train_loader, val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {results['train_loss']:.4f}")
        if results['val_loss']:
            print(f"Validation Loss: {results['val_loss']:.4f}")
    
    # Test prediction
    print("\nTesting prediction...")
    test_sequence = sequences[0].unsqueeze(0)  # Add batch dimension
    prediction = trainer.model.predict(test_sequence)
    
    print("\nSample Prediction:")
    print("Temperature:", prediction[0][0])
    print("pH:", prediction[0][1])
    print("Dissolved Oxygen:", prediction[0][2])
    print("Turbidity:", prediction[0][3])
    
    # Save model
    trainer.save_model('test_model.pth')
    print("\nModel saved to test_model.pth")
    
def evaluate_trained_model(model, test_loader):
    # Initialize evaluator
    evaluator = WaterQualityEvaluator()
    
    # Get predictions
    y_true_list = []
    y_pred_list = []
    
    model.eval()
    with torch.no_grad():
        for sequences, targets in test_loader:
            predictions = model(sequences)
            y_true_list.append(targets.numpy())
            y_pred_list.append(predictions.numpy())
    
    # Combine all batches
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    
    # Run evaluation
    results = evaluator.evaluate_model(y_true, y_pred, sequence_length=24)
    
    # Access specific metrics
    standards_compliance = results['water_quality_standards']
    ecological_metrics = results['ecological_impact']
    treatment_metrics = results['treatment_requirements']
    
    # Print summary
    print(evaluator.generate_evaluation_summary(results))
    
     # Save metrics for visualization
    metrics_file = save_metrics_for_visualization(results)
    print(f"Metrics saved for visualization at: {metrics_file}")
    
    return results

if __name__ == "__main__":
    # Load your trained model
    model = WaterQualityLSTM()
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Create test data loader
    test_data = generate_sample_data()
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)
    
    # Evaluate model
    evaluation_results = evaluate_trained_model(model, test_loader)