# src/shanduko/models/train_enhanced_model.py

import sys
from pathlib import Path
import pandas as pd
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from shanduko.models.enhanced_lstm import EnhancedWaterQualityLSTM
from shanduko.models.training_pipeline import WaterQualityTrainer
from shanduko.data.data_processor import WaterQualityDataProcessor

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main training script"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # 1. Initialize components
        logger.info("Initializing model and trainer...")
        model = EnhancedWaterQualityLSTM()
        trainer = WaterQualityTrainer(model)
        data_processor = WaterQualityDataProcessor()
        
        # 2. Load and process data
        logger.info("Loading and processing data...")
        data_path = project_root / "data" / "raw" / "water_quality_data.csv"
        
        if not data_path.exists():
            logger.info("No real data found, generating synthetic data for testing...")
            # Generate synthetic data for testing
            from shanduko.data.test_processor import generate_sample_data
            data = generate_sample_data(num_days=30)
            data.to_csv(data_path, index=False)
        
        # Load and process data
        data = data_processor.load_csv_data(str(data_path))
        data = data_processor.handle_missing_values(data)
        
        # 3. Prepare data loaders
        logger.info("Preparing data loaders...")
        train_loader, val_loader, test_loader = trainer.prepare_data(data)
        
        # 4. Train model
        logger.info("Starting model training...")
        train_losses, val_losses = trainer.train(train_loader, val_loader)
        
        # 5. Evaluate model
        logger.info("Evaluating model...")
        results = trainer.evaluate(test_loader)
        
        logger.info(f"Final test loss: {results['test_loss']:.6f}")
        
        # 6. Save final model
        trainer.save_checkpoint('final_model.pth')
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()