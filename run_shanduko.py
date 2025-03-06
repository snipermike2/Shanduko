# run_shanduko.py - Fixed version

import sys
import os
from pathlib import Path
import logging
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("shanduko.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set project root and add to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

def check_requirements():
    """Check if requirements are installed"""
    try:
        import torch
        import numpy
        import pandas
        import matplotlib
        import ttkbootstrap
        import sqlalchemy
        logger.info("All required packages are installed.")
        return True
    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        logger.info("Please run: pip install -e .")
        return False

def init_model():
    """Initialize model by running the model init script"""
    try:
        logger.info("Initializing model...")
        model_script = PROJECT_ROOT / "init_model.py"
        
        if not model_script.exists():
            logger.error(f"Model initialization script not found at {model_script}")
            return False
        
        # Create checkpoints directory if it doesn't exist
        checkpoint_dir = PROJECT_ROOT / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if model already exists
        checkpoint_path = checkpoint_dir / "best_model.pth"
        if checkpoint_path.exists():
            logger.info(f"Model file already exists at {checkpoint_path}")
            return True
            
        # Run the initialization script
        result = subprocess.run([sys.executable, str(model_script)], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Model initialization failed: {result.stderr}")
            return False
            
        # Verify the model file was created
        if not checkpoint_path.exists():
            logger.error(f"Model initialization didn't create the expected file at {checkpoint_path}")
            return False
            
        logger.info(f"Model initialized successfully: {result.stdout.strip()}")
        return True
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        return False

def init_database():
    """Initialize database by running the database init script"""
    try:
        logger.info("Initializing database...")
        from src.shanduko.database.database import init_db
        init_db()
        logger.info("Database initialized.")
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False

def run_application():
    """Run the main application"""
    try:
        logger.info("Starting Shanduko water quality monitoring application...")
        run_script = PROJECT_ROOT / "run.py"
        
        if not run_script.exists():
            logger.error(f"Application script not found at {run_script}")
            return False
            
        # Run in the current process so we can see the GUI
        os.chdir(PROJECT_ROOT)  # Change to project root directory
        logger.info(f"Executing: {sys.executable} {run_script} --test")
        
        # Run in test mode to avoid login issues for initial testing
        os.execl(sys.executable, sys.executable, str(run_script), "--test")
        
        # Note: Code after this point won't be executed due to exec
        return True
    except Exception as e:
        logger.error(f"Error running application: {e}")
        return False

def main():
    """Main function to run the entire system"""
    logger.info("=" * 50)
    logger.info("Starting Shanduko Water Quality Monitoring System")
    logger.info("=" * 50)
    
    # Check requirements
    if not check_requirements():
        logger.error("Missing required packages. Please install requirements.")
        sys.exit(1)
    
    # Initialize model
    if not init_model():
        logger.error("Model initialization failed. See log for details.")
        sys.exit(1)
    
    # Initialize database
    if not init_database():
        logger.warning("Database initialization had issues. Will attempt to continue.")
        # Don't exit here, as we may be able to run with existing DB
    
    # Run application
    if not run_application():
        logger.error("Application startup failed. See log for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()