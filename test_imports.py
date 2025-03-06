#test_imports.py
"""
Test imports for Shanduko package
"""
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from src.shanduko.gui.app import WaterQualityDashboard
    print("GUI import successful")
    
    from src.shanduko.database.database import init_db
    print("Database import successful")
    
    from src.shanduko.models.water_quality_lstm import WaterQualityLSTM
    print("Model import successful")
    
    print("\nAll imports successful!")
except ImportError as e:
    print(f"Import error: {e}")
    
    # Print debugging information
    import sys
    print("\nPython path:")
    for path in sys.path:
        print(f"  {path}")