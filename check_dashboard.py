# Create a file called check_dashboard.py in the same directory as run.py
from src.shanduko.gui.app import WaterQualityDashboard
import inspect

def check_dashboard():
    """Check the dashboard class for issues"""
    print("Checking WaterQualityDashboard class...")
    
    # Check initialization parameters
    sig = inspect.signature(WaterQualityDashboard.__init__)
    print(f"__init__ method signature: {sig}")
    print(f"Parameters: {list(sig.parameters.keys())}")
    
    try:
        # Try creating an instance with a mock user
        class MockUser:
            def __init__(self):
                self.username = "test"
                self.role = "ADMIN"
                
        print("\nTrying to create dashboard instance...")
        dashboard = WaterQualityDashboard(current_user=MockUser())
        print("Dashboard instance created successfully")
        
        # Check if run method exists
        if hasattr(dashboard, 'run'):
            print("Dashboard has 'run' method")
        else:
            print("WARNING: Dashboard does not have 'run' method")
            
    except Exception as e:
        print(f"Error creating dashboard instance: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_dashboard()