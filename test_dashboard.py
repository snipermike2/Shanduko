# test_dashboard.py
import tkinter as tk
from src.shanduko.auth.models import FallbackUser
from src.shanduko.gui.app import WaterQualityDashboard

# Create a simple test window first to make sure GUI works
root = tk.Tk()
root.title("Test Window")
root.geometry("300x200")
label = tk.Label(root, text="Test window works!")
label.pack(pady=50)
root.update()  # Force update to make sure window appears

# Create a test user
test_user = FallbackUser(username="admin", role="admin")
test_user.generate_session_token()

# Now try to create the dashboard
print("Creating dashboard...")
dashboard = WaterQualityDashboard(current_user=test_user)

# Run the dashboard
print("Running dashboard...")
dashboard.run()