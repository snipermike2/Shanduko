# Create a file called test_app.py in the same directory as run.py
import tkinter as tk
from tkinter import ttk, messagebox

def run_test_app():
    """Run a simple test application to verify Tkinter is working"""
    root = tk.Tk()
    root.title("Test Application")
    root.geometry("400x300")
    
    # Create a simple label
    ttk.Label(root, text="Test Application is Running", font=("Helvetica", 14)).pack(pady=20)
    
    # Create a button to close the app
    ttk.Button(root, text="Close", command=root.destroy).pack(pady=10)
    
    # Print to console
    print("Test application window created successfully")
    
    # Start the main loop
    root.mainloop()
    print("Test application closed")

if __name__ == "__main__":
    run_test_app()