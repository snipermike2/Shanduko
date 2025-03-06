# simple_dashboard.py
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class SimpleWaterQualityDashboard:
    def __init__(self, username="Guest"):
        # Create main window
        self.root = tk.Tk()
        self.root.title("Shanduko Water Quality Dashboard")
        self.root.geometry("1200x800")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill="both", expand=True)
        
        # Header with user info
        header = ttk.Frame(main_frame)
        header.pack(fill="x", pady=(0, 10))
        
        title = ttk.Label(header, text="Water Quality Monitoring Dashboard", font=("Helvetica", 20, "bold"))
        title.pack(side="left")
        
        user_info = ttk.Label(header, text=f"User: {username}", font=("Helvetica", 12))
        user_info.pack(side="right")
        
        # Create metrics display
        metrics_frame = ttk.LabelFrame(main_frame, text="Current Readings", padding=10)
        metrics_frame.pack(fill="x", pady=(0, 10))
        
        metrics_inner = ttk.Frame(metrics_frame)
        metrics_inner.pack(fill="x")
        
        # Create metric cards in a grid
        params = [
            ("Temperature", "24.5 °C"),
            ("pH Level", "7.2"),
            ("Dissolved Oxygen", "8.1 mg/L"),
            ("Turbidity", "2.3 NTU")
        ]
        
        for i, (param, value) in enumerate(params):
            frame = ttk.LabelFrame(metrics_inner, text=param, padding=5)
            frame.grid(row=0, column=i, padx=10, pady=5, sticky="ew")
            
            val_label = ttk.Label(frame, text=value, font=("Helvetica", 18, "bold"))
            val_label.pack(pady=5)
        
        # Equal column widths
        for i in range(4):
            metrics_inner.columnconfigure(i, weight=1)
        
        # Chart area
        chart_frame = ttk.LabelFrame(main_frame, text="Water Quality Trends", padding=10)
        chart_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        # Create a simple matplotlib chart
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Sample data
        time = list(range(10))
        temp = [24, 24.2, 24.5, 24.8, 25, 24.7, 24.5, 24.3, 24.1, 24]
        
        ax.plot(time, temp, label="Temperature")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Embed chart in the UI
        canvas = FigureCanvasTkAgg(fig, chart_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Control buttons
        controls = ttk.Frame(main_frame)
        controls.pack(fill="x", pady=(0, 10))
        
        self.monitor_button = ttk.Button(
            controls, 
            text="Start Monitoring",
            command=self.toggle_monitoring
        )
        self.monitor_button.pack(side="left", padx=5)
        
        ttk.Button(
            controls,
            text="Export Data"
        ).pack(side="left", padx=5)
        
        # Status bar
        self.status_label = ttk.Label(main_frame, text="Status: Ready", font=("Helvetica", 10))
        self.status_label.pack(side="left", pady=(0, 5))
        
        # Initialize state
        self.is_monitoring = False
    
    def toggle_monitoring(self):
        """Toggle monitoring state"""
        self.is_monitoring = not self.is_monitoring
        if self.is_monitoring:
            self.monitor_button.configure(text="Stop Monitoring")
            self.status_label.configure(text="Status: Monitoring")
        else:
            self.monitor_button.configure(text="Start Monitoring")
            self.status_label.configure(text="Status: Ready")
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = SimpleWaterQualityDashboard("Admin")
    app.run()