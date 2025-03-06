import tkinter as tk
from tkinter import ttk, messagebox
import ttkbootstrap as ttkb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime, timedelta
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewWaterQualityDashboard:
    """A new implementation of the water quality dashboard"""
    
    def __init__(self, current_user=None):
        """Initialize dashboard"""
        logger.info("Initializing new dashboard...")
        self.current_user = current_user
        
        # Create the main window
        self.root = ttkb.Window(themename="cosmo")
        self.root.title("Shanduko - Water Quality Dashboard")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        
        # Initialize variables
        self.is_monitoring = False
        self.update_interval = 1000  # milliseconds
        self.sensor_data = {
            'timestamp': [],
            'temperature': [],
            'ph': [],
            'dissolved_oxygen': [],
            'turbidity': []
        }
        
        # Create the main content
        self.create_dashboard()
        logger.info("Dashboard created successfully")
    
    def create_dashboard(self):
        """Create the dashboard layout"""
        # Create main container
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill="both", expand=True)
        
        # Create header
        self.create_header(main_frame)
        
        # Create metrics and charts
        self.create_metrics(main_frame)
        self.create_charts(main_frame)
        
        # Create controls
        self.create_controls(main_frame)
    
    def create_header(self, parent):
        """Create dashboard header"""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill="x", pady=(0, 10))
        
        # Title
        title = ttk.Label(header_frame, text="Water Quality Monitoring Dashboard", 
                         font=("Helvetica", 16, "bold"))
        title.pack(side="left")
        
        # User info
        if self.current_user:
            user_text = f"User: {self.current_user.username} | Role: {self.current_user.role}"
        else:
            user_text = "Demo Mode"
            
        user_info = ttk.Label(header_frame, text=user_text)
        user_info.pack(side="right")
        
        # Status label
        self.status_label = ttk.Label(header_frame, text="Status: Stopped")
        self.status_label.pack(side="right", padx=20)
    
    def create_metrics(self, parent):
        """Create metric cards for water quality parameters"""
        metrics_frame = ttk.Frame(parent)
        metrics_frame.pack(fill="x", pady=10)
        
        # Configure equal columns
        for i in range(4):
            metrics_frame.columnconfigure(i, weight=1, uniform="metric")
        
        # Create metric cards
        self.temp_card = self.create_metric_card(metrics_frame, "Temperature", "°C", 0)
        self.ph_card = self.create_metric_card(metrics_frame, "pH Level", "", 1)
        self.oxygen_card = self.create_metric_card(metrics_frame, "Dissolved Oxygen", "mg/L", 2)
        self.turbidity_card = self.create_metric_card(metrics_frame, "Turbidity", "NTU", 3)
    
    def create_metric_card(self, parent, title, unit, column):
        """Create a metric card for a parameter"""
        frame = ttk.LabelFrame(parent, text=title, padding=10)
        frame.grid(row=0, column=column, padx=5, sticky="ew")
        
        value = ttk.Label(frame, text="--", font=("Helvetica", 18, "bold"))
        value.pack(pady=5)
        
        if unit:
            unit_label = ttk.Label(frame, text=unit)
            unit_label.pack()
        
        return {"frame": frame, "value": value}
    
    def create_charts(self, parent):
        """Create charts for data visualization"""
        charts_frame = ttk.Frame(parent)
        charts_frame.pack(fill="both", expand=True, pady=10)
        
        # Create real-time chart
        self.create_realtime_chart(charts_frame)
        
        # Create prediction chart
        self.create_prediction_chart(charts_frame)
    
    def create_realtime_chart(self, parent):
        """Create real-time data chart"""
        # Create frame
        chart_frame = ttk.LabelFrame(parent, text="Real-time Measurements", padding=10)
        chart_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        # Create matplotlib figure
        self.fig_realtime, self.ax_realtime = plt.subplots(figsize=(10, 4))
        self.ax_realtime.set_title("Water Quality Parameters")
        self.ax_realtime.set_xlabel("Time")
        self.ax_realtime.set_ylabel("Value")
        self.ax_realtime.grid(True, alpha=0.3)
        
        # Create canvas
        self.canvas_realtime = FigureCanvasTkAgg(self.fig_realtime, chart_frame)
        self.canvas_realtime.draw()
        self.canvas_realtime.get_tk_widget().pack(fill="both", expand=True)
    
    def create_prediction_chart(self, parent):
        """Create prediction chart"""
        # Create frame
        chart_frame = ttk.LabelFrame(parent, text="24-Hour Predictions", padding=10)
        chart_frame.pack(fill="both", expand=True)
        
        # Create matplotlib figure
        self.fig_prediction, self.ax_prediction = plt.subplots(figsize=(10, 4))
        self.ax_prediction.set_title("Predicted Values")
        self.ax_prediction.set_xlabel("Time")
        self.ax_prediction.set_ylabel("Value")
        self.ax_prediction.grid(True, alpha=0.3)
        
        # Create canvas
        self.canvas_prediction = FigureCanvasTkAgg(self.fig_prediction, chart_frame)
        self.canvas_prediction.draw()
        self.canvas_prediction.get_tk_widget().pack(fill="both", expand=True)
    
    def create_controls(self, parent):
        """Create control buttons"""
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill="x", pady=10)
        
        # Monitor button
        self.monitor_button = ttk.Button(controls_frame, text="Start Monitoring", 
                                       command=self.toggle_monitoring, width=20)
        self.monitor_button.pack(side="left", padx=5)
        
        # Export data button
        export_button = ttk.Button(controls_frame, text="Export Data", 
                                 command=self.export_data, width=20)
        export_button.pack(side="left", padx=5)
    
    def toggle_monitoring(self):
        """Toggle monitoring state"""
        self.is_monitoring = not self.is_monitoring
        
        if self.is_monitoring:
            self.monitor_button.configure(text="Stop Monitoring")
            self.status_label.configure(text="Status: Monitoring")
            self.update_data()
        else:
            self.monitor_button.configure(text="Start Monitoring")
            self.status_label.configure(text="Status: Stopped")
    
    def update_data(self):
        """Update data with simulated values"""
        if not self.is_monitoring:
            return
        
        # Generate sample data
        new_data = self.generate_sample_data()
        
        # Update stored data
        for key in self.sensor_data:
            if key in new_data:
                self.sensor_data[key].append(new_data[key])
                # Keep only the last 100 points
                if len(self.sensor_data[key]) > 100:
                    self.sensor_data[key] = self.sensor_data[key][-100:]
        
        # Update UI
        self.update_metric_cards(new_data)
        self.update_charts()
        
        # Schedule next update
        self.root.after(self.update_interval, self.update_data)
    
    def generate_sample_data(self):
        """Generate sample data for demonstration"""
        # Current time
        now = datetime.now()
        
        # Base values with some daily cycle patterns
        hour = now.hour
        base_temp = 25 + 2 * np.sin(hour * np.pi / 12)  # Temperature varies by time of day
        base_ph = 7.0 + 0.3 * np.sin(hour * np.pi / 24)
        base_do = 8.0 - 0.1 * base_temp  # Inverse relationship with temperature
        base_turb = 3.0 + 0.5 * np.sin(hour * np.pi / 12)
        
        # Add random noise
        return {
            'timestamp': now,
            'temperature': base_temp + np.random.normal(0, 0.3),
            'ph': base_ph + np.random.normal(0, 0.1),
            'dissolved_oxygen': base_do + np.random.normal(0, 0.2),
            'turbidity': base_turb + np.random.normal(0, 0.2)
        }
    
    def update_metric_cards(self, data):
        """Update metric cards with new data"""
        self.temp_card['value'].configure(text=f"{data['temperature']:.1f}")
        self.ph_card['value'].configure(text=f"{data['ph']:.1f}")
        self.oxygen_card['value'].configure(text=f"{data['dissolved_oxygen']:.1f}")
        self.turbidity_card['value'].configure(text=f"{data['turbidity']:.1f}")
    
    def update_charts(self):
        """Update charts with new data"""
        # Clear previous plots
        self.ax_realtime.clear()
        
        # Get the last 50 data points
        timestamps = self.sensor_data['timestamp'][-50:]
        
        # Plot each parameter
        params = [
            ('temperature', 'Temperature (°C)', 'blue'),
            ('ph', 'pH', 'green'),
            ('dissolved_oxygen', 'Dissolved Oxygen (mg/L)', 'orange'),
            ('turbidity', 'Turbidity (NTU)', 'red')
        ]
        
        for param, label, color in params:
            values = self.sensor_data[param][-50:]
            if values:  # Only plot if we have data
                self.ax_realtime.plot(timestamps, values, label=label, color=color)
        
        # Configure plot
        self.ax_realtime.set_title('Real-time Water Quality Parameters')
        self.ax_realtime.set_xlabel('Time')
        self.ax_realtime.set_ylabel('Value')
        self.ax_realtime.legend(loc='upper right')
        self.ax_realtime.grid(True, alpha=0.3)
        self.ax_realtime.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        
        # Redraw canvas
        self.fig_realtime.tight_layout()
        self.canvas_realtime.draw()
        
        # Generate and update predictions (simplified)
        self.update_prediction_chart()
    
    def update_prediction_chart(self):
        """Update prediction chart with simple predictions"""
        # Clear previous plot
        self.ax_prediction.clear()
        
        # Only generate predictions if we have enough data
        if not all(len(self.sensor_data[key]) > 0 for key in ['temperature', 'ph', 'dissolved_oxygen', 'turbidity']):
            return
        
        # Get current values (last data point)
        current_values = {
            'temperature': self.sensor_data['temperature'][-1],
            'ph': self.sensor_data['ph'][-1],
            'dissolved_oxygen': self.sensor_data['dissolved_oxygen'][-1],
            'turbidity': self.sensor_data['turbidity'][-1]
        }
        
        # Generate future times
        now = datetime.now()
        future_times = [now + timedelta(hours=i) for i in range(24)]
        
        # Plot each parameter's prediction (simple flat line for demo)
        params = [
            ('temperature', 'Temperature (°C)', 'blue'),
            ('ph', 'pH', 'green'),
            ('dissolved_oxygen', 'Dissolved Oxygen (mg/L)', 'orange'),
            ('turbidity', 'Turbidity (NTU)', 'red')
        ]
        
        for param_name, label, color in params:
            # Just use current value as prediction (simple demo)
            predicted_value = current_values[param_name]
            predicted_values = [predicted_value] * 24  # Same value for 24 hours
            
            self.ax_prediction.plot(future_times, predicted_values, 
                                   label=f"Predicted {label}", 
                                   color=color, linestyle='--')
        
        # Configure plot
        self.ax_prediction.set_title('24-Hour Predictions')
        self.ax_prediction.set_xlabel('Time')
        self.ax_prediction.set_ylabel('Value')
        self.ax_prediction.legend(loc='upper right')
        self.ax_prediction.grid(True, alpha=0.3)
        self.ax_prediction.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        
        # Redraw canvas
        self.fig_prediction.tight_layout()
        self.canvas_prediction.draw()
    
    def export_data(self):
        """Export data to CSV"""
        try:
            import pandas as pd
            
            # Create DataFrame from sensor data
            df = pd.DataFrame({
                'timestamp': self.sensor_data['timestamp'],
                'temperature': self.sensor_data['temperature'],
                'ph': self.sensor_data['ph'],
                'dissolved_oxygen': self.sensor_data['dissolved_oxygen'],
                'turbidity': self.sensor_data['turbidity']
            })
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"water_quality_data_{timestamp}.csv"
            
            # Save to CSV
            df.to_csv(filename, index=False)
            messagebox.showinfo("Success", f"Data exported to {filename}")
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            messagebox.showerror("Error", f"Failed to export data: {str(e)}")
    
    def run(self):
        """Run the dashboard application"""
        logger.info("Starting dashboard mainloop")
        try:
            self.root.mainloop()
            logger.info("Dashboard mainloop ended")
        except Exception as e:
            logger.error(f"Error in mainloop: {e}")