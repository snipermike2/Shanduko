# interpretation_panel.py
"""
Implementation of the data interpretation panel for the water quality dashboard.
Add this code to the WaterQualityDashboard class in src/shanduko/gui/app.py
"""

def create_interpretation_panel(self, parent):
    """Create panel for water quality data interpretation"""
    import tkinter as tk
    from tkinter import ttk
    
    interpretation_frame = ttk.LabelFrame(parent, text="Water Quality Interpretation", padding="5")
    interpretation_frame.pack(fill="x", pady=(5, 0))
    
    # Create text widget for interpretations
    self.interpretation_text = tk.Text(interpretation_frame, height=4, wrap="word", font=("Helvetica", 9))
    self.interpretation_text.pack(fill="x", padx=5, pady=5)
    self.interpretation_text.config(state="disabled")  # Make read-only

def interpret_data(self, data):
    """Interpret water quality parameters and provide analysis"""
    interpretations = []
    
    # Temperature interpretation
    temp = data['temperature']
    if 20 <= temp <= 30:
        temp_status = "optimal"
    elif temp < 20:
        temp_status = "low"
    else:
        temp_status = "high"
    interpretations.append(f"Water temperature is {temp_status} ({temp:.1f}°C)")
    
    # pH interpretation
    ph = data['ph']
    if 6.5 <= ph <= 8.5:
        ph_status = "normal"
    elif ph < 6.5:
        ph_status = "acidic"
    else:
        ph_status = "alkaline"
    interpretations.append(f"pH level is {ph_status} ({ph:.1f})")
    
    # Dissolved oxygen interpretation
    do = data['dissolved_oxygen']
    if do >= 8:
        do_status = "excellent"
    elif do >= 6:
        do_status = "good"
    else:
        do_status = "concerning"
    interpretations.append(f"Dissolved oxygen level is {do_status} ({do:.1f} mg/L)")
    
    # Turbidity interpretation
    turb = data['turbidity']
    if turb <= 5:
        turb_status = "clear"
    elif turb <= 10:
        turb_status = "slightly cloudy"
    else:
        turb_status = "turbid"
    interpretations.append(f"Water clarity is {turb_status} ({turb:.1f} NTU)")
    
    # Update interpretation text
    self.interpretation_text.config(state="normal")
    self.interpretation_text.delete(1.0, .END)
    self.interpretation_text.insert(tk.END, " • " + "\n • ".join(interpretations))
    self.interpretation_text.config(state="disabled")

# Modify the update_data method to include interpretation
def update_data_with_interpretation(self):
    """Modified update_data method that includes interpretation"""
    if not self.is_monitoring:
        return
        
    # Generate sample data (replace with real sensor data)
    new_data = self.generate_sample_data()
    
    # Update data storage
    for key in self.sensor_data:
        if key in new_data:
            self.sensor_data[key].append(new_data[key])
            if len(self.sensor_data[key]) > 100:
                self.sensor_data[key] = self.sensor_data[key][-100:]
                
    # Update visualizations
    self.update_metric_cards(new_data)
    self.update_charts()
    
    # Add interpretation
    self.interpret_data(new_data)
    
    # Make prediction
    self.make_prediction()
    
    # Store in database
    self.store_data(new_data)
    
    # Schedule next update
    if self.is_monitoring:
        self.root.after(self.update_interval, self.update_data)