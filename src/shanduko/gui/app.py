#src/shanduko/gui/app.py
import tkinter as tk
from tkinter import ttk, messagebox
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime, timedelta
import numpy as np
import os
import torch
import logging  
import sys
from pathlib import Path
import threading
import queue

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

print("Loading app.py module...")

from src.shanduko.models.water_quality_lstm import WaterQualityLSTM
from src.shanduko.models.model_training import WaterQualityTrainer
from src.shanduko.database.database import get_db, SensorReading, PredictionResult
# imports for authentication
from src.shanduko.auth.auth_service import AuthService
from src.shanduko.database.database import UserRole

class WaterQualityDashboard:
    def __init__(self, current_user=None):
        """Initialize the dashboard with authenticated user"""
        print("Starting dashboard initialization...")
        self.root = ttkb.Window(themename="cosmo")
        print("Created root window")
        self.root.title("Shanduko - Water Quality Monitoring System")
        print("Set window title")
        self.root.geometry("1400x800")
        print("Set window geometry")
        self.root.minsize(1200, 700)
        print("Set window minimum size")
        
        # Store current user
        self.current_user = current_user
        print(f"Stored current user: {current_user.username if current_user else 'None'}")
        
        # Initialize data storage
        self.sensor_data = {
            'timestamp': [],
            'temperature': [],
            'ph': [],
            'dissolved_oxygen': [],
            'turbidity': []
        }
        print("Initialized sensor data storage")
        
        # Add proper shutdown handling
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        print("Added window close protocol")
        
        # Initialize thread-safe queue for predictions
        self.prediction_queue = queue.Queue()
        self.prediction_thread = None
        self.prediction_active = False
        
        # Create GUI components
        print("Creating header...")
        self.create_header()
        print("Header created")
        
        print("Creating dashboard components...")
        self.create_dashboard()
        print("Dashboard components created")
        
        print("Creating control buttons...")
        self.create_controls()
        print("Control buttons created")
        
        # Load model
        print("Loading model...")
        self.load_model()
        print("Model loaded")
        
        # Start monitoring
        self.is_monitoring = False
        self.update_interval = 1000  # 1 second
        
        # Start prediction checker
        self.root.after(100, self.check_prediction_queue)
        
        print("Dashboard initialization complete!")

    def on_closing(self):
        """Handle window closing properly with safer error handling"""
        try:
            print("Window closing event triggered")
            
            # Stop monitoring if active
            if hasattr(self, 'is_monitoring') and self.is_monitoring:
                print("Stopping monitoring")
                try:
                    self.toggle_monitoring()  # Stop monitoring if active
                except Exception as e:
                    print(f"Error stopping monitoring: {e}")
                    
            # Perform logout if needed
            if hasattr(self, 'current_user') and self.current_user:
                if hasattr(self.current_user, 'session_token') and self.current_user.session_token:
                    print("Logging out user")
                    try:
                        from src.shanduko.auth.auth_service import AuthService
                        AuthService.logout(self.current_user.session_token)
                    except Exception as e:
                        print(f"Error during logout: {e}")
                        
            # Close matplotlib figures
            try:
                import matplotlib.pyplot as plt
                plt.close('all')  # Close all matplotlib figures
            except Exception as e:
                print(f"Error closing matplotlib figures: {e}")
                
            # Cancel any pending after calls
            if hasattr(self, 'root') and self.root:
                try:
                    # Cancel all "after" scheduled callbacks
                    for after_id in self.root.tk.call('after', 'info'):
                        self.root.after_cancel(after_id)
                except Exception as e:
                    print(f"Error canceling scheduled tasks: {e}")
                    
            # Destroy the window
            if hasattr(self, 'root') and self.root:
                try:
                    self.root.destroy()  # Destroy the window
                    print("Window destroyed successfully")
                except Exception as e:
                    print(f"Error destroying window: {e}")
            
        except Exception as e:
            print(f"Error during shutdown: {e}")
            import traceback
            traceback.print_exc()
            # Last resort attempt to destroy window
            try:
                if hasattr(self, 'root') and self.root:
                    self.root.destroy()
            except:
                print("Could not destroy window in exception handler")

    def run(self):
        """Run the application with better error handling"""
        try:
            print("Starting application mainloop...")
            self.root.lift()  # Bring window to front
            self.root.update()  # Force an update
            print("Window should be visible now")
            self.root.mainloop()
            print("Mainloop ended normally")
        except Exception as e:
            print(f"Error in application mainloop: {e}")
            import traceback
            traceback.print_exc()
        
    def create_header(self):
        """Create header with user information"""
        header = ttk.Frame(self.root, padding="10")
        header.pack(fill="x")
        
        # Left side: Title
        title = ttk.Label(header, text="Water Quality Monitoring Dashboard",
                         font=("Helvetica", 24, "bold"))
        title.pack(side="left")
        
        # Right side: User info and controls
        user_frame = ttk.Frame(header)
        user_frame.pack(side="right")
        
        # Status label
        self.status_label = ttk.Label(user_frame, text="Status: Stopped",
                                    font=("Helvetica", 12))
        self.status_label.pack(side="left", padx=(0, 20))
        
        # User info
        if self.current_user:
            # Show user info
            user_info = ttk.Label(user_frame, 
                                text=f"User: {self.current_user.username} ({self.current_user.role})",
                                font=("Helvetica", 12))
            user_info.pack(side="left", padx=(0, 10))
            
            # Add admin controls if user is admin
            user_role = self.current_user.role
            print(f"User role: {user_role}, type: {type(user_role)}")
            
            # Handle different role formats (string or enum value)
            is_admin = False
            if isinstance(user_role, str):
                is_admin = user_role.lower() == "admin"
            else:
                is_admin = user_role == UserRole.ADMIN.value
                
            if is_admin:
                print("Adding admin button for admin user")
                admin_button = ttk.Button(user_frame, text="Admin", 
                                       command=self.show_admin_panel)
                admin_button.pack(side="left", padx=(0, 10))
            
            # Logout button
            logout_button = ttk.Button(user_frame, text="Logout", 
                                     command=self.logout)
            logout_button.pack(side="left")
            
    def show_admin_panel(self):
        """Show admin panel for user management with improved error handling"""
        try:
            print("Opening admin panel...")
            user_role = self.current_user.role
            
            # Handle different role formats
            is_admin = False
            if isinstance(user_role, str):
                is_admin = user_role.lower() == "admin"
            else:
                is_admin = user_role == UserRole.ADMIN.value
                    
            if not is_admin:
                messagebox.showerror("Access Denied", "You do not have permission to access admin features.")
                return
            
            try:
                # Import here to avoid circular imports
                from src.shanduko.auth.user_manager import UserManagementWindow
                
                # Make sure the main window still exists before creating child window
                if not hasattr(self, 'root') or not self.root.winfo_exists():
                    print("Main window no longer exists, cannot open admin panel")
                    return
                
                # Open user management window - use non-modal approach
                UserManagementWindow(self.root, self.current_user)
                print("Admin panel opened successfully")
            except ImportError as e:
                print(f"Error importing user management: {e}")
                messagebox.showerror("Error", f"Could not open admin panel: Module not found")
            except Exception as e:
                print(f"Error opening admin panel: {e}")
                messagebox.showerror("Error", f"Could not open admin panel: {str(e)}")
        except Exception as e:
            print(f"Unexpected error in show_admin_panel: {e}")
            import traceback
            traceback.print_exc()
    
    def logout(self):
        """Log out current user and exit application"""
        if self.current_user and hasattr(self.current_user, 'session_token') and self.current_user.session_token:
            # Invalidate session
            AuthService.logout(self.current_user.session_token)
            
            # Delete saved session file if exists
            session_file = Path.home() / ".shanduko_session"
            if session_file.exists():
                try:
                    session_file.unlink()
                except:
                    pass
        
        # Confirm logout
        if messagebox.askyesno("Logout", "Are you sure you want to log out?"):
            self.root.destroy()
            
            # Restart application
            python = sys.executable
            os.execl(python, python, *sys.argv)
        
    def create_scrollable_frame(self, parent):
        """Create a scrollable frame container"""
        # Create canvas and scrollbars
        canvas = tk.Canvas(parent)
        v_scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        h_scrollbar = ttk.Scrollbar(parent, orient="horizontal", command=canvas.xview)
        
        # Create inner frame for content
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        # Add frame to canvas
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Grid layout for scrollbars and canvas
        canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        # Configure grid weights
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        
        # Bind mouse wheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        return scrollable_frame

    def create_dashboard(self):
        """Create main dashboard with scrolling"""
        # Create container for scrollable content
        container = ttk.Frame(self.root)
        container.pack(fill="both", expand=True, pady=(0, 50))  # Space for bottom controls
        
        # Create scrollable frame
        dashboard = self.create_scrollable_frame(container)
        
        # Add components to scrollable frame
        self.create_metric_cards(dashboard)
        self.create_interpretation_panel(dashboard)
        self.create_charts(dashboard)
        
    def create_metric_cards(self, parent):
        """Create metric cards with improved layout"""
        # Create metrics container with reduced padding
        metrics_frame = ttk.LabelFrame(parent, text="Current Readings", padding="5")
        metrics_frame.pack(fill="x", pady=(0, 5))
        
        # Grid for metric cards
        metrics = ttk.Frame(metrics_frame)
        metrics.pack(fill="x", padx=5)
        
        for i in range(4):
            metrics.grid_columnconfigure(i, weight=1, uniform="metrics")
        
        # Create metric cards with adjusted padding
        self.temp_card = self.create_metric_card(metrics, "Temperature", "°C", 0)
        self.ph_card = self.create_metric_card(metrics, "pH Level", "", 1)
        self.oxygen_card = self.create_metric_card(metrics, "Dissolved Oxygen", "mg/L", 2)
        self.turbidity_card = self.create_metric_card(metrics, "Turbidity", "NTU", 3)
        
    def create_metric_card(self, parent, title, unit, column):
        """Create individual metric card with adjusted sizes"""
        frame = ttk.LabelFrame(parent, text=title, padding=5)
        frame.grid(row=0, column=column, padx=2, pady=2, sticky="nsew")
        
        value = ttk.Label(frame, text="--", font=("Helvetica", 20, "bold"))
        value.pack(pady=1)
        
        if unit:
            unit_label = ttk.Label(frame, text=unit, font=("Helvetica", 9))
            unit_label.pack(pady=1)
        
        trend = ttk.Label(frame, text="")
        trend.pack(pady=1)
        
        return {"frame": frame, "value": value, "trend": trend}
        
    def create_charts(self, parent):
        """Create charts with optimized layout"""
        try:
            plt.close('all')  # Close any existing plots
            
            # Main charts container
            self.charts_frame = ttk.Frame(parent)  # Store reference to charts_frame
            self.charts_frame.pack(fill="both", expand=True)
            
            # Real-time chart frame
            realtime_frame = ttk.LabelFrame(self.charts_frame, text="Real-time Monitoring", padding="5")
            realtime_frame.pack(fill="both", expand=True, pady=(0, 5))
            
            self.fig_realtime, self.ax_realtime = plt.subplots(figsize=(12, 2.8))
            self.canvas_realtime = FigureCanvasTkAgg(self.fig_realtime, realtime_frame)
            self.canvas_realtime.get_tk_widget().pack(fill="both", expand=True)
            
            # Prediction chart frame
            prediction_frame = ttk.LabelFrame(self.charts_frame, text="24-Hour Predictions", padding="5")
            prediction_frame.pack(fill="both", expand=True)
            
            # Create prediction figure with adjusted dimensions
            self.fig_prediction = plt.figure(figsize=(12, 3.5))
            self.ax_prediction = self.fig_prediction.add_subplot(111)
            self.canvas_prediction = FigureCanvasTkAgg(self.fig_prediction, prediction_frame)
            self.canvas_prediction.get_tk_widget().pack(fill="both", expand=True)
            
            # Configure charts with more precise spacing
            for ax in [self.ax_realtime, self.ax_prediction]:
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=8)
                ax.tick_params(axis='x', rotation=30)
            
            # Explicitly set figure margins
            for fig in [self.fig_realtime, self.fig_prediction]:
                fig.subplots_adjust(
                    left=0.1,    # Left margin
                    right=0.85,  # Right margin for legend
                    bottom=0.25, # Bottom margin for x-labels
                    top=0.9     # Top margin for title
                )
            
        except Exception as e:
            print(f"Error creating charts: {e}")
            messagebox.showerror("Error", "Failed to create charts. See console for details.")
    
    def create_controls(self):
        """Create control buttons"""
        try:
            # Create a button frame
            button_frame = ttk.Frame(self.root)
            button_frame.pack(side="bottom", pady=10)
            
            # Create monitor button
            self.monitor_button = ttk.Button(
                button_frame,
                text="Start Monitoring",
                command=self.toggle_monitoring,
                style="primary.TButton",
                width=20
            )
            self.monitor_button.pack(side="left", padx=5)
            
            # Add export button
            self.export_button = ttk.Button(
                button_frame,
                text="Export Data",
                command=self.export_data,
                width=20
            )
            self.export_button.pack(side="left", padx=5)
            
            print("Control buttons created successfully")
        except Exception as e:
            print(f"Error creating controls: {e}")
            import traceback
            traceback.print_exc()
        
    def load_model(self):
        try:
            self.trainer = WaterQualityTrainer()
            model_path = 'data/models/trained_models/water_quality_model.pth'
            
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            if os.path.exists(model_path):
                self.trainer.load_model(model_path)
                print("Model loaded successfully")
            else:
                print("No pre-trained model found. Using new model.")
                
                # Initialize a basic model
                self.trainer.model = WaterQualityLSTM()
                print("New model initialized")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            # Initialize a basic model as fallback
            self.trainer = WaterQualityTrainer()
            self.trainer.model = WaterQualityLSTM()
            print("Fallback model initialized")
            
    def toggle_monitoring(self):
        # Check if user has permission to control monitoring
        if self.current_user:
            user_role = self.current_user.role
            print(f"Toggle monitoring - User role: {user_role}")
            
            # Handle different role formats (enum value or string)
            has_permission = False
            if isinstance(user_role, str):
                has_permission = user_role.lower() in ["operator", "admin"]
            else:
                has_permission = user_role in [UserRole.OPERATOR.value, UserRole.ADMIN.value]
                
            if not has_permission:
                messagebox.showerror("Access Denied", "You do not have permission to control monitoring.")
                return
            
        self.is_monitoring = not self.is_monitoring
        if self.is_monitoring:
            self.monitor_button.configure(text="Stop Monitoring")
            self.status_label.configure(text="Status: Monitoring")
            self.update_data()
        else:
            self.monitor_button.configure(text="Start Monitoring")
            self.status_label.configure(text="Status: Stopped")
            
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
        import tkinter as tk  # Ensure tk is imported here as well
        
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
        self.interpretation_text.delete(1.0, tk.END)  # Fixed: .END to tk.END
        self.interpretation_text.insert(tk.END, " • " + "\n • ".join(interpretations))
        self.interpretation_text.config(state="disabled")

    def update_data(self):
        """Update data with new readings and update visualizations"""
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
        
        # Update interpretation text
        try:
            # Convert dictionary format if needed
            interpretation_data = {
                'temperature': new_data['temperature'],
                'ph': new_data['ph'],
                'dissolved_oxygen': new_data['dissolved_oxygen'],
                'turbidity': new_data['turbidity']
            }
            self.interpret_data(interpretation_data)
            print(f"Updated interpretation for values: Temp={new_data['temperature']:.1f}, pH={new_data['ph']:.1f}")
        except Exception as e:
            print(f"Error updating interpretation: {e}")
        
        # Make prediction if we have enough data
        if len(self.sensor_data['temperature']) >= 24:
            self.make_prediction()
        
        # Store in database
        try:
            self.store_data(new_data)
        except Exception as e:
            print(f"Error storing data: {e}")
        
        # Schedule next update
        if self.is_monitoring:
            self.root.after(self.update_interval, self.update_data)
            
    def validate_sensor_data(self, data):
        """Validate sensor readings are within acceptable ranges"""
        ranges = {
            'temperature': (0, 50),     # °C
            'ph': (0, 14),             # pH scale
            'dissolved_oxygen': (0, 20), # mg/L
            'turbidity': (0, 100)       # NTU
        }
        
        try:
            for param, (min_val, max_val) in ranges.items():
                if param in data:
                    value = float(data[param])
                    if not min_val <= value <= max_val:
                        print(f"Warning: {param} reading {value} outside normal range [{min_val}, {max_val}]")
                        # Could add alerting here if needed
            return True
        except Exception as e:
            print(f"Error validating sensor data: {e}")
            return False

            
    def generate_sample_data(self):
        """Generate synthetic data with realistic patterns"""
        try:
            # Base values
            base_temp = 25 + 2 * np.sin(datetime.now().hour * np.pi / 12)  # Daily cycle
            base_ph = 7.0 + 0.5 * np.sin(datetime.now().hour * np.pi / 24)
            base_do = 8.0 - 0.1 * (base_temp - 25)  # Inverse relationship with temperature
            base_turb = 3.0
            
            # Add random noise
            data = {
                'timestamp': datetime.now(),
                'temperature': base_temp + np.random.normal(0, 0.5),
                'ph': base_ph + np.random.normal(0, 0.1),
                'dissolved_oxygen': base_do + np.random.normal(0, 0.2),
                'turbidity': base_turb + np.random.normal(0, 0.3)
            }
            
            # Validate before returning
            if self.validate_sensor_data(data):
                return data
            else:
                raise ValueError("Generated data failed validation")
                
        except Exception as e:
            print(f"Error generating sample data: {e}")
            # Return safe default values if generation fails
            return {
                'timestamp': datetime.now(),
                'temperature': 25.0,
                'ph': 7.0,
                'dissolved_oxygen': 8.0,
                'turbidity': 3.0
            }
        
    def update_metric_cards(self, data):
        self.temp_card['value'].configure(text=f"{data['temperature']:.1f}")
        self.ph_card['value'].configure(text=f"{data['ph']:.1f}")
        self.oxygen_card['value'].configure(text=f"{data['dissolved_oxygen']:.1f}")
        self.turbidity_card['value'].configure(text=f"{data['turbidity']:.1f}")
        
    def update_charts(self):
        """Update charts with improved styling"""
        self.ax_realtime.clear()
        
        timestamps = self.sensor_data['timestamp'][-50:]
        
        if timestamps:
            min_time = min(timestamps)
            max_time = max(timestamps)
            if min_time != max_time:
                self.ax_realtime.set_xlim(min_time, max_time)
            else:
                buffer = timedelta(minutes=1)
                self.ax_realtime.set_xlim(min_time - buffer, max_time + buffer)
        
        parameter_configs = {
            'temperature': {'color': '#1f77b4', 'min': 15, 'max': 35},
            'ph': {'color': '#ff7f0e', 'min': 0, 'max': 14},
            'dissolved_oxygen': {'color': '#2ca02c', 'min': 0, 'max': 15},
            'turbidity': {'color': '#d62728', 'min': 0, 'max': 10}
        }
        
        for param, config in parameter_configs.items():
            values = self.sensor_data[param][-50:]
            if values:
                self.ax_realtime.plot(timestamps, values, 
                                    label=param.replace('_', ' ').title(),
                                    color=config['color'],
                                    linewidth=1.5)
        
        self.ax_realtime.set_title('Real-time Water Quality Parameters', pad=5, fontsize=9)
        self.ax_realtime.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
        self.ax_realtime.grid(True, alpha=0.3)
        self.ax_realtime.set_ylabel('Values', fontsize=8)
        self.ax_realtime.set_xlabel('Time', fontsize=8)
        
        self.ax_realtime.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        self.ax_realtime.set_ylim(0, 35)
        
        plt.setp(self.ax_realtime.get_xticklabels(), rotation=30, ha='right', fontsize=7)
        plt.setp(self.ax_realtime.get_yticklabels(), fontsize=7)
        
        self.fig_realtime.tight_layout()
        self.canvas_realtime.draw()
        
    def check_prediction_queue(self):
        """Check for completed predictions in the queue"""
        try:
            # Check if there are completed predictions in the queue
            while not self.prediction_queue.empty():
                # Get prediction result from queue
                prediction = self.prediction_queue.get_nowait()
                
                # Update prediction chart with the result
                self.update_prediction_chart(prediction)
                
                # Mark task as done
                self.prediction_queue.task_done()
                
        except Exception as e:
            print(f"Error checking prediction queue: {e}")
    
        # Schedule next check
        self.root.after(100, self.check_prediction_queue)
    
    def make_prediction(self):
        """Start a prediction in a separate thread"""
        if len(self.sensor_data['temperature']) < 24:  # Need at least 24 points for prediction
            return
            
        # Don't start a new prediction if one is already running
        if self.prediction_active:
            return
            
        try:
            # Prepare input sequence
            sequence = []
            for i in range(-24, 0):
                sequence.append([
                    self.sensor_data['temperature'][i],
                    self.sensor_data['ph'][i],
                    self.sensor_data['dissolved_oxygen'][i],
                    self.sensor_data['turbidity'][i]
                ])
            
            # Convert to tensor
            sequence_tensor = torch.FloatTensor([sequence])
            
            # Set prediction active flag
            self.prediction_active = True
            
            # Start prediction in a new thread
            self.prediction_thread = threading.Thread(
                target=self._prediction_worker,
                args=(sequence_tensor,)
            )
            self.prediction_thread.daemon = True  # Thread will exit when main thread exits
            self.prediction_thread.start()
        
        except Exception as e:
            print(f"Error starting prediction: {e}")
            self.prediction_active = False
            
    def _prediction_worker(self, sequence_tensor):
        """Worker function to run prediction in a separate thread"""
        try:
            # Make prediction
            prediction = self.trainer.model.predict(sequence_tensor)
            
            # Put result in queue
            self.prediction_queue.put(prediction[0])
        except Exception as e:
            print(f"Error in prediction worker: {e}")
        finally:
            # Reset prediction active flag
            self.prediction_active = False
            
    def denormalize_predictions(self, prediction):
        """Denormalize prediction values to their original ranges"""
        parameter_ranges = {
            'Temperature': (15, 35),
            'pH': (0, 14),
            'Dissolved Oxygen': (0, 15),
            'Turbidity': (0, 10)
        }
        
        denormalized = {}
        for i, (param, (min_val, max_val)) in enumerate(parameter_ranges.items()):
            # Convert from -1,1 range to actual range
            normalized_val = float(prediction[i])
            denormalized[param] = min_val + (normalized_val + 1) * (max_val - min_val) / 2
        
        return denormalized

    def update_prediction_chart(self, prediction):
        """Update prediction chart with improved visualization"""
        # Clear existing content
        self.ax_prediction.clear()
        
        # Generate future timestamps
        current_time = datetime.now()
        future_times = [current_time + timedelta(hours=i) for i in range(24)]
        
        # Denormalize predictions
        denorm_predictions = self.denormalize_predictions(prediction)
        
        # Define parameters with their configurations and y-axis ranges
        parameter_configs = {
            'Temperature': {
                'color': '#1f77b4', 
                'unit': '°C',
                'y_range': (15, 35)
            },
            'pH': {
                'color': '#ff7f0e', 
                'unit': '',
                'y_range': (0, 14)
            },
            'Dissolved Oxygen': {
                'color': '#2ca02c', 
                'unit': 'mg/L',
                'y_range': (0, 15)
            },
            'Turbidity': {
                'color': '#d62728', 
                'unit': 'NTU',
                'y_range': (0, 10)
            }
        }
        
        # Plot each parameter's prediction
        for param, config in parameter_configs.items():
            value = denorm_predictions[param]
            # Create dashed line for prediction
            self.ax_prediction.plot(
                future_times,
                [value] * 24,
                label=f'{param}: {value:.1f} {config["unit"]}',
                linestyle='--',
                color=config['color'],
                linewidth=2
            )
            
            # Add horizontal guides for parameter's range
            self.ax_prediction.axhspan(
                config['y_range'][0], 
                config['y_range'][1],
                color=config['color'],
                alpha=0.1
            )
        
        # Set y-axis range to show common range
        self.ax_prediction.set_ylim(0, 35)
        
        # Configure axes and labels
        self.ax_prediction.set_title('24-Hour Predictions', pad=10, fontsize=10)
        self.ax_prediction.legend(
            bbox_to_anchor=(1.02, 1), 
            loc='upper left', 
            fontsize=8,
            framealpha=1.0
        )
        self.ax_prediction.grid(True, alpha=0.3)
        self.ax_prediction.set_ylabel('Values', fontsize=8)
        self.ax_prediction.set_xlabel('Time', fontsize=8)
        
        # Format time axis
        self.ax_prediction.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        
        # Style axis labels
        plt.setp(self.ax_prediction.get_xticklabels(), rotation=30, ha='right', fontsize=7)
        plt.setp(self.ax_prediction.get_yticklabels(), fontsize=7)
        
        # Adjust layout
        self.fig_prediction.subplots_adjust(right=0.85, bottom=0.2)
        self.canvas_prediction.draw()
        
        # Debug output
        print("Denormalized predictions:", denorm_predictions)
        
    def store_data(self, data):
        """Store sensor readings in database"""
        try:
            with get_db() as db:
                reading = SensorReading(
                    temperature=float(data['temperature']),
                    ph_level=float(data['ph']),
                    dissolved_oxygen=float(data['dissolved_oxygen']),
                    turbidity=float(data['turbidity']),
                    timestamp=data['timestamp']
                )
                db.add(reading)
                db.commit()
                logger.info("Data stored successfully")
        except Exception as e:
            logger.error(f"Error storing data: {e}")
            if self.is_monitoring:
                messagebox.showwarning(
                    "Database Warning",
                    "Failed to store measurement in database. Monitoring will continue."
                )
            
    def export_data(self):
        """Export data to CSV"""
        try:
            from datetime import datetime
            import pandas as pd
            
            # Get current timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"water_quality_data_{timestamp}.csv"
            
            # Create DataFrame from sensor data
            df = pd.DataFrame({
                'timestamp': self.sensor_data['timestamp'],
                'temperature': self.sensor_data['temperature'],
                'ph': self.sensor_data['ph'],
                'dissolved_oxygen': self.sensor_data['dissolved_oxygen'],
                'turbidity': self.sensor_data['turbidity']
            })
            
            # Save to CSV
            df.to_csv(filename, index=False)
            messagebox.showinfo("Success", f"Data exported to {filename}")
            
        except Exception as e:
            print(f"Error exporting data: {e}")
            messagebox.showerror("Error", "Failed to export data. See console for details.")