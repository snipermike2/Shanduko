# src/shanduko/gui/login.py
import tkinter as tk
from tkinter import ttk, messagebox
import ttkbootstrap as ttkb
from PIL import Image, ImageTk, ImageDraw, ImageFilter, ImageEnhance
import os
from pathlib import Path
import math
from datetime import datetime

from src.shanduko.auth.auth_service import AuthService

class LoginScreen:
    def __init__(self, on_login_success):
        """
        Initialize login screen with water-themed background
        
        Args:
            on_login_success: Callback function to run after successful login
        """
        self.on_login_success = on_login_success
        self.root = ttkb.Window(themename="cosmo")
        self.root.title("Shanduko - Login")
        # Make the window taller to fit all content
        self.root.geometry("400x700")
        
        # Set window icon
        self.set_window_icon()
        
        # Create water-themed background
        self.create_water_background()
        
        # Create scrollable container instead of a fixed container
        self.create_scrollable_container()
        
        # Session storage
        self.session_token = None
        self.current_user = None
        
        # Check for saved session
        self.check_saved_session()
    
    def create_water_background(self):
        """Create a water-themed background"""
        # Create a canvas for the background
        canvas = tk.Canvas(self.root, highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        
        # Create gradient blue background
        canvas.create_rectangle(0, 0, 400, 700, fill="#0078D7", outline="")
        
        # Add water-like pattern lines
        for y in range(0, 700, 15):
            # Create wavy lines with lighter blue
            points = []
            for x in range(0, 420, 20):
                amp = 5 + (y % 30) / 6  # Amplitude varies slightly
                shift = y / 50.0  # Phase shift based on y position
                wave_y = y + amp * math.sin((x/40.0) + shift)
                points.extend([x, wave_y])
            
            canvas.create_line(points, fill="#40A9FF", width=1.5, smooth=True)
    
    def create_scrollable_container(self):
        """Create a scrollable container for all content"""
        # Create a frame that will contain the scrollable canvas
        self.outer_frame = tk.Frame(self.root)
        self.outer_frame.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.9, relheight=0.9)
        
        # Create canvas that can scroll
        self.canvas = tk.Canvas(self.outer_frame, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.outer_frame, orient="vertical", command=self.canvas.yview)
        
        # Configure canvas and scrollbar
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Create the white container frame inside the canvas
        self.container = tk.Frame(self.canvas, bg="white")
        
        # Add the container to the canvas
        self.canvas_window = self.canvas.create_window((0, 0), window=self.container, anchor="nw", width=350)
        
        # Configure canvas scrolling
        self.container.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        
        # Enable mousewheel scrolling
        self.root.bind_all("<MouseWheel>", self.on_mousewheel)
        
        # Create login content inside the container
        self.create_login_content()
    
    def on_frame_configure(self, event):
        """Update the scrollregion when the frame changes size"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def on_canvas_configure(self, event):
        """Adjust the container width when canvas changes size"""
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)
    
    def on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def create_login_content(self):
        """Create the login form content"""
        # Logo at the top
        logo_frame = tk.Frame(self.container, bg="#F0F8FF", width=200, height=150)
        logo_frame.pack(pady=(20, 5), fill="x")
        self.load_logo(logo_frame)
        
        # Tagline
        title_frame = tk.Frame(self.container, bg="white")
        title_frame.pack(pady=(0, 10))
        
        tk.Label(title_frame, text="WATER QUALITY", font=("Helvetica", 12), 
                bg="white", fg="#333333").pack()
        tk.Label(title_frame, text="MANAGEMENT SYSTEM", font=("Helvetica", 12), 
                bg="white", fg="#333333").pack()
        
        # Welcome message
        welcome_frame = tk.Frame(self.container, bg="white", padx=20)
        welcome_frame.pack(fill="x")
        
        tk.Label(welcome_frame, text="welcome to Shanduko", font=("Helvetica", 14, "bold"), 
                bg="white", fg="#333333").pack(anchor="w")
        
        welcome_text = "Harnessing AI, IoT, and community action to protect Lake Chivero and beyond. Join us in shaping a sustainable future—one byte at a time!"
        welcome_label = tk.Label(welcome_frame, text=welcome_text, wraplength=310, 
                               justify="left", bg="white", fg="#555555")
        welcome_label.pack(pady=(5, 15), anchor="w")
        
        # Login section header
        login_header = tk.Frame(self.container, bg="#0078D7", height=30)
        login_header.pack(fill="x", pady=(5, 15))
        
        tk.Label(login_header, text="USER LOGIN", font=("Helvetica", 12, "bold"), 
                bg="#0078D7", fg="white").pack(pady=5)
        
        # Username field
        username_frame = tk.Frame(self.container, bg="white")
        username_frame.pack(pady=5, padx=20, fill="x")
        
        tk.Label(username_frame, text="👤", bg="white", fg="#555", font=("Arial", 14)).pack(side="left", padx=(0, 5))
        self.username_var = tk.StringVar()
        username_entry = ttk.Entry(username_frame, textvariable=self.username_var, width=30)
        username_entry.pack(side="left", ipady=5, fill="x", expand=True)
        
        # Password field
        password_frame = tk.Frame(self.container, bg="white")
        password_frame.pack(pady=5, padx=20, fill="x")
        
        tk.Label(password_frame, text="🔒", bg="white", fg="#555", font=("Arial", 14)).pack(side="left", padx=(0, 5))
        self.password_var = tk.StringVar()
        password_entry = ttk.Entry(password_frame, textvariable=self.password_var, show="*", width=30)
        password_entry.pack(side="left", ipady=5, fill="x", expand=True)
        
        # Remember me and forgot password
        options_frame = tk.Frame(self.container, bg="white")
        options_frame.pack(pady=10, padx=20, fill="x")
        
        self.remember_var = tk.BooleanVar()
        remember_check = ttk.Checkbutton(options_frame, text="Remember", variable=self.remember_var)
        remember_check.pack(side="left")
        
        forgot_link = tk.Label(options_frame, text="Forgot Password?", 
                            bg="white", fg="#0078D7", cursor="hand2")
        forgot_link.pack(side="right")
        forgot_link.bind("<Button-1>", lambda e: self.forgot_password())
        
        # Login button
        button_frame = tk.Frame(self.container, bg="white")
        button_frame.pack(pady=10)
        
        login_button = tk.Button(button_frame, text="Login", bg="#0078D7", fg="white", 
                              font=("Helvetica", 12), relief="flat", command=self.authenticate,
                              activebackground="#005FA3", activeforeground="white", width=20, height=1)
        login_button.pack(pady=5)
        
        # Create account link
        create_frame = tk.Frame(self.container, bg="white")
        create_frame.pack(pady=5)
        
        create_account = tk.Label(create_frame, text="Create Account", 
                               bg="white", fg="#0078D7", cursor="hand2")
        create_account.pack()
        create_account.bind("<Button-1>", lambda e: self.create_account())
        
        # Add extra empty space at the bottom to ensure all content is visible when scrolling
        tk.Frame(self.container, height=40, bg="white").pack()
        
        # Set focus to username field
        username_entry.focus()
    
       
    def set_window_icon(self):
        """Set the window icon in the title bar"""
        try:
            # Look for the logo file
            logo_path = Path("assets/logo.png")
            
            if logo_path.exists():
                # Load icon for the window
                icon = Image.open(logo_path)
                # Resize to appropriate icon size
                icon = icon.resize((32, 32))
                icon_photo = ImageTk.PhotoImage(icon)
                # Set as window icon
                self.root.iconphoto(True, icon_photo)
                # Save reference to prevent garbage collection
                self.icon_photo = icon_photo
            else:
                print(f"Logo file not found at {logo_path}")
        except Exception as e:
            print(f"Error setting window icon: {e}")
            
    def load_logo(self, parent):
        """Load and display the Shanduko logo"""
        try:
            # Try to find logo in common locations
            logo_paths = [
                Path("assets/logo.png"),
                Path("src/shanduko/assets/logo.png"),
                Path(__file__).parent.parent.parent.parent / "assets" / "logo.png"
            ]
            
            logo_path = None
            for path in logo_paths:
                if path.exists():
                    logo_path = path
                    break
            
            if logo_path:
                # Load and resize logo
                logo_img = Image.open(logo_path)
                logo_img = logo_img.resize((100, 100))
                self.logo_photo = ImageTk.PhotoImage(logo_img)
                
                # Display logo on the background
                logo_label = tk.Label(parent, image=self.logo_photo, bg="#F0F8FF")
                logo_label.pack(padx=10, pady=10)
            else:
                # Fallback to text logo
                logo_label = tk.Label(parent, text="SHANDUKO", font=("Helvetica", 24, "bold"),
                                   bg="#F0F8FF", fg="#0078D7")
                logo_label.pack(padx=10, pady=10)
                print("Logo file not found, using text instead")
        except Exception as e:
            print(f"Error loading logo: {e}")
            # Fallback to text logo
            logo_label = tk.Label(parent, text="SHANDUKO", font=("Helvetica", 24, "bold"),
                               bg="#F0F8FF", fg="#0078D7")
            logo_label.pack(padx=10, pady=10)
    
    # In src/shanduko/gui/login.py, update the authenticate method:

    def authenticate(self):
        """Verify login credentials"""
        username = self.username_var.get()
        password = self.password_var.get()
        
        if not username or not password:
            messagebox.showerror("Error", "Username and password are required")
            return
        
        try:
            # Authenticate using the AuthService
            success, user, message = AuthService.authenticate(username, password)
            
            if success and user:
                # Create a FallbackUser to avoid SQLAlchemy session issues
                from src.shanduko.auth.models import FallbackUser
                fallback_user = FallbackUser(
                    username=user.username if hasattr(user, 'username') else username,
                    role=user.role if hasattr(user, 'role') else "viewer"
                )
                
                # Copy other attributes if available
                if hasattr(user, 'email'):
                    fallback_user.email = user.email
                if hasattr(user, 'full_name'):
                    fallback_user.full_name = user.full_name
                if hasattr(user, 'id'):
                    fallback_user.id = user.id
                
                # Generate session token
                fallback_user.generate_session_token()
                
                # Save session if remember me is checked
                if self.remember_var.get():
                    self.save_session(fallback_user.session_token)
                
                # Close login window
                self.root.destroy()
                
                # Call success callback with fallback user object
                self.on_login_success(fallback_user)
            else:
                messagebox.showerror("Authentication Failed", message)
                
        except Exception as e:
            print(f"Authentication error: {e}")
            
            # For testing - create a mock user
            test_mode = messagebox.askyesno(
                "Authentication Error", 
                f"Error: {str(e)}\n\nWould you like to continue in test mode?"
            )
            
            if test_mode:
                from src.shanduko.auth.models import FallbackUser
                mock_user = FallbackUser(username=username, role="viewer")
                mock_user.generate_session_token()
                
                # Close login window
                self.root.destroy()
                
                # Call success callback with mock user
                self.on_login_success(mock_user)
    
    def check_saved_session(self):
        """Check for saved session token and try to auto-login"""
        try:
            # Check if session file exists
            session_file = Path.home() / ".shanduko_session"
            if not session_file.exists():
                return
            
            # Read session token
            token = session_file.read_text().strip()
            if not token:
                return
            
            # Verify token
            success, user = AuthService.verify_session(token)
            if success:
                # Store current user and token
                self.current_user = user
                self.session_token = token
                
                # Auto-login after a short delay
                self.root.after(500, lambda: self.auto_login(user))
                
        except Exception as e:
            print(f"Error checking saved session: {e}")
    
    def auto_login(self, user):
        """Perform auto-login with the given user"""
        # Close login window
        self.root.destroy()
        
        # Call success callback with user object
        self.on_login_success(user)
    
    def save_session(self, token):
        """Save session token for future auto-login"""
        try:
            # Create session file
            session_file = Path.home() / ".shanduko_session"
            session_file.write_text(token)
        except Exception as e:
            print(f"Error saving session: {e}")
    
    def forgot_password(self):
        """Handle forgot password click"""
        messagebox.showinfo("Forgot Password", 
                          "Please contact the system administrator to reset your password.")
    
    def create_account(self):
        """Handle create account click"""
        messagebox.showinfo("Create Account", 
                          "Please contact the system administrator to create a new account.")
    
    def run(self):
        """Run the login window"""
        self.root.mainloop()

class MockUser:
    """Mock user object for testing or fallback"""
    def __init__(self, username="admin", role="admin"):
        self.id = "mock_id"
        self.username = username
        self.email = f"{username}@example.com"
        self.full_name = username.capitalize()
        self.role = role
        self.is_active = True
        self.last_login = datetime.now()
        self.session_token = "mock_token_" + username
        
    # Add this property to match the expected interface
    @property
    def value(self):
        return self.role

    def __str__(self):
        return f"User({self.username})"