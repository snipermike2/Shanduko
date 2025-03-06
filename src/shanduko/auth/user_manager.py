# src/shanduko/auth/user_manager.py

import tkinter as tk
from tkinter import ttk, messagebox
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *

from src.shanduko.database.database import UserRole
from src.shanduko.auth.auth_service import AuthService

class UserManagementWindow:
    """User management window for administrators"""
    
    # Modify the __init__ and create_window methods in UserManagementWindow

    def __init__(self, parent, current_user):
        """
        Initialize user management window
        
        Args:
            parent: Parent window
            current_user: Current user object
        """
        self.parent = parent
        self.current_user = current_user
        
        # Check if user is admin
        is_admin = False
        if hasattr(current_user, 'role'):
            if isinstance(current_user.role, str):
                is_admin = current_user.role.lower() == "admin"
            else:
                from src.shanduko.database.database import UserRole
                is_admin = current_user.role == UserRole.ADMIN.value
        
        if not is_admin:
            from tkinter import messagebox
            messagebox.showerror("Access Denied", "You do not have permission to access user management.")
            return
        
        # Ensure parent window still exists
        try:
            parent.winfo_exists()  # Will raise error if parent is destroyed
            self.create_window()
        except Exception as e:
            print(f"Cannot create UserManagementWindow: Parent window no longer exists - {e}")
        
    def create_window(self):
        """Create the user management window"""
        import tkinter as tk
        from tkinter import ttk
        import ttkbootstrap as ttkb
        
        try:
            self.window = ttkb.Toplevel(self.parent)
            self.window.title("User Management")
            self.window.geometry("800x600")
            self.window.minsize(600, 400)
            
            # Main frame with padding
            main_frame = ttk.Frame(self.window, padding=10)
            main_frame.pack(fill="both", expand=True)
            
            # Title
            title_label = ttk.Label(main_frame, text="User Management", font=("Helvetica", 16, "bold"))
            title_label.pack(pady=(0, 10))
            
            # Create top button frame
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(fill="x", pady=(0, 10))
            
            # Add User button
            add_button = ttk.Button(button_frame, text="Add User", command=self.show_add_user_dialog)
            add_button.pack(side="left", padx=(0, 10))
            
            # Refresh button
            refresh_button = ttk.Button(button_frame, text="Refresh", command=self.load_users)
            refresh_button.pack(side="left")
            
            # Create user table
            self.create_user_table(main_frame)
            
            # Load users
            self.load_users()
            
            # Make window transient of parent
            self.window.transient(self.parent)
            
            # Don't make it grab_set (modal) to avoid blocking the main window
            # self.window.grab_set()
            
            self.window.focus_set()
            
        except Exception as e:
            print(f"Error creating user management window: {e}")
            import traceback
            traceback.print_exc()
        
    def create_user_table(self, parent):
        """Create the user table"""
        # Frame for table with scrollbar
        table_frame = ttk.Frame(parent)
        table_frame.pack(fill="both", expand=True)
        
        # Create scrollbars
        y_scrollbar = ttk.Scrollbar(table_frame)
        y_scrollbar.pack(side="right", fill="y")
        
        x_scrollbar = ttk.Scrollbar(table_frame, orient="horizontal")
        x_scrollbar.pack(side="bottom", fill="x")
        
        # Create treeview (table)
        self.user_table = ttk.Treeview(table_frame, 
                                       columns=("username", "full_name", "email", "role", "status", "last_login"),
                                       yscrollcommand=y_scrollbar.set,
                                       xscrollcommand=x_scrollbar.set)
        
        # Configure scrollbars
        y_scrollbar.config(command=self.user_table.yview)
        x_scrollbar.config(command=self.user_table.xview)
        
        # Configure columns
        self.user_table.column("#0", width=0, stretch=tk.NO)  # Hidden ID column
        self.user_table.column("username", width=100, anchor=tk.W)
        self.user_table.column("full_name", width=150, anchor=tk.W)
        self.user_table.column("email", width=200, anchor=tk.W)
        self.user_table.column("role", width=100, anchor=tk.W)
        self.user_table.column("status", width=80, anchor=tk.CENTER)
        self.user_table.column("last_login", width=150, anchor=tk.W)
        
        # Configure column headings
        self.user_table.heading("#0", text="", anchor=tk.CENTER)
        self.user_table.heading("username", text="Username", anchor=tk.W)
        self.user_table.heading("full_name", text="Full Name", anchor=tk.W)
        self.user_table.heading("email", text="Email", anchor=tk.W)
        self.user_table.heading("role", text="Role", anchor=tk.W)
        self.user_table.heading("status", text="Status", anchor=tk.CENTER)
        self.user_table.heading("last_login", text="Last Login", anchor=tk.W)
        
        # Pack the table
        self.user_table.pack(fill="both", expand=True)
        
        # Bind events
        self.user_table.bind("<Double-1>", self.on_user_double_click)
        self.user_table.bind("<Button-3>", self.show_context_menu)  # Right-click
        
    def load_users(self):
        """Load users into the table"""
        # Clear existing items
        for item in self.user_table.get_children():
            self.user_table.delete(item)
        
        # Get users
        users = AuthService.list_users(include_inactive=True)
        
        # Add users to table
        for user in users:
            status = "Active" if user.is_active else "Inactive"
            last_login = user.last_login.strftime("%Y-%m-%d %H:%M") if user.last_login else "Never"
            
            self.user_table.insert("", "end", iid=user.id, values=(
                user.username,
                user.full_name or "",
                user.email or "",
                user.role,
                status,
                last_login
            ))
    
    def on_user_double_click(self, event):
        """Handle double-click on user row"""
        item_id = self.user_table.identify('item', event.x, event.y)
        if item_id:
            self.show_edit_user_dialog(item_id)
    
    def show_context_menu(self, event):
        """Show context menu on right-click"""
        item_id = self.user_table.identify('item', event.x, event.y)
        if item_id:
            # Select the item
            self.user_table.selection_set(item_id)
            
            # Create context menu
            menu = tk.Menu(self.window, tearoff=0)
            menu.add_command(label="Edit User", command=lambda: self.show_edit_user_dialog(item_id))
            
            # Get user status
            user = AuthService.get_user(user_id=item_id)
            if user:
                if user.is_active:
                    menu.add_command(label="Deactivate User", 
                                    command=lambda: self.toggle_user_status(item_id, False))
                else:
                    menu.add_command(label="Activate User", 
                                    command=lambda: self.toggle_user_status(item_id, True))
            
            menu.add_separator()
            menu.add_command(label="Reset Password", 
                           command=lambda: self.show_reset_password_dialog(item_id))
            
            # Show menu
            menu.tk_popup(event.x_root, event.y_root)
    
    def show_add_user_dialog(self):
        """Show dialog to add a new user"""
        # Create dialog window
        dialog = ttkb.Toplevel(self.window)
        dialog.title("Add User")
        dialog.geometry("400x450")
        dialog.resizable(False, False)
        
        # Make dialog modal
        dialog.transient(self.window)
        dialog.grab_set()
        
        # Main frame with padding
        frame = ttk.Frame(dialog, padding=20)
        frame.pack(fill="both", expand=True)
        
        # Title
        title_label = ttk.Label(frame, text="Add New User", font=("Helvetica", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Username field
        username_frame = ttk.Frame(frame)
        username_frame.pack(fill="x", pady=5)
        
        username_label = ttk.Label(username_frame, text="Username:", width=15, anchor="w")
        username_label.pack(side="left")
        
        username_var = tk.StringVar()
        username_entry = ttk.Entry(username_frame, textvariable=username_var)
        username_entry.pack(side="left", fill="x", expand=True)
        
        # Full name field
        fullname_frame = ttk.Frame(frame)
        fullname_frame.pack(fill="x", pady=5)
        
        fullname_label = ttk.Label(fullname_frame, text="Full Name:", width=15, anchor="w")
        fullname_label.pack(side="left")
        
        fullname_var = tk.StringVar()
        fullname_entry = ttk.Entry(fullname_frame, textvariable=fullname_var)
        fullname_entry.pack(side="left", fill="x", expand=True)
        
        # Email field
        email_frame = ttk.Frame(frame)
        email_frame.pack(fill="x", pady=5)
        
        email_label = ttk.Label(email_frame, text="Email:", width=15, anchor="w")
        email_label.pack(side="left")
        
        email_var = tk.StringVar()
        email_entry = ttk.Entry(email_frame, textvariable=email_var)
        email_entry.pack(side="left", fill="x", expand=True)
        
        # Password field
        password_frame = ttk.Frame(frame)
        password_frame.pack(fill="x", pady=5)
        
        password_label = ttk.Label(password_frame, text="Password:", width=15, anchor="w")
        password_label.pack(side="left")
        
        password_var = tk.StringVar()
        password_entry = ttk.Entry(password_frame, textvariable=password_var, show="*")
        password_entry.pack(side="left", fill="x", expand=True)
        
        # Confirm password field
        confirm_frame = ttk.Frame(frame)
        confirm_frame.pack(fill="x", pady=5)
        
        confirm_label = ttk.Label(confirm_frame, text="Confirm Password:", width=15, anchor="w")
        confirm_label.pack(side="left")
        
        confirm_var = tk.StringVar()
        confirm_entry = ttk.Entry(confirm_frame, textvariable=confirm_var, show="*")
        confirm_entry.pack(side="left", fill="x", expand=True)
        
        # Role selection
        role_frame = ttk.Frame(frame)
        role_frame.pack(fill="x", pady=5)
        
        role_label = ttk.Label(role_frame, text="Role:", width=15, anchor="w")
        role_label.pack(side="left")
        
        role_var = tk.StringVar(value=UserRole.VIEWER.value)
        role_combo = ttk.Combobox(role_frame, textvariable=role_var, state="readonly")
        role_combo["values"] = [role.value for role in UserRole]
        role_combo.pack(side="left", fill="x", expand=True)
        
        # Active checkbox
        active_frame = ttk.Frame(frame)
        active_frame.pack(fill="x", pady=5)
        
        active_label = ttk.Label(active_frame, text="Status:", width=15, anchor="w")
        active_label.pack(side="left")
        
        active_var = tk.BooleanVar(value=True)
        active_check = ttk.Checkbutton(active_frame, text="Active", variable=active_var)
        active_check.pack(side="left")
        
        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill="x", pady=(20, 0))
        
        def on_cancel():
            dialog.destroy()
        
        def on_save():
            # Validate fields
            username = username_var.get().strip()
            password = password_var.get()
            confirm = confirm_var.get()
            email = email_var.get().strip()
            full_name = fullname_var.get().strip()
            role = role_var.get()
            is_active = active_var.get()
            
            if not username:
                messagebox.showerror("Error", "Username is required")
                return
            
            if not password:
                messagebox.showerror("Error", "Password is required")
                return
            
            if password != confirm:
                messagebox.showerror("Error", "Passwords do not match")
                return
            
            # Create user
            success, user_id, message = AuthService.create_user(
                username=username,
                password=password,
                email=email if email else None,
                full_name=full_name if full_name else None,
                role=role
            )
            
            if success:
                messagebox.showinfo("Success", "User created successfully")
                dialog.destroy()
                self.load_users()
            else:
                messagebox.showerror("Error", message)
        
        # Cancel button
        cancel_button = ttk.Button(button_frame, text="Cancel", command=on_cancel)
        cancel_button.pack(side="left", padx=(0, 10))
        
        # Save button
        save_button = ttk.Button(button_frame, text="Save", style="success.TButton", command=on_save)
        save_button.pack(side="right")
        
        # Focus on username field
        username_entry.focus_set()
    
    def show_edit_user_dialog(self, user_id):
        """Show dialog to edit a user"""
        # Get user
        user = AuthService.get_user(user_id=user_id)
        if not user:
            messagebox.showerror("Error", "User not found")
            return
        
        # Create dialog window
        dialog = ttkb.Toplevel(self.window)
        dialog.title("Edit User")
        dialog.geometry("400x400")
        dialog.resizable(False, False)
        
        # Make dialog modal
        dialog.transient(self.window)
        dialog.grab_set()
        
        # Main frame with padding
        frame = ttk.Frame(dialog, padding=20)
        frame.pack(fill="both", expand=True)
        
        # Title
        title_label = ttk.Label(frame, text=f"Edit User: {user.username}", font=("Helvetica", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Username (read-only)
        username_frame = ttk.Frame(frame)
        username_frame.pack(fill="x", pady=5)
        
        username_label = ttk.Label(username_frame, text="Username:", width=15, anchor="w")
        username_label.pack(side="left")
        
        username_var = tk.StringVar(value=user.username)
        username_entry = ttk.Entry(username_frame, textvariable=username_var, state="readonly")
        username_entry.pack(side="left", fill="x", expand=True)
        
        # Full name field
        fullname_frame = ttk.Frame(frame)
        fullname_frame.pack(fill="x", pady=5)
        
        fullname_label = ttk.Label(fullname_frame, text="Full Name:", width=15, anchor="w")
        fullname_label.pack(side="left")
        
        fullname_var = tk.StringVar(value=user.full_name or "")
        fullname_entry = ttk.Entry(fullname_frame, textvariable=fullname_var)
        fullname_entry.pack(side="left", fill="x", expand=True)
        
        # Email field
        email_frame = ttk.Frame(frame)
        email_frame.pack(fill="x", pady=5)
        
        email_label = ttk.Label(email_frame, text="Email:", width=15, anchor="w")
        email_label.pack(side="left")
        
        email_var = tk.StringVar(value=user.email or "")
        email_entry = ttk.Entry(email_frame, textvariable=email_var)
        email_entry.pack(side="left", fill="x", expand=True)
        
        # Role selection
        role_frame = ttk.Frame(frame)
        role_frame.pack(fill="x", pady=5)
        
        role_label = ttk.Label(role_frame, text="Role:", width=15, anchor="w")
        role_label.pack(side="left")
        
        role_var = tk.StringVar(value=user.role)
        role_combo = ttk.Combobox(role_frame, textvariable=role_var, state="readonly")
        role_combo["values"] = [role.value for role in UserRole]
        role_combo.pack(side="left", fill="x", expand=True)
        
        # Active checkbox
        active_frame = ttk.Frame(frame)
        active_frame.pack(fill="x", pady=5)
        
        active_label = ttk.Label(active_frame, text="Status:", width=15, anchor="w")
        active_label.pack(side="left")
        
        active_var = tk.BooleanVar(value=user.is_active)
        active_check = ttk.Checkbutton(active_frame, text="Active", variable=active_var)
        active_check.pack(side="left")
        
        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill="x", pady=(20, 0))
        
        def on_cancel():
            dialog.destroy()
        
        def on_save():
            # Get updated values
            full_name = fullname_var.get().strip()
            email = email_var.get().strip()
            role = role_var.get()
            is_active = active_var.get()
            
            # Update user
            success, message = AuthService.update_user(
                user_id=user_id,
                full_name=full_name if full_name else None,
                email=email if email else None,
                role=role,
                is_active=is_active
            )
            
            if success:
                messagebox.showinfo("Success", "User updated successfully")
                dialog.destroy()
                self.load_users()
            else:
                messagebox.showerror("Error", message)
        
        # Cancel button
        cancel_button = ttk.Button(button_frame, text="Cancel", command=on_cancel)
        cancel_button.pack(side="left", padx=(0, 10))
        
        # Save button
        save_button = ttk.Button(button_frame, text="Save", style="success.TButton", command=on_save)
        save_button.pack(side="right")
        
        # Reset password button
        reset_pwd_button = ttk.Button(button_frame, text="Reset Password", 
                                   command=lambda: self.show_reset_password_dialog(user_id))
        reset_pwd_button.pack(side="right", padx=(0, 10))
        
    def show_reset_password_dialog(self, user_id):
        """Show dialog to reset password"""
        # Get user
        user = AuthService.get_user(user_id=user_id)
        if not user:
            messagebox.showerror("Error", "User not found")
            return
        
        # Create dialog window
        dialog = ttkb.Toplevel(self.window)
        dialog.title("Reset Password")
        dialog.geometry("400x250")
        dialog.resizable(False, False)
        
        # Make dialog modal
        dialog.transient(self.window)
        dialog.grab_set()
        # Main frame with padding
        frame = ttk.Frame(dialog, padding=20)
        frame.pack(fill="both", expand=True)
        
        # Title
        title_label = ttk.Label(frame, text=f"Reset Password: {user.username}", 
                               font=("Helvetica", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # New password field
        password_frame = ttk.Frame(frame)
        password_frame.pack(fill="x", pady=5)
        
        password_label = ttk.Label(password_frame, text="New Password:", width=15, anchor="w")
        password_label.pack(side="left")
        
        password_var = tk.StringVar()
        password_entry = ttk.Entry(password_frame, textvariable=password_var, show="*")
        password_entry.pack(side="left", fill="x", expand=True)
        
        # Confirm password field
        confirm_frame = ttk.Frame(frame)
        confirm_frame.pack(fill="x", pady=5)
        
        confirm_label = ttk.Label(confirm_frame, text="Confirm Password:", width=15, anchor="w")
        confirm_label.pack(side="left")
        
        confirm_var = tk.StringVar()
        confirm_entry = ttk.Entry(confirm_frame, textvariable=confirm_var, show="*")
        confirm_entry.pack(side="left", fill="x", expand=True)
        
        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill="x", pady=(20, 0))
        
        def on_cancel():
            dialog.destroy()
        
        def on_save():
            # Validate fields
            password = password_var.get()
            confirm = confirm_var.get()
            
            if not password:
                messagebox.showerror("Error", "New password is required")
                return
            
            if password != confirm:
                messagebox.showerror("Error", "Passwords do not match")
                return
            
            # Update password
            success, message = AuthService.update_user(
                user_id=user_id,
                password=password
            )
            
            if success:
                messagebox.showinfo("Success", "Password reset successfully")
                dialog.destroy()
            else:
                messagebox.showerror("Error", message)
        
        # Cancel button
        cancel_button = ttk.Button(button_frame, text="Cancel", command=on_cancel)
        cancel_button.pack(side="left", padx=(0, 10))
        
        # Save button
        save_button = ttk.Button(button_frame, text="Reset Password", 
                              style="danger.TButton", command=on_save)
        save_button.pack(side="right")
        
        # Focus on password field
        password_entry.focus_set()
    
    def toggle_user_status(self, user_id, is_active):
        """Toggle user active status"""
        # Get user
        user = AuthService.get_user(user_id=user_id)
        if not user:
            messagebox.showerror("Error", "User not found")
            return
        
        # Confirm action
        action = "activate" if is_active else "deactivate"
        confirm = messagebox.askyesno(
            "Confirm Action", 
            f"Are you sure you want to {action} user '{user.username}'?"
        )
        
        if not confirm:
            return
        
        # Update user status
        success, message = AuthService.update_user(
            user_id=user_id,
            is_active=is_active
        )
        
        if success:
            status = "activated" if is_active else "deactivated"
            messagebox.showinfo("Success", f"User {status} successfully")
            self.load_users()
        else:
            messagebox.showerror("Error", message)