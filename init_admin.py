# init_admin.py
import sys
import getpass
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src directory to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from src.shanduko.database.database import init_db, UserRole, get_db, User
from src.shanduko.auth.auth_service import AuthService

def init_admin():
    """Create an admin user if none exists"""
    print("Checking for existing admin users...")
    
    # Initialize database
    init_db()
    
    # Check if admin user already exists
    with get_db() as db:
        admin_user = db.query(User).filter(User.role == UserRole.ADMIN.value).first()
        
        if admin_user:
            print(f"Admin user already exists: {admin_user.username}")
            return admin_user
    
    # No admin user found, create one
    print("No admin user found. Creating admin user...")
    username = input("Enter admin username [admin]: ") or "admin"
    
    # Check if username already exists
    with get_db() as db:
        existing_user = db.query(User).filter(User.username == username).first()
        if existing_user:
            print(f"Error: Username '{username}' already exists.")
            return None
    
    # Get password
    password = getpass.getpass("Enter admin password: ")
    confirm_password = getpass.getpass("Confirm admin password: ")
    
    if password != confirm_password:
        print("Error: Passwords do not match.")
        return None
    
    # Get optional information
    full_name = input("Enter full name [Administrator]: ") or "Administrator"
    email = input("Enter email [admin@shanduko.org]: ") or "admin@shanduko.org"
    
    # Create admin user
    success, user_id, message = AuthService.create_user(
        username=username,
        password=password,
        email=email,
        full_name=full_name,
        role=UserRole.ADMIN.value  # Make sure to use .value here
    )
    
    if success:
        print(f"Admin user '{username}' created successfully!")
        return AuthService.get_user(user_id=user_id)
    else:
        print(f"Error creating admin user: {message}")
        return None

# Alternative simplified function for non-interactive use
def create_admin_user_default():
    """Create admin user with default values if none exists"""
    # Initialize database first
    init_db()
    
    # Use the auth service to check if admin exists
    users = AuthService.list_users(include_inactive=True)
    admin_exists = any(user.role == UserRole.ADMIN.value for user in users)
    
    if admin_exists:
        logger.info("Admin user already exists")
        return
    
    # Create admin user with default values
    success, user_id, message = AuthService.create_user(
        username="admin",
        password="password",  # CHANGE THIS IN PRODUCTION!
        email="admin@shanduko.org",
        full_name="System Administrator",
        role=UserRole.ADMIN.value
    )
    
    if success:
        logger.info(f"Admin user created: {message}")
    else:
        logger.error(f"Failed to create admin user: {message}")

if __name__ == "__main__":
    # Choose which function to use
    use_interactive = True  # Set to False to use default values without prompts
    
    if use_interactive:
        admin_user = init_admin()
        if admin_user:
            print("Admin user creation complete!")
        else:
            print("Admin user creation failed.")
    else:
        create_admin_user_default()
        print("Default admin user creation attempted - check logs for details")