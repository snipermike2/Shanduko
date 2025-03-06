# src/shanduko/auth/models.py
from datetime import datetime, timedelta
from src.shanduko.database.database import UserRole

class FallbackUser:
    """Fallback user class for testing or when database authentication fails"""
    def __init__(self, username="test", role="viewer"):
        self.id = 1
        self.username = username
        self.password = "password"  # Plain text for testing only
        self.email = f"{username}@example.com"
        self.full_name = f"Test {username.capitalize()}"
        
        # Handle role in different formats for compatibility
        if isinstance(role, str):
            self.role = role.lower()  # Ensure lowercase to match database values
        else:
            self.role = role
        
        self.is_active = True
        self.last_login = datetime.utcnow()
        self.session_token = None
        self.token_expiry = None
        
    def verify_password(self, password):
        """Simple password verification for testing"""
        return self.password == password
        
    def generate_session_token(self):
        """Generate a session token"""
        self.session_token = f"fallback_{str(datetime.utcnow().timestamp())}"
        self.token_expiry = datetime.utcnow() + timedelta(days=1)
        return self.session_token
        
    def is_token_valid(self):
        """Check if token is valid"""
        if not self.session_token or not self.token_expiry:
            return False
        return self.token_expiry > datetime.utcnow()
    
    def __str__(self):
        """String representation of the user"""
        return f"User({self.username}, role={self.role})"