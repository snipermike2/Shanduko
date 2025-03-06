# src/shanduko/auth/auth_service.py

import logging
from datetime import datetime, timedelta
from sqlalchemy.exc import SQLAlchemyError
from src.shanduko.database.database import get_db, User, UserRole

logger = logging.getLogger(__name__)

class AuthService:
    """Service for handling authentication and user management"""
    
   
    # In src/shanduko/auth/auth_service.py, modify the authenticate method:

    @staticmethod
    # src/shanduko/auth/auth_service.py

    @staticmethod
    def authenticate(username, password):
        """
        Authenticate a user
        
        Args:
            username: Username
            password: Plain text password
            
        Returns:
            Tuple of (success, user, message)
        """
        try:
            with get_db() as db:
                user = db.query(User).filter(User.username == username).first()
                
                if not user:
                    return False, None, "User not found"
                
                if not user.is_active:
                    return False, None, "Account is inactive"
                
                # Verify password
                if not user.verify_password(password):
                    return False, None, "Invalid password"
                
                # Update last login time and generate session token
                user.last_login = datetime.utcnow()
                token = None
                if hasattr(user, 'generate_session_token'):
                    token = user.generate_session_token()
                db.commit()
                
                # Create a FallbackUser to avoid the session binding issue
                from src.shanduko.auth.models import FallbackUser
                fallback = FallbackUser(
                    username=user.username,
                    role=user.role
                )
                
                # Copy other attributes
                if hasattr(user, 'email'):
                    fallback.email = user.email
                if hasattr(user, 'full_name'):
                    fallback.full_name = user.full_name
                if hasattr(user, 'id'):
                    fallback.id = user.id
                
                # Set session token
                fallback.session_token = token or fallback.generate_session_token()
                
                return True, fallback, "Authentication successful"
                    
        except SQLAlchemyError as e:
            logger.error(f"Database error during authentication: {e}")
            return False, None, f"Database error: {str(e)}"
        except Exception as e:
            logger.error(f"Error during authentication: {e}")
            return False, None, f"Authentication error: {str(e)}"
    
    @staticmethod
    def verify_session(token):
        """
        Verify a session token
        
        Args:
            token: Session token
            
        Returns:
            Tuple of (success, user)
        """
        try:
            with get_db() as db:
                user = db.query(User).filter(User.session_token == token).first()
                
                if not user:
                    return False, None
                
                # Check if token is valid, with fallback
                if hasattr(user, 'is_token_valid'):
                    if not user.is_token_valid():
                        return False, None
                else:
                    # Simple expiry check as fallback
                    if hasattr(user, 'token_expiry') and user.token_expiry < datetime.utcnow():
                        return False, None
                
                return True, user
                
        except Exception as e:
            logger.error(f"Error verifying session: {e}")
            return False, None
    
    @staticmethod
    def logout(token):
        """
        Invalidate a session token
        
        Args:
            token: Session token
        """
        try:
            with get_db() as db:
                user = db.query(User).filter(User.session_token == token).first()
                
                if user:
                    user.session_token = None
                    user.token_expiry = None
                    db.commit()
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Error during logout: {e}")
            return False
    
    @staticmethod
    def create_user(username, password, email=None, full_name=None, role=UserRole.VIEWER):
        """
        Create a new user
        
        Args:
            username: Username
            password: Plain text password
            email: Email address
            full_name: Full name
            role: User role
            
        Returns:
            Tuple of (success, user_id or None, message)
        """
        try:
            with get_db() as db:
                # Check if username already exists
                if db.query(User).filter(User.username == username).first():
                    return False, None, "Username already exists"
                
                # Check if email already exists (if provided)
                if email and db.query(User).filter(User.email == email).first():
                    return False, None, "Email already exists"
                
                # Create new user
                password_hash = User.hash_password(password) if hasattr(User, 'hash_password') else password
                
                # Create user object with appropriate handling for different User implementations
                try:
                    new_user = User(
                        username=username,
                        password_hash=password_hash,
                        email=email,
                        full_name=full_name,
                        role=role.value if isinstance(role, UserRole) else role
                    )
                except Exception as e:
                    logger.warning(f"Error creating user with standard constructor: {e}")
                    # Try fallback approach
                    new_user = User()
                    new_user.username = username
                    new_user.password_hash = password_hash
                    new_user.email = email
                    new_user.full_name = full_name
                    new_user.role = role.value if isinstance(role, UserRole) else role
                
                db.add(new_user)
                db.commit()
                
                return True, new_user.id, "User created successfully"
                
        except SQLAlchemyError as e:
            logger.error(f"Database error creating user: {e}")
            return False, None, "Database error"
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return False, None, "Error creating user"
    
    @staticmethod
    def get_user(user_id=None, username=None):
        """
        Get user by ID or username
        
        Args:
            user_id: User ID
            username: Username
            
        Returns:
            User object or None
        """
        try:
            with get_db() as db:
                if user_id:
                    return db.query(User).filter(User.id == user_id).first()
                elif username:
                    return db.query(User).filter(User.username == username).first()
                return None
                
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None
    
    @staticmethod
    def update_user(user_id, **kwargs):
        """
        Update user
        
        Args:
            user_id: User ID
            **kwargs: Fields to update
            
        Returns:
            Tuple of (success, message)
        """
        allowed_fields = {'email', 'full_name', 'role', 'is_active'}
        update_data = {k: v for k, v in kwargs.items() if k in allowed_fields}
        
        try:
            with get_db() as db:
                user = db.query(User).filter(User.id == user_id).first()
                
                if not user:
                    return False, "User not found"
                
                # Handle password change separately
                if 'password' in kwargs:
                    if hasattr(User, 'hash_password'):
                        user.password_hash = User.hash_password(kwargs['password'])
                    else:
                        user.password_hash = kwargs['password']
                
                # Update other fields
                for key, value in update_data.items():
                    setattr(user, key, value)
                
                db.commit()
                return True, "User updated successfully"
                
        except Exception as e:
            logger.error(f"Error updating user: {e}")
            return False, f"Error updating user: {str(e)}"
    
    @staticmethod
    def list_users(include_inactive=False):
        """
        List all users
        
        Args:
            include_inactive: Whether to include inactive users
            
        Returns:
            List of user objects
        """
        try:
            with get_db() as db:
                query = db.query(User)
                
                if not include_inactive:
                    query = query.filter(User.is_active == True)
                
                return query.all()
                
        except Exception as e:
            logger.error(f"Error listing users: {e}")
            return []