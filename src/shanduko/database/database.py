"""
src/shanduko/database/database.py
Database module for Water Quality Monitoring System
"""
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import uuid
from datetime import datetime, timedelta
import bcrypt  # Make sure to install this with pip
import logging
from enum import Enum
from contextlib import contextmanager
import secrets

_database_initialized = False
# Set up logging
logger = logging.getLogger(__name__)

# Create Base class for declarative models
Base = declarative_base()

# Create global engine and session factory
engine = create_engine('sqlite:///water_quality.db', connect_args={'check_same_thread': False})
SessionLocal = sessionmaker(bind=engine)

def generate_uuid():
    return str(uuid.uuid4())

# Add this Enum for user roles
class UserRole(Enum):
    VIEWER = "viewer"     # Can only view data
    OPERATOR = "operator" # Can input data and control monitoring
    ADMIN = "admin"       # Full access including user management

# Add this User model class
class User(Base):
    __tablename__ = 'users'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True)
    password_hash = Column(String(128), nullable=False)
    full_name = Column(String(100))
    role = Column(String(20), default=UserRole.VIEWER.value)
    is_active = Column(Boolean, default=True)
    last_login = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Session token for authentication
    session_token = Column(String(64))
    token_expiry = Column(DateTime)
    
    def __repr__(self):
        return f"<User(username='{self.username}', role='{self.role}')>"
    
    @classmethod
    def hash_password(cls, password):
        """Generate a secure hash for the password"""
        # Generate salt and hash the password
        password_bytes = password.encode('utf-8')
        salt = bcrypt.gensalt()
        password_hash = bcrypt.hashpw(password_bytes, salt)
        return password_hash.decode('utf-8')
    
    def verify_password(self, password):
        """Verify password against stored hash"""
        password_bytes = password.encode('utf-8')
        hash_bytes = self.password_hash.encode('utf-8')
        return bcrypt.checkpw(password_bytes, hash_bytes)
    
    def generate_session_token(self, expiry_days=1):
        """Generate a new session token"""
        self.session_token = secrets.token_hex(32)
        self.token_expiry = datetime.utcnow() + timedelta(days=expiry_days)
        return self.session_token
    
    def is_token_valid(self):
        """Check if the current session token is valid"""
        if not self.session_token or not self.token_expiry:
            return False
        return datetime.utcnow() <= self.token_expiry

class Location(Base):
    __tablename__ = 'locations'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(100), nullable=False)
    latitude = Column(Float)
    longitude = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Location(name='{self.name}', lat={self.latitude}, lon={self.longitude})>"

class SensorReading(Base):
    __tablename__ = 'sensor_readings'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    temperature = Column(Float)
    ph_level = Column(Float)
    dissolved_oxygen = Column(Float)
    turbidity = Column(Float)
    location_id = Column(String(36), ForeignKey("locations.id"))
    
    location = relationship("Location")

    def __repr__(self):
        return f"<SensorReading(temp={self.temperature}, ph={self.ph_level})>"

class PredictionResult(Base):
    __tablename__ = 'prediction_results'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    location_id = Column(String(36), ForeignKey("locations.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    predicted_temperature = Column(Float)
    predicted_ph = Column(Float)
    predicted_oxygen = Column(Float)
    predicted_turbidity = Column(Float)
    confidence_score = Column(Float)
    
    location = relationship("Location")

    def __repr__(self):
        return f"<PredictionResult(temp={self.predicted_temperature}, ph={self.predicted_ph})>"

# Add this near the top of the file, after imports


# Add this new function
def ensure_db_initialized():
    """
    Ensure database is initialized only once.
    Returns True if initialization was performed, False if already initialized.
    """
    global _database_initialized
    
    if _database_initialized:
        logger.debug("Database already initialized, skipping")
        return False
        
    try:
        # Create all tables
        Base.metadata.create_all(engine)
        logger.info("Database tables created successfully")
        
        # Create default location if it doesn't exist
        with get_db() as db:
            if not db.query(Location).first():
                default_location = Location(
                    name="Default Location",
                    latitude=0.0,
                    longitude=0.0
                )
                db.add(default_location)
                db.commit()
                logger.info("Default location created")
                
        _database_initialized = True
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

# Replace your existing init_db function with this simpler version
def init_db():
    """Initialize database and create tables"""
    return ensure_db_initialized()

@contextmanager
def get_db():
    """Database session context manager"""
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def test_database():
    """Test database connection and basic operations"""
    try:
        # Initialize database
        init_db()
        logger.info("Database initialized for testing")
        
        # Test CRUD operations
        with get_db() as db:
            # Create test reading
            test_reading = SensorReading(
                temperature=25.0,
                ph_level=7.0,
                dissolved_oxygen=8.0,
                turbidity=2.0,
                timestamp=datetime.now()
            )
            db.add(test_reading)
            db.commit()
            logger.info("Test reading created")
            
            # Read test reading
            queried_reading = db.query(SensorReading).first()
            if queried_reading:
                logger.info("Test reading retrieved successfully")
            
            # Test successful
            return True
            
    except Exception as e:
        logger.error(f"Database test failed: {e}")
        return False

_database_initialized = False

def ensure_db_initialized():
    """
    Ensure database is initialized only once.
    Returns True if initialization was performed, False if already initialized.
    """
    global _database_initialized
    
    if _database_initialized:
        logger.debug("Database already initialized, skipping")
        return False
        
    try:
        # Create all tables
        Base.metadata.create_all(engine)
        logger.info("Database tables created successfully")
        
        # Create default location if it doesn't exist
        with get_db() as db:
            if not db.query(Location).first():
                default_location = Location(
                    name="Default Location",
                    latitude=0.0,
                    longitude=0.0
                )
                db.add(default_location)
                db.commit()
                logger.info("Default location created")
                
        _database_initialized = True
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise
class Zone(Base):
    __tablename__ = 'zones'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    zone_code = Column(String(20), unique=True, nullable=False)  # Unique code for the QR
    name = Column(String(100), nullable=False)  # User-friendly name (e.g., "North Shore Zone 3")
    description = Column(Text)  # Description of the zone
    water_body = Column(String(100))  # e.g., "Lake Chivero"
    center_latitude = Column(Float)  # Approximate center coordinates 
    center_longitude = Column(Float)  # (for mapping, not user tracking)
    created_at = Column(DateTime, default=datetime.utcnow)
       
class CommunityReport(Base):
    __tablename__ = 'community_reports'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    zone_id = Column(String(36), ForeignKey("zones.id"), nullable=False)
    report_type = Column(String(50), nullable=False)
    description = Column(Text)
    image_url = Column(String(200))
    timestamp = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default="pending")
    verified_by = Column(String(36), ForeignKey("users.id"), nullable=True)
    
    # Fix the relationships by explicitly specifying foreign keys
    user = relationship("User", foreign_keys=[user_id])
    verifier = relationship("User", foreign_keys=[verified_by])
    zone = relationship("Zone")
    
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run database test
    success = test_database()
    if success:
        print("Database test completed successfully!")
    else:
        print("Database test failed. Check logs for details.")
        
