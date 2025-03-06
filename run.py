#run.py
"""
Main entry point for Shanduko application with authentication
"""
import sys
import argparse
from pathlib import Path
import logging
import tkinter as tk
from tkinter import ttk, messagebox

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src directory to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

try:
    # Import necessary components
    from src.shanduko.gui.login import LoginScreen
    from src.shanduko.gui.app import WaterQualityDashboard
    from src.shanduko.database.database import init_db
    from src.shanduko.auth.auth_service import AuthService
    from src.shanduko.database.database import UserRole
    from src.shanduko.auth.models import FallbackUser
    
    # Modify the launch_main_app function in run.py:

    def launch_main_app(user):
        """Launch the main application after successful login"""
        try:
            if user:
                logger.info(f"User authenticated: {user.username} (Role: {user.role})")
            else:
                logger.warning("No user provided, launching in demo mode")
                # Create a fallback user
                from src.shanduko.auth.models import FallbackUser
                user = FallbackUser(username="guest", role="viewer")
            
            # Initialize database
            try:
                init_db()
                logger.info("Database initialized successfully")
            except Exception as e:
                logger.error(f"Database initialization error: {e}")
                logger.info("Continuing without database initialization")
            
            try:
                # Try to use the regular dashboard first
                logger.info("Creating dashboard...")
                app = WaterQualityDashboard(current_user=user)
                logger.info("Running dashboard...")
                app.run()
                logger.info("Dashboard closed normally")
            except Exception as e:
                # If that fails, try the simple dashboard
                logger.error(f"Error starting main dashboard: {e}")
                try:
                    # Check if we have a simple dashboard
                    try:
                        from simple_dashboard import SimpleWaterQualityDashboard
                        logger.info("Falling back to simple dashboard...")
                        app = SimpleWaterQualityDashboard(username=user.username)
                        app.run()
                    except ImportError:
                        # If no simple dashboard, show a message box
                        logger.error("Simple dashboard not found")
                        if tk:
                            tk.messagebox.showerror(
                                "Application Error",
                                "The main application could not be started. Please contact your administrator."
                            )
                except Exception as ex:
                    logger.critical(f"Could not start any dashboard: {ex}")
                    
        except Exception as e:
            logger.error(f"Error launching application: {e}")
            import traceback
            traceback.print_exc()
    
    def test_mode():
        """Run the application in test mode with a mock user"""
        logger.info("Running in test mode (with mock user)")
        
        # Create a mock user with admin privileges
        mock_user = FallbackUser(username="admin", role="admin")
        
        # Generate a session token for the mock user
        if hasattr(mock_user, 'generate_session_token'):
            mock_user.generate_session_token()
            
        # Launch the main application with the mock user
        launch_main_app(mock_user)
    
    def main():
        """Main application entry point"""
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Shanduko Water Quality Monitoring System")
        parser.add_argument("--test", action="store_true", help="Run in test mode (skip login)")
        args = parser.parse_args()
        
        # Initialize database (will only initialize once)
        try:
            init_db()
            logger.info("Database initialization checked")
        except Exception as e:
            logger.warning(f"Error checking database: {e}")
        
        if args.test:
            # Run in test mode
            test_mode()
        else:
            # Show login screen
            logger.info("Launching login screen...")
            try:
                login = LoginScreen(on_login_success=launch_main_app)
                login.run()
            except Exception as e:
                logger.error(f"Error with login screen: {e}")
                # Fall back to test mode
                logger.info("Falling back to test mode...")
                test_mode()
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"Error importing modules: {str(e)}")
    sys.exit(1)
except Exception as e:
    print(f"Error running application: {str(e)}")
    import traceback
    traceback.print_exc()