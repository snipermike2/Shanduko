import sys
import traceback
import logging
from src.shanduko.gui.app import WaterQualityDashboard

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DebugWaterQualityDashboard(WaterQualityDashboard):
    """Debug wrapper for WaterQualityDashboard with enhanced error handling"""
    
    def __init__(self, current_user=None):
        """Initialize with try/except to catch errors"""
        logger.info("Initializing DebugWaterQualityDashboard...")
        try:
            # Call the parent's __init__ with explicit step logging
            logger.info("Starting parent initialization...")
            super().__init__(current_user)
            logger.info("Parent initialization completed")
        except Exception as e:
            logger.error(f"Error during dashboard initialization: {e}")
            traceback.print_exc()
            raise
    
    def run(self):
        """Run with enhanced error handling"""
        logger.info("Starting dashboard run method...")
        try:
            # Verify we have a root window
            if not hasattr(self, 'root'):
                logger.error("No root window found in dashboard!")
                raise ValueError("Dashboard has no root window")
                
            # Check if root is already destroyed
            try:
                title = self.root.title()
                logger.info(f"Window title is: {title}")
            except Exception as e:
                logger.error(f"Root window may be destroyed: {e}")
                raise
                
            # Make sure mainloop is being called
            logger.info("Calling root.mainloop()...")
            if hasattr(self.root, 'mainloop'):
                self.root.mainloop()
                logger.info("Mainloop completed")
            else:
                logger.error("Root has no mainloop method!")
                
        except Exception as e:
            logger.error(f"Error running dashboard: {e}")
            traceback.print_exc()
            raise