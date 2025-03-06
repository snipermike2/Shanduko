#test_database.py
"""
Test script for database functionality
"""
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Import and test database
from src.shanduko.database.database import test_database

def main():
    print("Testing database connection and operations...")
    if test_database():
        print("Database test successful!")
    else:
        print("Database test failed!")

if __name__ == "__main__":
    main()