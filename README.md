# Shanduko: AI-Powered Water Quality Monitoring System

## Overview

Shanduko is an AI-powered water quality monitoring system designed to address critical water pollution levels in Zimbabwe through real-time monitoring, prediction, and management of water quality parameters. The system leverages deep learning, IoT technologies, and community participation to detect, predict, and mitigate water contamination, directly supporting several Sustainable Development Goals (SDGs).

![Water Quality Dashboard](assets/dashboard_screenshot.png)

## Key Features

- **Real-time Monitoring**: Track key water quality parameters (temperature, pH, dissolved oxygen, turbidity) with a responsive dashboard
- **AI-Powered Predictions**: LSTM neural networks to predict future water quality conditions 24 hours in advance
- **Anomaly Detection**: Identify unusual water quality conditions that might indicate pollution events
- **Data Analysis**: Comprehensive data quality assessment and visualization tools
- **User Management**: Role-based access control for administrators, operators, and viewers
- **Interactive Dashboard**: Intuitive interface for monitoring and analyzing water quality data

## Project Background

Zimbabwe's economy, ecosystems, and public health are at risk due to critical levels of water pollution. Lake Chivero, which serves as Harare's primary water source for over two million people, has been identified as one of the most polluted bodies of water in the region. This pollution stems from:

- **Urbanization**: Untreated effluent from Harare leads to extreme nutrient loads and eutrophication
- **Industrial Discharge**: Improperly treated waste containing heavy metals and toxic substances
- **Illegal Dumping**: Plastics and solid waste contributing to pollution

These issues have led to waterborne diseases, destruction of aquatic ecosystems, increased water purification costs, and economic losses in agriculture, tourism, and fisheries.

## Installation

### Prerequisites

- Python 3.8 or higher
- Git

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/shanduko.git
   cd shanduko
   ```

2. Install the package and its dependencies:
   ```bash
   pip install -e .
   ```

This will install all required dependencies listed in the `pyproject.toml` file, including:
- PyTorch
- NumPy
- Pandas
- Matplotlib
- ttkbootstrap
- SQLAlchemy
- and other necessary packages

## Running the Application

For the easiest experience, use the main runner script:

```bash
python run_shanduko.py
```

This script will:
1. Initialize the LSTM model
2. Set up the database
3. Launch the monitoring dashboard

### Alternative Manual Steps

If you prefer to run each step manually:

1. Initialize the model:
   ```bash
   python init_model.py
   ```

2. Initialize the database:
   ```bash
   python init_database.py
   ```

3. Run the application:
   ```bash
   python run.py
   ```

## Using the Dashboard

1. **Login**: Use your credentials to log in. Default admin account is `admin` with password `password` (change this in production!)
2. **Start Monitoring**: Click "Start Monitoring" to begin collecting and displaying water quality data
3. **Dashboard Features**:
   - Current parameter values with real-time updates
   - Historical trends with interactive charts
   - 24-hour predictions for all parameters
   - Water quality interpretations and status alerts
4. **Export Data**: Use the "Export Data" button to save collected data to a CSV file
5. **Admin Functions**: Administrators can manage users and system settings

## Project Structure

- `src/shanduko/models/`: Deep learning models for water quality prediction
- `src/shanduko/database/`: Database models and interactions
- `src/shanduko/gui/`: User interface components
- `src/shanduko/data/`: Data processing and quality assessment utilities
- `src/shanduko/visualization/`: Data visualization tools
- `src/shanduko/auth/`: Authentication and user management
- `src/shanduko/evaluation/`: Model evaluation metrics
- `src/shanduko/utils/`: Utility functions and helpers

## Prototype Limitations

The current prototype:
- Uses simulated data instead of real sensor readings
- Has a simplified LSTM model that will be enhanced in future versions
- Provides basic analysis with plans for more advanced analytics

## Future Enhancements

- Integration with real IoT water quality sensors
- Advanced anomaly detection algorithms
- Mobile application for alerts and monitoring
- Expanded dashboard with additional metrics and visualizations
- Community engagement features for pollution reporting

## Contributing

We welcome contributions to the Shanduko project! Please feel free to submit issues or pull requests.

## Team Members

- Panashe Dinha
- Takunda Shumbanhete
- Paidamoyo Makamba
- Shantel Piosi
- Chiedza Sagonda

## Acknowledgments

This project was developed to address water conservation in Zimbabwe, targeting Sustainable Development Goals 6 (Clean Water and Sanitation), 3 (Good Health and Well-Being), 14 (Life Below Water), and 9 (Industry, Innovation, and Infrastructure).

## License

[MIT License](LICENSE)#   S h a n d u k o  
 