#src/shanduko/models/analyze_water_quality.py
import sys
import os
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from src.shanduko.models.train_water_quality import generate_synthetic_data, prepare_sequences

class WaterQualityAnalyzer:
    def __init__(self, num_days=30):
        """Initialize with synthetic data"""
        self.data = generate_synthetic_data(num_days=num_days)
        self.parameters = ['Temperature', 'pH', 'Dissolved Oxygen', 'Turbidity']
        
    def plot_time_series(self):
        """Plot detailed time series for each parameter"""
        plt.figure(figsize=(15, 10))
        
        for i, param in enumerate(self.parameters):
            plt.subplot(4, 1, i+1)
            plt.plot(self.data[:, i], label=param, linewidth=2)
            plt.title(f'{param} Over Time')
            plt.xlabel('Hours')
            plt.ylabel('Value')
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    def plot_daily_patterns(self):
        """Visualize daily patterns for each parameter"""
        # Reshape data into daily chunks
        hours_per_day = 24
        num_days = len(self.data) // hours_per_day
        daily_data = self.data[:num_days * hours_per_day].reshape(num_days, hours_per_day, -1)
        
        plt.figure(figsize=(15, 10))
        
        for i, param in enumerate(self.parameters):
            plt.subplot(2, 2, i+1)
            
            # Plot each day as a thin line
            for day in range(num_days):
                plt.plot(daily_data[day, :, i], alpha=0.2, color='blue')
            
            # Plot average daily pattern
            plt.plot(daily_data[:, :, i].mean(axis=0), 
                    color='red', linewidth=3, label='Average')
            
            plt.title(f'Daily Pattern: {param}')
            plt.xlabel('Hour of Day')
            plt.ylabel('Value')
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    def plot_correlations(self):
        """Analyze and visualize parameter correlations"""
        plt.figure(figsize=(12, 8))
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(self.data.T)
        
        # Create heatmap
        sns.heatmap(corr_matrix, 
                   xticklabels=self.parameters,
                   yticklabels=self.parameters,
                   annot=True, 
                   cmap='coolwarm',
                   vmin=-1, 
                   vmax=1,
                   center=0)
        
        plt.title('Parameter Correlations')
        plt.show()
        
        # Print detailed correlations
        print("\nDetailed Correlations:")
        for i in range(len(self.parameters)):
            for j in range(i+1, len(self.parameters)):
                correlation = corr_matrix[i, j]
                print(f"{self.parameters[i]} vs {self.parameters[j]}: {correlation:.3f}")
    
    def analyze_distributions(self):
        """Analyze the distribution of each parameter"""
        plt.figure(figsize=(15, 10))
        
        for i, param in enumerate(self.parameters):
            plt.subplot(2, 2, i+1)
            
            # Plot histogram with KDE
            sns.histplot(self.data[:, i], kde=True)
            
            # Add statistical information
            mean = np.mean(self.data[:, i])
            std = np.std(self.data[:, i])
            skew = stats.skew(self.data[:, i])
            
            plt.title(f'{param} Distribution\n'
                     f'Mean: {mean:.2f}, Std: {std:.2f}\n'
                     f'Skewness: {skew:.2f}')
            plt.xlabel('Value')
            plt.ylabel('Count')
            
        plt.tight_layout()
        plt.show()
    
    def analyze_extreme_events(self):
        """Analyze extreme events and anomalies"""
        plt.figure(figsize=(15, 10))
        
        for i, param in enumerate(self.parameters):
            plt.subplot(2, 2, i+1)
            
            data = self.data[:, i]
            mean = np.mean(data)
            std = np.std(data)
            
            # Define thresholds for extreme events
            upper_threshold = mean + 2*std
            lower_threshold = mean - 2*std
            
            # Plot data
            plt.plot(data, label='Data', alpha=0.7)
            plt.axhline(y=upper_threshold, color='r', linestyle='--', 
                       label='Upper Threshold')
            plt.axhline(y=lower_threshold, color='r', linestyle='--', 
                       label='Lower Threshold')
            
            # Highlight extreme events
            extremes = data[(data > upper_threshold) | (data < lower_threshold)]
            extreme_indices = np.where((data > upper_threshold) | 
                                     (data < lower_threshold))[0]
            plt.scatter(extreme_indices, extremes, color='red', 
                       label='Extreme Events')
            
            plt.title(f'Extreme Events: {param}\n'
                     f'Number of extremes: {len(extremes)}')
            plt.xlabel('Hours')
            plt.ylabel('Value')
            plt.legend()
            
        plt.tight_layout()
        plt.show()
    
    def run_full_analysis(self):
        """Run all analyses"""
        print("Starting comprehensive water quality data analysis...")
        
        print("\n1. Plotting time series data...")
        self.plot_time_series()
        
        print("\n2. Analyzing daily patterns...")
        self.plot_daily_patterns()
        
        print("\n3. Analyzing parameter correlations...")
        self.plot_correlations()
        
        print("\n4. Analyzing parameter distributions...")
        self.analyze_distributions()
        
        print("\n5. Analyzing extreme events...")
        self.analyze_extreme_events()
        
        print("\nAnalysis complete!")

if __name__ == "__main__":
    # Create analyzer instance
    analyzer = WaterQualityAnalyzer(num_days=30)
    
    # Run full analysis
    analyzer.run_full_analysis()