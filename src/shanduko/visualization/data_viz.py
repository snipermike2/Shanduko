"""
src/shanduko/visualization/data_viz.py
Visualization tools for water quality monitoring system
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from pathlib import Path

class WaterQualityVisualizer:
    def __init__(self, style: str = 'seaborn'):
        """Initialize visualizer with style settings"""
        plt.style.use(style)
        self.default_figsize = (12, 8)
        self.colors = sns.color_palette("husl", 8)
        
    def plot_time_series(self, 
                        df: pd.DataFrame,
                        parameters: Optional[List[str]] = None,
                        title: str = "Water Quality Parameters Over Time",
                        save_path: Optional[str] = None):
        """
        Plot time series data for multiple parameters
        
        Parameters:
            df: DataFrame with timestamp and parameter columns
            parameters: List of parameters to plot (default: all numeric columns)
            title: Plot title
            save_path: Optional path to save the plot
        """
        if parameters is None:
            parameters = df.select_dtypes(include=[np.number]).columns
            
        plt.figure(figsize=self.default_figsize)
        
        for i, param in enumerate(parameters):
            plt.plot(df['timestamp'], df[param], 
                    label=param, color=self.colors[i % len(self.colors)])
            
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_correlation_heatmap(self, 
                               df: pd.DataFrame,
                               parameters: Optional[List[str]] = None,
                               title: str = "Parameter Correlations",
                               save_path: Optional[str] = None):
        """Plot correlation heatmap between parameters"""
        if parameters is None:
            parameters = df.select_dtypes(include=[np.number]).columns
            
        corr_matrix = df[parameters].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', square=True)
        
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_daily_patterns(self,
                          df: pd.DataFrame,
                          parameters: Optional[List[str]] = None,
                          title: str = "Daily Patterns",
                          save_path: Optional[str] = None):
        """Plot average daily patterns for each parameter"""
        if parameters is None:
            parameters = df.select_dtypes(include=[np.number]).columns
            
        # Extract hour from timestamp
        df = df.copy()
        df['hour'] = df['timestamp'].dt.hour
        
        fig, axes = plt.subplots(len(parameters), 1, 
                                figsize=(12, 4*len(parameters)),
                                sharex=True)
        
        if len(parameters) == 1:
            axes = [axes]
            
        for ax, param in zip(axes, parameters):
            # Calculate mean and std for each hour
            hourly_stats = df.groupby('hour')[param].agg(['mean', 'std']).reset_index()
            
            # Plot mean line
            ax.plot(hourly_stats['hour'], hourly_stats['mean'],
                   label='Mean', linewidth=2)
            
            # Add confidence interval
            ax.fill_between(hourly_stats['hour'],
                          hourly_stats['mean'] - hourly_stats['std'],
                          hourly_stats['mean'] + hourly_stats['std'],
                          alpha=0.2, label='±1 std')
            
            ax.set_title(f"{param} Daily Pattern")
            ax.set_xlabel("Hour of Day")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_parameter_distributions(self,
                                   df: pd.DataFrame,
                                   parameters: Optional[List[str]] = None,
                                   title: str = "Parameter Distributions",
                                   save_path: Optional[str] = None):
        """Plot distribution of each parameter"""
        if parameters is None:
            parameters = df.select_dtypes(include=[np.number]).columns
            
        n_params = len(parameters)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols,
                                figsize=(5*n_cols, 4*n_rows))
        if n_params == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, param in enumerate(parameters):
            sns.histplot(data=df, x=param, kde=True, ax=axes[i])
            axes[i].set_title(f"{param} Distribution")
            
        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_boxplots(self,
                     df: pd.DataFrame,
                     parameters: Optional[List[str]] = None,
                     by: str = 'hour',
                     title: str = "Parameter Variations",
                     save_path: Optional[str] = None):
        """Plot boxplots for parameters grouped by time unit"""
        if parameters is None:
            parameters = df.select_dtypes(include=[np.number]).columns
            
        df = df.copy()
        if by == 'hour':
            df['group'] = df['timestamp'].dt.hour
            group_label = "Hour of Day"
        elif by == 'weekday':
            df['group'] = df['timestamp'].dt.day_name()
            group_label = "Day of Week"
        elif by == 'month':
            df['group'] = df['timestamp'].dt.month_name()
            group_label = "Month"
            
        n_params = len(parameters)
        fig, axes = plt.subplots(n_params, 1,
                                figsize=(12, 4*n_params),
                                sharex=True)
        if n_params == 1:
            axes = [axes]
            
        for ax, param in zip(axes, parameters):
            sns.boxplot(data=df, x='group', y=param, ax=ax)
            ax.set_title(f"{param} by {group_label}")
            ax.set_xlabel(group_label)
            ax.set_ylabel(param)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_anomalies(self,
                      df: pd.DataFrame,
                      parameters: Optional[List[str]] = None,
                      threshold: float = 2.0,
                      title: str = "Anomaly Detection",
                      save_path: Optional[str] = None):
        """Plot time series with highlighted anomalies"""
        if parameters is None:
            parameters = df.select_dtypes(include=[np.number]).columns
            
        n_params = len(parameters)
        fig, axes = plt.subplots(n_params, 1,
                                figsize=(12, 4*n_params),
                                sharex=True)
        if n_params == 1:
            axes = [axes]
            
        for ax, param in zip(axes, parameters):
            # Calculate mean and std
            mean = df[param].mean()
            std = df[param].std()
            
            # Identify anomalies
            anomalies = df[abs(df[param] - mean) > threshold * std]
            
            # Plot normal data
            ax.plot(df['timestamp'], df[param], label='Normal', alpha=0.7)
            
            # Plot anomalies
            ax.scatter(anomalies['timestamp'], anomalies[param],
                      color='red', label='Anomalies')
            
            ax.set_title(f"{param} Anomalies (threshold: {threshold} std)")
            ax.set_xlabel("Time")
            ax.set_ylabel(param)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def create_dashboard(self,
                        df: pd.DataFrame,
                        output_dir: str = "reports",
                        prefix: str = "dashboard"):
        """
        Create a comprehensive dashboard with all visualizations
        
        Parameters:
            df: Input DataFrame
            output_dir: Directory to save plots
            prefix: Prefix for plot filenames
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate all plots
        plots = {
            'time_series': self.plot_time_series,
            'correlations': self.plot_correlation_heatmap,
            'daily_patterns': self.plot_daily_patterns,
            'distributions': self.plot_parameter_distributions,
            'hourly_variations': lambda df: self.plot_boxplots(df, by='hour'),
            'anomalies': self.plot_anomalies
        }
        
        for name, plot_func in plots.items():
            save_path = output_path / f"{prefix}_{name}.png"
            plot_func(df, save_path=str(save_path))
            plt.close()  # Close the figure to free memory