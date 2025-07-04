"""
Data Plots Package
-----------------
Comprehensive data visualization and analytics for Smart Privacy Cam

Structure:
- data_tracker.py: Collects and stores various metrics during app sessions
- plot_generator.py: Generates comprehensive data visualizations
- plots/: Directory containing generated plot files (PNG format)
- session_data.json: Session data storage for analysis
"""

from .data_tracker import DataTracker
from .plot_generator import PlotGenerator

__all__ = ['DataTracker', 'PlotGenerator'] 