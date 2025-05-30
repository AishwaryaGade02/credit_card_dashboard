# src/__init__.py
"""
Credit Card Dashboard - Data Processing Package

This package contains modules for:
- Data preprocessing and cleaning
- KPI calculations and analytics
- Big data processing with PySpark
"""

__version__ = "1.0.0"
__author__ = "Credit Card Dashboard Team"

from .data_preprocessing import DataPreprocessor
from .kpi_calculator import KPICalculator

__all__ = ['DataPreprocessor', 'KPICalculator']