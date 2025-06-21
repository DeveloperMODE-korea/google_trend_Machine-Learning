"""
Configuration management for Google Trends ML Predictor.

This module handles all configuration settings including API parameters,
model hyperparameters, and data processing options.
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import yaml


@dataclass
class DataConfig:
    """Configuration for data collection and processing."""
    
    # Google Trends parameters
    geo: str = ""  # Empty string for worldwide
    timeframe: str = "today 5-y"  # Default: last 5 years
    language: str = "en-US"
    
    # Data processing
    normalize: bool = True
    interpolate_missing: bool = True
    smoothing_window: int = 7  # Weekly smoothing
    
    # Storage paths
    raw_data_path: str = "./data/raw"
    processed_data_path: str = "./data/processed"


@dataclass
class ModelConfig:
    """Configuration for machine learning models."""
    
    # Common parameters
    test_size: float = 0.2
    random_state: int = 42
    validation_split: float = 0.1
    
    # LSTM parameters
    lstm_units: List[int] = None
    lstm_dropout: float = 0.2
    lstm_epochs: int = 100
    lstm_batch_size: int = 32
    
    # Prophet parameters
    prophet_changepoint_prior_scale: float = 0.05
    prophet_seasonality_mode: str = "additive"
    prophet_yearly_seasonality: bool = True
    prophet_weekly_seasonality: bool = True
    
    # Model paths
    model_save_path: str = "./models"
    
    def __post_init__(self):
        if self.lstm_units is None:
            self.lstm_units = [128, 64, 32]


@dataclass
class AppConfig:
    """Main application configuration."""
    
    # Subconfigurations
    data: DataConfig = None
    model: ModelConfig = None
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "google_trends_ml.log"
    
    # Visualization
    plot_style: str = "seaborn"
    figure_size: tuple = (12, 6)
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'AppConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        
        # Create nested dataclass instances
        if 'data' in config_dict:
            config_dict['data'] = DataConfig(**config_dict['data'])
        if 'model' in config_dict:
            config_dict['model'] = ModelConfig(**config_dict['model'])
        
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'log_level': self.log_level,
            'log_file': self.log_file,
            'plot_style': self.plot_style,
            'figure_size': list(self.figure_size)
        }
        
        with open(yaml_path, 'w') as file:
            yaml.dump(config_dict, file, default_flow_style=False)


# Default configuration instance
default_config = AppConfig()


def get_timeframe_dates(timeframe: str) -> tuple:
    """
    Convert timeframe string to start and end dates.
    
    Args:
        timeframe: Timeframe string (e.g., "today 5-y", "2020-01-01 2024-01-01")
    
    Returns:
        Tuple of (start_date, end_date) as strings
    """
    if ' ' in timeframe and len(timeframe.split(' ')) == 2:
        # Handle relative timeframes like "today 5-y"
        period, unit = timeframe.split(' ')
        
        if period.lower() == 'today':
            end_date = datetime.now()
            
            # Parse the unit
            if unit.endswith('-y'):
                years = int(unit[:-2])
                start_date = end_date - timedelta(days=365 * years)
            elif unit.endswith('-m'):
                months = int(unit[:-2])
                start_date = end_date - timedelta(days=30 * months)
            elif unit.endswith('-d'):
                days = int(unit[:-2])
                start_date = end_date - timedelta(days=days)
            else:
                raise ValueError(f"Invalid timeframe unit: {unit}")
            
            return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    
    # Handle absolute date ranges
    dates = timeframe.split(' ')
    if len(dates) == 2:
        return dates[0], dates[1]
    
    raise ValueError(f"Invalid timeframe format: {timeframe}") 