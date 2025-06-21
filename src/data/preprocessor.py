"""
Data preprocessing module for Google Trends time series data.

This module provides functionality to clean, normalize, and prepare
trend data for machine learning models.
"""

import os
import logging
from typing import List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import json

from ..utils.config import DataConfig


logger = logging.getLogger(__name__)


class TrendsPreprocessor:
    """
    Preprocessor for Google Trends time series data.
    
    Handles data cleaning, normalization, feature engineering,
    and train/test splitting for ML models.
    """
    
    def __init__(self, config: Optional[DataConfig] = None):
        """
        Initialize the preprocessor.
        
        Args:
            config: Data configuration object.
        """
        self.config = config or DataConfig()
        self.scalers = {}  # Store scalers for each column
        
    def preprocess(self, data: pd.DataFrame, 
                  target_column: Optional[str] = None,
                  save_processed: bool = True) -> pd.DataFrame:
        """
        Apply full preprocessing pipeline to the data.
        
        Args:
            data: Raw trends data with datetime index.
            target_column: Specific column to use as target. If None, processes all.
            save_processed: Whether to save processed data.
        
        Returns:
            Preprocessed DataFrame.
        """
        logger.info("Starting preprocessing pipeline")
        
        # Copy to avoid modifying original
        processed_data = data.copy()
        
        # Handle missing values
        processed_data = self._handle_missing_values(processed_data)
        
        # Apply smoothing if configured
        if self.config.smoothing_window > 1:
            processed_data = self._apply_smoothing(processed_data, 
                                                 window=self.config.smoothing_window)
        
        # Normalize data if configured
        if self.config.normalize:
            processed_data = self._normalize_data(processed_data)
        
        # Add time-based features
        processed_data = self._add_time_features(processed_data)
        
        # Select target column if specified
        if target_column and target_column in processed_data.columns:
            # Keep target and time features
            time_features = [col for col in processed_data.columns 
                           if col.startswith(('year', 'month', 'day', 'week'))]
            keep_columns = [target_column] + time_features
            processed_data = processed_data[keep_columns]
        
        # Save processed data
        if save_processed:
            self._save_processed_data(processed_data)
        
        logger.info("Preprocessing complete. Shape: %s", processed_data.shape)
        return processed_data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data."""
        initial_missing = data.isnull().sum().sum()
        
        if initial_missing > 0:
            logger.warning("Found %d missing values", initial_missing)
            
            if self.config.interpolate_missing:
                # Interpolate missing values
                data = data.interpolate(method='time', limit_direction='both')
                
                # Fill any remaining NaN values
                data = data.fillna(method='ffill').fillna(method='bfill')
            else:
                # Simply drop rows with missing values
                data = data.dropna()
            
            final_missing = data.isnull().sum().sum()
            logger.info("Missing values after handling: %d", final_missing)
        
        return data
    
    def _apply_smoothing(self, data: pd.DataFrame, window: int) -> pd.DataFrame:
        """Apply rolling window smoothing."""
        logger.info("Applying smoothing with window size: %d", window)
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            # Apply rolling mean
            data[f'{col}_smooth'] = data[col].rolling(
                window=window, 
                center=True, 
                min_periods=1
            ).mean()
            
            # Keep original column for reference
            data[f'{col}_original'] = data[col]
            data[col] = data[f'{col}_smooth']
            data = data.drop(f'{col}_smooth', axis=1)
        
        return data
    
    def _normalize_data(self, data: pd.DataFrame, 
                       method: str = 'minmax') -> pd.DataFrame:
        """
        Normalize numerical data.
        
        Args:
            data: DataFrame to normalize.
            method: 'minmax' or 'standard' normalization.
        
        Returns:
            Normalized DataFrame.
        """
        logger.info("Normalizing data using %s method", method)
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'standard':
                scaler = StandardScaler()
            else:
                raise ValueError(f"Unknown normalization method: {method}")
            
            # Fit and transform
            data[col] = scaler.fit_transform(data[[col]])
            
            # Store scaler for inverse transform
            self.scalers[col] = scaler
        
        return data
    
    def inverse_transform(self, data: pd.DataFrame, 
                         columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Inverse transform normalized data back to original scale.
        
        Args:
            data: Normalized data.
            columns: Specific columns to inverse transform. If None, transforms all.
        
        Returns:
            Data in original scale.
        """
        data_copy = data.copy()
        
        if columns is None:
            columns = [col for col in data.columns if col in self.scalers]
        
        for col in columns:
            if col in self.scalers and col in data_copy.columns:
                data_copy[col] = self.scalers[col].inverse_transform(data_copy[[col]])
        
        return data_copy
    
    def _add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features for better prediction."""
        logger.info("Adding time-based features")
        
        # Extract time components
        data['year'] = data.index.year
        data['month'] = data.index.month
        data['day'] = data.index.day
        data['day_of_week'] = data.index.dayofweek
        data['week_of_year'] = data.index.isocalendar().week
        
        # Add cyclical encoding for periodic features
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        
        data['day_sin'] = np.sin(2 * np.pi * data['day'] / 31)
        data['day_cos'] = np.cos(2 * np.pi * data['day'] / 31)
        
        data['dow_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['dow_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        
        return data
    
    def create_sequences(self, data: pd.DataFrame,
                        target_column: str,
                        sequence_length: int = 30,
                        prediction_length: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series models like LSTM.
        
        Args:
            data: Preprocessed data.
            target_column: Column to predict.
            sequence_length: Number of time steps to look back.
            prediction_length: Number of time steps to predict.
        
        Returns:
            Tuple of (X, y) arrays suitable for LSTM training.
        """
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Convert to numpy array
        values = data.values
        target_idx = data.columns.get_loc(target_column)
        
        X, y = [], []
        
        for i in range(sequence_length, len(data) - prediction_length + 1):
            # Input sequence: all features
            X.append(values[i - sequence_length:i])
            
            # Output: only target column
            if prediction_length == 1:
                y.append(values[i, target_idx])
            else:
                y.append(values[i:i + prediction_length, target_idx])
        
        return np.array(X), np.array(y)
    
    def train_test_split_temporal(self, data: pd.DataFrame,
                                 test_size: float = 0.2,
                                 validation_size: float = 0.1) -> Tuple[pd.DataFrame, ...]:
        """
        Split time series data into train/validation/test sets.
        
        Maintains temporal order - no random shuffling.
        
        Args:
            data: Preprocessed data.
            test_size: Fraction of data for testing.
            validation_size: Fraction of training data for validation.
        
        Returns:
            Tuple of (train_data, val_data, test_data).
        """
        n = len(data)
        
        # Calculate split points
        test_split = int(n * (1 - test_size))
        val_split = int(test_split * (1 - validation_size))
        
        # Split maintaining temporal order
        train_data = data.iloc[:val_split]
        val_data = data.iloc[val_split:test_split]
        test_data = data.iloc[test_split:]
        
        logger.info("Data split - Train: %d, Val: %d, Test: %d",
                   len(train_data), len(val_data), len(test_data))
        
        return train_data, val_data, test_data
    
    def _save_processed_data(self, data: pd.DataFrame) -> None:
        """Save processed data with metadata."""
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"processed_trends_{timestamp}.csv"
        filepath = os.path.join(self.config.processed_data_path, filename)
        
        # Ensure directory exists
        os.makedirs(self.config.processed_data_path, exist_ok=True)
        
        # Save data
        data.to_csv(filepath)
        
        # Save preprocessing metadata
        metadata = {
            'processed_at': timestamp,
            'shape': list(data.shape),
            'columns': list(data.columns),
            'date_range': [str(data.index.min()), str(data.index.max())],
            'preprocessing_config': {
                'normalized': self.config.normalize,
                'interpolated': self.config.interpolate_missing,
                'smoothing_window': self.config.smoothing_window
            },
            'scalers': list(self.scalers.keys())
        }
        
        metadata_file = filepath.replace('.csv', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save scalers for later use
        if self.scalers:
            import joblib
            scaler_file = filepath.replace('.csv', '_scalers.joblib')
            joblib.dump(self.scalers, scaler_file)
        
        logger.info("Saved processed data to: %s", filepath)


def load_processed_data(filepath: str, 
                       load_scalers: bool = True) -> Tuple[pd.DataFrame, Optional[dict]]:
    """
    Load previously processed data along with its scalers.
    
    Args:
        filepath: Path to the processed CSV file.
        load_scalers: Whether to load associated scalers.
    
    Returns:
        Tuple of (data, scalers_dict).
    """
    # Load data
    data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    # Load scalers if requested and available
    scalers = None
    if load_scalers:
        scaler_file = filepath.replace('.csv', '_scalers.joblib')
        if os.path.exists(scaler_file):
            import joblib
            scalers = joblib.load(scaler_file)
    
    return data, scalers


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess Google Trends data")
    parser.add_argument("--input", type=str, required=True,
                       help="Input CSV file path")
    parser.add_argument("--target", type=str,
                       help="Target column name for prediction")
    parser.add_argument("--no-normalize", action="store_true",
                       help="Skip normalization")
    parser.add_argument("--smoothing", type=int, default=7,
                       help="Smoothing window size")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load data
    data = pd.read_csv(args.input, index_col=0, parse_dates=True)
    print(f"Loaded data shape: {data.shape}")
    
    # Configure preprocessing
    config = DataConfig()
    config.normalize = not args.no_normalize
    config.smoothing_window = args.smoothing
    
    # Preprocess
    preprocessor = TrendsPreprocessor(config)
    processed_data = preprocessor.preprocess(data, target_column=args.target)
    
    print(f"\nProcessed data shape: {processed_data.shape}")
    print(f"Columns: {list(processed_data.columns)}")
    print(f"\nFirst few rows:")
    print(processed_data.head()) 