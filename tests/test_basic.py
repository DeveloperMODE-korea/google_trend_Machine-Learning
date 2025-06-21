"""
Basic tests for Google Trends ML Predictor.
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import AppConfig, DataConfig, ModelConfig
from src.data.preprocessor import TrendsPreprocessor
import pandas as pd
import numpy as np


class TestConfig(unittest.TestCase):
    """Test configuration classes."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = AppConfig()
        self.assertIsNotNone(config.data)
        self.assertIsNotNone(config.model)
        self.assertEqual(config.log_level, "INFO")
    
    def test_data_config(self):
        """Test data configuration."""
        config = DataConfig()
        self.assertEqual(config.geo, "")  # Worldwide
        self.assertEqual(config.timeframe, "today 5-y")
        self.assertTrue(config.normalize)
    
    def test_model_config(self):
        """Test model configuration."""
        config = ModelConfig()
        self.assertEqual(config.test_size, 0.2)
        self.assertEqual(config.lstm_units, [128, 64, 32])
        self.assertEqual(config.lstm_epochs, 100)


class TestPreprocessor(unittest.TestCase):
    """Test data preprocessing functionality."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample time series data
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        self.data = pd.DataFrame({
            'keyword1': np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 50 + 50,
            'keyword2': np.cos(np.arange(len(dates)) * 2 * np.pi / 365) * 30 + 60
        }, index=dates)
        
        # Add some missing values
        self.data.iloc[10:15, 0] = np.nan
    
    def test_preprocessing(self):
        """Test basic preprocessing."""
        preprocessor = TrendsPreprocessor()
        processed = preprocessor.preprocess(self.data, save_processed=False)
        
        # Check no missing values after preprocessing
        self.assertEqual(processed.isnull().sum().sum(), 0)
        
        # Check time features added
        self.assertIn('month', processed.columns)
        self.assertIn('day', processed.columns)
        self.assertIn('day_of_week', processed.columns)
    
    def test_sequence_creation(self):
        """Test sequence creation for LSTM."""
        preprocessor = TrendsPreprocessor()
        processed = preprocessor.preprocess(self.data, target_column='keyword1', save_processed=False)
        
        # Create sequences
        X, y = preprocessor.create_sequences(processed, 'keyword1', sequence_length=30)
        
        # Check shapes
        self.assertEqual(X.shape[1], 30)  # sequence length
        self.assertEqual(X.shape[2], processed.shape[1])  # features
        self.assertEqual(len(y), len(X))
    
    def test_train_test_split(self):
        """Test temporal train/test split."""
        preprocessor = TrendsPreprocessor()
        processed = preprocessor.preprocess(self.data, save_processed=False)
        
        train, val, test = preprocessor.train_test_split_temporal(
            processed, test_size=0.2, validation_size=0.1
        )
        
        # Check sizes
        total_size = len(processed)
        self.assertAlmostEqual(len(test) / total_size, 0.2, places=1)
        
        # Check temporal order
        self.assertTrue(train.index.max() < val.index.min())
        self.assertTrue(val.index.max() < test.index.min())


class TestDataCollection(unittest.TestCase):
    """Test data collection functionality."""
    
    def test_timeframe_parsing(self):
        """Test timeframe date parsing."""
        from src.utils.config import get_timeframe_dates
        
        # Test relative timeframe
        start, end = get_timeframe_dates("today 5-y")
        self.assertIsInstance(start, str)
        self.assertIsInstance(end, str)
        
        # Test absolute timeframe
        start, end = get_timeframe_dates("2020-01-01 2021-01-01")
        self.assertEqual(start, "2020-01-01")
        self.assertEqual(end, "2021-01-01")


if __name__ == '__main__':
    unittest.main() 