"""
Base model interface for Google Trends prediction models.

This module defines the abstract base class that all prediction models
must implement to ensure consistency across different model types.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
import json


logger = logging.getLogger(__name__)


class BaseTrendsModel(ABC):
    """
    Abstract base class for all Google Trends prediction models.
    
    This class defines the interface that all models must implement,
    ensuring consistent behavior across different model types.
    """
    
    def __init__(self, name: str, model_type: str):
        """
        Initialize base model.
        
        Args:
            name: Model name for identification.
            model_type: Type of model (e.g., 'lstm', 'prophet', 'arima').
        """
        self.name = name
        self.model_type = model_type
        self.is_trained = False
        self.model = None
        self.history = {'loss': [], 'val_loss': [], 'metrics': {}}
        self.config = {}
        self.feature_names = None
        self.target_name = None
        
    @abstractmethod
    def build_model(self, input_shape: Tuple[int, ...], **kwargs) -> None:
        """
        Build the model architecture.
        
        Args:
            input_shape: Shape of input data.
            **kwargs: Additional model-specific parameters.
        """
        pass
    
    @abstractmethod
    def train(self, X_train: Union[np.ndarray, pd.DataFrame],
              y_train: Union[np.ndarray, pd.Series],
              X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
              y_val: Optional[Union[np.ndarray, pd.Series]] = None,
              **kwargs) -> Dict:
        """
        Train the model.
        
        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features (optional).
            y_val: Validation targets (optional).
            **kwargs: Additional training parameters.
        
        Returns:
            Dictionary containing training history and metrics.
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame],
                **kwargs) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features.
            **kwargs: Additional prediction parameters.
        
        Returns:
            Array of predictions.
        """
        pass
    
    @abstractmethod
    def evaluate(self, X_test: Union[np.ndarray, pd.DataFrame],
                 y_test: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features.
            y_test: True values.
        
        Returns:
            Dictionary of evaluation metrics.
        """
        pass
    
    def save(self, filepath: str) -> None:
        """
        Save model and metadata.
        
        Args:
            filepath: Path to save the model.
        """
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model-specific files
        self._save_model(filepath)
        
        # Save metadata
        metadata = {
            'name': self.name,
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'config': self.config,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'history': self.history,
            'saved_at': datetime.now().isoformat()
        }
        
        metadata_path = filepath.replace('.h5', '').replace('.pkl', '') + '_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Saved model to: %s", filepath)
    
    @abstractmethod
    def _save_model(self, filepath: str) -> None:
        """Save model-specific files."""
        pass
    
    def load(self, filepath: str) -> None:
        """
        Load model and metadata.
        
        Args:
            filepath: Path to load the model from.
        """
        # Load metadata
        metadata_path = filepath.replace('.h5', '').replace('.pkl', '') + '_metadata.json'
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.name = metadata.get('name', self.name)
            self.is_trained = metadata.get('is_trained', False)
            self.config = metadata.get('config', {})
            self.feature_names = metadata.get('feature_names')
            self.target_name = metadata.get('target_name')
            self.history = metadata.get('history', {'loss': [], 'val_loss': [], 'metrics': {}})
        
        # Load model-specific files
        self._load_model(filepath)
        
        logger.info("Loaded model from: %s", filepath)
    
    @abstractmethod
    def _load_model(self, filepath: str) -> None:
        """Load model-specific files."""
        pass
    
    def get_params(self) -> Dict:
        """Get model parameters."""
        return self.config.copy()
    
    def set_params(self, **params) -> None:
        """Set model parameters."""
        self.config.update(params)
    
    def summary(self) -> str:
        """Get model summary."""
        summary_lines = [
            f"Model: {self.name}",
            f"Type: {self.model_type}",
            f"Trained: {self.is_trained}",
            f"Target: {self.target_name}",
            f"Features: {self.feature_names}"
        ]
        
        if self.history.get('metrics'):
            summary_lines.append("\nLatest Metrics:")
            for metric, value in self.history['metrics'].items():
                if isinstance(value, (int, float)):
                    summary_lines.append(f"  {metric}: {value:.4f}")
        
        return '\n'.join(summary_lines)
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, 
                         y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate common evaluation metrics.
        
        Args:
            y_true: True values.
            y_pred: Predicted values.
        
        Returns:
            Dictionary of metrics.
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        # Ensure arrays are 1D
        y_true = np.squeeze(y_true)
        y_pred = np.squeeze(y_pred)
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE (avoid division by zero)
        mask = y_true != 0
        if np.any(mask):
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.nan
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'r2': r2
        }
    
    def plot_predictions(self, y_true: np.ndarray, 
                        y_pred: np.ndarray,
                        dates: Optional[pd.DatetimeIndex] = None,
                        title: Optional[str] = None) -> None:
        """
        Plot predictions vs actual values.
        
        Args:
            y_true: True values.
            y_pred: Predicted values.
            dates: Date index for x-axis.
            title: Plot title.
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        
        if dates is not None:
            plt.plot(dates, y_true, label='Actual', alpha=0.7)
            plt.plot(dates, y_pred, label='Predicted', alpha=0.7)
            plt.xlabel('Date')
        else:
            plt.plot(y_true, label='Actual', alpha=0.7)
            plt.plot(y_pred, label='Predicted', alpha=0.7)
            plt.xlabel('Time Step')
        
        plt.ylabel('Value')
        plt.title(title or f'{self.name} Predictions vs Actual')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self) -> None:
        """Plot training history if available."""
        if not self.history.get('loss'):
            logger.warning("No training history available")
            return
        
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(self.history['loss'], label='Training Loss')
        if self.history.get('val_loss'):
            plt.plot(self.history['val_loss'], label='Validation Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{self.name} Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


class ModelRegistry:
    """Registry for managing multiple models."""
    
    _models = {}
    
    @classmethod
    def register(cls, model_class: type, model_type: str) -> None:
        """Register a model class."""
        cls._models[model_type] = model_class
    
    @classmethod
    def get_model(cls, model_type: str, **kwargs) -> BaseTrendsModel:
        """Get a model instance by type."""
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available: {list(cls._models.keys())}")
        
        return cls._models[model_type](**kwargs)
    
    @classmethod
    def list_models(cls) -> List[str]:
        """List available model types."""
        return list(cls._models.keys()) 