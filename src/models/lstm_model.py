"""
LSTM model implementation for Google Trends time series prediction.

This module implements a Long Short-Term Memory (LSTM) neural network
for predicting future trends based on historical data.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.optimizers import Adam
import joblib

from .base_model import BaseTrendsModel, ModelRegistry
from ..utils.config import ModelConfig


logger = logging.getLogger(__name__)


class LSTMTrendsModel(BaseTrendsModel):
    """
    LSTM model for Google Trends prediction.
    
    Implements a multi-layer LSTM architecture with dropout for
    time series forecasting.
    """
    
    def __init__(self, name: str = "LSTM_Trends_Model", 
                 config: Optional[ModelConfig] = None):
        """
        Initialize LSTM model.
        
        Args:
            name: Model name.
            config: Model configuration.
        """
        super().__init__(name, "lstm")
        self.model_config = config or ModelConfig()
        self.sequence_length = None
        self.n_features = None
        
        # Update base config with LSTM-specific parameters
        self.config.update({
            'units': self.model_config.lstm_units,
            'dropout': self.model_config.lstm_dropout,
            'epochs': self.model_config.lstm_epochs,
            'batch_size': self.model_config.lstm_batch_size,
            'learning_rate': 0.001,
            'early_stopping_patience': 10,
            'reduce_lr_patience': 5
        })
    
    def build_model(self, input_shape: Tuple[int, int], **kwargs) -> None:
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: (sequence_length, n_features)
            **kwargs: Additional parameters like units, dropout, etc.
        """
        self.sequence_length, self.n_features = input_shape
        
        # Update config with any provided kwargs
        self.config.update(kwargs)
        
        # Build sequential model
        self.model = keras.Sequential(name=self.name)
        
        # Input layer
        self.model.add(layers.Input(shape=input_shape))
        
        # LSTM layers
        units = self.config['units']
        dropout = self.config['dropout']
        
        # Add LSTM layers based on units configuration
        for i, n_units in enumerate(units[:-1]):
            self.model.add(layers.LSTM(
                n_units,
                return_sequences=True,
                dropout=dropout,
                recurrent_dropout=dropout,
                name=f'lstm_{i}'
            ))
        
        # Last LSTM layer (no return_sequences)
        self.model.add(layers.LSTM(
            units[-1],
            dropout=dropout,
            recurrent_dropout=dropout,
            name=f'lstm_{len(units)-1}'
        ))
        
        # Dense layers for output
        self.model.add(layers.Dense(64, activation='relu', name='dense_1'))
        self.model.add(layers.Dropout(dropout, name='dropout_final'))
        self.model.add(layers.Dense(1, name='output'))
        
        # Compile model
        optimizer = Adam(learning_rate=self.config['learning_rate'])
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        logger.info("Built LSTM model with shape: %s", input_shape)
        logger.info("Model architecture:\n%s", self.model.summary())
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              **kwargs) -> Dict:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training sequences of shape (samples, time_steps, features).
            y_train: Training targets.
            X_val: Validation sequences.
            y_val: Validation targets.
            **kwargs: Additional training parameters.
        
        Returns:
            Training history dictionary.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Update training parameters
        epochs = kwargs.get('epochs', self.config['epochs'])
        batch_size = kwargs.get('batch_size', self.config['batch_size'])
        verbose = kwargs.get('verbose', 1)
        
        # Prepare callbacks
        callback_list = []
        
        # Early stopping
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=self.config['early_stopping_patience'],
            restore_best_weights=True,
            verbose=verbose
        )
        callback_list.append(early_stop)
        
        # Reduce learning rate on plateau
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=self.config['reduce_lr_patience'],
            min_lr=1e-7,
            verbose=verbose
        )
        callback_list.append(reduce_lr)
        
        # Model checkpoint
        if 'checkpoint_path' in kwargs:
            checkpoint = callbacks.ModelCheckpoint(
                kwargs['checkpoint_path'],
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True,
                verbose=verbose
            )
            callback_list.append(checkpoint)
        
        # Training
        logger.info("Starting LSTM training for %d epochs", epochs)
        
        # Ensure data types are correct
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        
        if X_val is not None and y_val is not None:
            X_val = np.array(X_val, dtype=np.float32)
            y_val = np.array(y_val, dtype=np.float32)
            validation_data = (X_val, y_val)
        else:
            validation_data = None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=verbose
        )
        
        # Update training history
        self.history['loss'] = history.history['loss']
        if 'val_loss' in history.history:
            self.history['val_loss'] = history.history['val_loss']
        
        # Store final metrics
        final_metrics = {}
        for metric in history.history:
            final_metrics[metric] = float(history.history[metric][-1])
        self.history['metrics'] = final_metrics
        
        self.is_trained = True
        logger.info("Training completed. Final loss: %.4f", final_metrics['loss'])
        
        return history.history
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input sequences of shape (samples, time_steps, features).
            **kwargs: Additional parameters.
        
        Returns:
            Predictions array.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Make predictions
        predictions = self.model.predict(X, **kwargs)
        
        return predictions.squeeze()
    
    def predict_sequence(self, initial_sequence: np.ndarray,
                        n_steps: int,
                        feature_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Predict multiple steps into the future using recursive prediction.
        
        Args:
            initial_sequence: Initial sequence of shape (time_steps, features).
            n_steps: Number of steps to predict.
            feature_indices: Indices of features to update with predictions.
                           If None, assumes single feature prediction.
        
        Returns:
            Array of predictions.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = []
        current_sequence = initial_sequence.copy()
        
        for _ in range(n_steps):
            # Prepare input
            X = current_sequence.reshape(1, self.sequence_length, self.n_features)
            
            # Predict next step
            pred = self.model.predict(X, verbose=0)
            predictions.append(pred[0, 0])
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1, axis=0)
            
            if feature_indices is None:
                # Single feature - update the last time step
                current_sequence[-1, 0] = pred[0, 0]
            else:
                # Multiple features - update specified indices
                for idx in feature_indices:
                    current_sequence[-1, idx] = pred[0, 0]
        
        return np.array(predictions)
    
    def evaluate(self, X_test: np.ndarray, 
                y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test sequences.
            y_test: True values.
        
        Returns:
            Dictionary of evaluation metrics.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred)
        
        # Add model-specific evaluation
        test_loss, test_mae, test_mape = self.model.evaluate(
            X_test, y_test, verbose=0
        )
        
        metrics['test_loss'] = test_loss
        metrics['test_mae'] = test_mae
        metrics['test_mape'] = test_mape
        
        return metrics
    
    def _save_model(self, filepath: str) -> None:
        """Save LSTM model."""
        if self.model is not None:
            # Save Keras model
            model_path = filepath.replace('.pkl', '.h5')
            self.model.save(model_path)
            
            # Save additional attributes
            attrs = {
                'sequence_length': self.sequence_length,
                'n_features': self.n_features,
                'model_config': self.model_config
            }
            attrs_path = filepath.replace('.pkl', '_attrs.joblib').replace('.h5', '_attrs.joblib')
            joblib.dump(attrs, attrs_path)
    
    def _load_model(self, filepath: str) -> None:
        """Load LSTM model."""
        # Load Keras model
        model_path = filepath.replace('.pkl', '.h5')
        if os.path.exists(model_path):
            self.model = keras.models.load_model(model_path)
            
            # Load additional attributes
            attrs_path = filepath.replace('.pkl', '_attrs.joblib').replace('.h5', '_attrs.joblib')
            if os.path.exists(attrs_path):
                attrs = joblib.load(attrs_path)
                self.sequence_length = attrs.get('sequence_length')
                self.n_features = attrs.get('n_features')
                self.model_config = attrs.get('model_config', ModelConfig())
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores (not directly available for LSTM).
        
        Returns a placeholder message.
        """
        return {
            "message": "Feature importance not directly available for LSTM models. "
                      "Consider using attention mechanisms or SHAP values for interpretation."
        }


# Register the model
ModelRegistry.register(LSTMTrendsModel, 'lstm')


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create sample data
    n_samples = 1000
    sequence_length = 30
    n_features = 5
    
    X = np.random.randn(n_samples, sequence_length, n_features)
    y = np.random.randn(n_samples)
    
    # Split data
    train_size = int(0.8 * n_samples)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Create and train model
    model = LSTMTrendsModel(name="test_lstm")
    model.build_model(input_shape=(sequence_length, n_features))
    
    # Train with reduced epochs for testing
    config = ModelConfig()
    config.lstm_epochs = 5
    model.model_config = config
    model.config['epochs'] = 5
    
    history = model.train(X_train, y_train, X_val, y_val, verbose=1)
    
    # Evaluate
    metrics = model.evaluate(X_val, y_val)
    print("\nValidation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Make predictions
    predictions = model.predict(X_val[:10])
    print(f"\nFirst 10 predictions: {predictions}")
    
    # Save model
    model.save("./test_lstm_model.h5")
    print("\nModel saved successfully") 