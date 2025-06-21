"""
Prophet model implementation for Google Trends time series prediction.

This module implements Facebook's Prophet model for time series forecasting,
which is particularly good at handling seasonality and holidays.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import joblib
import json

from .base_model import BaseTrendsModel, ModelRegistry
from ..utils.config import ModelConfig


logger = logging.getLogger(__name__)


class ProphetTrendsModel(BaseTrendsModel):
    """
    Prophet model for Google Trends prediction.
    
    Implements Facebook's Prophet algorithm which excels at forecasting
    time series data with strong seasonal patterns.
    """
    
    def __init__(self, name: str = "Prophet_Trends_Model",
                 config: Optional[ModelConfig] = None):
        """
        Initialize Prophet model.
        
        Args:
            name: Model name.
            config: Model configuration.
        """
        super().__init__(name, "prophet")
        self.model_config = config or ModelConfig()
        
        # Update base config with Prophet-specific parameters
        self.config.update({
            'changepoint_prior_scale': self.model_config.prophet_changepoint_prior_scale,
            'seasonality_mode': self.model_config.prophet_seasonality_mode,
            'yearly_seasonality': self.model_config.prophet_yearly_seasonality,
            'weekly_seasonality': self.model_config.prophet_weekly_seasonality,
            'daily_seasonality': False,
            'interval_width': 0.95,
            'mcmc_samples': 0  # Use MAP estimation by default
        })
        
        # Store training data info
        self.train_dates = None
        self.data_frequency = None
    
    def build_model(self, **kwargs) -> None:
        """
        Build Prophet model.
        
        Args:
            **kwargs: Model parameters to override defaults.
        """
        # Update config with any provided kwargs
        self.config.update(kwargs)
        
        # Initialize Prophet model
        self.model = Prophet(
            changepoint_prior_scale=self.config['changepoint_prior_scale'],
            seasonality_mode=self.config['seasonality_mode'],
            yearly_seasonality=self.config['yearly_seasonality'],
            weekly_seasonality=self.config['weekly_seasonality'],
            daily_seasonality=self.config['daily_seasonality'],
            interval_width=self.config['interval_width'],
            mcmc_samples=self.config['mcmc_samples']
        )
        
        # Add custom seasonalities if specified
        if 'custom_seasonalities' in kwargs:
            for seasonality in kwargs['custom_seasonalities']:
                self.model.add_seasonality(**seasonality)
        
        logger.info("Built Prophet model with config: %s", self.config)
    
    def _prepare_data(self, dates: pd.DatetimeIndex, 
                     values: Union[np.ndarray, pd.Series]) -> pd.DataFrame:
        """
        Prepare data in Prophet's required format.
        
        Args:
            dates: DateTime index.
            values: Target values.
        
        Returns:
            DataFrame with 'ds' and 'y' columns.
        """
        df = pd.DataFrame({
            'ds': dates,
            'y': values
        })
        
        # Handle any missing values
        df = df.dropna()
        
        return df
    
    def train(self, X_train: Union[pd.DataFrame, pd.DatetimeIndex],
              y_train: Union[np.ndarray, pd.Series],
              X_val: Optional[Union[pd.DataFrame, pd.DatetimeIndex]] = None,
              y_val: Optional[Union[np.ndarray, pd.Series]] = None,
              **kwargs) -> Dict:
        """
        Train the Prophet model.
        
        Args:
            X_train: Training dates (DatetimeIndex or DataFrame with dates).
            y_train: Training target values.
            X_val: Validation dates (not used by Prophet, kept for interface compatibility).
            y_val: Validation values (not used by Prophet).
            **kwargs: Additional parameters.
        
        Returns:
            Training history dictionary.
        """
        if self.model is None:
            self.build_model(**kwargs)
        
        # Extract dates if DataFrame is passed
        if isinstance(X_train, pd.DataFrame):
            if 'ds' in X_train.columns:
                dates = pd.to_datetime(X_train['ds'])
            else:
                dates = X_train.index
        else:
            dates = X_train
        
        # Store training dates for later use
        self.train_dates = dates
        
        # Detect data frequency
        if len(dates) > 1:
            freq_counts = pd.Series(np.diff(dates)).value_counts()
            self.data_frequency = freq_counts.index[0]
        
        # Prepare data
        train_df = self._prepare_data(dates, y_train)
        
        # Fit model
        logger.info("Training Prophet model on %d samples", len(train_df))
        self.model.fit(train_df)
        
        self.is_trained = True
        
        # Prophet doesn't provide traditional training history
        # We can perform cross-validation to get metrics
        if kwargs.get('perform_cv', False):
            cv_results = self._perform_cross_validation()
            self.history['metrics'] = cv_results
        
        # Store feature info
        self.target_name = 'y'
        self.feature_names = ['ds']
        
        logger.info("Prophet model training completed")
        
        return {'status': 'completed', 'samples': len(train_df)}
    
    def predict(self, X: Union[pd.DataFrame, pd.DatetimeIndex, int],
                **kwargs) -> np.ndarray:
        """
        Make predictions using Prophet.
        
        Args:
            X: Either:
               - DatetimeIndex or DataFrame with dates to predict
               - Integer number of periods to forecast
            **kwargs: Additional parameters like 'freq' for period specification.
        
        Returns:
            Array of predictions.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create future dataframe
        if isinstance(X, int):
            # Predict n periods into the future
            periods = X
            freq = kwargs.get('freq', self.data_frequency or 'D')
            future = self.model.make_future_dataframe(periods=periods, freq=freq)
        else:
            # Predict for specific dates
            if isinstance(X, pd.DataFrame):
                if 'ds' in X.columns:
                    dates = pd.to_datetime(X['ds'])
                else:
                    dates = X.index
            else:
                dates = X
            
            future = pd.DataFrame({'ds': dates})
        
        # Make predictions
        forecast = self.model.predict(future)
        
        # Extract predictions (yhat)
        predictions = forecast['yhat'].values
        
        # If predicting future periods, return only the future predictions
        if isinstance(X, int) and self.train_dates is not None:
            n_train = len(self.train_dates)
            predictions = predictions[n_train:]
        
        return predictions
    
    def predict_with_uncertainty(self, X: Union[pd.DataFrame, pd.DatetimeIndex, int],
                               **kwargs) -> Dict[str, np.ndarray]:
        """
        Make predictions with uncertainty intervals.
        
        Args:
            X: Input for prediction (same as predict method).
            **kwargs: Additional parameters.
        
        Returns:
            Dictionary with 'yhat', 'yhat_lower', and 'yhat_upper'.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create future dataframe
        if isinstance(X, int):
            periods = X
            freq = kwargs.get('freq', self.data_frequency or 'D')
            future = self.model.make_future_dataframe(periods=periods, freq=freq)
        else:
            if isinstance(X, pd.DataFrame):
                if 'ds' in X.columns:
                    dates = pd.to_datetime(X['ds'])
                else:
                    dates = X.index
            else:
                dates = X
            
            future = pd.DataFrame({'ds': dates})
        
        # Make predictions
        forecast = self.model.predict(future)
        
        # Extract results
        result = {
            'yhat': forecast['yhat'].values,
            'yhat_lower': forecast['yhat_lower'].values,
            'yhat_upper': forecast['yhat_upper'].values
        }
        
        # If predicting future periods, return only the future predictions
        if isinstance(X, int) and self.train_dates is not None:
            n_train = len(self.train_dates)
            for key in result:
                result[key] = result[key][n_train:]
        
        return result
    
    def evaluate(self, X_test: Union[pd.DataFrame, pd.DatetimeIndex],
                 y_test: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test dates.
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
        
        # Add Prophet-specific metrics if available
        if hasattr(self.model, 'params'):
            metrics['changepoint_prior_scale'] = self.model.changepoint_prior_scale
        
        return metrics
    
    def _perform_cross_validation(self, initial: str = '730 days',
                                period: str = '180 days',
                                horizon: str = '90 days') -> Dict[str, float]:
        """
        Perform time series cross-validation.
        
        Args:
            initial: Initial training period.
            period: Period between cutoff dates.
            horizon: Forecast horizon.
        
        Returns:
            Dictionary of cross-validation metrics.
        """
        try:
            # Perform cross-validation
            df_cv = cross_validation(self.model, initial=initial, 
                                   period=period, horizon=horizon)
            
            # Calculate performance metrics
            df_p = performance_metrics(df_cv)
            
            # Return mean metrics
            metrics = {
                'cv_rmse': df_p['rmse'].mean(),
                'cv_mae': df_p['mae'].mean(),
                'cv_mape': df_p['mape'].mean(),
                'cv_coverage': df_p['coverage'].mean()
            }
            
            return metrics
            
        except Exception as e:
            logger.warning("Cross-validation failed: %s", str(e))
            return {}
    
    def plot_components(self, **kwargs) -> None:
        """Plot the forecast components (trend, seasonalities)."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Need to make a forecast first
        future = self.model.make_future_dataframe(periods=0)
        forecast = self.model.predict(future)
        
        # Plot components
        self.model.plot_components(forecast, **kwargs)
    
    def _save_model(self, filepath: str) -> None:
        """Save Prophet model."""
        if self.model is not None:
            # Prophet models can be serialized with joblib
            model_path = filepath.replace('.h5', '.pkl')
            joblib.dump(self.model, model_path)
            
            # Save additional attributes
            attrs = {
                'train_dates': self.train_dates.tolist() if self.train_dates is not None else None,
                'data_frequency': str(self.data_frequency) if self.data_frequency is not None else None,
                'model_config': self.model_config
            }
            attrs_path = filepath.replace('.h5', '_attrs.json').replace('.pkl', '_attrs.json')
            with open(attrs_path, 'w') as f:
                json.dump(attrs, f, default=str)
    
    def _load_model(self, filepath: str) -> None:
        """Load Prophet model."""
        # Load Prophet model
        model_path = filepath.replace('.h5', '.pkl')
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            
            # Load additional attributes
            attrs_path = filepath.replace('.h5', '_attrs.json').replace('.pkl', '_attrs.json')
            if os.path.exists(attrs_path):
                with open(attrs_path, 'r') as f:
                    attrs = json.load(f)
                
                if attrs.get('train_dates'):
                    self.train_dates = pd.DatetimeIndex(attrs['train_dates'])
                if attrs.get('data_frequency'):
                    self.data_frequency = pd.Timedelta(attrs['data_frequency'])
                self.model_config = attrs.get('model_config', ModelConfig())
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get component importance from Prophet model.
        
        Returns:
            Dictionary with component contributions.
        """
        if not self.is_trained:
            return {"message": "Model not trained yet."}
        
        # Make a forecast to get components
        future = self.model.make_future_dataframe(periods=0)
        forecast = self.model.predict(future)
        
        # Calculate relative importance of components
        components = {}
        
        # Trend
        if 'trend' in forecast.columns:
            components['trend'] = np.std(forecast['trend'])
        
        # Seasonalities
        for col in forecast.columns:
            if col.endswith('_seasonal') or col in ['yearly', 'weekly', 'daily']:
                components[col] = np.std(forecast[col])
        
        # Normalize to percentages
        total = sum(components.values())
        if total > 0:
            components = {k: v/total * 100 for k, v in components.items()}
        
        return components


# Register the model
ModelRegistry.register(ProphetTrendsModel, 'prophet')


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create sample data
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    trend = np.linspace(100, 200, len(dates))
    seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    noise = np.random.normal(0, 5, len(dates))
    values = trend + seasonal + noise
    
    # Split data
    split_date = '2023-06-01'
    train_mask = dates < split_date
    
    X_train = dates[train_mask]
    y_train = values[train_mask]
    X_test = dates[~train_mask]
    y_test = values[~train_mask]
    
    # Create and train model
    model = ProphetTrendsModel(name="test_prophet")
    model.build_model()
    
    history = model.train(X_train, y_train)
    print("Training completed:", history)
    
    # Make predictions
    predictions = model.predict(30)  # Predict next 30 days
    print(f"\nPredicted next 30 days: {predictions[:5]}... (showing first 5)")
    
    # Evaluate on test set
    metrics = model.evaluate(X_test, y_test)
    print("\nTest Metrics:")
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{metric}: {value:.4f}")
    
    # Get component importance
    importance = model.get_feature_importance()
    print("\nComponent Importance:")
    for component, value in importance.items():
        print(f"{component}: {value:.2f}%")
    
    # Save model
    model.save("./test_prophet_model.pkl")
    print("\nModel saved successfully") 