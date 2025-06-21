"""
Main prediction pipeline for Google Trends forecasting.

This module orchestrates the entire workflow from data collection
to model training and prediction generation.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

from .data.collector import TrendsCollector, batch_collect_keywords
from .data.collector_safe import SafeTrendsCollector, batch_collect_keywords_safe
from .data.preprocessor import TrendsPreprocessor, load_processed_data
from .models.base_model import ModelRegistry
from .models.lstm_model import LSTMTrendsModel
from .models.prophet_model import ProphetTrendsModel
from .utils.config import AppConfig, DataConfig, ModelConfig
from .utils.visualizer import TrendsVisualizer, plot_quick_summary


logger = logging.getLogger(__name__)


class TrendsPredictor:
    """
    Main class for Google Trends prediction pipeline.
    
    Handles the complete workflow from data collection to prediction.
    """
    
    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize the predictor.
        
        Args:
            config: Application configuration.
        """
        self.config = config or AppConfig()
        # Use SafeTrendsCollector for better compatibility
        try:
            self.collector = SafeTrendsCollector(self.config.data)
            logger.info("Using SafeTrendsCollector")
        except Exception as e:
            logger.warning("SafeTrendsCollector failed, falling back to TrendsCollector: %s", str(e))
            self.collector = TrendsCollector(self.config.data)
        self.preprocessor = TrendsPreprocessor(self.config.data)
        self.visualizer = TrendsVisualizer(self.config)
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        
        # Set up logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.log_file),
                logging.StreamHandler()
            ]
        )
    
    def collect_data(self, keywords: List[str],
                    timeframe: Optional[str] = None,
                    geo: Optional[str] = None) -> pd.DataFrame:
        """
        Collect Google Trends data for specified keywords.
        
        Args:
            keywords: List of keywords to collect.
            timeframe: Time period for collection.
            geo: Geographic location.
        
        Returns:
            DataFrame with collected data.
        """
        logger.info("Starting data collection for keywords: %s", keywords)
        
        # Handle multiple keywords with batching
        if len(keywords) > 5:
            try:
                data = batch_collect_keywords_safe(keywords, self.config.data)
            except Exception as e:
                logger.warning("Safe batch collection failed, trying regular: %s", str(e))
                data = batch_collect_keywords(keywords, self.config.data)
        else:
            data = self.collector.collect_trends(
                keywords, timeframe=timeframe, geo=geo
            )
        
        if data.empty:
            raise ValueError("No data collected from Google Trends")
        
        logger.info("Data collection completed. Shape: %s", data.shape)
        return data
    
    def prepare_data(self, data: pd.DataFrame,
                    target_keyword: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Preprocess data and split into train/val/test sets.
        
        Args:
            data: Raw trends data.
            target_keyword: Keyword to predict.
        
        Returns:
            Tuple of (train_data, val_data, test_data).
        """
        logger.info("Preprocessing data for target: %s", target_keyword)
        
        # Preprocess data
        processed_data = self.preprocessor.preprocess(
            data, target_column=target_keyword
        )
        
        # Split data
        train_data, val_data, test_data = self.preprocessor.train_test_split_temporal(
            processed_data,
            test_size=self.config.model.test_size,
            validation_size=self.config.model.validation_split
        )
        
        logger.info("Data split - Train: %d, Val: %d, Test: %d",
                   len(train_data), len(val_data), len(test_data))
        
        return train_data, val_data, test_data
    
    def train_models(self, train_data: pd.DataFrame,
                    val_data: pd.DataFrame,
                    target_column: str,
                    model_types: Optional[List[str]] = None) -> Dict[str, object]:
        """
        Train multiple models on the data.
        
        Args:
            train_data: Training data.
            val_data: Validation data.
            target_column: Column to predict.
            model_types: List of model types to train. If None, trains all.
        
        Returns:
            Dictionary of trained models.
        """
        if model_types is None:
            model_types = ['lstm', 'prophet']
        
        logger.info("Training models: %s", model_types)
        
        for model_type in model_types:
            try:
                if model_type == 'lstm':
                    self._train_lstm(train_data, val_data, target_column)
                elif model_type == 'prophet':
                    self._train_prophet(train_data, val_data, target_column)
                else:
                    logger.warning("Unknown model type: %s", model_type)
            except Exception as e:
                logger.error("Failed to train %s model: %s", model_type, str(e))
        
        return self.models
    
    def _train_lstm(self, train_data: pd.DataFrame,
                   val_data: pd.DataFrame,
                   target_column: str) -> None:
        """Train LSTM model."""
        logger.info("Training LSTM model")
        
        # Create sequences
        sequence_length = min(30, len(train_data) // 4)  # Adaptive sequence length
        
        # Prepare training sequences
        X_train, y_train = self.preprocessor.create_sequences(
            train_data, target_column, sequence_length
        )
        
        # Check if we have enough validation data for sequences
        if len(val_data) > sequence_length:
            X_val, y_val = self.preprocessor.create_sequences(
                val_data, target_column, sequence_length
            )
        else:
            # If validation data is too small, use part of training data
            logger.warning("Validation data too small for sequences, using training split")
            split_point = int(0.8 * len(X_train))
            X_val, y_val = X_train[split_point:], y_train[split_point:]
            X_train, y_train = X_train[:split_point], y_train[:split_point]
        
        # Create and train model
        lstm_model = LSTMTrendsModel(
            name=f"LSTM_{target_column}",
            config=self.config.model
        )
        
        # Build model
        n_features = train_data.shape[1]
        lstm_model.build_model(input_shape=(sequence_length, n_features))
        
        # Validate data before training
        if len(X_train) == 0:
            raise ValueError("Training data is empty after sequence creation")
        if len(X_val) == 0:
            logger.warning("Validation data is empty, training without validation")
            X_val, y_val = None, None
        
        logger.info("Training LSTM with %d training samples, %d validation samples", 
                   len(X_train), len(X_val) if X_val is not None else 0)
        
        # Train
        history = lstm_model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=self.config.model.lstm_epochs,
            batch_size=self.config.model.lstm_batch_size
        )
        
        # Store model
        self.models['lstm'] = lstm_model
        
        # Save model
        model_path = os.path.join(
            self.config.model.model_save_path,
            f"lstm_{target_column}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
        )
        os.makedirs(self.config.model.model_save_path, exist_ok=True)
        lstm_model.save(model_path)
        
        logger.info("LSTM model trained and saved to: %s", model_path)
    
    def _train_prophet(self, train_data: pd.DataFrame,
                      val_data: pd.DataFrame,
                      target_column: str) -> None:
        """Train Prophet model."""
        logger.info("Training Prophet model")
        
        # Prophet needs dates and target values
        # Combine train and val for Prophet (it doesn't use validation split)
        full_train = pd.concat([train_data, val_data])
        
        # Create and train model
        prophet_model = ProphetTrendsModel(
            name=f"Prophet_{target_column}",
            config=self.config.model
        )
        
        # Train
        prophet_model.train(
            full_train.index,
            full_train[target_column]
        )
        
        # Store model
        self.models['prophet'] = prophet_model
        
        # Save model
        model_path = os.path.join(
            self.config.model.model_save_path,
            f"prophet_{target_column}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        )
        os.makedirs(self.config.model.model_save_path, exist_ok=True)
        prophet_model.save(model_path)
        
        logger.info("Prophet model trained and saved to: %s", model_path)
    
    def evaluate_models(self, test_data: pd.DataFrame,
                       target_column: str) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models on test data.
        
        Args:
            test_data: Test data.
            target_column: Target column name.
        
        Returns:
            Dictionary of model metrics.
        """
        logger.info("Evaluating models on test data")
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'lstm':
                    # Create test sequences
                    X_test, y_test = self.preprocessor.create_sequences(
                        test_data, target_column, sequence_length=30
                    )
                    metrics = model.evaluate(X_test, y_test)
                    
                elif model_name == 'prophet':
                    # Prophet evaluation
                    metrics = model.evaluate(
                        test_data.index,
                        test_data[target_column]
                    )
                
                self.metrics[model_name] = metrics
                
                logger.info("%s metrics: %s", model_name, 
                           {k: f"{v:.4f}" for k, v in metrics.items() 
                            if isinstance(v, (int, float))})
                
            except Exception as e:
                logger.error("Failed to evaluate %s: %s", model_name, str(e))
                self.metrics[model_name] = {'error': str(e)}
        
        return self.metrics
    
    def predict_future(self, periods: int,
                      last_data: pd.DataFrame,
                      target_column: str) -> Dict[str, pd.Series]:
        """
        Generate future predictions using all trained models.
        
        Args:
            periods: Number of periods to predict.
            last_data: Most recent data for prediction base.
            target_column: Target column to predict.
        
        Returns:
            Dictionary of model predictions.
        """
        logger.info("Generating predictions for %d periods", periods)
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'lstm':
                    # LSTM prediction
                    # Use last sequence_length points
                    sequence_length = 30
                    last_sequence = last_data.iloc[-sequence_length:].values
                    
                    # Predict recursively
                    target_idx = last_data.columns.get_loc(target_column)
                    predictions = model.predict_sequence(
                        last_sequence, 
                        n_steps=periods,
                        feature_indices=[target_idx]
                    )
                    
                    # Create date index for predictions
                    last_date = last_data.index[-1]
                    freq = pd.infer_freq(last_data.index)
                    if freq is None or 'W-SUN' in str(freq):
                        freq = 'D'  # Default to daily if frequency can't be inferred
                    
                    # Create safe timedelta
                    if freq == 'W':
                        delta = pd.Timedelta(days=7)
                    elif freq == 'M':
                        delta = pd.Timedelta(days=30)
                    else:
                        delta = pd.Timedelta(days=1)
                    
                    future_dates = pd.date_range(
                        start=last_date + delta,
                        periods=periods,
                        freq=freq
                    )
                    
                    # Inverse transform if needed
                    if target_column in self.preprocessor.scalers:
                        predictions = self.preprocessor.scalers[target_column].inverse_transform(
                            predictions.reshape(-1, 1)
                        ).flatten()
                    
                    self.predictions[model_name] = pd.Series(
                        predictions, index=future_dates
                    )
                    
                elif model_name == 'prophet':
                    # Prophet prediction
                    predictions = model.predict(periods)
                    
                    # Get dates
                    last_date = last_data.index[-1]
                    freq = pd.infer_freq(last_data.index)
                    if freq is None or 'W-SUN' in str(freq):
                        freq = 'D'  # Default to daily if frequency can't be inferred
                    
                    # Create safe timedelta
                    if freq == 'W':
                        delta = pd.Timedelta(days=7)
                    elif freq == 'M':
                        delta = pd.Timedelta(days=30)
                    else:
                        delta = pd.Timedelta(days=1)
                    
                    future_dates = pd.date_range(
                        start=last_date + delta,
                        periods=periods,
                        freq=freq
                    )
                    
                    # Inverse transform if needed
                    if target_column in self.preprocessor.scalers:
                        predictions = self.preprocessor.scalers[target_column].inverse_transform(
                            predictions.reshape(-1, 1)
                        ).flatten()
                    
                    self.predictions[model_name] = pd.Series(
                        predictions, index=future_dates
                    )
                
                logger.info("%s predicted %d future values", 
                           model_name, len(self.predictions[model_name]))
                
            except Exception as e:
                logger.error("Failed to predict with %s: %s", model_name, str(e))
        
        return self.predictions
    
    def visualize_results(self, historical_data: pd.DataFrame,
                         target_column: str,
                         save_plots: bool = True) -> None:
        """
        Create visualizations of results.
        
        Args:
            historical_data: Historical data for context.
            target_column: Target column that was predicted.
            save_plots: Whether to save plots to files.
        """
        logger.info("Creating visualizations")
        
        # Plot predictions comparison
        if self.predictions:
            # Get recent historical data for comparison
            recent_data = historical_data[target_column].iloc[-100:]
            
            self.visualizer.plot_predictions_comparison(
                recent_data,
                self.predictions,
                title=f"Predictions for {target_column}",
                save_path="predictions_comparison.png" if save_plots else None
            )
        
        # Plot model metrics
        if self.metrics:
            self.visualizer.plot_model_metrics(
                self.metrics,
                metric_names=['rmse', 'mae', 'mape', 'r2'],
                save_path="model_metrics.png" if save_plots else None
            )
        
        # Create interactive dashboard
        dashboard = self.visualizer.create_dashboard(
            historical_data,
            self.predictions,
            self.metrics
        )
        
        if save_plots:
            dashboard.write_html("trends_dashboard.html")
            logger.info("Saved interactive dashboard to trends_dashboard.html")
        
        return dashboard
    
    def run_complete_pipeline(self, keywords: List[str],
                            target_keyword: str,
                            prediction_periods: int = 30,
                            model_types: Optional[List[str]] = None) -> Dict:
        """
        Run the complete prediction pipeline.
        
        Args:
            keywords: Keywords to collect data for.
            target_keyword: Specific keyword to predict.
            prediction_periods: Number of periods to predict.
            model_types: Models to use.
        
        Returns:
            Dictionary with results.
        """
        logger.info("Starting complete pipeline")
        
        # 1. Collect data
        logger.info("Step 1: Collecting data")
        raw_data = self.collect_data(keywords)
        
        # 2. Prepare data
        logger.info("Step 2: Preparing data")
        train_data, val_data, test_data = self.prepare_data(raw_data, target_keyword)
        
        # 3. Train models
        logger.info("Step 3: Training models")
        self.train_models(train_data, val_data, target_keyword, model_types)
        
        # 4. Evaluate models
        logger.info("Step 4: Evaluating models")
        self.evaluate_models(test_data, target_keyword)
        
        # 5. Generate predictions
        logger.info("Step 5: Generating predictions")
        # Use all available data for final predictions
        all_processed_data = self.preprocessor.preprocess(raw_data, target_keyword, save_processed=False)
        self.predict_future(prediction_periods, all_processed_data, target_keyword)
        
        # 6. Visualize results
        logger.info("Step 6: Creating visualizations")
        self.visualize_results(raw_data, target_keyword)
        
        # 7. Save results summary
        results = {
            'timestamp': datetime.now().isoformat(),
            'keywords': keywords,
            'target_keyword': target_keyword,
            'data_shape': list(raw_data.shape),
            'date_range': [str(raw_data.index.min()), str(raw_data.index.max())],
            'models_trained': list(self.models.keys()),
            'metrics': {k: {mk: float(mv) for mk, mv in v.items() if isinstance(mv, (int, float))} 
                       for k, v in self.metrics.items()},
            'prediction_periods': prediction_periods,
            'predictions_saved': True
        }
        
        # Save predictions
        predictions_df = pd.DataFrame(self.predictions)
        predictions_df.to_csv(f"predictions_{target_keyword}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
        # Save results summary
        with open(f"results_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("Pipeline completed successfully!")
        
        return results


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Google Trends ML Predictor")
    parser.add_argument("--keywords", type=str, required=True,
                       help="Comma-separated list of keywords")
    parser.add_argument("--target", type=str, required=True,
                       help="Target keyword to predict")
    parser.add_argument("--periods", type=int, default=30,
                       help="Number of periods to predict")
    parser.add_argument("--models", type=str, default="lstm,prophet",
                       help="Comma-separated list of models to use")
    parser.add_argument("--config", type=str,
                       help="Path to configuration YAML file")
    
    args = parser.parse_args()
    
    # Parse arguments
    keywords = [k.strip() for k in args.keywords.split(',')]
    model_types = [m.strip() for m in args.models.split(',')]
    
    # Load config if provided
    if args.config:
        config = AppConfig.from_yaml(args.config)
    else:
        config = AppConfig()
    
    # Create predictor and run pipeline
    predictor = TrendsPredictor(config)
    
    results = predictor.run_complete_pipeline(
        keywords=keywords,
        target_keyword=args.target,
        prediction_periods=args.periods,
        model_types=model_types
    )
    
    print("\n=== Pipeline Results ===")
    print(f"Data collected: {results['data_shape'][0]} time points, {results['data_shape'][1]} keywords")
    print(f"Date range: {results['date_range'][0]} to {results['date_range'][1]}")
    print(f"Models trained: {', '.join(results['models_trained'])}")
    print("\nModel Performance:")
    for model, metrics in results['metrics'].items():
        print(f"\n{model}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    print(f"\nPredictions generated for next {results['prediction_periods']} periods")
    print("\nResults saved to files.")


if __name__ == "__main__":
    main() 