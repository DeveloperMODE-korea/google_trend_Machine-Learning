"""
Example usage of Google Trends ML Predictor.

This script demonstrates how to use the system to collect data,
train models, and generate predictions.
"""

from src.predictor import TrendsPredictor
from src.utils.config import AppConfig
from src.utils.visualizer import plot_quick_summary


def example_basic_usage():
    """Basic usage example with default settings."""
    print("=== Basic Usage Example ===\n")
    
    # Create predictor with default config
    predictor = TrendsPredictor()
    
    # Define keywords to analyze
    keywords = ["machine learning", "artificial intelligence", "deep learning"]
    target_keyword = "machine learning"
    
    # Run complete pipeline
    results = predictor.run_complete_pipeline(
        keywords=keywords,
        target_keyword=target_keyword,
        prediction_periods=30,  # Predict next 30 days
        model_types=["lstm", "prophet"]
    )
    
    print("\nPipeline completed!")
    print(f"Check the generated files for detailed results.")


def example_custom_config():
    """Example with custom configuration."""
    print("\n=== Custom Configuration Example ===\n")
    
    # Load custom config from file
    config = AppConfig.from_yaml("config.yaml")
    
    # Modify some settings
    config.model.lstm_epochs = 50  # Reduce epochs for faster demo
    config.data.smoothing_window = 3  # Less smoothing
    
    # Create predictor
    predictor = TrendsPredictor(config)
    
    # Tech-related keywords
    keywords = ["python programming", "javascript", "data science", "cloud computing"]
    target_keyword = "python programming"
    
    # Collect and analyze data
    print("Collecting data...")
    data = predictor.collect_data(keywords)
    
    print(f"Collected {len(data)} data points")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    
    # Quick visualization
    plot_quick_summary(data, target_keyword)
    
    # Prepare and train
    print("\nPreparing data...")
    train_data, val_data, test_data = predictor.prepare_data(data, target_keyword)
    
    print("Training models...")
    predictor.train_models(train_data, val_data, target_keyword, ["prophet"])
    
    print("Evaluating...")
    metrics = predictor.evaluate_models(test_data, target_keyword)
    
    print("\nModel Performance:")
    for model, model_metrics in metrics.items():
        print(f"\n{model}:")
        for metric, value in model_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")


def example_data_collection_only():
    """Example of just collecting and visualizing data."""
    print("\n=== Data Collection Only Example ===\n")
    
    from src.data.collector import TrendsCollector
    from src.utils.visualizer import TrendsVisualizer
    
    # Create collector
    collector = TrendsCollector()
    
    # Collect data for specific keywords
    keywords = ["covid vaccine", "pandemic", "health"]
    data = collector.collect_trends(
        keywords=keywords,
        timeframe="2020-01-01 2024-01-01"  # Specific date range
    )
    
    print(f"Collected data shape: {data.shape}")
    
    # Visualize
    visualizer = TrendsVisualizer()
    visualizer.plot_trends(data, title="COVID-related Search Trends")
    visualizer.plot_correlation_heatmap(data)
    
    # Get related queries
    related = collector.get_related_queries("covid vaccine")
    print("\nTop related queries for 'covid vaccine':")
    if not related['top'].empty:
        print(related['top'].head())


def example_model_comparison():
    """Example comparing different models."""
    print("\n=== Model Comparison Example ===\n")
    
    from src.models.lstm_model import LSTMTrendsModel
    from src.models.prophet_model import ProphetTrendsModel
    from src.data.preprocessor import TrendsPreprocessor
    
    # Create predictor
    predictor = TrendsPredictor()
    
    # Simple keywords for faster processing
    keywords = ["bitcoin", "cryptocurrency"]
    target = "bitcoin"
    
    # Collect data
    print("Collecting data...")
    data = predictor.collect_data(keywords, timeframe="today 2-y")
    
    # Preprocess
    preprocessor = TrendsPreprocessor()
    processed_data = preprocessor.preprocess(data, target)
    
    # Split data
    train_data, val_data, test_data = preprocessor.train_test_split_temporal(processed_data)
    
    # Train both models
    print("\nTraining models...")
    predictor.train_models(train_data, val_data, target, ["lstm", "prophet"])
    
    # Evaluate
    print("\nEvaluating models...")
    metrics = predictor.evaluate_models(test_data, target)
    
    # Visualize comparison
    predictor.visualizer.plot_model_metrics(
        metrics,
        metric_names=["rmse", "mae", "r2"],
        title="LSTM vs Prophet Performance"
    )
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = predictor.predict_future(30, processed_data, target)
    
    # Plot predictions
    predictor.visualizer.plot_predictions_comparison(
        test_data[target],
        predictions,
        title="Model Predictions Comparison"
    )


def example_batch_keywords():
    """Example with many keywords (batching)."""
    print("\n=== Batch Keywords Example ===\n")
    
    from src.data.collector import batch_collect_keywords
    
    # Many keywords (will be batched automatically)
    keywords = [
        "machine learning", "deep learning", "neural network",
        "artificial intelligence", "computer vision", "natural language processing",
        "reinforcement learning", "tensorflow", "pytorch", "scikit-learn"
    ]
    
    print(f"Collecting data for {len(keywords)} keywords...")
    print("This will be done in batches of 5...")
    
    # Collect with batching
    data = batch_collect_keywords(keywords)
    
    print(f"\nCollected data shape: {data.shape}")
    print(f"Successfully collected: {list(data.columns)}")
    
    # Show correlation between ML terms
    from src.utils.visualizer import TrendsVisualizer
    visualizer = TrendsVisualizer()
    visualizer.plot_correlation_heatmap(
        data,
        title="Correlation between ML/AI Terms"
    )


if __name__ == "__main__":
    import sys
    
    print("Google Trends ML Predictor - Examples")
    print("=====================================\n")
    
    if len(sys.argv) > 1:
        example_name = sys.argv[1]
        
        examples = {
            "basic": example_basic_usage,
            "custom": example_custom_config,
            "collect": example_data_collection_only,
            "compare": example_model_comparison,
            "batch": example_batch_keywords
        }
        
        if example_name in examples:
            examples[example_name]()
        else:
            print(f"Unknown example: {example_name}")
            print(f"Available examples: {', '.join(examples.keys())}")
    else:
        print("Running all examples...\n")
        print("(To run specific example, use: python example_usage.py [example_name])\n")
        
        # Run basic example only by default
        example_basic_usage()
        
        print("\n" + "="*50)
        print("To run other examples:")
        print("  python example_usage.py custom    # Custom configuration")
        print("  python example_usage.py collect   # Data collection only")
        print("  python example_usage.py compare   # Model comparison")
        print("  python example_usage.py batch     # Batch keywords") 