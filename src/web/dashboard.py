"""
Streamlit Web Dashboard for Google Trends ML Predictor.

This module provides a web-based interface for collecting Google Trends data,
training ML models, and visualizing predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import sys
import os
from datetime import datetime, timedelta
import time
import json

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.predictor import TrendsPredictor
from src.data.collector import TrendsCollector
from src.utils.config import AppConfig, DataConfig, ModelConfig
from src.utils.visualizer import TrendsVisualizer


# Page configuration
st.set_page_config(
    page_title="Google Trends ML Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clear cache on startup to avoid JavaScript issues
if 'startup_complete' not in st.session_state:
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.startup_complete = True

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.3rem solid #1f77b4;
    }
    .stAlert > div {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None


def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Google Trends ML Predictor</h1>', 
                unsafe_allow_html=True)
    st.markdown("**Predict future search trends using machine learning**")
    
    # Sidebar for configuration
    setup_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs([
        "🔍 Data Collection", 
        "🤖 Model Training", 
        "📊 Predictions"
    ])
    
    with tab1:
        data_collection_tab()
    
    with tab2:
        model_training_tab()
    
    with tab3:
        predictions_tab()


def setup_sidebar():
    """Setup sidebar with configuration options."""
    
    st.sidebar.header("🛠️ Configuration")
    
    # Keywords input
    st.sidebar.subheader("Keywords")
    keywords_input = st.sidebar.text_area(
        "Enter keywords (one per line):",
        value="machine learning\nartificial intelligence\ndeep learning",
        height=100,
        help="Enter up to 5 keywords, one per line"
    )
    
    # Process keywords
    keywords = [k.strip() for k in keywords_input.split('\n') if k.strip()]
    if len(keywords) > 5:
        st.sidebar.warning("⚠️ Maximum 5 keywords allowed. Using first 5.")
        keywords = keywords[:5]
    
    st.session_state.keywords = keywords
    
    # Target keyword selection
    if keywords:
        # Add "ALL" option to predict all keywords
        keyword_options = ["ALL"] + keywords
        target_selection = st.sidebar.selectbox(
            "Target keyword for prediction:",
            keyword_options,
            help="Select which keyword to predict, or choose ALL to predict all keywords"
        )
        
        # Set target keyword based on selection
        if target_selection == "ALL":
            st.session_state.target_keyword = "ALL"
            st.session_state.all_keywords = keywords
        else:
            st.session_state.target_keyword = target_selection
            st.session_state.all_keywords = [target_selection]
    
    # Data collection settings
    st.sidebar.subheader("Data Collection")
    
    timeframe = st.sidebar.selectbox(
        "Time Period:",
        ["today 5-y", "today 3-y", "today 1-y", "today 12-m", "today 6-m"],
        help="Select the time period for data collection"
    )
    
    geo = st.sidebar.selectbox(
        "Geographic Location:",
        [("Worldwide", ""), ("United States", "US"), ("United Kingdom", "GB"), 
         ("Germany", "DE"), ("Japan", "JP"), ("South Korea", "KR")],
        format_func=lambda x: x[0],
        help="Select geographic focus"
    )[1]
    
    # Model settings
    st.sidebar.subheader("Model Configuration")
    
    model_types = st.sidebar.multiselect(
        "Select Models:",
        ["LSTM", "Prophet"],
        default=["LSTM", "Prophet"],
        help="Choose which models to train"
    )
    
    prediction_periods = st.sidebar.slider(
        "Prediction Periods:",
        min_value=7, max_value=365, value=30,
        help="Number of days to predict into the future"
    )
    
    # Advanced settings
    with st.sidebar.expander("⚙️ Advanced Settings"):
        lstm_epochs = st.slider("LSTM Epochs:", 10, 200, 50)
        smoothing_window = st.slider("Smoothing Window:", 1, 14, 7)
        test_size = st.slider("Test Size:", 0.1, 0.3, 0.2)
    
    # Store configuration in session state
    st.session_state.config = {
        'timeframe': timeframe,
        'geo': geo,
        'model_types': [m.lower() for m in model_types],
        'prediction_periods': prediction_periods,
        'lstm_epochs': lstm_epochs,
        'smoothing_window': smoothing_window,
        'test_size': test_size
    }


def data_collection_tab():
    """Data collection and exploration tab."""
    
    st.header("🔍 Data Collection & Exploration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Collect Data", type="primary", use_container_width=True):
            collect_data()
    
    with col2:
        if st.session_state.data is not None:
            st.success(f"✅ Data loaded: {st.session_state.data.shape[0]} rows")
    
    # Display data if available
    if st.session_state.data is not None:
        display_data_overview()


def collect_data():
    """Collect Google Trends data."""
    
    if not st.session_state.keywords:
        st.error("❌ Please enter at least one keyword")
        return
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize predictor
        status_text.text("🔧 Initializing predictor...")
        progress_bar.progress(20)
        
        config = create_config()
        predictor = TrendsPredictor(config)
        st.session_state.predictor = predictor
        
        # Collect data
        status_text.text("📡 Collecting Google Trends data...")
        progress_bar.progress(60)
        
        data = predictor.collect_data(
            keywords=st.session_state.keywords,
            timeframe=st.session_state.config['timeframe'],
            geo=st.session_state.config['geo']
        )
        
        st.session_state.data = data
        
        progress_bar.progress(100)
        status_text.text("✅ Data collection completed!")
        
        # Auto-advance to next step after delay
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"🎉 Successfully collected data for {len(data.columns)} keywords!")
        st.rerun()
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error collecting data: {str(e)}")


def display_data_overview():
    """Display overview of collected data."""
    
    data = st.session_state.data
    
    # Data summary
    st.subheader("📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Keywords", len(data.columns))
    
    with col2:
        st.metric("Data Points", len(data))
    
    with col3:
        st.metric("Date Range", f"{data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}")
    
    with col4:
        st.metric("Missing Values", data.isnull().sum().sum())
    
    # Interactive time series chart
    st.subheader("📈 Trends Visualization")
    
    fig = go.Figure()
    
    for column in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[column],
            mode='lines',
            name=column,
            hovertemplate='%{x|%Y-%m-%d}<br>%{y}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Google Trends Search Interest Over Time",
        xaxis_title="Date",
        yaxis_title="Search Interest",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    if len(data.columns) > 1:
        st.subheader("🔗 Keyword Correlations")
        
        corr_matrix = data.corr()
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z}<extra></extra>'
        ))
        
        fig_heatmap.update_layout(
            title="Keyword Correlation Matrix",
            height=400
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Data preview
    with st.expander("🔍 Raw Data Preview"):
        st.dataframe(data.tail(10), use_container_width=True)
    
    # Download option
    csv = data.to_csv()
    st.download_button(
        label="💾 Download Data as CSV",
        data=csv,
        file_name=f"trends_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


def model_training_tab():
    """Model training tab."""
    
    st.header("🤖 Model Training")
    
    if st.session_state.data is None:
        st.info("📋 Please collect data first in the Data Collection tab.")
        return
    
    if not hasattr(st.session_state, 'target_keyword'):
        st.error("❌ Please select a target keyword in the sidebar.")
        return
    
    # Display selected target information
    if st.session_state.target_keyword == "ALL":
        st.info(f"🎯 Training models for ALL keywords: {', '.join(st.session_state.all_keywords)}")
    else:
        st.info(f"🎯 Training models for: {st.session_state.target_keyword}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Train Models", type="primary", use_container_width=True):
            train_models()
    
    with col2:
        if st.session_state.metrics is not None:
            st.success("✅ Models trained successfully!")
    
    # Display training results
    if st.session_state.metrics is not None:
        display_training_results()


def train_models():
    """Train machine learning models."""
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        predictor = st.session_state.predictor
        if predictor is None:
            config = create_config()
            predictor = TrendsPredictor(config)
            st.session_state.predictor = predictor
        
        # Handle ALL keywords or single keyword
        target_keywords = st.session_state.all_keywords
        all_metrics = {}
        
        total_keywords = len(target_keywords)
        
        for idx, target_keyword in enumerate(target_keywords):
            # Calculate safe progress values (0-90 range for processing)
            base_progress = int((idx / total_keywords) * 90)
            
            status_text.text(f"🔄 Processing keyword {idx+1}/{total_keywords}: {target_keyword}")
            progress_bar.progress(min(base_progress + 5, 90))
            
            # Prepare data for this keyword
            train_data, val_data, test_data = predictor.prepare_data(
                st.session_state.data, 
                target_keyword
            )
            
            # Train models for this keyword
            status_text.text(f"🧠 Training models for: {target_keyword}")
            progress_bar.progress(min(base_progress + 15, 90))
            
            predictor.train_models(
                train_data, val_data, target_keyword,
                model_types=st.session_state.config['model_types']
            )
            
            # Evaluate models for this keyword
            status_text.text(f"📊 Evaluating models for: {target_keyword}")
            progress_bar.progress(min(base_progress + 25, 90))
            
            keyword_metrics = predictor.evaluate_models(test_data, target_keyword)
            
            # Store metrics with keyword prefix
            for model_name, metrics in keyword_metrics.items():
                if st.session_state.target_keyword == "ALL":
                    all_metrics[f"{model_name}_{target_keyword}"] = metrics
                else:
                    all_metrics[model_name] = metrics
        
        st.session_state.metrics = all_metrics
        
        progress_bar.progress(100)
        status_text.text("✅ Model training completed!")
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        if st.session_state.target_keyword == "ALL":
            st.success(f"🎉 Models trained successfully for all {total_keywords} keywords!")
        else:
            st.success("🎉 Models trained successfully!")
        st.rerun()
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error training models: {str(e)}")


def display_training_results():
    """Display model training results."""
    
    st.subheader("📊 Model Performance")
    
    metrics = st.session_state.metrics
    
    # Create metrics comparison chart
    if len(metrics) > 1:
        
        # Prepare data for comparison
        models = list(metrics.keys())
        metric_names = ['mae', 'rmse', 'mape', 'r2']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Mean Absolute Error', 'Root Mean Square Error', 
                          'Mean Absolute Percentage Error', 'R² Score'],
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for idx, metric in enumerate(metric_names):
            values = []
            model_names = []
            
            for model in models:
                if metric in metrics[model] and not np.isnan(metrics[model][metric]):
                    values.append(metrics[model][metric])
                    model_names.append(model.upper())
            
            if values:
                row, col = positions[idx]
                
                # Color based on metric type
                colors = ['green' if metric == 'r2' else 'red' for _ in values]
                
                fig.add_trace(
                    go.Bar(x=model_names, y=values, 
                          marker_color=colors, showlegend=False),
                    row=row, col=col
                )
        
        fig.update_layout(height=600, title_text="Model Performance Comparison")
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics table
    st.subheader("📋 Detailed Metrics")
    
    metrics_df = pd.DataFrame(metrics).T
    # Round numerical values
    for col in metrics_df.columns:
        if metrics_df[col].dtype in ['float64', 'float32']:
            metrics_df[col] = metrics_df[col].round(4)
    
    st.dataframe(metrics_df, use_container_width=True)


def predictions_tab():
    """Predictions and forecasting tab."""
    
    st.header("📊 Predictions & Forecasting")
    
    if st.session_state.metrics is None:
        st.info("📋 Please train models first in the Model Training tab.")
        return
    
    # Display selected target information
    if st.session_state.target_keyword == "ALL":
        st.info(f"🎯 Generating predictions for ALL keywords: {', '.join(st.session_state.all_keywords)}")
    else:
        st.info(f"🎯 Generating predictions for: {st.session_state.target_keyword}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🔮 Generate Predictions", type="primary", use_container_width=True):
            generate_predictions()
    
    with col2:
        if st.session_state.predictions is not None:
            st.success("✅ Predictions generated!")
    
    # Display predictions
    if st.session_state.predictions is not None:
        display_predictions()


def generate_predictions():
    """Generate future predictions."""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        predictor = st.session_state.predictor
        
        # Handle ALL keywords or single keyword
        target_keywords = st.session_state.all_keywords
        all_predictions = {}
        
        total_keywords = len(target_keywords)
        
        for idx, target_keyword in enumerate(target_keywords):
            # Calculate safe progress values (0-90 range for processing)
            base_progress = int((idx / total_keywords) * 90)
            
            status_text.text(f"🔮 Generating predictions {idx+1}/{total_keywords}: {target_keyword}")
            progress_bar.progress(min(base_progress + 10, 90))
            
            # Prepare data for this keyword
            all_processed_data = predictor.preprocessor.preprocess(
                st.session_state.data, 
                target_keyword, 
                save_processed=False
            )
            
            # Generate predictions for this keyword
            keyword_predictions = predictor.predict_future(
                st.session_state.config['prediction_periods'],
                all_processed_data,
                target_keyword
            )
            
            # Store predictions with keyword prefix
            for model_name, pred in keyword_predictions.items():
                if st.session_state.target_keyword == "ALL":
                    all_predictions[f"{model_name}_{target_keyword}"] = pred
                else:
                    all_predictions[model_name] = pred
        
        st.session_state.predictions = all_predictions
        
        progress_bar.progress(100)
        status_text.text("✅ Predictions generated!")
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        if st.session_state.target_keyword == "ALL":
            st.success(f"🎉 Predictions generated for all {total_keywords} keywords!")
        else:
            st.success("🎉 Predictions generated successfully!")
        st.rerun()
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error generating predictions: {str(e)}")


def display_predictions():
    """Display prediction results."""
    
    st.subheader("🔮 Future Predictions")
    
    predictions = st.session_state.predictions
    data = st.session_state.data
    target_keyword = st.session_state.target_keyword
    
    if target_keyword == "ALL":
        # Display predictions for all keywords
        display_all_predictions(predictions, data)
    else:
        # Display predictions for single keyword
        display_single_prediction(predictions, data, target_keyword)
    
    # Download predictions
    if predictions:
        predictions_df = pd.DataFrame(predictions)
        csv = predictions_df.to_csv()
        
        filename = f"predictions_{'ALL' if target_keyword == 'ALL' else target_keyword}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        st.download_button(
            label="💾 Download Predictions as CSV",
            data=csv,
            file_name=filename,
            mime="text/csv"
        )


def display_single_prediction(predictions, data, target_keyword):
    """Display predictions for a single keyword."""
    
    # Create prediction visualization
    fig = go.Figure()
    
    # Historical data
    historical = data[target_keyword].tail(100)  # Last 100 points
    fig.add_trace(go.Scatter(
        x=historical.index,
        y=historical.values,
        mode='lines',
        name='Historical',
        line=dict(color='blue', width=2)
    ))
    
    # Predictions from each model
    colors = ['red', 'green', 'orange', 'purple']
    
    for idx, (model_name, pred) in enumerate(predictions.items()):
        fig.add_trace(go.Scatter(
            x=pred.index,
            y=pred.values,
            mode='lines',
            name=f'{model_name.upper()} Prediction',
            line=dict(color=colors[idx % len(colors)], dash='dash', width=2)
        ))
    
    # Add vertical line at prediction start
    if len(predictions) > 0:
        first_pred = list(predictions.values())[0]
        if len(first_pred) > 0:
            prediction_start = first_pred.index[0]
            fig.add_shape(
                type="line",
                x0=prediction_start,
                x1=prediction_start,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color="gray", width=2, dash="dot"),
            )
            fig.add_annotation(
                x=prediction_start,
                y=1.02,
                yref="paper",
                text="Prediction Start",
                showarrow=False,
                font=dict(size=12, color="gray")
            )
    
    fig.update_layout(
        title=f"Predictions for '{target_keyword}'",
        xaxis_title="Date",
        yaxis_title="Search Interest",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction summary
    st.subheader("📈 Prediction Summary")
    
    col1, col2, col3 = st.columns(3)
    
    for idx, (model_name, pred) in enumerate(predictions.items()):
        with [col1, col2, col3][idx % 3]:
            avg_prediction = pred.mean()
            trend = "📈" if pred.iloc[-1] > pred.iloc[0] else "📉"
            
            st.metric(
                f"{model_name.upper()} Average",
                f"{avg_prediction:.1f}",
                f"{trend} {pred.iloc[-1] - pred.iloc[0]:.1f}"
            )


def display_all_predictions(predictions, data):
    """Display predictions for all keywords."""
    
    # Group predictions by keyword
    keywords = st.session_state.all_keywords
    
    # Create tabs for each keyword
    if len(keywords) > 1:
        tabs = st.tabs([f"📊 {keyword}" for keyword in keywords])
        
        for idx, keyword in enumerate(keywords):
            with tabs[idx]:
                # Filter predictions for this keyword
                keyword_predictions = {
                    k.split('_')[0]: v for k, v in predictions.items() 
                    if k.endswith(f"_{keyword}")
                }
                
                if keyword_predictions:
                    display_single_prediction(keyword_predictions, data, keyword)
                else:
                    st.warning(f"No predictions available for {keyword}")
    else:
        # Single keyword case
        keyword = keywords[0]
        keyword_predictions = {
            k.split('_')[0]: v for k, v in predictions.items() 
            if k.endswith(f"_{keyword}")
        }
        display_single_prediction(keyword_predictions, data, keyword)
    
    # Overall summary
    st.subheader("🌟 Overall Summary")
    
    summary_data = []
    for keyword in keywords:
        keyword_predictions = {
            k.split('_')[0]: v for k, v in predictions.items() 
            if k.endswith(f"_{keyword}")
        }
        
        for model_name, pred in keyword_predictions.items():
            if len(pred) > 0:
                avg_pred = pred.mean()
                trend_value = pred.iloc[-1] - pred.iloc[0]
                trend_direction = "상승" if trend_value > 0 else "하락"
                
                summary_data.append({
                    "키워드": keyword,
                    "모델": model_name.upper(),
                    "평균 예측값": f"{avg_pred:.1f}",
                    "트렌드": f"{trend_direction} ({abs(trend_value):.1f})"
                })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)


def create_config():
    """Create configuration object from session state."""
    
    # Data configuration
    data_config = DataConfig()
    data_config.timeframe = st.session_state.config['timeframe']
    data_config.geo = st.session_state.config['geo']
    data_config.smoothing_window = st.session_state.config['smoothing_window']
    
    # Model configuration
    model_config = ModelConfig()
    model_config.test_size = st.session_state.config['test_size']
    model_config.lstm_epochs = st.session_state.config['lstm_epochs']
    
    # App configuration
    app_config = AppConfig()
    app_config.data = data_config
    app_config.model = model_config
    
    return app_config


if __name__ == "__main__":
    main() 