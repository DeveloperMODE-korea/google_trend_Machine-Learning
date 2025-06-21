# Google Trends Machine Learning Predictor

A comprehensive machine learning system that leverages Google Trends API to collect global search volume data and predict future trends using advanced ML algorithms with an intuitive web dashboard interface.

## 🎯 Project Overview

This project fetches keyword search volume data from Google Trends API, processes it using advanced preprocessing techniques, and employs machine learning models (LSTM and Prophet) to predict future search trends. The system features a modern web dashboard built with Streamlit for easy interaction and visualization.

## ✨ Key Features

- **🌐 Interactive Web Dashboard**: User-friendly Streamlit interface with real-time progress tracking
- **🔍 Multi-keyword Analysis**: Support for up to 5 keywords simultaneously with correlation analysis
- **🤖 Advanced ML Models**: LSTM neural networks and Facebook Prophet for accurate predictions
- **📊 Rich Visualizations**: Interactive Plotly charts with zoom, pan, and hover features
- **📈 ALL Keywords Mode**: Train and predict for all keywords at once
- **💾 Data Export**: Download collected data and predictions as CSV files
- **🛡️ Robust Error Handling**: Safe collectors and fallback mechanisms for stability
- **📱 Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices

## 📁 Project Structure

```
google_trend_Machine-Learning/
│
├── src/
│   ├── data/
│   │   ├── collector.py          # Google Trends data collection
│   │   ├── collector_safe.py     # Safe collector with error handling
│   │   └── preprocessor.py       # Data preprocessing and feature engineering
│   │
│   ├── models/
│   │   ├── base_model.py         # Abstract base model interface
│   │   ├── lstm_model.py         # LSTM neural network implementation
│   │   └── prophet_model.py      # Facebook Prophet model
│   │
│   ├── utils/
│   │   ├── config.py             # Configuration management
│   │   └── visualizer.py         # Advanced visualization utilities
│   │
│   ├── web/
│   │   └── dashboard.py          # Streamlit web dashboard
│   │
│   └── predictor.py              # Main prediction pipeline orchestrator
│
├── data/                         # Data storage
│   ├── raw/                      # Raw Google Trends data
│   └── processed/                # Preprocessed data ready for ML
│
├── models/                       # Saved trained models
├── notebooks/                    # Jupyter notebooks for analysis
├── tests/                        # Unit tests
├── .streamlit/                   # Streamlit configuration
├── demo_dashboard.py             # Interactive demo launcher
├── run_dashboard.py              # Dashboard launcher script
├── requirements.txt              # Project dependencies
├── setup.py                      # Package installation
└── config.yaml                   # Default configuration
```

## 🚀 Quick Start

### Option 1: Web Dashboard (Recommended)

The easiest way to get started is with our interactive web dashboard:

```bash
# Clone the repository
git clone https://github.com/DeveloperMODE-korea/google_trend_Machine-Learning.git
cd google_trend_Machine-Learning

# Install dependencies
pip install -r requirements.txt

# Launch the interactive demo
python demo_dashboard.py

# Or launch directly
python run_dashboard.py
```

The dashboard will open at `http://localhost:8502` with these features:

#### 🛠️ Configuration Sidebar
- Enter up to 5 keywords (one per line)
- Select "ALL" to predict all keywords simultaneously
- Choose time periods (6 months to 5 years)
- Configure geographic regions
- Adjust model parameters

#### 📊 Three Main Tabs

**1. 🔍 Data Collection**
- Real-time Google Trends data collection
- Interactive trend visualizations
- Correlation heatmaps between keywords
- Data quality overview and download options

**2. 🤖 Model Training**
- Train LSTM and Prophet models
- Real-time progress tracking for each keyword
- Model performance comparison charts
- Detailed metrics tables

**3. 📈 Predictions**
- Generate future trend predictions
- Interactive charts with historical context
- Compare predictions across models
- Summary statistics and trend analysis

### Option 2: Python API

For programmatic usage or integration into existing workflows:

```python
from src.predictor import TrendsPredictor
from src.utils.config import AppConfig

# Initialize predictor
config = AppConfig.from_yaml("config.yaml")
predictor = TrendsPredictor(config)

# Run complete pipeline
results = predictor.run_complete_pipeline(
    keywords=["machine learning", "artificial intelligence", "deep learning"],
    target_keyword="machine learning",
    prediction_periods=30,
    model_types=["lstm", "prophet"]
)

# Access results
print(f"Models trained: {results['models_trained']}")
print(f"Prediction accuracy: {results['metrics']}")
```

## 📋 Requirements

### System Requirements
- Python 3.8+
- 4GB RAM minimum (8GB recommended)
- Internet connection for Google Trends API

### Dependencies
```
pytrends>=4.9.2          # Google Trends API
streamlit>=1.46.0         # Web dashboard
pandas>=2.1.4,<2.3.0     # Data manipulation
tensorflow==2.15.0       # LSTM models
prophet==1.1.5            # Time series forecasting
plotly==5.18.0           # Interactive visualizations
scikit-learn==1.3.2      # ML utilities
```

## 🔧 Installation

### Method 1: Standard Installation

```bash
# Clone repository
git clone https://github.com/DeveloperMODE-korea/google_trend_Machine-Learning.git
cd google_trend_Machine-Learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Method 2: Development Installation

```bash
# Install in development mode
pip install -e .

# Install additional development tools
pip install pytest black flake8 isort
```

## 💻 Usage Examples

### Web Dashboard Workflow

1. **Launch Dashboard**
   ```bash
   python run_dashboard.py
   ```

2. **Configure Keywords**
   - Enter keywords like "bitcoin", "cryptocurrency", "blockchain"
   - Select "ALL" to analyze all keywords simultaneously

3. **Collect Data**
   - Choose time period (e.g., "today 2-y" for 2 years)
   - Select geographic region or worldwide
   - Click "Collect Data" and watch real-time progress

4. **Train Models**
   - Select LSTM, Prophet, or both models
   - Monitor training progress for each keyword
   - Compare model performance metrics

5. **Generate Predictions**
   - Set prediction horizon (7-365 days)
   - View interactive charts with confidence intervals
   - Download results as CSV

### Command Line Usage

```bash
# Quick prediction for single keyword
python -m src.predictor \
    --keywords "machine learning,AI,deep learning" \
    --target "machine learning" \
    --periods 30 \
    --models lstm,prophet

# Batch processing for multiple keywords
python example_usage.py batch
```

### Advanced Configuration

Create custom `config.yaml`:

```yaml
data:
  timeframe: "today 3-y"
  geo: "US"  # United States only
  smoothing_window: 14  # 2-week smoothing

model:
  lstm_epochs: 100
  lstm_units: [256, 128, 64]
  test_size: 0.15

plot_style: "seaborn-v0_8"
```

## 📊 Model Performance

### LSTM Neural Network
- **Best for**: Complex non-linear patterns, long sequences
- **Architecture**: Multi-layer LSTM with dropout regularization
- **Features**: Automatic early stopping, learning rate scheduling
- **Typical Accuracy**: MAPE 15-25% for stable trends

### Facebook Prophet
- **Best for**: Seasonal patterns, holiday effects
- **Features**: Automatic seasonality detection, trend changepoints
- **Robustness**: Handles missing data and outliers well
- **Typical Accuracy**: MAPE 10-20% for seasonal data

### Performance Metrics
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error  
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of determination

## 🌟 Advanced Features

### ALL Keywords Mode
Train and predict for multiple keywords simultaneously:
- Efficient batch processing
- Comparative analysis across keywords
- Consolidated reporting and visualization
- Individual model performance tracking

### Safe Data Collection
Robust error handling for various scenarios:
- Network connectivity issues
- API rate limiting
- Version compatibility problems
- Automatic fallback mechanisms

### Interactive Visualizations
- **Plotly Integration**: Zoom, pan, hover, and select
- **Correlation Analysis**: Heatmaps and relationship matrices
- **Time Series Decomposition**: Trend, seasonal, and residual components
- **Model Comparison**: Side-by-side performance visualization

### Export and Integration
- **CSV Export**: Raw data and predictions
- **JSON Reports**: Comprehensive results with metadata
- **API Integration**: Programmatic access to all functionality
- **Batch Processing**: Handle large keyword lists efficiently

## 🛠️ Troubleshooting

### Common Issues

**1. JavaScript Module Loading Error**
```bash
# Clear Streamlit cache and restart
python run_dashboard.py  # Automatic cache clearing included
```

**2. Matplotlib Style Warnings**
- Automatically handled with safe style fallbacks
- No action required

**3. LSTM Training Issues**
- Adaptive sequence length based on data size
- Automatic validation data handling
- Progress tracking shows detailed status

**4. Prophet Frequency Errors**
- Safe frequency conversion (W-SUN → W)
- Automatic fallback to daily frequency
- Error logging for debugging

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the Repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/google_trend_Machine-Learning.git
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make Changes**
   - Follow PEP 8 style guidelines
   - Add tests for new functionality
   - Update documentation

4. **Test Changes**
   ```bash
   pytest tests/
   python -m flake8 src/
   ```

5. **Submit Pull Request**
   - Describe changes clearly
   - Include screenshots for UI changes
   - Reference related issues

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
black src/
isort src/

# Lint code
flake8 src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Trends**: For providing the comprehensive search data API
- **pytrends**: Excellent Python library for Google Trends access
- **Streamlit**: Amazing framework for rapid web app development
- **Facebook Prophet**: Robust time series forecasting algorithm
- **TensorFlow**: Powerful machine learning framework
- **Plotly**: Interactive visualization library
- **Open Source Community**: For continuous inspiration and support

## 📞 Contact & Support

- **Repository**: [https://github.com/DeveloperMODE-korea/google_trend_Machine-Learning](https://github.com/DeveloperMODE-korea/google_trend_Machine-Learning)
- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: Join community discussions for questions and ideas

## 🔮 Future Roadmap

- [ ] **Additional Models**: ARIMA, XGBoost, Transformer models
- [ ] **Real-time Updates**: Live data streaming and predictions
- [ ] **Advanced Analytics**: Anomaly detection, event correlation
- [ ] **Cloud Deployment**: Docker containers, cloud hosting options
- [ ] **API Endpoints**: RESTful API for external integration
- [ ] **Mobile App**: React Native mobile application
- [ ] **Enterprise Features**: Multi-user support, role-based access

---

**⭐ Star this repository if you find it useful!**

Made with ❤️ by [DeveloperMODE-korea](https://github.com/DeveloperMODE-korea)
