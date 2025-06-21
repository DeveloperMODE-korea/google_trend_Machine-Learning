# Google Trends Machine Learning Predictor

A machine learning system that leverages Google Trends API to collect global search volume data and predict future trends using advanced ML algorithms.

## 🎯 Project Overview

This project fetches keyword search volume data from Google Trends API (focusing on worldwide data), processes it, and uses machine learning models to predict future search trends. The system is designed with modularity and maintainability in mind.

## 📁 Project Structure

```
google_trend_Machine-Learning/
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── collector.py      # Google Trends data collection
│   │   └── preprocessor.py   # Data preprocessing and cleaning
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py     # Base model interface
│   │   ├── lstm_model.py     # LSTM model for time series
│   │   └── prophet_model.py  # Facebook Prophet model
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py         # Configuration management
│   │   └── visualizer.py     # Data visualization utilities
│   │
│   └── predictor.py          # Main prediction pipeline
│
├── data/                      # Data storage directory
│   ├── raw/                   # Raw data from Google Trends
│   └── processed/             # Preprocessed data
│
├── models/                    # Saved model files
├── notebooks/                 # Jupyter notebooks for analysis
├── tests/                     # Unit tests
├── requirements.txt           # Project dependencies
├── setup.py                   # Package setup
└── README.md                  # This file
```

## 🚀 Features

- **Global Focus**: Collects worldwide search trend data (not limited to specific regions)
- **Multiple Keywords**: Support for tracking multiple keywords simultaneously
- **Time Range Flexibility**: Customizable date ranges for data collection
- **Multiple ML Models**: Implements various models (LSTM, Prophet, etc.) for comparison
- **Modular Architecture**: Clean separation of concerns for easy maintenance
- **Visualization**: Built-in plotting capabilities for trends and predictions

## 📋 Requirements

- Python 3.8+
- pytrends
- pandas
- numpy
- scikit-learn
- tensorflow/keras
- prophet
- matplotlib
- seaborn

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/DeveloperMODE-korea/google_trend_Machine-Learning.git
cd google_trend_Machine-Learning
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Basic Example

```python
from src.data.collector import TrendsCollector
from src.predictor import TrendsPredictor

# Initialize collector
collector = TrendsCollector()

# Collect data for keywords
keywords = ['machine learning', 'artificial intelligence']
data = collector.collect_trends(
    keywords=keywords,
    timeframe='2020-01-01 2024-01-01',
    geo=''  # Empty string for worldwide
)

# Initialize predictor
predictor = TrendsPredictor(model_type='lstm')

# Train and predict
predictor.train(data)
predictions = predictor.predict(periods=30)  # Predict next 30 days
```

### Command Line Interface

```bash
# Collect data
python -m src.data.collector --keywords "AI,ML" --timeframe "today 5-y"

# Train model
python -m src.predictor --train --model lstm --data ./data/processed/trends_data.csv

# Make predictions
python -m src.predictor --predict --periods 30 --output ./predictions.csv
```

## 📊 Models

### 1. LSTM (Long Short-Term Memory)
- Best for capturing long-term dependencies in time series
- Handles non-linear patterns well
- Requires more data for training

### 2. Prophet
- Developed by Facebook for time series forecasting
- Handles seasonality and holidays automatically
- Works well with missing data

### 3. ARIMA (Coming Soon)
- Traditional time series model
- Good baseline for comparison

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

## 📈 Performance Metrics

The system evaluates models using:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- R² Score

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Trends for providing the data API
- pytrends library maintainers
- Open source ML community

## 📞 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter) - email@example.com

Project Link: [https://github.com/yourusername/google_trend_Machine-Learning](https://github.com/yourusername/google_trend_Machine-Learning) 