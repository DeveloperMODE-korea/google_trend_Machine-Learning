"""
Visualization utilities for Google Trends data and predictions.

This module provides functions for creating various plots and visualizations
to analyze trends data and model predictions.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..utils.config import AppConfig


logger = logging.getLogger(__name__)


class TrendsVisualizer:
    """Class for creating visualizations of trends data and predictions."""
    
    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize visualizer with configuration.
        
        Args:
            config: Application configuration.
        """
        self.config = config or AppConfig()
        
        # Set plotting style
        plt.style.use(self.config.plot_style)
        sns.set_palette("husl")
        
        # Default figure size
        self.figsize = self.config.figure_size
    
    def plot_trends(self, data: pd.DataFrame, 
                   title: Optional[str] = None,
                   save_path: Optional[str] = None) -> None:
        """
        Plot raw trends data.
        
        Args:
            data: DataFrame with trends data (columns are keywords).
            title: Plot title.
            save_path: Path to save the plot.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot each keyword
        for column in data.columns:
            ax.plot(data.index, data[column], label=column, alpha=0.8)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Search Interest')
        ax.set_title(title or 'Google Trends Search Interest Over Time')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("Saved plot to: %s", save_path)
        
        plt.show()
    
    def plot_predictions_comparison(self, 
                                  actual: pd.Series,
                                  predictions: Dict[str, pd.Series],
                                  title: Optional[str] = None,
                                  save_path: Optional[str] = None) -> None:
        """
        Compare predictions from multiple models.
        
        Args:
            actual: Actual values.
            predictions: Dictionary of model_name -> predictions.
            title: Plot title.
            save_path: Path to save the plot.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot actual values
        ax.plot(actual.index, actual.values, label='Actual', 
                color='black', linewidth=2, alpha=0.8)
        
        # Plot predictions from each model
        colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))
        for (model_name, pred), color in zip(predictions.items(), colors):
            ax.plot(pred.index, pred.values, label=model_name, 
                   color=color, alpha=0.7, linestyle='--')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.set_title(title or 'Model Predictions Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_forecast_with_confidence(self,
                                    historical: pd.Series,
                                    forecast: pd.Series,
                                    lower_bound: Optional[pd.Series] = None,
                                    upper_bound: Optional[pd.Series] = None,
                                    title: Optional[str] = None,
                                    save_path: Optional[str] = None) -> None:
        """
        Plot forecast with confidence intervals.
        
        Args:
            historical: Historical data.
            forecast: Forecasted values.
            lower_bound: Lower confidence bound.
            upper_bound: Upper confidence bound.
            title: Plot title.
            save_path: Path to save the plot.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot historical data
        ax.plot(historical.index, historical.values, 
               label='Historical', color='blue', alpha=0.8)
        
        # Plot forecast
        ax.plot(forecast.index, forecast.values, 
               label='Forecast', color='red', alpha=0.8)
        
        # Add confidence interval if provided
        if lower_bound is not None and upper_bound is not None:
            ax.fill_between(forecast.index, 
                          lower_bound.values, 
                          upper_bound.values,
                          alpha=0.2, color='red', 
                          label='Confidence Interval')
        
        # Add vertical line at forecast start
        if len(historical) > 0 and len(forecast) > 0:
            forecast_start = forecast.index[0]
            ax.axvline(x=forecast_start, color='gray', 
                      linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.set_title(title or 'Forecast with Confidence Intervals')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_interactive_trends(self, data: pd.DataFrame,
                              title: Optional[str] = None) -> go.Figure:
        """
        Create interactive plot using Plotly.
        
        Args:
            data: DataFrame with trends data.
            title: Plot title.
        
        Returns:
            Plotly Figure object.
        """
        fig = go.Figure()
        
        # Add trace for each keyword
        for column in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[column],
                mode='lines',
                name=column,
                hovertemplate='%{x|%Y-%m-%d}<br>Value: %{y}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=title or 'Google Trends - Interactive View',
            xaxis_title='Date',
            yaxis_title='Search Interest',
            hovermode='x unified',
            width=1200,
            height=600
        )
        
        # Add range slider
        fig.update_xaxes(rangeslider_visible=True)
        
        return fig
    
    def plot_correlation_heatmap(self, data: pd.DataFrame,
                               title: Optional[str] = None,
                               save_path: Optional[str] = None) -> None:
        """
        Plot correlation heatmap between keywords.
        
        Args:
            data: DataFrame with trends data.
            title: Plot title.
            save_path: Path to save the plot.
        """
        # Calculate correlation matrix
        corr_matrix = data.corr()
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, 
                   mask=mask,
                   annot=True, 
                   fmt='.2f',
                   cmap='coolwarm',
                   center=0,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": 0.8})
        
        plt.title(title or 'Keyword Correlation Heatmap')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_seasonal_decomposition(self, data: pd.Series,
                                  period: Optional[int] = None,
                                  title: Optional[str] = None,
                                  save_path: Optional[str] = None) -> None:
        """
        Plot seasonal decomposition of time series.
        
        Args:
            data: Time series data.
            period: Seasonal period (if None, will be inferred).
            title: Plot title.
            save_path: Path to save the plot.
        """
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Perform decomposition
        if period is None:
            # Try to infer period (weekly=7, monthly=30, yearly=365)
            if len(data) > 730:
                period = 365
            elif len(data) > 60:
                period = 30
            else:
                period = 7
        
        decomposition = seasonal_decompose(data, model='additive', period=period)
        
        # Create subplots
        fig, axes = plt.subplots(4, 1, figsize=(self.figsize[0], 10))
        
        # Original data
        data.plot(ax=axes[0], title=title or 'Seasonal Decomposition')
        axes[0].set_ylabel('Original')
        
        # Trend
        decomposition.trend.plot(ax=axes[1])
        axes[1].set_ylabel('Trend')
        
        # Seasonal
        decomposition.seasonal.plot(ax=axes[2])
        axes[2].set_ylabel('Seasonal')
        
        # Residual
        decomposition.resid.plot(ax=axes[3])
        axes[3].set_ylabel('Residual')
        axes[3].set_xlabel('Date')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_metrics(self, metrics_dict: Dict[str, Dict[str, float]],
                         metric_names: Optional[List[str]] = None,
                         title: Optional[str] = None,
                         save_path: Optional[str] = None) -> None:
        """
        Plot comparison of model metrics.
        
        Args:
            metrics_dict: Dictionary of model_name -> metrics dict.
            metric_names: List of metrics to plot. If None, plots all.
            title: Plot title.
            save_path: Path to save the plot.
        """
        # Extract metrics to plot
        if metric_names is None:
            # Get all unique metrics
            all_metrics = set()
            for metrics in metrics_dict.values():
                all_metrics.update(metrics.keys())
            metric_names = sorted(list(all_metrics))
        
        # Prepare data for plotting
        models = list(metrics_dict.keys())
        n_metrics = len(metric_names)
        
        # Create subplots
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metric_names):
            values = []
            valid_models = []
            
            for model in models:
                if metric in metrics_dict[model]:
                    value = metrics_dict[model][metric]
                    if not np.isnan(value):
                        values.append(value)
                        valid_models.append(model)
            
            if values:
                ax = axes[idx]
                bars = ax.bar(valid_models, values)
                
                # Color bars based on metric type
                if 'r2' in metric.lower() or 'accuracy' in metric.lower():
                    # Higher is better - green
                    colors = plt.cm.Greens(np.linspace(0.4, 0.8, len(values)))
                else:
                    # Lower is better - red
                    colors = plt.cm.Reds_r(np.linspace(0.4, 0.8, len(values)))
                
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
                
                ax.set_title(metric.upper())
                ax.set_ylabel('Value')
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom')
        
        fig.suptitle(title or 'Model Performance Comparison', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_dashboard(self, data: pd.DataFrame,
                        predictions: Optional[Dict[str, pd.Series]] = None,
                        metrics: Optional[Dict[str, Dict[str, float]]] = None) -> go.Figure:
        """
        Create an interactive dashboard with multiple visualizations.
        
        Args:
            data: Historical trends data.
            predictions: Model predictions.
            metrics: Model metrics.
        
        Returns:
            Plotly Figure object with dashboard.
        """
        # Create subplots
        n_rows = 2 if metrics else 1
        n_cols = 2
        
        subplot_titles = ['Historical Trends', 'Correlation Matrix']
        if metrics:
            subplot_titles.extend(['Model Predictions', 'Model Metrics'])
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=subplot_titles,
            specs=[[{'type': 'scatter'}, {'type': 'heatmap'}]] + 
                  [[{'type': 'scatter'}, {'type': 'bar'}]] if metrics else []
        )
        
        # 1. Historical trends
        for column in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data[column], 
                         mode='lines', name=column),
                row=1, col=1
            )
        
        # 2. Correlation heatmap
        corr_matrix = data.corr()
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0
            ),
            row=1, col=2
        )
        
        # 3. Predictions comparison (if provided)
        if predictions and metrics:
            # Add actual values
            last_keyword = data.columns[-1]
            last_n = min(100, len(data))
            actual = data[last_keyword].iloc[-last_n:]
            
            fig.add_trace(
                go.Scatter(x=actual.index, y=actual.values,
                         mode='lines', name='Actual',
                         line=dict(color='black', width=2)),
                row=2, col=1
            )
            
            # Add predictions
            for model_name, pred in predictions.items():
                if len(pred) > 0:
                    fig.add_trace(
                        go.Scatter(x=pred.index, y=pred.values,
                                 mode='lines', name=model_name,
                                 line=dict(dash='dash')),
                        row=2, col=1
                    )
        
        # 4. Metrics comparison (if provided)
        if metrics:
            # Use RMSE as the main metric for comparison
            metric_name = 'rmse'
            models = []
            values = []
            
            for model, model_metrics in metrics.items():
                if metric_name in model_metrics:
                    models.append(model)
                    values.append(model_metrics[metric_name])
            
            if models:
                fig.add_trace(
                    go.Bar(x=models, y=values, name='RMSE'),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Google Trends Analysis Dashboard"
        )
        
        return fig


def plot_quick_summary(data: pd.DataFrame, 
                      keyword: str,
                      predictions: Optional[pd.Series] = None) -> None:
    """
    Quick summary plot for a single keyword.
    
    Args:
        data: Historical data.
        keyword: Keyword to plot.
        predictions: Future predictions.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Time series plot
    ax1.plot(data.index, data[keyword], label='Historical', alpha=0.8)
    if predictions is not None:
        ax1.plot(predictions.index, predictions.values, 
                label='Forecast', color='red', alpha=0.8)
        # Add forecast start line
        ax1.axvline(x=predictions.index[0], color='gray', 
                   linestyle='--', alpha=0.5)
    
    ax1.set_title(f'Google Trends: {keyword}')
    ax1.set_ylabel('Search Interest')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Distribution plot
    ax2.hist(data[keyword].values, bins=30, alpha=0.7, edgecolor='black')
    ax2.axvline(data[keyword].mean(), color='red', 
               linestyle='--', label=f'Mean: {data[keyword].mean():.1f}')
    ax2.axvline(data[keyword].median(), color='green', 
               linestyle='--', label=f'Median: {data[keyword].median():.1f}')
    ax2.set_xlabel('Search Interest')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Search Interest')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    import numpy as np
    
    # Create sample data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    data = pd.DataFrame({
        'keyword1': np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 50 + 50 + np.random.randn(len(dates)) * 5,
        'keyword2': np.cos(np.arange(len(dates)) * 2 * np.pi / 365) * 30 + 60 + np.random.randn(len(dates)) * 3,
        'keyword3': np.sin(np.arange(len(dates)) * 2 * np.pi / 180) * 40 + 45 + np.random.randn(len(dates)) * 4
    }, index=dates)
    
    # Create visualizer
    visualizer = TrendsVisualizer()
    
    # Plot trends
    visualizer.plot_trends(data, title="Sample Trends Data")
    
    # Plot correlation heatmap
    visualizer.plot_correlation_heatmap(data)
    
    # Create interactive plot
    fig = visualizer.plot_interactive_trends(data)
    fig.show()
    
    print("Visualization examples completed!") 