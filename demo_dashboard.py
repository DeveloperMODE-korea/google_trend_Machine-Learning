#!/usr/bin/env python3
"""
Demo script for Google Trends ML Predictor Web Dashboard.

This script demonstrates the dashboard features with sample data.
"""

import os
import sys
import webbrowser
import time
from pathlib import Path

def print_banner():
    """Print welcome banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║           📈 Google Trends ML Predictor - Demo                   ║
    ║                                                                  ║
    ║           🌐 Interactive Web Dashboard                           ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def show_features():
    """Display dashboard features."""
    features = [
        "🔍 Real-time Google Trends data collection",
        "🤖 Machine Learning model training (LSTM & Prophet)",
        "📊 Interactive data visualizations",
        "📈 Future trend predictions",
        "💾 Data export functionality",
        "📱 Mobile-responsive design",
        "🛠️ Easy configuration interface"
    ]
    
    print("✨ Dashboard Features:")
    for feature in features:
        print(f"   {feature}")
    print()

def show_usage_guide():
    """Show quick usage guide."""
    guide = """
📋 Quick Start Guide:

1️⃣  Launch Dashboard
   • Run: python run_dashboard.py
   • Or: streamlit run src/web/dashboard.py

2️⃣  Configure Keywords (Sidebar)
   • Enter up to 5 keywords (one per line)
   • Example: "machine learning", "artificial intelligence"
   • Select target keyword for prediction

3️⃣  Collect Data (Tab 1)
   • Choose time period (1 year to 5 years)
   • Select geographic region
   • Click "Collect Data" button

4️⃣  Train Models (Tab 2)
   • Select models (LSTM, Prophet, or both)
   • Adjust training parameters
   • Click "Train Models" button

5️⃣  Generate Predictions (Tab 3)
   • Set prediction period (7-365 days)
   • Click "Generate Predictions"
   • Download results as CSV

💡 Pro Tips:
   • Use related keywords for better correlation analysis
   • Longer time periods provide more training data
   • Compare multiple models for best results
   • Export data for further analysis
    """
    print(guide)

def check_requirements():
    """Check if all requirements are installed."""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'pytrends',
        'tensorflow',
        'prophet',
        'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   • {package}")
        print("\n💡 Install missing packages:")
        print("   pip install -r requirements.txt")
        return False
    else:
        print("✅ All required packages are installed!")
        return True

def main():
    """Main demo function."""
    print_banner()
    show_features()
    
    print("🔍 Checking requirements...")
    if not check_requirements():
        return
    
    print("\n🚀 Ready to launch dashboard!")
    show_usage_guide()
    
    # Ask user if they want to launch the dashboard
    try:
        response = input("Would you like to launch the dashboard now? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            print("\n🌐 Launching dashboard...")
            
            # Import and run the dashboard launch script
            sys.path.append('.')
            from run_dashboard import main as launch_dashboard
            launch_dashboard()
        else:
            print("\n💡 To launch later, run: python run_dashboard.py")
            print("📖 Documentation: README.md")
    
    except KeyboardInterrupt:
        print("\n👋 Demo cancelled. Thanks for checking out the project!")

if __name__ == "__main__":
    main() 