#!/usr/bin/env python3
"""
Launch script for Google Trends ML Predictor Web Dashboard.

This script provides an easy way to launch the Streamlit web interface.
"""

import subprocess
import sys
import os
import webbrowser
import time

def check_dependencies():
    """Check if Streamlit is installed."""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def clear_streamlit_cache():
    """Clear Streamlit cache to resolve potential issues."""
    try:
        cache_dir = os.path.expanduser("~/.streamlit")
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir, ignore_errors=True)
            print("✅ Cleared Streamlit cache")
    except Exception as e:
        print(f"⚠️ Could not clear cache: {e}")

def main():
    """Launch the web dashboard."""
    
    print("🚀 Google Trends ML Predictor - Web Dashboard")
    print("=" * 50)
    
    # Check if Streamlit is installed
    if not check_dependencies():
        print("❌ Streamlit is not installed!")
        print("Please install it using: pip install streamlit")
        return
    
    # Clear cache to resolve potential issues
    print("🧹 Clearing Streamlit cache...")
    clear_streamlit_cache()
    
    # Get the dashboard path
    dashboard_path = os.path.join("src", "web", "dashboard.py")
    
    if not os.path.exists(dashboard_path):
        print(f"❌ Dashboard file not found: {dashboard_path}")
        return
    
    print("✅ Starting web dashboard...")
    print("📱 The dashboard will open in your default web browser")
    print("🔗 URL: http://localhost:8502")
    print("\n💡 To stop the dashboard, press Ctrl+C in this terminal")
    print("=" * 50)
    
    # Launch Streamlit with enhanced parameters
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            dashboard_path,
            "--server.headless", "false",
            "--server.port", "8502",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false",
            "--browser.gatherUsageStats", "false",
            "--global.developmentMode", "false",
            "--server.enableWebsocketCompression", "false"
        ]
        
        print(f"🔧 Running command: {' '.join(cmd)}")
        
        # Start the process
        process = subprocess.Popen(cmd)
        
        # Wait a moment then open browser
        time.sleep(3)
        try:
            webbrowser.open("http://localhost:8502")
            print("🌐 Opened browser automatically")
        except Exception:
            print("⚠️ Could not open browser automatically. Please visit http://localhost:8502 manually")
        
        # Wait for the process
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Clear your browser cache and cookies")
        print("2. Try opening http://localhost:8502 in an incognito/private window")
        print("3. Restart your browser completely")
        print("4. Try a different browser (Chrome, Firefox, Edge)")

if __name__ == "__main__":
    main() 