"""
Safe Google Trends data collector with better error handling.

This module provides a more robust version of TrendsCollector that handles
various compatibility issues with different versions of pytrends and urllib3.
"""

import os
import time
import logging
from typing import List, Dict, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False
    logging.warning("pytrends not available")

from ..utils.config import DataConfig


logger = logging.getLogger(__name__)


class SafeTrendsCollector:
    """
    Safe version of TrendsCollector with better error handling.
    
    This version handles various compatibility issues and provides fallbacks.
    """
    
    def __init__(self, config: Optional[DataConfig] = None):
        """
        Initialize the safe collector.
        
        Args:
            config: Data configuration object.
        """
        self.config = config or DataConfig()
        self.pytrends = None
        
        if not PYTRENDS_AVAILABLE:
            raise ImportError("pytrends library is required but not available")
        
        self._initialize_pytrends()
        
        logger.info("SafeTrendsCollector initialized")
    
    def _initialize_pytrends(self):
        """Initialize pytrends with fallback options."""
        
        # Try different initialization methods
        init_methods = [
            # Method 1: Full parameters (newest versions)
            lambda: TrendReq(
                hl=self.config.language,
                tz=0,
                timeout=(10, 25),
                proxies=None,
                retries=3,
                backoff_factor=0.1
            ),
            # Method 2: Basic parameters
            lambda: TrendReq(
                hl=self.config.language,
                tz=0,
                timeout=(10, 25)
            ),
            # Method 3: Minimal parameters
            lambda: TrendReq(
                hl=self.config.language,
                tz=0
            ),
            # Method 4: Default initialization
            lambda: TrendReq()
        ]
        
        for i, init_method in enumerate(init_methods):
            try:
                self.pytrends = init_method()
                logger.info("Successfully initialized pytrends with method %d", i + 1)
                return
            except Exception as e:
                logger.warning("Initialization method %d failed: %s", i + 1, str(e))
                continue
        
        raise RuntimeError("Failed to initialize pytrends with any method")
    
    def collect_trends(self, keywords: List[str], 
                      timeframe: Optional[str] = None,
                      geo: Optional[str] = None,
                      save_raw: bool = True,
                      max_retries: int = 3) -> pd.DataFrame:
        """
        Collect Google Trends data with retry logic.
        
        Args:
            keywords: List of keywords (max 5).
            timeframe: Time period for data collection.
            geo: Geographic location.
            save_raw: Whether to save raw data.
            max_retries: Maximum number of retries.
        
        Returns:
            DataFrame with trend data.
        """
        if len(keywords) > 5:
            raise ValueError("Google Trends API accepts maximum 5 keywords at a time")
        
        timeframe = timeframe or self.config.timeframe
        geo = geo if geo is not None else self.config.geo
        
        logger.info("Collecting trends for keywords: %s", keywords)
        logger.info("Timeframe: %s, Geo: %s", timeframe, geo or "Worldwide")
        
        for attempt in range(max_retries):
            try:
                # Build payload with error handling
                self._build_payload_safe(keywords, timeframe, geo)
                
                # Get interest over time
                trends_data = self._get_interest_over_time_safe()
                
                if trends_data.empty:
                    logger.warning("No data returned from Google Trends (attempt %d)", attempt + 1)
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        return pd.DataFrame()
                
                # Clean data
                trends_data = self._clean_trends_data(trends_data)
                
                logger.info("Successfully collected data with shape: %s", trends_data.shape)
                
                # Save raw data if requested
                if save_raw:
                    self._save_raw_data(trends_data, keywords)
                
                return trends_data
                
            except Exception as e:
                logger.warning("Attempt %d failed: %s", attempt + 1, str(e))
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    # Try to reinitialize pytrends
                    try:
                        self._initialize_pytrends()
                    except Exception:
                        pass
                else:
                    logger.error("All attempts failed for keywords: %s", keywords)
                    raise
        
        return pd.DataFrame()
    
    def _build_payload_safe(self, keywords: List[str], timeframe: str, geo: str):
        """Safely build pytrends payload."""
        try:
            self.pytrends.build_payload(
                kw_list=keywords,
                cat=0,
                timeframe=timeframe,
                geo=geo,
                gprop=''
            )
        except Exception as e:
            logger.error("Failed to build payload: %s", str(e))
            raise
    
    def _get_interest_over_time_safe(self) -> pd.DataFrame:
        """Safely get interest over time data."""
        try:
            return self.pytrends.interest_over_time()
        except Exception as e:
            logger.error("Failed to get interest over time: %s", str(e))
            raise
    
    def _clean_trends_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate trends data."""
        # Remove 'isPartial' column if present
        if 'isPartial' in data.columns:
            data = data.drop('isPartial', axis=1)
        
        # Handle any data type issues
        for col in data.columns:
            if data[col].dtype == 'object':
                try:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                except Exception:
                    pass
        
        # Fill any NaN values
        data = data.fillna(0)
        
        return data
    
    def get_related_queries_safe(self, keyword: str,
                                timeframe: Optional[str] = None,
                                geo: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Safely get related queries.
        
        Args:
            keyword: The keyword to analyze.
            timeframe: Time period.
            geo: Geographic location.
        
        Returns:
            Dictionary with 'top' and 'rising' DataFrames.
        """
        timeframe = timeframe or self.config.timeframe
        geo = geo if geo is not None else self.config.geo
        
        try:
            self._build_payload_safe([keyword], timeframe, geo)
            related_queries = self.pytrends.related_queries()
            
            result = {
                'top': pd.DataFrame(),
                'rising': pd.DataFrame()
            }
            
            if keyword in related_queries:
                if related_queries[keyword]['top'] is not None:
                    result['top'] = related_queries[keyword]['top']
                if related_queries[keyword]['rising'] is not None:
                    result['rising'] = related_queries[keyword]['rising']
            
            return result
            
        except Exception as e:
            logger.error("Failed to get related queries for %s: %s", keyword, str(e))
            return {'top': pd.DataFrame(), 'rising': pd.DataFrame()}
    
    def _save_raw_data(self, data: pd.DataFrame, keywords: List[str]) -> None:
        """Save raw data with metadata."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trends_{'_'.join(keywords)[:50]}_{timestamp}.csv"
            filepath = os.path.join(self.config.raw_data_path, filename)
            
            # Ensure directory exists
            os.makedirs(self.config.raw_data_path, exist_ok=True)
            
            # Save data
            data.to_csv(filepath)
            
            # Save metadata
            metadata = {
                'keywords': keywords,
                'timeframe': self.config.timeframe,
                'geo': self.config.geo or 'Worldwide',
                'collected_at': timestamp,
                'shape': list(data.shape),
                'date_range': [str(data.index.min()), str(data.index.max())]
            }
            
            metadata_file = filepath.replace('.csv', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("Saved raw data to: %s", filepath)
            
        except Exception as e:
            logger.warning("Failed to save raw data: %s", str(e))


def batch_collect_keywords_safe(keywords: List[str], 
                               config: Optional[DataConfig] = None,
                               batch_size: int = 5) -> pd.DataFrame:
    """
    Safely collect trends for many keywords by batching them.
    
    Args:
        keywords: List of all keywords to collect.
        config: Data configuration.
        batch_size: Maximum keywords per batch (max 5).
    
    Returns:
        Combined DataFrame with all keyword data.
    """
    if batch_size > 5:
        batch_size = 5
        logger.warning("Batch size reduced to 5 (Google Trends API limit)")
    
    collector = SafeTrendsCollector(config)
    all_data = []
    
    # Process keywords in batches
    for i in tqdm(range(0, len(keywords), batch_size), desc="Collecting batches"):
        batch = keywords[i:i + batch_size]
        
        try:
            batch_data = collector.collect_trends(batch, save_raw=False)
            if not batch_data.empty:
                all_data.append(batch_data)
            
            # Delay to avoid rate limiting
            if i + batch_size < len(keywords):
                time.sleep(3)  # Increased delay for safety
                
        except Exception as e:
            logger.error("Failed to collect batch %s: %s", batch, str(e))
            continue
    
    if not all_data:
        logger.warning("No data collected for any batch")
        return pd.DataFrame()
    
    # Combine all data
    try:
        combined = pd.concat(all_data, axis=1)
        
        # Save the combined data
        if config:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"batch_trends_{len(keywords)}_keywords_{timestamp}.csv"
            filepath = os.path.join(config.raw_data_path, filename)
            os.makedirs(config.raw_data_path, exist_ok=True)
            combined.to_csv(filepath)
            logger.info("Saved batch collection to: %s", filepath)
        
        return combined
        
    except Exception as e:
        logger.error("Failed to combine batch data: %s", str(e))
        return pd.DataFrame()


if __name__ == "__main__":
    # Test the safe collector
    logging.basicConfig(level=logging.INFO)
    
    try:
        collector = SafeTrendsCollector()
        keywords = ["python", "javascript"]
        data = collector.collect_trends(keywords, timeframe="today 12-m")
        
        if not data.empty:
            print(f"Successfully collected data: {data.shape}")
            print(data.head())
        else:
            print("No data collected")
            
    except Exception as e:
        print(f"Error: {e}") 