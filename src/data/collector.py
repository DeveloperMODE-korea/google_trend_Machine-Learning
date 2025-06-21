"""
Google Trends data collector module.

This module provides functionality to collect search trend data from Google Trends API
using the pytrends library. It focuses on worldwide data collection and supports
multiple keywords and custom time ranges.
"""

import os
import time
import logging
from typing import List, Dict, Optional, Union
from datetime import datetime
import pandas as pd
from pytrends.request import TrendReq
from tqdm import tqdm
import json

from ..utils.config import DataConfig, get_timeframe_dates


logger = logging.getLogger(__name__)


class TrendsCollector:
    """
    Collector class for Google Trends data.
    
    This class handles the connection to Google Trends API and provides methods
    to collect trend data for specified keywords and time periods.
    """
    
    def __init__(self, config: Optional[DataConfig] = None, 
                 retries: int = 3, backoff_factor: float = 2.0):
        """
        Initialize the TrendsCollector.
        
        Args:
            config: Data configuration object. If None, uses default config.
            retries: Number of retry attempts for failed requests.
            backoff_factor: Factor for exponential backoff between retries.
        """
        self.config = config or DataConfig()
        self.retries = retries
        self.backoff_factor = backoff_factor
        
        # Initialize pytrends with retries
        try:
            # Try new parameter name first (newer versions)
            self.pytrends = TrendReq(
                hl=self.config.language,
                tz=0,  # UTC timezone
                retries=retries,
                backoff_factor=backoff_factor
            )
        except Exception as e:
            # Fallback to basic initialization without retry parameters
            logger.warning("Failed to initialize TrendReq with retry parameters: %s", str(e))
            try:
                self.pytrends = TrendReq(
                    hl=self.config.language,
                    tz=0
                )
            except Exception as e2:
                # Last resort - minimal initialization
                logger.warning("Failed basic initialization, using minimal setup: %s", str(e2))
                self.pytrends = TrendReq()
        
        logger.info("TrendsCollector initialized with language: %s", self.config.language)
    
    def collect_trends(self, keywords: List[str], 
                      timeframe: Optional[str] = None,
                      geo: Optional[str] = None,
                      save_raw: bool = True) -> pd.DataFrame:
        """
        Collect Google Trends data for specified keywords.
        
        Args:
            keywords: List of keywords to collect trends for (max 5 at a time).
            timeframe: Time period for data collection. If None, uses config default.
            geo: Geographic location. Empty string for worldwide. If None, uses config.
            save_raw: Whether to save raw data to file.
        
        Returns:
            DataFrame with trend data. Columns are keywords, index is datetime.
        
        Raises:
            ValueError: If more than 5 keywords are provided.
            Exception: If data collection fails after retries.
        """
        if len(keywords) > 5:
            raise ValueError("Google Trends API accepts maximum 5 keywords at a time")
        
        timeframe = timeframe or self.config.timeframe
        geo = geo if geo is not None else self.config.geo
        
        logger.info("Collecting trends for keywords: %s", keywords)
        logger.info("Timeframe: %s, Geo: %s", timeframe, geo or "Worldwide")
        
        # Build payload
        try:
            self.pytrends.build_payload(
                kw_list=keywords,
                cat=0,  # All categories
                timeframe=timeframe,
                geo=geo,
                gprop=''  # Web search
            )
            
            # Get interest over time
            trends_data = self.pytrends.interest_over_time()
            
            if trends_data.empty:
                logger.warning("No data returned from Google Trends")
                return pd.DataFrame()
            
            # Remove 'isPartial' column if present
            if 'isPartial' in trends_data.columns:
                trends_data = trends_data.drop('isPartial', axis=1)
            
            logger.info("Successfully collected data with shape: %s", trends_data.shape)
            
            # Save raw data if requested
            if save_raw:
                self._save_raw_data(trends_data, keywords)
            
            return trends_data
            
        except Exception as e:
            logger.error("Failed to collect trends data: %s", str(e))
            raise
    
    def collect_multiple_periods(self, keywords: List[str],
                               periods: List[str],
                               geo: Optional[str] = None) -> pd.DataFrame:
        """
        Collect trends data for multiple time periods and combine them.
        
        Useful for getting more granular data by collecting shorter periods
        and then combining them.
        
        Args:
            keywords: List of keywords to collect.
            periods: List of timeframe strings.
            geo: Geographic location.
        
        Returns:
            Combined DataFrame with all period data.
        """
        all_data = []
        
        for period in tqdm(periods, desc="Collecting periods"):
            try:
                period_data = self.collect_trends(keywords, timeframe=period, geo=geo, save_raw=False)
                if not period_data.empty:
                    all_data.append(period_data)
                
                # Add delay to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.warning("Failed to collect data for period %s: %s", period, str(e))
                continue
        
        if not all_data:
            logger.warning("No data collected for any period")
            return pd.DataFrame()
        
        # Combine all data
        combined_data = pd.concat(all_data, axis=0)
        combined_data = combined_data[~combined_data.index.duplicated(keep='first')]
        combined_data = combined_data.sort_index()
        
        return combined_data
    
    def get_related_queries(self, keyword: str,
                          timeframe: Optional[str] = None,
                          geo: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Get related queries for a keyword.
        
        Args:
            keyword: The keyword to get related queries for.
            timeframe: Time period for analysis.
            geo: Geographic location.
        
        Returns:
            Dictionary with 'top' and 'rising' DataFrames of related queries.
        """
        timeframe = timeframe or self.config.timeframe
        geo = geo if geo is not None else self.config.geo
        
        try:
            self.pytrends.build_payload(
                kw_list=[keyword],
                timeframe=timeframe,
                geo=geo
            )
            
            related_queries = self.pytrends.related_queries()
            
            return {
                'top': related_queries[keyword]['top'],
                'rising': related_queries[keyword]['rising']
            }
            
        except Exception as e:
            logger.error("Failed to get related queries for %s: %s", keyword, str(e))
            return {'top': pd.DataFrame(), 'rising': pd.DataFrame()}
    
    def get_trending_searches(self, geo: str = 'US') -> pd.DataFrame:
        """
        Get currently trending searches for a specific country.
        
        Args:
            geo: Country code (e.g., 'US', 'GB', 'KR'). 
                 Note: This doesn't work for worldwide.
        
        Returns:
            DataFrame with trending searches.
        """
        try:
            trending = self.pytrends.trending_searches(pn=geo.lower())
            return trending
        except Exception as e:
            logger.error("Failed to get trending searches for %s: %s", geo, str(e))
            return pd.DataFrame()
    
    def _save_raw_data(self, data: pd.DataFrame, keywords: List[str]) -> None:
        """Save raw data to file with metadata."""
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


def batch_collect_keywords(keywords: List[str], 
                         config: Optional[DataConfig] = None,
                         batch_size: int = 5) -> pd.DataFrame:
    """
    Collect trends for many keywords by batching them.
    
    Google Trends allows max 5 keywords per request, so this function
    batches larger keyword lists and combines the results.
    
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
    
    collector = TrendsCollector(config)
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
                time.sleep(2)
                
        except Exception as e:
            logger.error("Failed to collect batch %s: %s", batch, str(e))
            continue
    
    if not all_data:
        return pd.DataFrame()
    
    # Combine all data
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


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect Google Trends data")
    parser.add_argument("--keywords", type=str, required=True,
                       help="Comma-separated list of keywords")
    parser.add_argument("--timeframe", type=str, default="today 5-y",
                       help="Timeframe for data collection")
    parser.add_argument("--geo", type=str, default="",
                       help="Geographic location (empty for worldwide)")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Collect data
    keywords = [k.strip() for k in args.keywords.split(',')]
    collector = TrendsCollector()
    
    data = collector.collect_trends(
        keywords=keywords,
        timeframe=args.timeframe,
        geo=args.geo
    )
    
    if not data.empty:
        print(f"\nCollected data shape: {data.shape}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        print(f"\nFirst few rows:")
        print(data.head())
    else:
        print("No data collected") 