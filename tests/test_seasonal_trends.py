"""
Unit tests for seasonal trends analysis module.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from seasonal_trends import SeasonalTrendsAnalyzer, create_seasonal_sample_data


class TestSeasonalTrendsAnalyzer(unittest.TestCase):
    """Test cases for SeasonalTrendsAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SeasonalTrendsAnalyzer()
        self.sample_data = create_seasonal_sample_data(200)  # Small dataset for testing
    
    def test_seasonal_sample_data_creation(self):
        """Test that seasonal sample data is created correctly."""
        self.assertEqual(len(self.sample_data), 200)
        self.assertIn('date', self.sample_data.columns)
        self.assertIn('price', self.sample_data.columns)
        self.assertIn('month', self.sample_data.columns)
        
        # Check date format
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.sample_data['date']))
        
        # Check price constraints
        self.assertTrue(self.sample_data['price'].min() >= 30)
    
    def test_analyze_monthly_trends(self):
        """Test monthly trends analysis."""
        monthly_trends = self.analyzer.analyze_monthly_trends(self.sample_data)
        
        self.assertIsInstance(monthly_trends, pd.DataFrame)
        self.assertTrue(len(monthly_trends) > 0)
        self.assertIn('month', monthly_trends.columns)
        self.assertIn('month_name', monthly_trends.columns)
        self.assertIn('listings_count', monthly_trends.columns)
        self.assertIn('mean', monthly_trends.columns)
        
        # Check that analyzer has stored the results
        self.assertIsNotNone(self.analyzer.monthly_trends_)
    
    def test_analyze_monthly_trends_with_groupby(self):
        """Test monthly trends analysis with grouping."""
        monthly_trends = self.analyzer.analyze_monthly_trends(
            self.sample_data, group_by='room_type'
        )
        
        self.assertIsInstance(monthly_trends, pd.DataFrame)
        self.assertIn('room_type', monthly_trends.columns)
    
    def test_identify_seasonal_patterns(self):
        """Test seasonal patterns identification."""
        patterns = self.analyzer.identify_seasonal_patterns(self.sample_data)
        
        self.assertIsInstance(patterns, dict)
        self.assertIn('seasonal_stats', patterns)
        self.assertIn('monthly_stats', patterns)
        self.assertIn('seasonal_patterns', patterns)
        
        # Check seasonal patterns structure
        seasonal_patterns = patterns['seasonal_patterns']
        self.assertIn('peak_months', seasonal_patterns)
        self.assertIn('off_peak_months', seasonal_patterns)
        self.assertIn('highest_price_month', seasonal_patterns)
        self.assertIn('lowest_price_month', seasonal_patterns)
        self.assertIn('price_variation_pct', seasonal_patterns)
        
        # Check that analyzer has stored the results
        self.assertIsNotNone(self.analyzer.seasonal_stats_)
        self.assertIsNotNone(self.analyzer.peak_periods_)
    
    def test_analyze_demand_patterns(self):
        """Test demand patterns analysis."""
        demand_patterns = self.analyzer.analyze_demand_patterns(self.sample_data)
        
        self.assertIsInstance(demand_patterns, dict)
        self.assertIn('demand_stats', demand_patterns)
        self.assertIn('high_demand_months', demand_patterns)
        self.assertIn('low_demand_months', demand_patterns)
        self.assertIn('peak_occupancy_month', demand_patterns)
        self.assertIn('lowest_occupancy_month', demand_patterns)
        
        # Check demand stats structure
        demand_stats = demand_patterns['demand_stats']
        self.assertIsInstance(demand_stats, pd.DataFrame)
        self.assertIn('avg_occupancy_rate', demand_stats.columns)
    
    def test_compare_seasonal_trends_by_type(self):
        """Test seasonal trends comparison by type."""
        comparison = self.analyzer.compare_seasonal_trends_by_type(
            self.sample_data, type_col='room_type'
        )
        
        self.assertIsInstance(comparison, pd.DataFrame)
        self.assertIn('room_type', comparison.columns)
        self.assertIn('month', comparison.columns)
        self.assertIn('month_name', comparison.columns)
        self.assertIn('mean', comparison.columns)
    
    def test_visualize_monthly_trends(self):
        """Test monthly trends visualization."""
        # First run monthly trends analysis
        self.analyzer.analyze_monthly_trends(self.sample_data)
        
        # Test visualization creation
        fig = self.analyzer.visualize_monthly_trends()
        self.assertIsNotNone(fig)
    
    def test_visualize_seasonal_patterns(self):
        """Test seasonal patterns visualization."""
        # First run seasonal patterns analysis
        self.analyzer.identify_seasonal_patterns(self.sample_data)
        
        # Test visualization creation
        fig = self.analyzer.visualize_seasonal_patterns()
        self.assertIsNotNone(fig)
    
    def test_generate_seasonal_insights_report(self):
        """Test seasonal insights report generation."""
        # Run analyses first
        self.analyzer.analyze_monthly_trends(self.sample_data)
        self.analyzer.identify_seasonal_patterns(self.sample_data)
        
        report = self.analyzer.generate_seasonal_insights_report(self.sample_data)
        self.assertIsInstance(report, str)
        self.assertTrue(len(report) > 0)
        self.assertIn('SEASONAL TRENDS ANALYSIS REPORT', report)
        self.assertIn('Seasonal Price Analysis', report)
    
    def test_date_conversion(self):
        """Test proper date conversion in analysis."""
        # Create data with string dates
        test_data = self.sample_data.copy()
        test_data['date'] = test_data['date'].astype(str)
        
        # Should still work with string dates
        monthly_trends = self.analyzer.analyze_monthly_trends(test_data)
        self.assertIsInstance(monthly_trends, pd.DataFrame)
        self.assertTrue(len(monthly_trends) > 0)
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty dataframe."""
        empty_df = pd.DataFrame({'date': [], 'price': []})
        
        # Should either raise an exception or return empty results
        try:
            result = self.analyzer.analyze_monthly_trends(empty_df)
            # If it doesn't raise an exception, should return empty or minimal results
            if isinstance(result, pd.DataFrame):
                self.assertTrue(len(result) == 0 or result.empty)
        except (ValueError, KeyError, IndexError):
            # This is also acceptable behavior
            pass
    
    def test_missing_required_columns(self):
        """Test handling of missing required columns."""
        df_no_price = self.sample_data.drop('price', axis=1)
        
        with self.assertRaises(KeyError):
            self.analyzer.analyze_monthly_trends(df_no_price)
        
        df_no_date = self.sample_data.drop('date', axis=1)
        
        with self.assertRaises(KeyError):
            self.analyzer.analyze_monthly_trends(df_no_date)


if __name__ == '__main__':
    unittest.main()