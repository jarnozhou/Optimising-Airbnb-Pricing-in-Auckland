"""
Unit tests for pricing drivers analysis module.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pricing_drivers import PricingDriversAnalyzer, create_sample_data


class TestPricingDriversAnalyzer(unittest.TestCase):
    """Test cases for PricingDriversAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = PricingDriversAnalyzer()
        self.sample_data = create_sample_data(100)  # Small dataset for testing
    
    def test_sample_data_creation(self):
        """Test that sample data is created correctly."""
        self.assertEqual(len(self.sample_data), 100)
        self.assertIn('price', self.sample_data.columns)
        self.assertIn('accommodates', self.sample_data.columns)
        self.assertTrue(self.sample_data['price'].min() >= 25)  # Minimum price constraint
    
    def test_feature_importance_analysis(self):
        """Test feature importance analysis."""
        importance = self.analyzer.analyze_feature_importance(self.sample_data)
        
        self.assertIsInstance(importance, dict)
        self.assertTrue(len(importance) > 0)
        
        # Check that importance scores are between 0 and 1
        for score in importance.values():
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)
        
        # Check that analyzer has stored the results
        self.assertIsNotNone(self.analyzer.feature_importance_)
        self.assertIsNotNone(self.analyzer.model_)
    
    def test_correlation_analysis(self):
        """Test correlation analysis."""
        correlations = self.analyzer.correlation_analysis(self.sample_data)
        
        self.assertIsInstance(correlations, pd.DataFrame)
        self.assertTrue(len(correlations) > 0)
        self.assertIn('feature', correlations.columns)
        self.assertIn('correlation', correlations.columns)
        self.assertIn('abs_correlation', correlations.columns)
        
        # Check that analyzer has stored the correlation matrix
        self.assertIsNotNone(self.analyzer.correlation_matrix_)
    
    def test_price_distribution_analysis(self):
        """Test price distribution analysis."""
        categorical_features = ['room_type', 'property_type']
        distributions = self.analyzer.price_distribution_analysis(
            self.sample_data, categorical_features=categorical_features
        )
        
        self.assertIsInstance(distributions, dict)
        self.assertIn('room_type', distributions)
        self.assertIn('property_type', distributions)
        
        # Check that each category has statistical measures
        for feature, stats in distributions.items():
            for category, measures in stats.items():
                self.assertIn('mean', measures)
                self.assertIn('count', measures)
                self.assertIn('std', measures)
    
    def test_visualize_top_drivers(self):
        """Test visualization creation."""
        # First run feature importance analysis
        self.analyzer.analyze_feature_importance(self.sample_data)
        
        # Test visualization creation
        fig = self.analyzer.visualize_top_drivers(top_n=5)
        self.assertIsNotNone(fig)
    
    def test_create_correlation_heatmap(self):
        """Test correlation heatmap creation."""
        # First run correlation analysis
        self.analyzer.correlation_analysis(self.sample_data)
        
        # Test heatmap creation
        fig = self.analyzer.create_correlation_heatmap(top_n=8)
        self.assertIsNotNone(fig)
    
    def test_generate_insights_report(self):
        """Test insights report generation."""
        # Run analyses first
        self.analyzer.analyze_feature_importance(self.sample_data)
        
        report = self.analyzer.generate_pricing_insights_report(self.sample_data)
        self.assertIsInstance(report, str)
        self.assertTrue(len(report) > 0)
        self.assertIn('PRICING DRIVERS ANALYSIS REPORT', report)
        self.assertIn('Top 10 Pricing Drivers', report)
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty dataframe."""
        empty_df = pd.DataFrame()
        
        with self.assertRaises((ValueError, KeyError)):
            self.analyzer.analyze_feature_importance(empty_df)
    
    def test_missing_target_column(self):
        """Test handling of missing target column."""
        df_no_price = self.sample_data.drop('price', axis=1)
        
        with self.assertRaises(KeyError):
            self.analyzer.analyze_feature_importance(df_no_price)


if __name__ == '__main__':
    unittest.main()