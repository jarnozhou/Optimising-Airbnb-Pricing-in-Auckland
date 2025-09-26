"""
Seasonal Trends Analysis Module

This module provides comprehensive analysis of seasonal patterns in Airbnb pricing,
including temporal trends, demand cycles, and seasonal pricing strategies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class SeasonalTrendsAnalyzer:
    """
    Comprehensive analyzer for identifying seasonal trends in Airbnb pricing and demand.
    
    This class provides methods to:
    - Analyze monthly and seasonal pricing patterns
    - Identify peak and off-peak periods
    - Compare trends across different property types and regions
    - Forecast seasonal demand patterns
    """
    
    def __init__(self):
        self.seasonal_stats_ = None
        self.monthly_trends_ = None
        self.peak_periods_ = None
        
    def analyze_monthly_trends(self, 
                              df: pd.DataFrame, 
                              date_col: str = 'date',
                              price_col: str = 'price',
                              group_by: Optional[str] = None) -> pd.DataFrame:
        """
        Analyze monthly pricing and demand trends.
        
        Args:
            df: DataFrame containing the data
            date_col: Name of the date column
            price_col: Name of the price column
            group_by: Optional column to group analysis by (e.g., 'room_type', 'neighbourhood')
            
        Returns:
            DataFrame with monthly trend statistics
        """
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Extract month information
        df_analysis = df.copy()
        df_analysis['month'] = df_analysis[date_col].dt.month
        df_analysis['month_name'] = df_analysis[date_col].dt.month_name()
        
        if group_by:
            # Group by month and specified column
            monthly_stats = df_analysis.groupby(['month', 'month_name', group_by])[price_col].agg([
                'count', 'mean', 'median', 'std', 'min', 'max'
            ]).round(2)
        else:
            # Group by month only
            monthly_stats = df_analysis.groupby(['month', 'month_name'])[price_col].agg([
                'count', 'mean', 'median', 'std', 'min', 'max'
            ]).round(2)
        
        monthly_stats = monthly_stats.reset_index()
        monthly_stats.rename(columns={'count': 'listings_count'}, inplace=True)
        
        self.monthly_trends_ = monthly_stats
        return monthly_stats
    
    def identify_seasonal_patterns(self, 
                                 df: pd.DataFrame, 
                                 date_col: str = 'date',
                                 price_col: str = 'price') -> Dict:
        """
        Identify distinct seasonal patterns and classify periods.
        
        Args:
            df: DataFrame containing the data
            date_col: Name of the date column
            price_col: Name of the price column
            
        Returns:
            Dictionary containing seasonal pattern analysis
        """
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Extract seasonal information
        df_seasonal = df.copy()
        df_seasonal['month'] = df_seasonal[date_col].dt.month
        df_seasonal['quarter'] = df_seasonal[date_col].dt.quarter
        
        # Define seasons (Southern Hemisphere - Auckland)
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Summer'
            elif month in [3, 4, 5]:
                return 'Autumn'
            elif month in [6, 7, 8]:
                return 'Winter'
            else:  # [9, 10, 11]
                return 'Spring'
        
        df_seasonal['season'] = df_seasonal['month'].apply(get_season)
        
        # Calculate seasonal statistics
        seasonal_stats = df_seasonal.groupby('season')[price_col].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(2)
        
        # Calculate monthly statistics
        monthly_stats = df_seasonal.groupby('month')[price_col].agg([
            'count', 'mean', 'median', 'std'
        ]).round(2)
        
        # Identify peak and off-peak periods
        mean_prices = monthly_stats['mean'].to_dict()
        overall_mean = monthly_stats['mean'].mean()
        
        peak_months = [month for month, price in mean_prices.items() if price > overall_mean * 1.1]
        off_peak_months = [month for month, price in mean_prices.items() if price < overall_mean * 0.9]
        
        # Calculate seasonal variations
        seasonal_variation = {
            'peak_months': peak_months,
            'off_peak_months': off_peak_months,
            'highest_price_month': max(mean_prices, key=mean_prices.get),
            'lowest_price_month': min(mean_prices, key=mean_prices.get),
            'price_variation_pct': ((max(mean_prices.values()) - min(mean_prices.values())) / 
                                  min(mean_prices.values()) * 100)
        }
        
        self.seasonal_stats_ = seasonal_stats
        self.peak_periods_ = seasonal_variation
        
        return {
            'seasonal_stats': seasonal_stats,
            'monthly_stats': monthly_stats,
            'seasonal_patterns': seasonal_variation
        }
    
    def analyze_demand_patterns(self, 
                              df: pd.DataFrame, 
                              date_col: str = 'date',
                              availability_col: str = 'availability_365') -> Dict:
        """
        Analyze demand patterns based on availability and booking data.
        
        Args:
            df: DataFrame containing the data
            date_col: Name of the date column
            availability_col: Name of the availability column
            
        Returns:
            Dictionary containing demand pattern analysis
        """
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        df_demand = df.copy()
        df_demand['month'] = df_demand[date_col].dt.month
        df_demand['occupancy_rate'] = (365 - df_demand[availability_col]) / 365 * 100
        
        # Monthly demand statistics
        demand_stats = df_demand.groupby('month').agg({
            availability_col: ['mean', 'median'],
            'occupancy_rate': ['mean', 'median']
        }).round(2)
        
        demand_stats.columns = ['avg_availability', 'median_availability', 
                               'avg_occupancy_rate', 'median_occupancy_rate']
        
        # Identify high and low demand periods
        avg_occupancy = demand_stats['avg_occupancy_rate'].to_dict()
        overall_avg_occupancy = demand_stats['avg_occupancy_rate'].mean()
        
        high_demand_months = [month for month, rate in avg_occupancy.items() 
                             if rate > overall_avg_occupancy + 5]
        low_demand_months = [month for month, rate in avg_occupancy.items() 
                            if rate < overall_avg_occupancy - 5]
        
        return {
            'demand_stats': demand_stats.reset_index(),
            'high_demand_months': high_demand_months,
            'low_demand_months': low_demand_months,
            'peak_occupancy_month': max(avg_occupancy, key=avg_occupancy.get),
            'lowest_occupancy_month': min(avg_occupancy, key=avg_occupancy.get)
        }
    
    def compare_seasonal_trends_by_type(self, 
                                      df: pd.DataFrame, 
                                      date_col: str = 'date',
                                      price_col: str = 'price',
                                      type_col: str = 'room_type') -> pd.DataFrame:
        """
        Compare seasonal trends across different property types or categories.
        
        Args:
            df: DataFrame containing the data
            date_col: Name of the date column
            price_col: Name of the price column
            type_col: Name of the grouping column (e.g., room_type, neighbourhood)
            
        Returns:
            DataFrame with comparative seasonal analysis
        """
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        df_compare = df.copy()
        df_compare['month'] = df_compare[date_col].dt.month
        df_compare['month_name'] = df_compare[date_col].dt.month_name()
        
        # Group by month and type
        comparison_stats = df_compare.groupby(['month', 'month_name', type_col])[price_col].agg([
            'count', 'mean', 'median', 'std'
        ]).round(2)
        
        comparison_stats = comparison_stats.reset_index()
        comparison_stats.rename(columns={'count': 'listings_count'}, inplace=True)
        
        return comparison_stats
    
    def visualize_monthly_trends(self, 
                               figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
        """
        Create visualization of monthly pricing trends.
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure object
        """
        if self.monthly_trends_ is None:
            raise ValueError("Monthly trends analysis must be run first")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Monthly average prices
        ax1.plot(self.monthly_trends_['month'], self.monthly_trends_['mean'], 
                marker='o', linewidth=2, markersize=6)
        ax1.set_title('Average Monthly Prices')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Average Price ($)')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(1, 13))
        
        # Monthly listing counts
        ax2.bar(self.monthly_trends_['month'], self.monthly_trends_['listings_count'], 
               color='lightcoral', alpha=0.7)
        ax2.set_title('Monthly Listing Counts')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Number of Listings')
        ax2.set_xticks(range(1, 13))
        
        # Price variation (box plot style with std dev)
        ax3.errorbar(self.monthly_trends_['month'], self.monthly_trends_['mean'], 
                    yerr=self.monthly_trends_['std'], fmt='o', capsize=5, capthick=2)
        ax3.set_title('Monthly Price Variation (Mean ± Std Dev)')
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Price ($)')
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(range(1, 13))
        
        # Min-Max range
        ax4.fill_between(self.monthly_trends_['month'], 
                        self.monthly_trends_['min'], 
                        self.monthly_trends_['max'], 
                        alpha=0.3, color='skyblue', label='Min-Max Range')
        ax4.plot(self.monthly_trends_['month'], self.monthly_trends_['median'], 
                marker='s', color='red', label='Median')
        ax4.set_title('Monthly Price Range and Median')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Price ($)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(range(1, 13))
        
        plt.tight_layout()
        return fig
    
    def visualize_seasonal_patterns(self, 
                                  figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create visualization of seasonal patterns.
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure object
        """
        if self.seasonal_stats_ is None:
            raise ValueError("Seasonal patterns analysis must be run first")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Seasonal average prices
        seasons = self.seasonal_stats_.index
        prices = self.seasonal_stats_['mean']
        
        bars = ax1.bar(seasons, prices, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax1.set_title('Average Prices by Season')
        ax1.set_ylabel('Average Price ($)')
        ax1.set_xlabel('Season')
        
        # Add value labels on bars
        for bar, price in zip(bars, prices):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'${price:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Seasonal listing counts
        counts = self.seasonal_stats_['count']
        bars2 = ax2.bar(seasons, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax2.set_title('Number of Listings by Season')
        ax2.set_ylabel('Number of Listings')
        ax2.set_xlabel('Season')
        
        # Add value labels on bars
        for bar, count in zip(bars2, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def generate_seasonal_insights_report(self, df: pd.DataFrame) -> str:
        """
        Generate a comprehensive text report of seasonal insights.
        
        Args:
            df: DataFrame containing the data
            
        Returns:
            String containing the seasonal insights report
        """
        report = []
        report.append("=" * 60)
        report.append("AIRBNB SEASONAL TRENDS ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Dataset overview
        report.append(f"Analysis Period: {len(df):,} listings analyzed")
        report.append("")
        
        # Seasonal patterns
        if self.seasonal_stats_ is not None:
            report.append("Seasonal Price Analysis:")
            report.append("-" * 25)
            for season in self.seasonal_stats_.index:
                stats = self.seasonal_stats_.loc[season]
                report.append(f"{season:>8}: ${stats['mean']:6.2f} avg, "
                            f"{stats['count']:4.0f} listings")
            report.append("")
        
        # Peak periods
        if self.peak_periods_:
            report.append("Peak and Off-Peak Periods:")
            report.append("-" * 30)
            
            month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                          7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
            
            peak_months_str = ', '.join([month_names[m] for m in self.peak_periods_['peak_months']])
            off_peak_months_str = ', '.join([month_names[m] for m in self.peak_periods_['off_peak_months']])
            
            report.append(f"Peak months: {peak_months_str}")
            report.append(f"Off-peak months: {off_peak_months_str}")
            report.append(f"Highest prices: {month_names[self.peak_periods_['highest_price_month']]}")
            report.append(f"Lowest prices: {month_names[self.peak_periods_['lowest_price_month']]}")
            report.append(f"Price variation: {self.peak_periods_['price_variation_pct']:.1f}%")
            report.append("")
        
        # Key insights
        report.append("Key Seasonal Insights:")
        report.append("-" * 22)
        report.append("• Summer (Dec-Feb) typically shows highest demand and prices")
        report.append("• Winter (Jun-Aug) generally represents the off-peak period")
        report.append("• Shoulder seasons offer balanced pricing opportunities")
        report.append("• Holiday periods and events significantly impact pricing")
        report.append("• Regional variations exist within seasonal patterns")
        report.append("")
        
        # Recommendations
        report.append("Pricing Recommendations:")
        report.append("-" * 24)
        report.append("• Implement dynamic pricing strategies for peak seasons")
        report.append("• Offer competitive rates during off-peak periods")
        report.append("• Monitor local events and adjust pricing accordingly")
        report.append("• Consider minimum stay requirements during high demand")
        report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)


def create_seasonal_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Create sample Airbnb data with seasonal patterns for demonstration.
    
    Args:
        n_samples: Number of sample records to generate
        
    Returns:
        DataFrame with sample seasonal Airbnb data
    """
    np.random.seed(42)
    
    # Generate date range covering multiple seasons
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = pd.date_range(start_date, end_date, freq='D')
    
    # Generate sample data
    dates = np.random.choice(date_range, n_samples)
    
    data = {
        'date': dates,
        'accommodates': np.random.randint(1, 17, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'room_type': np.random.choice(['Entire home/apt', 'Private room', 'Shared room'], 
                                    n_samples, p=[0.6, 0.35, 0.05]),
        'neighbourhood_group': np.random.choice(['Central', 'North', 'South', 'East', 'West'], 
                                              n_samples, p=[0.3, 0.2, 0.2, 0.15, 0.15]),
        'availability_365': np.random.randint(30, 365, n_samples),
        'number_of_reviews': np.random.randint(0, 201, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create seasonal pricing patterns
    df['month'] = pd.to_datetime(df['date']).dt.month
    
    # Base price
    base_price = 80
    
    # Seasonal multipliers (Auckland patterns - higher in summer)
    seasonal_multipliers = {
        12: 1.3, 1: 1.4, 2: 1.3,  # Summer
        3: 1.1, 4: 1.0, 5: 0.9,   # Autumn
        6: 0.8, 7: 0.8, 8: 0.9,   # Winter
        9: 1.0, 10: 1.1, 11: 1.2  # Spring
    }
    
    # Apply seasonal patterns
    seasonal_factor = df['month'].map(seasonal_multipliers)
    
    # Generate price with seasonal patterns
    price = (
        base_price * seasonal_factor +
        df['accommodates'] * 12 +
        df['bedrooms'] * 15 +
        (df['room_type'] == 'Entire home/apt') * 25 +
        np.random.normal(0, 15, n_samples)  # Add noise
    )
    
    # Ensure minimum price
    df['price'] = np.maximum(price, 30).round(2)
    
    return df


if __name__ == "__main__":
    # Example usage
    print("Creating seasonal sample data...")
    sample_data = create_seasonal_sample_data(2000)
    
    print("Initializing seasonal trends analyzer...")
    analyzer = SeasonalTrendsAnalyzer()
    
    print("Analyzing monthly trends...")
    monthly_trends = analyzer.analyze_monthly_trends(sample_data)
    
    print("Identifying seasonal patterns...")
    seasonal_patterns = analyzer.identify_seasonal_patterns(sample_data)
    
    print("Analyzing demand patterns...")
    demand_patterns = analyzer.analyze_demand_patterns(sample_data)
    
    print("Generating seasonal insights report...")
    report = analyzer.generate_seasonal_insights_report(sample_data)
    
    print(report)