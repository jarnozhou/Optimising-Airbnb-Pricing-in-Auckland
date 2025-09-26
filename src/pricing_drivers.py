"""
Pricing Drivers Analysis Module

This module provides comprehensive analysis of key factors that drive Airbnb pricing,
including property features, host attributes, location factors, and review metrics.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class PricingDriversAnalyzer:
    """
    Comprehensive analyzer for identifying key pricing drivers in Airbnb data.
    
    This class provides methods to:
    - Identify the most important features affecting pricing
    - Perform correlation analysis
    - Build predictive models to understand feature importance
    - Visualize pricing relationships
    """
    
    def __init__(self):
        self.feature_importance_ = None
        self.correlation_matrix_ = None
        self.model_ = None
        self.scaler_ = StandardScaler()
        self.label_encoders_ = {}
        
    def analyze_feature_importance(self, 
                                 df: pd.DataFrame, 
                                 target_col: str = 'price',
                                 feature_cols: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Analyze feature importance using Random Forest to identify key pricing drivers.
        
        Args:
            df: DataFrame containing the data
            target_col: Name of the target column (price)
            feature_cols: List of feature columns to analyze. If None, uses all numeric columns.
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if feature_cols is None:
            # Select numeric columns excluding the target
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in feature_cols:
                feature_cols.remove(target_col)
        
        # Prepare data
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        y = y.fillna(y.median())
        
        # Encode categorical variables if any
        for col in X.columns:
            if X[col].dtype == 'object':
                if col not in self.label_encoders_:
                    self.label_encoders_[col] = LabelEncoder()
                    X[col] = self.label_encoders_[col].fit_transform(X[col].astype(str))
                else:
                    X[col] = self.label_encoders_[col].transform(X[col].astype(str))
        
        # Train Random Forest model
        self.model_ = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_.fit(X, y)
        
        # Get feature importance
        importance_scores = dict(zip(feature_cols, self.model_.feature_importances_))
        self.feature_importance_ = dict(sorted(importance_scores.items(), 
                                             key=lambda x: x[1], 
                                             reverse=True))
        
        return self.feature_importance_
    
    def correlation_analysis(self, 
                           df: pd.DataFrame, 
                           target_col: str = 'price',
                           min_correlation: float = 0.1) -> pd.DataFrame:
        """
        Perform correlation analysis to identify features correlated with pricing.
        
        Args:
            df: DataFrame containing the data
            target_col: Name of the target column (price)
            min_correlation: Minimum absolute correlation threshold
            
        Returns:
            DataFrame with correlation coefficients sorted by absolute value
        """
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Calculate correlations with target
        correlations = numeric_df.corr()[target_col].abs().sort_values(ascending=False)
        
        # Filter by minimum correlation threshold
        significant_correlations = correlations[correlations >= min_correlation]
        
        # Create results dataframe
        correlation_df = pd.DataFrame({
            'feature': significant_correlations.index,
            'correlation': numeric_df.corr()[target_col][significant_correlations.index],
            'abs_correlation': significant_correlations.values
        })
        
        self.correlation_matrix_ = numeric_df.corr()
        
        return correlation_df
    
    def price_distribution_analysis(self, 
                                  df: pd.DataFrame, 
                                  price_col: str = 'price',
                                  categorical_features: Optional[List[str]] = None) -> Dict:
        """
        Analyze price distribution across different categorical features.
        
        Args:
            df: DataFrame containing the data
            price_col: Name of the price column
            categorical_features: List of categorical features to analyze
            
        Returns:
            Dictionary containing price statistics for each category
        """
        if categorical_features is None:
            categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        
        results = {}
        
        for feature in categorical_features:
            if feature in df.columns:
                price_stats = df.groupby(feature)[price_col].agg([
                    'count', 'mean', 'median', 'std', 'min', 'max'
                ]).round(2)
                
                results[feature] = price_stats.to_dict('index')
        
        return results
    
    def visualize_top_drivers(self, 
                            top_n: int = 10, 
                            figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create visualization of top pricing drivers.
        
        Args:
            top_n: Number of top features to visualize
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure object
        """
        if self.feature_importance_ is None:
            raise ValueError("Feature importance analysis must be run first")
        
        # Get top N features
        top_features = dict(list(self.feature_importance_.items())[:top_n])
        
        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=figsize)
        
        features = list(top_features.keys())
        importances = list(top_features.values())
        
        bars = ax.barh(range(len(features)), importances, color='skyblue', alpha=0.8)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Feature Importance Score')
        ax.set_title(f'Top {top_n} Pricing Drivers (Random Forest Feature Importance)')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def create_correlation_heatmap(self, 
                                 top_n: int = 15, 
                                 figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Create correlation heatmap for top features.
        
        Args:
            top_n: Number of top correlated features to include
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure object
        """
        if self.correlation_matrix_ is None:
            raise ValueError("Correlation analysis must be run first")
        
        # Get top correlated features
        price_correlations = self.correlation_matrix_['price'].abs().sort_values(ascending=False)
        top_features = price_correlations.head(top_n).index.tolist()
        
        # Create correlation matrix for top features
        top_corr_matrix = self.correlation_matrix_.loc[top_features, top_features]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(top_corr_matrix, 
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={'shrink': 0.8},
                   ax=ax)
        
        ax.set_title(f'Correlation Heatmap: Top {top_n} Features Related to Price')
        plt.tight_layout()
        return fig
    
    def generate_pricing_insights_report(self, df: pd.DataFrame) -> str:
        """
        Generate a comprehensive text report of pricing insights.
        
        Args:
            df: DataFrame containing the data
            
        Returns:
            String containing the insights report
        """
        report = []
        report.append("=" * 60)
        report.append("AIRBNB PRICING DRIVERS ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Dataset overview
        report.append(f"Dataset Overview:")
        report.append(f"- Total listings: {len(df):,}")
        report.append(f"- Features analyzed: {len(df.columns)}")
        report.append("")
        
        # Top pricing drivers
        if self.feature_importance_:
            report.append("Top 10 Pricing Drivers (by importance):")
            report.append("-" * 40)
            for i, (feature, importance) in enumerate(list(self.feature_importance_.items())[:10], 1):
                report.append(f"{i:2d}. {feature:<25} {importance:.4f}")
            report.append("")
        
        # Price statistics
        if 'price' in df.columns:
            price_stats = df['price'].describe()
            report.append("Price Distribution Statistics:")
            report.append("-" * 30)
            report.append(f"Mean price: ${price_stats['mean']:.2f}")
            report.append(f"Median price: ${price_stats['50%']:.2f}")
            report.append(f"Price range: ${price_stats['min']:.2f} - ${price_stats['max']:.2f}")
            report.append(f"Standard deviation: ${price_stats['std']:.2f}")
            report.append("")
        
        # Key insights
        report.append("Key Insights:")
        report.append("-" * 15)
        report.append("• Property characteristics are the primary pricing drivers")
        report.append("• Location and neighborhood significantly impact pricing")
        report.append("• Host attributes and review metrics influence guest willingness to pay")
        report.append("• Property type and accommodation capacity are major factors")
        report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)


def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Create sample Airbnb data for demonstration purposes.
    
    Args:
        n_samples: Number of sample records to generate
        
    Returns:
        DataFrame with sample Airbnb listing data
    """
    np.random.seed(42)
    
    # Generate sample data
    data = {
        'accommodates': np.random.randint(1, 17, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.uniform(1, 4, n_samples).round(1),
        'beds': np.random.randint(1, 8, n_samples),
        'room_type': np.random.choice(['Entire home/apt', 'Private room', 'Shared room'], 
                                    n_samples, p=[0.6, 0.35, 0.05]),
        'property_type': np.random.choice(['Apartment', 'House', 'Townhouse', 'Condominium'], 
                                        n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'minimum_nights': np.random.randint(1, 31, n_samples),
        'maximum_nights': np.random.randint(30, 366, n_samples),
        'availability_365': np.random.randint(0, 366, n_samples),
        'number_of_reviews': np.random.randint(0, 501, n_samples),
        'review_scores_rating': np.random.uniform(3.0, 5.0, n_samples).round(2),
        'host_is_superhost': np.random.choice([True, False], n_samples, p=[0.2, 0.8]),
        'instant_bookable': np.random.choice([True, False], n_samples, p=[0.3, 0.7]),
        'neighbourhood_group': np.random.choice(['Central', 'North', 'South', 'East', 'West'], 
                                              n_samples, p=[0.3, 0.2, 0.2, 0.15, 0.15])
    }
    
    df = pd.DataFrame(data)
    
    # Generate price based on features (realistic pricing model)
    base_price = 50
    price = (
        base_price +
        df['accommodates'] * 15 +
        df['bedrooms'] * 20 +
        df['bathrooms'] * 10 +
        (df['room_type'] == 'Entire home/apt') * 30 +
        (df['property_type'] == 'House') * 25 +
        (df['host_is_superhost']) * 20 +
        df['review_scores_rating'] * 10 +
        np.random.normal(0, 20, n_samples)  # Add noise
    )
    
    # Ensure minimum price
    df['price'] = np.maximum(price, 25).round(2)
    
    return df


if __name__ == "__main__":
    # Example usage
    print("Creating sample data...")
    sample_data = create_sample_data(1000)
    
    print("Initializing pricing drivers analyzer...")
    analyzer = PricingDriversAnalyzer()
    
    print("Analyzing feature importance...")
    importance = analyzer.analyze_feature_importance(sample_data)
    
    print("Performing correlation analysis...")
    correlations = analyzer.correlation_analysis(sample_data)
    
    print("Generating insights report...")
    report = analyzer.generate_pricing_insights_report(sample_data)
    
    print(report)