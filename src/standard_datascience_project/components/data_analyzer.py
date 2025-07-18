import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
from ..entity.config_entity import FeatureConfig

class DataAnalyzer:
    """Component for exploratory data analysis and visualization"""
    
    def __init__(self, feature_config: FeatureConfig):
        self.feature_config = feature_config
        # Set display settings
        pd.set_option('display.max_columns', None)
        plt.style.use('ggplot')
        sns.set(style="whitegrid")
        
    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive data analysis"""
        
        analysis_results = {}
        
        # Basic info and missing values
        analysis_results['info'] = df.info()
        analysis_results['missing_values'] = df.isnull().sum()
        
        # Descriptive statistics by anomaly status
        analysis_results['descriptive_stats'] = df.groupby('Anomaly').describe()
        
        # Anomaly ratio
        analysis_results['anomaly_ratio'] = df['Anomaly'].value_counts(normalize=True) * 100
        
        # Correlation analysis
        df_temp = df.copy()
        df_temp['Login Status'] = df_temp['Login Status'].map({'Success': 1, 'Fail': 0})
        correlation_matrix = df_temp.select_dtypes(include=[np.number]).corr()
        analysis_results['correlation_matrix'] = correlation_matrix
        
        return analysis_results
    
    def print_analysis_summary(self, analysis_results: Dict[str, Any]):
        """Print analysis summary"""
        print("=== DATA ANALYSIS SUMMARY ===")
        print("\nMissing values in each column:")
        print(analysis_results['missing_values'])
        
        print("\nDescriptive Statistics by Anomaly Status:")
        print(analysis_results['descriptive_stats'])
        
        print("\nAnomaly Ratio (%):")
        print(analysis_results['anomaly_ratio'])
        
        print("\nCorrelation Matrix of all features:")
        print(analysis_results['correlation_matrix'])
    
    def create_visualizations(self, df: pd.DataFrame, save_plots: bool = False):
        """Create comprehensive visualizations"""
        
        # Density and box plots for numerical features
        for feature in self.feature_config.numerical_features:
            plt.figure(figsize=(20, 6))
            
            # Density plot
            plt.subplot(1, 2, 1)
            sns.histplot(
                data=df, 
                x=feature, 
                hue='Anomaly', 
                element='step', 
                stat='density', 
                common_norm=False, 
                kde=True, 
                palette={0: 'skyblue', 1: 'salmon'}
            )
            plt.title(f'Density Plot of {feature} by Anomaly Status')
            plt.xlabel(feature)
            plt.ylabel('Density')
            
            # Box plot
            plt.subplot(1, 2, 2)
            sns.boxplot(
                x='Anomaly', 
                y=feature, 
                data=df, 
                palette={'0': 'skyblue', '1': 'salmon'}
            )
            plt.title(f'Box Plot of {feature} by Anomaly Status')
            plt.xlabel('Anomaly Status')
            plt.ylabel(feature)
            
            if save_plots:
                plt.savefig(f'plots/{feature}_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Correlation heatmap
        df_temp = df.copy()
        df_temp['Login Status'] = df_temp['Login Status'].map({'Success': 1, 'Fail': 0})
        correlation_matrix = df_temp.select_dtypes(include=[np.number]).corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap='coolwarm', 
            cbar_kws={'label': 'Correlation Coefficient'}
        )
        plt.title('Correlation Matrix of Numerical Features')
        
        if save_plots:
            plt.savefig('plots/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Pairplot
        sns.pairplot(
            df, 
            hue='Anomaly', 
            vars=self.feature_config.numerical_features, 
            palette={0: 'skyblue', 1: 'salmon'}
        )
        plt.suptitle('Pairwise relationships by Anomaly Status', verticalalignment='top')
        
        if save_plots:
            plt.savefig('plots/pairplot.png', dpi=300, bbox_inches='tight')
        plt.show() 