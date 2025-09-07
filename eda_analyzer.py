import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class IPOExploratoryAnalyzer:
    """
    Class for comprehensive exploratory data analysis of IPO data
    """
    
    def __init__(self):
        plt.style.use('default')
        sns.set_palette("husl")
        
    def generate_summary_statistics(self, df):
        """
        Generate comprehensive summary statistics
        """
        print("="*60)
        print("COMPREHENSIVE SUMMARY STATISTICS")
        print("="*60)
        
        # Target variable statistics
        print("\nLISTING GAINS STATISTICS:")
        print("-" * 40)
        gains = df['listing_gains']
        
        stats_dict = {
            'Count': len(gains),
            'Mean': gains.mean(),
            'Median': gains.median(),
            'Std Dev': gains.std(),
            'Min': gains.min(),
            'Max': gains.max(),
            'Q1 (25%)': gains.quantile(0.25),
            'Q3 (75%)': gains.quantile(0.75),
            'IQR': gains.quantile(0.75) - gains.quantile(0.25),
            'Skewness': stats.skew(gains),
            'Kurtosis': stats.kurtosis(gains)
        }
        
        for stat, value in stats_dict.items():
            if stat in ['Count']:
                print(f"{stat:<15}: {value:,}")
            else:
                print(f"{stat:<15}: {value:8.2f}%")
        
        # Performance categories
        print(f"\nPERFORMANCE DISTRIBUTION:")
        print("-" * 40)
        categories = {
            'Negative Returns': (gains < 0).sum(),
            'Low Gains (0-20%)': ((gains >= 0) & (gains < 20)).sum(),
            'Medium Gains (20-50%)': ((gains >= 20) & (gains < 50)).sum(),
            'High Gains (50-100%)': ((gains >= 50) & (gains < 100)).sum(),
            'Very High Gains (>100%)': (gains >= 100).sum()
        }
        
        total = len(gains)
        for category, count in categories.items():
            percentage = (count / total) * 100
            print(f"{category:<25}: {count:3d} ({percentage:5.1f}%)")
        
        return stats_dict
    
    def plot_distribution_analysis(self, df):
        """
        Create distribution plots for listing gains
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('IPO Listing Gains - Distribution Analysis', fontsize=16, fontweight='bold')
        
        gains = df['listing_gains']
        
        # 1. Histogram with normal curve
        ax1 = axes[0, 0]
        n, bins, patches = ax1.hist(gains, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
        
        # Overlay normal distribution
        mu, sigma = stats.norm.fit(gains)
        x = np.linspace(gains.min(), gains.max(), 100)
        ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label=f'Normal (μ={mu:.1f}, σ={sigma:.1f})')
        
        ax1.axvline(gains.mean(), color='red', linestyle='--', alpha=0.8, label=f'Mean: {gains.mean():.1f}%')
        ax1.axvline(gains.median(), color='green', linestyle='--', alpha=0.8, label=f'Median: {gains.median():.1f}%')
        ax1.set_xlabel('Listing Gains (%)')
        ax1.set_ylabel('Density')
        ax1.set_title('Distribution with Normal Overlay')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Box plot
        ax2 = axes[0, 1]
        box_plot = ax2.boxplot(gains, patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightblue')
        ax2.set_ylabel('Listing Gains (%)')
        ax2.set_title('Box Plot - Identifying Outliers')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics to box plot
        q1, q3 = gains.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        outliers = gains[(gains < lower_fence) | (gains > upper_fence)]
        
        ax2.text(1.1, gains.max() * 0.8, f'Outliers: {len(outliers)}', transform=ax2.transData)
        
        # 3. Q-Q plot for normality check
        ax3 = axes[1, 0]
        stats.probplot(gains, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot - Normality Check')
        ax3.grid(True, alpha=0.3)
        
        # 4. Cumulative distribution
        ax4 = axes[1, 1]
        gains_sorted = np.sort(gains)
        cumulative_prob = np.arange(1, len(gains_sorted) + 1) / len(gains_sorted)
        ax4.plot(gains_sorted, cumulative_prob, linewidth=2, color='purple')
        ax4.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='50th Percentile')
        ax4.axhline(0.75, color='orange', linestyle='--', alpha=0.7, label='75th Percentile')
        ax4.set_xlabel('Listing Gains (%)')
        ax4.set_ylabel('Cumulative Probability')
        ax4.set_title('Cumulative Distribution Function')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_correlations(self, df):
        """
        Analyze correlations with listing gains
        """
        print("\nCORRELATION ANALYSIS")
        print("="*50)
        
        # Select numeric columns for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_with_gains = df[numeric_cols].corr()['listing_gains'].sort_values(key=abs, ascending=False)
        
        print("Correlation with Listing Gains:")
        print("-" * 40)
        for feature, corr in correlation_with_gains.items():
            if feature != 'listing_gains':
                strength = self._interpret_correlation(abs(corr))
                direction = "Positive" if corr > 0 else "Negative"
                print(f"{feature:<25}: {corr:7.3f} ({direction}, {strength})")
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        
        # Select top correlated features for better visualization
        top_features = correlation_with_gains.head(15).index.tolist()
        if 'listing_gains' not in top_features:
            top_features.append('listing_gains')
            
        correlation_matrix = df[top_features].corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix))
        
        # Generate heatmap
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdYlBu_r', 
                   center=0,
                   square=True,
                   fmt='.3f',
                   cbar_kws={"shrink": .8})
        
        plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
        
        return correlation_with_gains
    
    def _interpret_correlation(self, corr_value):
        """Helper method to interpret correlation strength"""
        if corr_value < 0.1:
            return "Very Weak"
        elif corr_value < 0.3:
            return "Weak"
        elif corr_value < 0.5:
            return "Moderate"
        elif corr_value < 0.7:
            return "Strong"
        else:
            return "Very Strong"
    
    def analyze_key_relationships(self, df):
        """
        Analyze key relationships affecting listing gains
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Key Relationships with Listing Gains', fontsize=16, fontweight='bold')
        
        # 1. Issue Size vs Listing Gains
        if 'issue_size_cr' in df.columns:
            ax = axes[0, 0]
            ax.scatter(df['issue_size_cr'], df['listing_gains'], alpha=0.6, color='blue')
            
            # Add trend line
            z = np.polyfit(np.log(df['issue_size_cr'].dropna()), 
                          df['listing_gains'][df['issue_size_cr'].notna()], 1)
            p = np.poly1d(z)
            x_trend = np.logspace(np.log10(df['issue_size_cr'].min()), 
                                 np.log10(df['issue_size_cr'].max()), 100)
            ax.plot(x_trend, p(np.log(x_trend)), "r--", alpha=0.8, linewidth=2)
            
            correlation = df['issue_size_cr'].corr(df['listing_gains'])
            ax.set_xlabel('Issue Size (₹ Crores)')
            ax.set_ylabel('Listing Gains (%)')
            ax.set_title(f'Issue Size vs Gains\nCorr: {correlation:.3f}')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
        
        # 2. Total Subscription vs Listing Gains
        if 'total_subscription' in df.columns:
            ax = axes[0, 1]
            # Cap subscription at 50x for better visualization
            subscription_capped = df['total_subscription'].clip(upper=50)
            ax.scatter(subscription_capped, df['listing_gains'], alpha=0.6, color='green')
            
            correlation = df['total_subscription'].corr(df['listing_gains'])
            ax.set_xlabel('Total Subscription (times)')
            ax.set_ylabel('Listing Gains (%)')
            ax.set_title(f'Subscription vs Gains\nCorr: {correlation:.3f}')
            ax.grid(True, alpha=0.3)
        
        # 3. PE Ratio vs Listing Gains
        if 'pe_ratio' in df.columns:
            ax = axes[0, 2]
            # Filter extreme PE ratios for better visualization
            valid_pe = df['pe_ratio'].between(0, 100)
            ax.scatter(df.loc[valid_pe, 'pe_ratio'], 
                      df.loc[valid_pe, 'listing_gains'], alpha=0.6, color='orange')
            
            correlation = df.loc[valid_pe, 'pe_ratio'].corr(df.loc[valid_pe, 'listing_gains'])
            ax.set_xlabel('P/E Ratio')
            ax.set_ylabel('Listing Gains (%)')
            ax.set_title(f'P/E Ratio vs Gains\nCorr: {correlation:.3f}')
            ax.grid(True, alpha=0.3)
        
        # 4. Sector Analysis
        if 'sector' in df.columns:
            ax = axes[1, 0]
            sector_stats = df.groupby('sector').agg({
                'listing_gains': ['mean', 'count', 'std']
            }).round(2)
            sector_stats.columns = ['Mean_Gain', 'Count', 'Std_Dev']
            sector_stats = sector_stats.sort_values('Mean_Gain', ascending=True)
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(sector_stats)))
            bars = ax.barh(range(len(sector_stats)), sector_stats['Mean_Gain'], color=colors)
            ax.set_yticks(range(len(sector_stats)))
            ax.set_yticklabels(sector_stats.index)
            ax.set_xlabel('Average Listing Gains (%)')
            ax.set_title('Average Gains by Sector')
            ax.grid(True, alpha=0.3)
            
            # Add count labels
            for i, (bar, count) in enumerate(zip(bars, sector_stats['Count'])):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                       f'n={count}', va='center', fontsize=8)
        
        # 5. Year-wise Analysis
        if 'listing_year' in df.columns:
            ax = axes[1, 1]
            yearly_stats = df.groupby('listing_year').agg({
                'listing_gains': ['mean', 'count', 'std']
            })
            yearly_stats.columns = ['Mean_Gain', 'Count', 'Std_Dev']
            
            ax.errorbar(yearly_stats.index, yearly_stats['Mean_Gain'], 
                       yerr=yearly_stats['Std_Dev'], marker='o', capsize=5, 
                       capthick=2, linewidth=2, markersize=8)
            ax.set_xlabel('Year')
            ax.set_ylabel('Average Listing Gains (%)')
            ax.set_title('Yearly Performance Trend')
            ax.grid(True, alpha=0.3)
            
            # Add count labels
            for year, stats in yearly_stats.iterrows():
                ax.annotate(f"n={stats['Count']}", 
                           (year, stats['Mean_Gain']), 
                           textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=8)
        
        # 6. Subscription Category Analysis
        if 'subscription_category' in df.columns:
            ax = axes[1, 2]
            sub_stats = df.groupby('subscription_category').agg({
                'listing_gains': ['mean', 'count', 'std']
            })
            sub_stats.columns = ['Mean_Gain', 'Count', 'Std_Dev']
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            ax.bar(range(len(sub_stats)), sub_stats['Mean_Gain'], 
                  color=colors[:len(sub_stats)], alpha=0.8)
            ax.set_xticks(range(len(sub_stats)))
            ax.set_xticklabels(sub_stats.index, rotation=45, ha='right')
            ax.set_ylabel('Average Listing Gains (%)')
            ax.set_title('Gains by Subscription Level')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_insights(self, df):
        """
        Generate key insights from the analysis
        """
        print("\n" + "="*60)
        print("KEY INSIGHTS AND FINDINGS")
        print("="*60)
        
        insights = []
        
        # Issue size insight
        if 'issue_size_cr' in df.columns:
            size_corr = df['issue_size_cr'].corr(df['listing_gains'])
            if size_corr < -0.1:
                insights.append(f"• NEGATIVE CORRELATION: Larger IPOs tend to have lower listing gains (r={size_corr:.3f})")
                insights.append("  This suggests investors are more cautious with large offerings")
            elif size_corr > 0.1:
                insights.append(f"• POSITIVE CORRELATION: Larger IPOs tend to have higher listing gains (r={size_corr:.3f})")
            else:
                insights.append(f"• WEAK CORRELATION: Issue size has minimal impact on listing gains (r={size_corr:.3f})")
        
        # Subscription insight
        if 'total_subscription' in df.columns:
            sub_corr = df['total_subscription'].corr(df['listing_gains'])
            if sub_corr > 0.2:
                insights.append(f"• STRONG DEMAND EFFECT: Higher subscription leads to better listing performance (r={sub_corr:.3f})")
            elif sub_corr > 0.1:
                insights.append(f"• MODERATE DEMAND EFFECT: Subscription positively impacts listing gains (r={sub_corr:.3f})")
            else:
                insights.append(f"• LIMITED DEMAND EFFECT: Subscription has weak correlation with gains (r={sub_corr:.3f})")
        
        # Performance statistics
        gains = df['listing_gains']
        positive_pct = (gains > 0).mean() * 100
        high_gains_pct = (gains > 50).mean() * 100
        
        insights.append(f"• OVERALL SUCCESS RATE: {positive_pct:.1f}% of IPOs had positive listing gains")
        insights.append(f"• HIGH PERFORMANCE RATE: {high_gains_pct:.1f}% of IPOs gained more than 50%")
        
        # Sector insights
        if 'sector' in df.columns:
            sector_performance = df.groupby('sector')['listing_gains'].mean().sort_values(ascending=False)
            best_sector = sector_performance.index[0]
            worst_sector = sector_performance.index[-1]
            insights.append(f"• BEST SECTOR: {best_sector} with average gains of {sector_performance.iloc[0]:.1f}%")
            insights.append(f"• CHALLENGING SECTOR: {worst_sector} with average gains of {sector_performance.iloc[-1]:.1f}%")
        
        # Timing insights
        if 'listing_year' in df.columns:
            yearly_performance = df.groupby('listing_year')['listing_gains'].mean()
            best_year = yearly_performance.idxmax()
            worst_year = yearly_performance.idxmin()
            insights.append(f"• BEST YEAR: {best_year} with average gains of {yearly_performance.loc[best_year]:.1f}%")
            insights.append(f"• TOUGHEST YEAR: {worst_year} with average gains of {yearly_performance.loc[worst_year]:.1f}%")
        
        # Risk insights
        volatility = gains.std()
        if volatility > 50:
            insights.append(f"• HIGH VOLATILITY: IPO gains are highly variable (σ={volatility:.1f}%)")
            insights.append("  This indicates significant risk in IPO investments")
        elif volatility > 30:
            insights.append(f"• MODERATE VOLATILITY: IPO gains show moderate variation (σ={volatility:.1f}%)")
        else:
            insights.append(f"• LOW VOLATILITY: IPO gains are relatively stable (σ={volatility:.1f}%)")
        
        # Print insights
        for insight in insights:
            print(insight)
        
        return insights
    
    def create_comprehensive_report(self, df):
        """
        Create a comprehensive EDA report
        """
        print("\n" + "="*60)
        print("COMPREHENSIVE IPO ANALYSIS REPORT")
        print("="*60)
        
        # Generate all analyses
        summary_stats = self.generate_summary_statistics(df)
        self.plot_distribution_analysis(df)
        correlations = self.analyze_correlations(df)
        self.analyze_key_relationships(df)
        insights = self.generate_insights(df)
        
        # Create summary report
        report = {
            'dataset_info': {
                'total_ipos': len(df),
                'date_range': f"{df['listing_date'].min().strftime('%Y-%m-%d')} to {df['listing_date'].max().strftime('%Y-%m-%d')}" if 'listing_date' in df.columns else 'N/A',
                'features': len(df.columns)
            },
            'performance_metrics': summary_stats,
            'key_correlations': correlations.head(10).to_dict(),
            'insights': insights
        }
        
        return report

# Example usage
if __name__ == "__main__":
    analyzer = IPOExploratoryAnalyzer()
    
    # Example with sample data
    print("EDA Analyzer ready to use!")
    print("Usage:")
    print("analyzer = IPOExploratoryAnalyzer()")
    print("report = analyzer.create_comprehensive_report(your_dataframe)")
