import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

class MerchantAnalyzer:
    """Enhanced merchant analysis with pattern detection and insights"""
    
    def __init__(self, transactions_df, mcc_df):
        self.transactions_df = transactions_df
        self.mcc_df = mcc_df
        
        # Debug: Print info about dataframes
        if not self.mcc_df.empty:
            print(f"MerchantAnalyzer - MCC DataFrame columns: {self.mcc_df.columns.tolist()}")
            print(f"MerchantAnalyzer - MCC DataFrame shape: {self.mcc_df.shape}")
        
        # Common merchant patterns for better identification
        self.merchant_patterns = {
            'grocery': ['walmart', 'target', 'kroger', 'safeway', 'grocery', 'market', 'food'],
            'gas': ['shell', 'exxon', 'bp', 'chevron', 'mobil', 'gas', 'fuel', 'station'],
            'restaurant': ['mcdonalds', 'starbucks', 'pizza', 'restaurant', 'cafe', 'diner'],
            'retail': ['amazon', 'target', 'walmart', 'costco', 'home depot', 'lowes'],
            'subscription': ['netflix', 'spotify', 'apple', 'google', 'microsoft', 'adobe']
        }
    
    def get_merchant_insights(self, user_id, time_period='3_months'):
        """Get comprehensive merchant insights"""
        if self.transactions_df.empty:
            return {}
        
        user_transactions = self.transactions_df[self.transactions_df['client_id'] == user_id].copy()
        
        if user_transactions.empty:
            return {}
        
        # Filter by time period
        user_transactions['date'] = pd.to_datetime(user_transactions['date'])
        latest_date = user_transactions['date'].max()
        
        if time_period == '1_month':
            start_date = latest_date - pd.Timedelta(days=30)
        elif time_period == '3_months':
            start_date = latest_date - pd.Timedelta(days=90)
        elif time_period == '6_months':
            start_date = latest_date - pd.Timedelta(days=180)
        elif time_period == 'Last Year':
            start_date = latest_date - pd.Timedelta(days=365)
        else:
            start_date = latest_date - pd.Timedelta(days=90)
        
        filtered_transactions = user_transactions[
            (user_transactions['date'] >= start_date) &
            (user_transactions['amount'] > 0)
        ]
        
        if filtered_transactions.empty:
            return {}
        
        # Basic merchant analysis
        merchant_stats = filtered_transactions.groupby('merchant_id').agg({
            'amount': ['sum', 'count', 'mean', 'std'],
            'date': ['min', 'max'],
            'mcc': 'first'
        }).round(2)
        
        merchant_stats.columns = ['total_spent', 'transaction_count', 'avg_amount', 'std_amount', 
                                'first_transaction', 'last_transaction', 'mcc']
        merchant_stats = merchant_stats.reset_index()
        
        # Add merchant categories if MCC data is available - FIXED VERSION
        if not self.mcc_df.empty and 'mcc' in merchant_stats.columns:
            # Ensure mcc columns are same type for joining
            merchant_stats['mcc'] = merchant_stats['mcc'].astype(int)
            mcc_df_copy = self.mcc_df.copy()
            mcc_df_copy['mcc'] = mcc_df_copy['mcc'].astype(int)
            
            merchant_stats = merchant_stats.merge(
                mcc_df_copy[['mcc', 'category']], 
                on='mcc', 
                how='left'
            )
            merchant_stats['category'] = merchant_stats['category'].fillna('Other')
        else:
            merchant_stats['category'] = 'Other'
        
        # Calculate additional metrics
        merchant_stats['consistency_score'] = self._calculate_consistency_score(merchant_stats)
        merchant_stats['frequency_score'] = self._calculate_frequency_score(merchant_stats, start_date, latest_date)
        merchant_stats['merchant_name'] = merchant_stats['merchant_id'].apply(
            lambda x: self._generate_merchant_name(x, merchant_stats)
        )
        
        # Sort by total spending
        merchant_stats = merchant_stats.sort_values('total_spent', ascending=False)
        
        return {
            'merchant_stats': merchant_stats,
            'total_merchants': len(merchant_stats),
            'top_merchant_concentration': self._calculate_concentration(merchant_stats),
            'subscription_merchants': self._identify_subscriptions(merchant_stats,time_period),
            'loyalty_opportunities': self._identify_loyalty_opportunities(merchant_stats),
            'spending_volatility': self._calculate_spending_volatility(filtered_transactions)
        }
    
    def _calculate_consistency_score(self, merchant_stats):
        """Calculate how consistent spending is at each merchant (lower std = more consistent)"""
        # Handle NaN and zero values in std_amount
        merchant_stats['std_amount'] = merchant_stats['std_amount'].fillna(0)
        
        # Normalize standard deviation by mean to get coefficient of variation
        # Avoid division by zero
        cv = np.where(
            merchant_stats['avg_amount'] > 0,
            merchant_stats['std_amount'] / merchant_stats['avg_amount'],
            0
        )
        
        # Convert to consistency score (higher = more consistent)
        consistency = 1 / (1 + cv)
        return pd.Series(consistency).round(3)
    
    def _calculate_frequency_score(self, merchant_stats, start_date, end_date):
        """Calculate how frequently user visits each merchant"""
        period_days = (end_date - start_date).days
        if period_days <= 0:
            return pd.Series([0] * len(merchant_stats))
        
        frequency = merchant_stats['transaction_count'] / period_days * 30  # Transactions per month
        return frequency.round(2)
    
    def _generate_merchant_name(self, merchant_id, merchant_stats):
        """Generate more readable merchant names"""
        # Get the category for context if available
        merchant_row = merchant_stats[merchant_stats['merchant_id'] == merchant_id]
        if not merchant_row.empty and 'category' in merchant_row.columns:
            category = merchant_row['category'].iloc[0]
        else:
            category = 'Unknown'
        
        # Simple mapping - in real scenario, you'd have a merchant lookup table
        merchant_names = {
            1: "Walmart Supercenter",
            2: "Shell Gas Station", 
            3: "Starbucks Coffee",
            4: "Amazon.com",
            5: "McDonald's",
            6: "Target Store",
            7: "Kroger Grocery",
            8: "Home Depot",
            9: "Netflix Subscription",
            10: "Uber Rides"
        }
        
        return merchant_names.get(merchant_id, f"{category} Merchant {merchant_id}")
    
    def _calculate_concentration(self, merchant_stats):
        """Calculate how concentrated spending is among top merchants"""
        total_spending = merchant_stats['total_spent'].sum()
        if total_spending == 0:
            return {}
        
        top_3_spending = merchant_stats.head(3)['total_spent'].sum()
        top_5_spending = merchant_stats.head(5)['total_spent'].sum()
        top_10_spending = merchant_stats.head(10)['total_spent'].sum()
        
        return {
            'top_3_percentage': (top_3_spending / total_spending * 100).round(1),
            'top_5_percentage': (top_5_spending / total_spending * 100).round(1),
            'top_10_percentage': (top_10_spending / total_spending * 100).round(1)
        }
    
    def _identify_subscriptions(self, merchant_stats, time_period):
        """Identify potential subscription merchants based on regularity"""
        
        # UPDATED: Handle 1-month period limitation
        if time_period == '1_month':
            # Return a special message indicating insufficient data
            return {
                'insufficient_data': True,
                'message': 'Select other time frame to know about your subscriptions',
                'recommendation': 'Please select 3 months or longer period for accurate subscription detection.',
                'data': pd.DataFrame()  # Empty DataFrame
            }
        
        # Original logic for periods longer than 1 month
        subscriptions = merchant_stats[
            (merchant_stats['consistency_score'] > 0.8) &  # Very consistent amounts
            (merchant_stats['frequency_score'] >= 0.8) &   # At least monthly
            (merchant_stats['transaction_count'] >= 2)      # Multiple transactions
        ].copy()
        
        if not subscriptions.empty:
            # Estimate monthly cost
            subscriptions['estimated_monthly_cost'] = (
                subscriptions['total_spent'] / subscriptions['transaction_count']
            ).round(2)
            
            subscriptions = subscriptions.sort_values('estimated_monthly_cost', ascending=False)
        
        return subscriptions
        
    def _identify_loyalty_opportunities(self, merchant_stats):
        """Identify merchants where loyalty programs might be beneficial"""
        if merchant_stats.empty:
            return pd.DataFrame()
        
        # High spending, frequent visits = good loyalty candidates
        spending_threshold = merchant_stats['total_spent'].quantile(0.7) if len(merchant_stats) > 0 else 0
        frequency_threshold = merchant_stats['frequency_score'].quantile(0.7) if len(merchant_stats) > 0 else 0
        
        loyalty_candidates = merchant_stats[
            (merchant_stats['total_spent'] >= spending_threshold) |
            (merchant_stats['frequency_score'] >= frequency_threshold)
        ].copy()
        
        if not loyalty_candidates.empty:
            loyalty_candidates['loyalty_score'] = (
                loyalty_candidates['total_spent'] * 0.6 + 
                loyalty_candidates['frequency_score'] * 20 + 
                loyalty_candidates['transaction_count'] * 2
            ).round(2)
            
            loyalty_candidates = loyalty_candidates.sort_values('loyalty_score', ascending=False)
        
        return loyalty_candidates
    
    def _calculate_spending_volatility(self, transactions):
        """Calculate overall spending volatility"""
        if transactions.empty:
            return {}
        
        # Daily spending aggregation
        daily_spending = transactions.groupby(transactions['date'].dt.date)['amount'].sum()
        
        if daily_spending.empty or len(daily_spending) < 2:
            return {
                'avg_daily_spending': 0,
                'std_daily_spending': 0,
                'max_daily_spending': 0,
                'min_daily_spending': 0,
                'volatility_coefficient': 0
            }
        
        return {
            'avg_daily_spending': daily_spending.mean().round(2),
            'std_daily_spending': daily_spending.std().round(2),
            'max_daily_spending': daily_spending.max().round(2),
            'min_daily_spending': daily_spending.min().round(2),
            'volatility_coefficient': (daily_spending.std() / daily_spending.mean()).round(3) if daily_spending.mean() > 0 else 0
        }
    
    def get_merchant_recommendations(self, user_id, user_cards_df, time_period='3_months'):
        """Get personalized merchant recommendations"""
        insights = self.get_merchant_insights(user_id, time_period)
        
        if not insights or insights.get('merchant_stats') is None:
            return {}
        
        merchant_stats = insights['merchant_stats']
        recommendations = []
        
        # Recommendation 1: Subscription optimization
        subscriptions = insights.get('subscription_merchants', pd.DataFrame())
        if not subscriptions.empty:
            total_subscription_cost = subscriptions['estimated_monthly_cost'].sum()
            recommendations.append({
                'type': 'Subscription Optimization',
                'title': f'Review {len(subscriptions)} Subscription Services',
                'description': f'You spend ~${total_subscription_cost:.2f}/month on subscriptions. Consider reviewing for unused services.',
                'potential_savings': total_subscription_cost * 0.2,  # Assume 20% potential savings
                'priority': 'High' if total_subscription_cost > 100 else 'Medium'
            })
        
        # Recommendation 2: Loyalty programs
        loyalty_opportunities = insights.get('loyalty_opportunities', pd.DataFrame())
        if not loyalty_opportunities.empty:
            top_loyalty = loyalty_opportunities.head(3)
            potential_rewards = top_loyalty['total_spent'].sum() * 0.02  # Assume 2% rewards
            recommendations.append({
                'type': 'Loyalty Programs',
                'title': f'Join loyalty programs at top {len(top_loyalty)} merchants',
                'description': f'Potential to earn ${potential_rewards:.2f} in rewards/cashback',
                'merchants': top_loyalty['merchant_name'].tolist(),
                'priority': 'Medium'
            })