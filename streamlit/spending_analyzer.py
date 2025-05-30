import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import streamlit as st

class SpendingAnalyzer:
    def __init__(self, dataframes):
        self.cards_df = dataframes.get('cards', pd.DataFrame())
        self.transactions_df = dataframes.get('transactions', pd.DataFrame())
        self.users_df = dataframes.get('users', pd.DataFrame())
        self.mcc_df = dataframes.get('mcc_codes', pd.DataFrame())
        self.fraud_df = dataframes.get('fraud_labels', pd.DataFrame())
        
        # Debug: Print MCC dataframe info
        if not self.mcc_df.empty:
            print(f"MCC DataFrame columns: {self.mcc_df.columns.tolist()}")
            print(f"MCC DataFrame shape: {self.mcc_df.shape}")
            print(f"Sample MCC data:\n{self.mcc_df.head()}")
        
        # Card rewards mapping (example data - you can expand this)
        self.card_rewards = {
            'Visa': {
                'Eating Places and Restaurants': 0.02,  # 2% cashback
                'Service Stations': 0.02,
                'Grocery Stores, Supermarkets': 0.015,
                'default': 0.01
            },
            'Mastercard': {
                'Eating Places and Restaurants': 0.03,  # 3% cashback
                'Travel': 0.02,
                'Amusement Parks, Carnivals, Circuses': 0.02,
                'default': 0.01
            },
            'American Express': {
                'Travel': 0.03,
                'Amusement Parks, Carnivals, Circuses': 0.02,
                'Grocery Stores, Supermarkets': 0.015,
                'default': 0.01
            },
            'Discover': {
                'Service Stations': 0.05,  # 5% rotating categories
                'Eating Places and Restaurants': 0.02,
                'Book Stores': 0.02,
                'default': 0.01
            }
        }
    
    def get_filtered_transactions(self, user_id, time_period='3_months', start_date=None, end_date=None):
        """Get filtered transactions based on time period"""
        if self.transactions_df.empty:
            return pd.DataFrame()
        
        user_transactions = self.transactions_df[self.transactions_df['client_id'] == user_id].copy()
        
        if user_transactions.empty:
            return pd.DataFrame()
        
        # Convert date column to datetime
        user_transactions['date'] = pd.to_datetime(user_transactions['date'])
        
        # Filter by time period
        if start_date and end_date:
            # Custom date range
            user_transactions = user_transactions[
                (user_transactions['date'] >= pd.to_datetime(start_date)) &
                (user_transactions['date'] <= pd.to_datetime(end_date))
            ]
        else:
            # Predefined periods
            latest_date = user_transactions['date'].max()
            
            if time_period == '1_month':
                start_date = latest_date - pd.Timedelta(days=30)
            elif time_period == '3_months':
                start_date = latest_date - pd.Timedelta(days=90)
            elif time_period == '6_months':
                start_date = latest_date - pd.Timedelta(days=180)
            elif time_period == '1_year':
                start_date = latest_date - pd.Timedelta(days=365)
            else:
                start_date = latest_date - pd.Timedelta(days=90)  # Default to 3 months
            
            user_transactions = user_transactions[user_transactions['date'] >= start_date]
        
        # Only positive amounts (spending)
        user_transactions = user_transactions[user_transactions['amount'] > 0]
        
        return user_transactions
    
    def get_category_breakdown(self, user_id, time_period='3_months', start_date=None, end_date=None):
        """Get spending breakdown by category with proper MCC handling"""
        transactions = self.get_filtered_transactions(user_id, time_period, start_date, end_date)
        
        if transactions.empty:
            return pd.DataFrame()
        
        # Join with MCC codes for categories - FIXED VERSION
        if not self.mcc_df.empty and 'mcc' in transactions.columns:
            # Ensure both mcc columns are the same type
            transactions['mcc'] = transactions['mcc'].astype(int)
            mcc_df_copy = self.mcc_df.copy()
            mcc_df_copy['mcc'] = mcc_df_copy['mcc'].astype(int)
            
            # Perform the join
            transactions_with_categories = transactions.merge(
                mcc_df_copy[['mcc', 'category']], 
                on='mcc', 
                how='left'
            )
            
            # Fill missing categories
            transactions_with_categories['category'] = transactions_with_categories['category'].fillna('Other')
            
            print(f"Categories found: {transactions_with_categories['category'].unique()}")
            
        else:
            # Fallback: create categories based on MCC codes if no MCC data
            transactions_with_categories = transactions.copy()
            transactions_with_categories['category'] = 'Other'
            
            # Add some basic MCC-to-category mapping as fallback
            mcc_mapping = {
                5812: 'Eating Places and Restaurants',
                5814: 'Eating Places and Restaurants',
                5541: 'Service Stations',
                5411: 'Grocery Stores, Supermarkets',
                4900: 'Utilities',
                5942: 'Book Stores',
                7996: 'Amusement Parks, Carnivals, Circuses'
            }
            
            if 'mcc' in transactions_with_categories.columns:
                transactions_with_categories['category'] = transactions_with_categories['mcc'].map(mcc_mapping).fillna('Other')
        
        # Group by category
        category_breakdown = transactions_with_categories.groupby('category').agg({
            'amount': ['sum', 'count', 'mean'],
        }).round(2)
        
        # Flatten column names
        category_breakdown.columns = ['total_spent', 'transaction_count', 'avg_transaction']
        category_breakdown = category_breakdown.reset_index()
        
        # Calculate percentage
        total_spending = category_breakdown['total_spent'].sum()
        if total_spending > 0:
            category_breakdown['percentage'] = (category_breakdown['total_spent'] / total_spending * 100).round(1)
        else:
            category_breakdown['percentage'] = 0
        
        # Sort by spending
        category_breakdown = category_breakdown.sort_values('total_spent', ascending=False)
        
        return category_breakdown
    
    def get_top_merchants(self, user_id, time_period='3_months', start_date=None, end_date=None, top_n=10):
        """Get top merchants by spending"""
        transactions = self.get_filtered_transactions(user_id, time_period, start_date, end_date)
        
        if transactions.empty:
            return pd.DataFrame()
        
        # Group by merchant_id and merchant name (if available)
        # Note: You might need to join with a merchants table if you have merchant names
        merchant_breakdown = transactions.groupby('merchant_id').agg({
            'amount': ['sum', 'count', 'mean'],
            'date': ['min', 'max']
        }).round(2)
        
        # Flatten column names
        merchant_breakdown.columns = ['total_spent', 'transaction_count', 'avg_transaction', 'first_transaction', 'last_transaction']
        merchant_breakdown = merchant_breakdown.reset_index()
        
        # Sort by spending and get top N
        merchant_breakdown = merchant_breakdown.sort_values('total_spent', ascending=False).head(top_n)
        
        # Add merchant names (placeholder - you can join with actual merchant data)
        merchant_breakdown['merchant_name'] = merchant_breakdown['merchant_id'].apply(
            lambda x: f"Merchant {x}"
        )
        
        return merchant_breakdown
    
    def get_spending_trends(self, user_id, time_period='1_year'):
        """Get spending trends over time with proper category handling"""
        transactions = self.get_filtered_transactions(user_id, time_period)
        
        if transactions.empty:
            return pd.DataFrame()
        
        # Add categories using the same logic as get_category_breakdown
        if not self.mcc_df.empty and 'mcc' in transactions.columns:
            transactions['mcc'] = transactions['mcc'].astype(int)
            mcc_df_copy = self.mcc_df.copy()
            mcc_df_copy['mcc'] = mcc_df_copy['mcc'].astype(int)
            
            transactions = transactions.merge(
                mcc_df_copy[['mcc', 'category']], 
                on='mcc', 
                how='left'
            )
            transactions['category'] = transactions['category'].fillna('Other')
        else:
            transactions['category'] = 'Other'
        
        # Create month-year grouping
        transactions['month_year'] = transactions['date'].dt.to_period('M')
        
        # Group by month and category
        monthly_trends = transactions.groupby(['month_year', 'category']).agg({
            'amount': 'sum'
        }).reset_index()
        
        monthly_trends['month_year_str'] = monthly_trends['month_year'].astype(str)
        
        return monthly_trends
    
    def calculate_potential_rewards(self, user_id, time_period='3_months', start_date=None, end_date=None):
        """Calculate potential rewards optimization"""
        transactions = self.get_filtered_transactions(user_id, time_period, start_date, end_date)
        
        if transactions.empty or self.cards_df.empty:
            return pd.DataFrame()
        
        # Get user's cards
        user_cards = self.cards_df[self.cards_df['client_id'] == user_id]
        
        if user_cards.empty:
            return pd.DataFrame()
        
        # Join transactions with categories using same logic
        if not self.mcc_df.empty and 'mcc' in transactions.columns:
            transactions['mcc'] = transactions['mcc'].astype(int)
            mcc_df_copy = self.mcc_df.copy()
            mcc_df_copy['mcc'] = mcc_df_copy['mcc'].astype(int)
            
            transactions_with_categories = transactions.merge(
                mcc_df_copy[['mcc', 'category']], 
                on='mcc', 
                how='left'
            )
            transactions_with_categories['category'] = transactions_with_categories['category'].fillna('Other')
        else:
            transactions_with_categories = transactions.copy()
            transactions_with_categories['category'] = 'Other'
        
        # Group by category
        category_spending = transactions_with_categories.groupby('category')['amount'].sum().reset_index()
        
        rewards_analysis = []
        
        for _, category_row in category_spending.iterrows():
            category = category_row['category']
            amount = category_row['amount']
            
            best_card = None
            best_reward_rate = 0
            current_rewards = amount * 0.01  # Assume 1% default
            
            # Find best card for this category
            for _, card in user_cards.iterrows():
                card_brand = card['card_brand']
                if card_brand in self.card_rewards:
                    reward_rate = self.card_rewards[card_brand].get(category, 
                                  self.card_rewards[card_brand]['default'])
                    
                    if reward_rate > best_reward_rate:
                        best_reward_rate = reward_rate
                        best_card = f"{card_brand} ({card['card_type']})"
            
            potential_rewards = amount * best_reward_rate
            additional_rewards = potential_rewards - current_rewards
            
            rewards_analysis.append({
                'category': category,
                'spending': amount,
                'current_rewards': current_rewards,
                'potential_rewards': potential_rewards,
                'additional_rewards': additional_rewards,
                'best_card': best_card or 'Current Card',
                'best_reward_rate': best_reward_rate * 100,
                'improvement_pct': (additional_rewards / current_rewards * 100) if current_rewards > 0 else 0
            })
        
        rewards_df = pd.DataFrame(rewards_analysis)
        rewards_df = rewards_df.sort_values('additional_rewards', ascending=False)
        
        return rewards_df
    
    def get_spending_patterns(self, user_id, time_period='3_months'):
        """Analyze spending patterns by day of week, time of day, etc."""
        transactions = self.get_filtered_transactions(user_id, time_period)
        
        if transactions.empty:
            return {}
        
        transactions['day_of_week'] = transactions['date'].dt.day_name()
        transactions['hour'] = transactions['date'].dt.hour
        transactions['day_of_month'] = transactions['date'].dt.day
        
        patterns = {
            'by_day_of_week': transactions.groupby('day_of_week')['amount'].sum().reset_index(),
            'by_hour': transactions.groupby('hour')['amount'].sum().reset_index(),
            'by_day_of_month': transactions.groupby('day_of_month')['amount'].sum().reset_index(),
            'total_transactions': len(transactions),
            'avg_transaction': transactions['amount'].mean(),
            'max_transaction': transactions['amount'].max(),
            'min_transaction': transactions['amount'].min()
        }
        
        return patterns
    
    def get_comparison_metrics(self, user_id, time_period='3_months'):
        """Get comparison metrics for the selected period vs previous period"""
        current_transactions = self.get_filtered_transactions(user_id, time_period)
        
        if current_transactions.empty:
            return {}
        
        # Calculate previous period
        # if time_period == '1_month':
        #     days_back = 60  # Current month + previous month
        # elif time_period == '3_months':
        #     days_back = 180  # Current 3 months + previous 3 months
        # elif time_period == '6_months':
        #     days_back = 365  # Current 6 months + previous 6 months
        # else:
        #     days_back = 180
        
        # Get all transactions for comparison
        if self.transactions_df.empty:
            return {}
            
        all_transactions = self.transactions_df[
            (self.transactions_df['client_id'] == user_id) &
            (self.transactions_df['amount'] > 0)
        ].copy()
        
        if all_transactions.empty:
            return {}
        
        all_transactions['date'] = pd.to_datetime(all_transactions['date'])
        latest_date = all_transactions['date'].max()
        
        # Split into current and previous periods
        # if time_period == '1_month':
        #     current_start = latest_date - pd.Timedelta(days=30)
        #     previous_start = latest_date - pd.Timedelta(days=60)
        #     previous_end = latest_date - pd.Timedelta(days=30)
        # elif time_period == '3_months':
        #     current_start = latest_date - pd.Timedelta(days=90)
        #     previous_start = latest_date - pd.Timedelta(days=180)
        #     previous_end = latest_date - pd.Timedelta(days=90)
        # else:
        #     current_start = latest_date - pd.Timedelta(days=90)
        #     previous_start = latest_date - pd.Timedelta(days=180)
        #     previous_end = latest_date - pd.Timedelta(days=90)

        days = {'1_month':30,'3_months':90,'6_months':180,'1_year':365}.get(time_period,90)
        current_start = latest_date - pd.Timedelta(days=days)
        previous_start = latest_date - pd.Timedelta(days=days * 2)
        previous_end = latest_date - pd.Timedelta(days=days)
        
        current_period = all_transactions[all_transactions['date'] >= current_start]
        previous_period = all_transactions[
            (all_transactions['date'] >= previous_start) & 
            (all_transactions['date'] < previous_end)
        ]
        
        current_total = current_period['amount'].sum() if not current_period.empty else 0
        previous_total = previous_period['amount'].sum() if not previous_period.empty else 0
        
        current_count = len(current_period)
        previous_count = len(previous_period)
        
        spending_change = ((current_total - previous_total) / previous_total * 100) if previous_total > 0 else 0
        transaction_change = ((current_count - previous_count) / previous_count * 100) if previous_count > 0 else 0
        
        return {
            'current_spending': current_total,
            'previous_spending': previous_total,
            'spending_change_pct': spending_change,
            'current_transactions': current_count,
            'previous_transactions': previous_count,
            'transaction_change_pct': transaction_change,
            'avg_transaction_current': current_total / current_count if current_count > 0 else 0,
            'avg_transaction_previous': previous_total / previous_count if previous_count > 0 else 0
        }