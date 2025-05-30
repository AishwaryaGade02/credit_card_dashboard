from pyspark.sql.functions import *
from pyspark.sql.types import *
from datetime import datetime, timedelta
import calendar

class KPICalculator:
    def __init__(self, dataframes):
        self.cards_df = dataframes['cards']
        self.transactions_df = dataframes['transactions']
        self.users_df = dataframes['users']
        self.mcc_df = dataframes['mcc_codes']
        self.fraud_df = dataframes['fraud_labels']
    
    def get_user_profile(self, user_id):
        """Get user profile information"""
        user_profile = self.users_df.filter(col("id") == user_id).collect()
        
        if not user_profile:
            return None
        
        user = user_profile[0]
        
        # Get user's cards
        user_cards = self.cards_df.filter(col("client_id") == user_id).collect()
        
        profile = {
            'user_id': user['id'],
            'age': user['current_age'],
            'gender': user['gender'],
            'address': user['address'],
            'yearly_income': user['yearly_income'],
            'total_debt': user['total_debt'],
            'credit_score': user['credit_score'],
            'num_credit_cards': user['num_credit_cards'],
            'cards': []
        }
        
        for card in user_cards:
            profile['cards'].append({
                'card_id': card['id'],
                'card_brand': card['card_brand'],
                'card_type': card['card_type'],
                'credit_limit': card['credit_limit'],
                'expires': card['expires']
            })
        
        return profile
    
    def calculate_monthly_spending(self, user_id, months_back=1):
        """Calculate total monthly spending for the user"""
        # Get current date (using latest transaction date as reference)
        latest_date = self.transactions_df.agg(max("date")).collect()[0][0]
        
        if latest_date is None:
            return 0.0
        
        # Calculate date range for the specified months back
        start_date = latest_date - timedelta(days=30 * months_back)
        
        monthly_spending = self.transactions_df.filter(
            (col("client_id") == user_id) &
            (col("date") >= start_date) &
            (col("date") <= latest_date) &
            (col("amount") > 0)  # Only positive amounts (spending)
        ).agg(sum("amount")).collect()[0][0]
        
        return monthly_spending if monthly_spending else 0.0
    
    def calculate_credit_utilization(self, user_id):
        """Calculate average credit utilization percentage"""
        # Get user's cards and their credit limits
        user_cards = self.cards_df.filter(col("client_id") == user_id)
        total_credit_limit = user_cards.agg(sum("credit_limit")).collect()[0][0]
        
        if not total_credit_limit:
            return 0.0
        
        # Get current month spending (as proxy for current balance)
        current_spending = self.calculate_monthly_spending(user_id, 1)
        
        # Calculate utilization percentage
        utilization = (current_spending / total_credit_limit) * 100
        return min(utilization, 100.0)  # Cap at 100%
    
    def calculate_interest_paid(self, user_id, months_back=12):
        """Estimate interest paid based on debt and typical credit card APR"""
        user_data = self.users_df.filter(col("id") == user_id).collect()
        
        if not user_data:
            return 0.0
        
        total_debt = user_data[0]['total_debt']
        if not total_debt:
            return 0.0
        
        # Estimate annual interest (assuming 18% APR as average)
        estimated_annual_interest = total_debt * 0.18
        monthly_interest = estimated_annual_interest / 12
        
        return monthly_interest * months_back
    
    def calculate_rewards_earned(self, user_id, months_back=12):
        """Calculate estimated rewards earned (assuming 1% cashback)"""
        # Get spending for the specified period
        total_spending = 0
        for month in range(months_back):
            monthly_spend = self.calculate_monthly_spending(user_id, month + 1)
            total_spending += monthly_spend
        
        # Assume 1% cashback rate
        rewards = total_spending * 0.01
        return rewards
    
    def get_spending_by_category(self, user_id, months_back=3):
        """Get spending breakdown by merchant category"""
        latest_date = self.transactions_df.agg(max("date")).collect()[0][0]
        if latest_date is None:
            return []
        
        start_date = latest_date - timedelta(days=30 * months_back)
        
        # Join transactions with MCC codes
        spending_by_category = self.transactions_df.filter(
            (col("client_id") == user_id) &
            (col("date") >= start_date) &
            (col("amount") > 0)
        ).join(
            self.mcc_df, 
            self.transactions_df.mcc == self.mcc_df.mcc, 
            "left"
        ).groupBy("category").agg(
            sum("amount").alias("total_spent"),
            count("*").alias("transaction_count")
        ).orderBy(desc("total_spent"))
        
        categories = spending_by_category.collect()
        
        result = []
        for category in categories:
            result.append({
                'category': category['category'] if category['category'] else 'Other',
                'amount': category['total_spent'],
                'transactions': category['transaction_count']
            })
        
        return result
    
    def get_fraud_risk_score(self, user_id):
        """Calculate fraud risk score based on transaction patterns"""
        # Get user's recent transactions
        user_transactions = self.transactions_df.filter(col("client_id") == user_id)
        
        # Join with fraud labels where available
        fraud_analysis = user_transactions.join(
            self.fraud_df,
            user_transactions.id == self.fraud_df.transaction_id,
            "left"
        )
        
        total_transactions = fraud_analysis.count()
        if total_transactions == 0:
            return 0.0
        
        fraud_transactions = fraud_analysis.filter(col("is_fraud") == True).count()
        
        # Calculate fraud percentage
        fraud_percentage = (fraud_transactions / total_transactions) * 100
        return fraud_percentage
    
    def get_monthly_trend(self, user_id, months=6):
        """Get monthly spending trend"""
        latest_date = self.transactions_df.agg(max("date")).collect()[0][0]
        if latest_date is None:
            return []
        
        trends = []
        for i in range(months):
            month_start = latest_date - timedelta(days=30 * (i + 1))
            month_end = latest_date - timedelta(days=30 * i)
            
            monthly_spend = self.transactions_df.filter(
                (col("client_id") == user_id) &
                (col("date") >= month_start) &
                (col("date") < month_end) &
                (col("amount") > 0)
            ).agg(sum("amount")).collect()[0][0]
            
            month_name = calendar.month_name[month_start.month]
            trends.append({
                'month': f"{month_name} {month_start.year}",
                'spending': monthly_spend if monthly_spend else 0.0
            })
        
        return list(reversed(trends))
    
    def get_all_kpis(self, user_id):
        """Get all KPIs for a user"""
        return {
            'total_monthly_spending': self.calculate_monthly_spending(user_id),
            'credit_utilization': self.calculate_credit_utilization(user_id),
            'interest_paid': self.calculate_interest_paid(user_id),
            'rewards_earned': self.calculate_rewards_earned(user_id),
            'credit_score': self.get_user_profile(user_id)['credit_score'] if self.get_user_profile(user_id) else 0,
            'spending_by_category': self.get_spending_by_category(user_id),
            'fraud_risk_score': self.get_fraud_risk_score(user_id),
            'monthly_trend': self.get_monthly_trend(user_id)
        }