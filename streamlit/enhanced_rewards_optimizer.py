import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json




class RewardsOptimizer:
    """Advanced rewards optimization with real credit card data"""
    
    def __init__(self, cards_df, transactions_df, mcc_df):
        self.cards_df = cards_df
        self.transactions_df = transactions_df
        self.mcc_df = mcc_df
        
        # Debug: Print MCC dataframe info
        if not self.mcc_df.empty:
            print(f"RewardsOptimizer - MCC DataFrame columns: {self.mcc_df.columns.tolist()}")
            print(f"RewardsOptimizer - MCC DataFrame shape: {self.mcc_df.shape}")
        
        # Comprehensive credit card rewards database
        self.card_rewards_database = {
            'Chase Freedom Unlimited': {
                'annual_fee': 0,
                'categories': {
                    'Eating Places and Restaurants': 0.03,
                    'Book Stores': 0.03,
                    'default': 0.015
                },
                'signup_bonus': 200,
                'spending_requirement': 500
            },
            'Chase Sapphire Preferred': {
                'annual_fee': 95,
                'categories': {
                    'Amusement Parks, Carnivals, Circuses': 0.02,
                    'Eating Places and Restaurants': 0.02,
                    'default': 0.01
                },
                'signup_bonus': 600,
                'spending_requirement': 4000,
                'transfer_partners': True
            },
            'American Express Gold': {
                'annual_fee': 250,
                'categories': {
                    'Eating Places and Restaurants': 0.04,
                    'Grocery Stores, Supermarkets': 0.04,
                    'default': 0.01
                },
                'signup_bonus': 600,
                'spending_requirement': 4000,
                'credits': 120  # Annual dining credit
            },
            'Citi Double Cash': {
                'annual_fee': 0,
                'categories': {
                    'default': 0.02
                },
                'signup_bonus': 200,
                'spending_requirement': 1500
            },
            'Capital One Savor': {
                'annual_fee': 95,
                'categories': {
                    'Eating Places and Restaurants': 0.04,
                    'Amusement Parks, Carnivals, Circuses': 0.04,
                    'Grocery Stores, Supermarkets': 0.02,
                    'default': 0.01
                },
                'signup_bonus': 300,
                'spending_requirement': 3000
            },
            'Discover it Cash Back': {
                'annual_fee': 0,
                'categories': {
                    'Service Stations': 0.05,  # Rotating category
                    'Eating Places and Restaurants': 0.05,   # Rotating category
                    'default': 0.01
                },
                'signup_bonus': 0,
                'cashback_match': True  # First year cashback matched
            }
        }
        
        # MCC to category mapping for better analysis
        self.mcc_category_mapping = {
            5411: 'Grocery Stores, Supermarkets',
            5542: 'Service Stations', 
            5812: 'Eating Places and Restaurants',
            5814: 'Eating Places and Restaurants',
            5815: 'Eating Places and Restaurants',
            7832: 'Amusement Parks, Carnivals, Circuses',
            7841: 'Amusement Parks, Carnivals, Circuses',
            7996: 'Amusement Parks, Carnivals, Circuses',
            5912: 'Book Stores',  
            5942: 'Book Stores',
            4900: 'Utilities',
            5999: 'Retail'
        }
    
    def optimize_card_portfolio(self, user_id, time_period='1_year', max_cards=3):
        """Optimize entire credit card portfolio for maximum rewards"""
        
        # Get user's spending pattern
        spending_analysis = self._analyze_user_spending(user_id, time_period)
        
        if not spending_analysis:
            return {}
        
        # Calculate current rewards with existing cards
        current_rewards = self._calculate_current_rewards(user_id, spending_analysis)
        
        # Find optimal card combination
        optimal_portfolio = self._find_optimal_portfolio(spending_analysis, max_cards)
        
        # Calculate potential improvement
        optimization_results = self._calculate_optimization_impact(
            current_rewards, optimal_portfolio, spending_analysis
        )
        
        return {
            'current_portfolio': self._get_current_portfolio(user_id),
            'current_annual_rewards': current_rewards['total_rewards'],
            'optimal_portfolio': optimal_portfolio,
            'optimization_results': optimization_results,
            'recommendations': self._generate_recommendations(optimization_results),
            'spending_analysis': spending_analysis
        }
    
    def _analyze_user_spending(self, user_id, time_period):
        """Analyze user spending patterns by category with proper MCC handling"""
        if self.transactions_df.empty:
            return None
        
        user_transactions = self.transactions_df[
            (self.transactions_df['client_id'] == user_id) &
            (self.transactions_df['amount'] > 0)
        ].copy()
        
        if user_transactions.empty:
            return None
        
        # Filter by time period
        user_transactions['date'] = pd.to_datetime(user_transactions['date'])
        latest_date = user_transactions['date'].max()
        
        if time_period == '3_months':
            start_date = latest_date - pd.Timedelta(days=90)
            multiplier = 4  # Annualize
        elif time_period == '6_months':
            start_date = latest_date - pd.Timedelta(days=180)
            multiplier = 2
        else:  # 1_year
            start_date = latest_date - pd.Timedelta(days=365)
            multiplier = 1
        
        period_transactions = user_transactions[user_transactions['date'] >= start_date]
        
        # Join with MCC codes for categories - FIXED VERSION
        if not self.mcc_df.empty and 'mcc' in period_transactions.columns:
            # Ensure both mcc columns are the same type
            period_transactions['mcc'] = period_transactions['mcc'].astype(int)
            mcc_df_copy = self.mcc_df.copy()
            mcc_df_copy['mcc'] = mcc_df_copy['mcc'].astype(int)
            
            transactions_with_categories = period_transactions.merge(
                mcc_df_copy[['mcc', 'category']], 
                on='mcc', 
                how='left'
            )
            transactions_with_categories['category'] = transactions_with_categories['category'].fillna('Other')
        else:
            transactions_with_categories = period_transactions.copy()
            # Use fallback mapping if MCC data is not available
            if 'mcc' in transactions_with_categories.columns:
                transactions_with_categories['category'] = transactions_with_categories['mcc'].map(
                    self.mcc_category_mapping
                ).fillna('Other')
            else:
                transactions_with_categories['category'] = 'Other'
        
        # Calculate annual spending by category
        category_spending = transactions_with_categories.groupby('category')['amount'].sum() * multiplier
        
        return {
            'annual_spending_by_category': category_spending.to_dict(),
            'total_annual_spending': category_spending.sum(),
            'top_categories': category_spending.nlargest(5).to_dict(),
            'period_analyzed': time_period,
            'transaction_count': len(period_transactions)
        }
    
    def _calculate_current_rewards(self, user_id, spending_analysis):
        """Calculate rewards with current card portfolio"""
        current_cards = self._get_current_portfolio(user_id)
        
        if not current_cards:
            # Assume basic 1% cashback if no card data
            return {
                'total_rewards': spending_analysis['total_annual_spending'] * 0.01,
                'rewards_by_category': {
                    cat: amount * 0.01 
                    for cat, amount in spending_analysis['annual_spending_by_category'].items()
                },
                'annual_fees': 0
            }
        
        # Calculate rewards with current cards (simplified)
        total_rewards = 0
        rewards_by_category = {}
        annual_fees = 0
        
        for category, spending in spending_analysis['annual_spending_by_category'].items():
            # Use best current card for each category (simplified logic)
            best_rate = 0.01  # Default 1%
            
            for card in current_cards:
                card_brand = card.get('card_brand', '').lower()
                if 'chase' in card_brand and category in ['Eating Places and Restaurants', 'Amusement Parks, Carnivals, Circuses']:
                    best_rate = max(best_rate, 0.02)
                elif 'amex' in card_brand and category in ['Eating Places and Restaurants', 'Grocery Stores, Supermarkets']:
                    best_rate = max(best_rate, 0.03)
                elif 'discover' in card_brand and category in ['Service Stations']:
                    best_rate = max(best_rate, 0.05)
            
            category_rewards = spending * best_rate
            rewards_by_category[category] = category_rewards
            total_rewards += category_rewards
        
        return {
            'total_rewards': total_rewards,
            'rewards_by_category': rewards_by_category,
            'annual_fees': annual_fees
        }
    
    def _find_optimal_portfolio(self, spending_analysis, max_cards):
        """Find optimal combination of cards for maximum rewards"""
        spending_by_category = spending_analysis['annual_spending_by_category']
        
        # Generate all possible combinations of cards
        from itertools import combinations
        
        card_names = list(self.card_rewards_database.keys())
        best_portfolio = None
        best_net_rewards = 0
        
        # Try all combinations up to max_cards
        for num_cards in range(1, max_cards + 1):
            for card_combo in combinations(card_names, num_cards):
                net_rewards = self._calculate_portfolio_rewards(card_combo, spending_by_category)
                
                if net_rewards > best_net_rewards:
                    best_net_rewards = net_rewards
                    best_portfolio = card_combo
        
        if best_portfolio:
            return {
                'cards': list(best_portfolio),
                'net_annual_rewards': best_net_rewards,
                'gross_rewards': self._calculate_gross_rewards(best_portfolio, spending_by_category),
                'total_annual_fees': sum([
                    self.card_rewards_database[card]['annual_fee'] 
                    for card in best_portfolio
                ]),
                'card_details': [
                    {
                        'name': card,
                        'annual_fee': self.card_rewards_database[card]['annual_fee'],
                        'categories': self.card_rewards_database[card]['categories']
                    }
                    for card in best_portfolio
                ]
            }
        
        return None
    
    def _calculate_portfolio_rewards(self, card_combo, spending_by_category):
        """Calculate net rewards for a specific card combination"""
        total_rewards = 0
        total_fees = 0
        
        for category, spending in spending_by_category.items():
            best_rate = 0
            
            # Find best rate among cards in portfolio for this category
            for card_name in card_combo:
                card_info = self.card_rewards_database[card_name]
                rate = card_info['categories'].get(category, card_info['categories']['default'])
                best_rate = max(best_rate, rate)
            
            total_rewards += spending * best_rate
        
        # Subtract annual fees
        for card_name in card_combo:
            total_fees += self.card_rewards_database[card_name]['annual_fee']
        
        return total_rewards - total_fees
    
    def _calculate_gross_rewards(self, card_combo, spending_by_category):
        """Calculate gross rewards (before fees) for portfolio"""
        total_rewards = 0
        rewards_by_category = {}
        
        for category, spending in spending_by_category.items():
            best_rate = 0
            best_card = None
            
            for card_name in card_combo:
                card_info = self.card_rewards_database[card_name]
                rate = card_info['categories'].get(category, card_info['categories']['default'])
                if rate > best_rate:
                    best_rate = rate
                    best_card = card_name
            
            category_rewards = spending * best_rate
            rewards_by_category[category] = {
                'rewards': category_rewards,
                'rate': best_rate,
                'best_card': best_card,
                'spending': spending
            }
            total_rewards += category_rewards
        
        return {
            'total': total_rewards,
            'by_category': rewards_by_category
        }
    
    def _calculate_optimization_impact(self, current_rewards, optimal_portfolio, spending_analysis):
        """Calculate the impact of optimization"""
        if not optimal_portfolio:
            return {}
        
        current_net = current_rewards['total_rewards'] - current_rewards['annual_fees']
        optimal_net = optimal_portfolio['net_annual_rewards']
        
        improvement = optimal_net - current_net
        improvement_pct = (improvement / current_net * 100) if current_net > 0 else 0
        
        # Calculate signup bonuses
        signup_bonuses = sum([
            self.card_rewards_database[card]['signup_bonus']
            for card in optimal_portfolio['cards']
        ])
        
        return {
            'annual_improvement': improvement,
            'improvement_percentage': improvement_pct,
            'signup_bonuses': signup_bonuses,
            'payback_period': self._calculate_payback_period(optimal_portfolio, improvement),
            'first_year_benefit': improvement + signup_bonuses,
            'break_even_spending': self._calculate_break_even(optimal_portfolio)
        }
    
    def _calculate_payback_period(self, optimal_portfolio, annual_improvement):
        """Calculate how long it takes to pay back annual fees"""
        total_fees = optimal_portfolio['total_annual_fees']
        
        if annual_improvement <= 0 or total_fees == 0:
            return 0
        
        return total_fees / annual_improvement
    
    def _calculate_break_even(self, optimal_portfolio):
        """Calculate spending needed to break even on annual fees"""
        break_even_by_card = {}
        
        for card_detail in optimal_portfolio['card_details']:
            card_name = card_detail['name']
            annual_fee = card_detail['annual_fee']
            
            if annual_fee > 0:
                # Find average reward rate for this card
                categories = card_detail['categories']
                avg_rate = sum(categories.values()) / len(categories)
                break_even_spending = annual_fee / avg_rate if avg_rate > 0 else 0
                break_even_by_card[card_name] = break_even_spending
        
        return break_even_by_card
    
    def _get_current_portfolio(self, user_id):
        """Get user's current credit card portfolio"""
        if self.cards_df.empty:
            return []
        
        user_cards = self.cards_df[self.cards_df['client_id'] == user_id]
        
        return [
            {
                'card_id': row['id'],
                'card_brand': row['card_brand'],
                'card_type': row['card_type'],
                'credit_limit': row['credit_limit']
            }
            for _, row in user_cards.iterrows()
        ]
    
    def _generate_recommendations(self, optimization_results):
        """Generate actionable recommendations"""
        recommendations = []
        
        if optimization_results.get('annual_improvement', 0) > 100:
            recommendations.append({
                'priority': 'High',
                'title': 'Significant Rewards Opportunity',
                'description': f'Optimizing your card portfolio could earn you ${optimization_results["annual_improvement"]:.2f} more per year.',
                'action': 'Consider applying for recommended cards'
            })
        
        if optimization_results.get('signup_bonuses', 0) > 500:
            recommendations.append({
                'priority': 'High',
                'title': 'Valuable Signup Bonuses Available',
                'description': f'You could earn ${optimization_results["signup_bonuses"]:.2f} in signup bonuses.',
                'action': 'Apply for cards with highest signup bonuses first'
            })
        
        payback_period = optimization_results.get('payback_period', 0)
        if payback_period > 0 and payback_period < 2:
            recommendations.append({
                'priority': 'Medium',
                'title': 'Fast Payback Period',
                'description': f'Annual fees will pay for themselves in {payback_period:.1f} years.',
                'action': 'Annual fees are justified by rewards earned'
            })
        
        return recommendations
    
    def generate_card_comparison_table(self, user_spending):
        """Generate comparison table of all cards for user's spending pattern"""
        comparison_data = []
        
        for card_name, card_info in self.card_rewards_database.items():
            total_rewards = 0
            
            for category, spending in user_spending['annual_spending_by_category'].items():
                rate = card_info['categories'].get(category, card_info['categories']['default'])
                total_rewards += spending * rate
            
            net_rewards = total_rewards - card_info['annual_fee']
            
            comparison_data.append({
                'Card': card_name,
                'Annual Fee': card_info['annual_fee'],
                'Gross Rewards': total_rewards,
                'Net Rewards': net_rewards,
                'Signup Bonus': card_info.get('signup_bonus', 0),
                'First Year Value': net_rewards + card_info.get('signup_bonus', 0)
            })
        
        return pd.DataFrame(comparison_data).sort_values('Net Rewards', ascending=False)