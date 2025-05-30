import streamlit as st
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
#sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data_preprocessing import DataPreprocessor
from src.kpi_calculator import KPICalculator

# Import all analysis modules
from spending_analysis_page import display_spending_analysis_page
from advanced_spending_page import display_advanced_spending_analysis
from merchant_analyzer import MerchantAnalyzer
from enhanced_rewards_optimizer import RewardsOptimizer

# Page configuration
st.set_page_config(
    page_title="Credit Card Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stMetric > label {
        font-size: 14px !important;
        font-weight: 600 !important;
    }
    .stMetric > div {
        font-size: 28px !important;
        font-weight: 700 !important;
    }
    .nav-tabs {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .feature-badge {
        background-color: #28a745;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_preprocessor():
    """Get or create preprocessor instance"""
    preprocessor = DataPreprocessor()
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    preprocessor.set_data_path(data_path)
    return preprocessor

@st.cache_data
def get_user_list():
    """Get list of users for dropdown"""
    try:
        preprocessor = get_preprocessor()
        return preprocessor.get_user_list()
    except Exception as e:
        st.error(f"Error loading user list: {str(e)}")
        return []

@st.cache_data
def load_user_data(user_id):
    """Load data for specific user only"""
    try:
        preprocessor = get_preprocessor()
        spark_dataframes = preprocessor.load_user_specific_data(user_id)
        
        # Convert to pandas for caching and faster operations
        pandas_dataframes = {}
        for key, spark_df in spark_dataframes.items():
            if spark_df.count() > 0:  # Only convert if DataFrame has data
                pandas_dataframes[key] = spark_df.toPandas()
            else:
                # Create empty pandas DataFrame with proper structure
                pandas_dataframes[key] = pd.DataFrame()
        
        return pandas_dataframes
    except Exception as e:
        st.error(f"Error loading user data: {str(e)}")
        return None

def display_user_profile(profile):
    """Display user profile information"""
    if not profile:
        st.error("User profile not found")
        return
    
    st.subheader("👤 User Profile")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Age", f"{profile['age']} years")
        st.metric("Gender", profile['gender'])
    
    with col2:
        st.metric("Credit Score", profile['credit_score'])
        st.metric("Number of Cards", profile['num_credit_cards'])
    
    with col3:
        st.metric("Yearly Income", f"${profile['yearly_income']:,.0f}")
        st.metric("Total Debt", f"${profile['total_debt']:,.0f}")
    
    with col4:
        st.metric("Address", profile['address'], label_visibility="visible")
    
    # Display cards information
    if profile['cards']:
        st.subheader("💳 User's Credit Cards")
        cards_df = pd.DataFrame(profile['cards'])
        st.dataframe(cards_df, use_container_width=True)

def display_kpi_cards(kpis):
    """Display KPI cards"""
    st.subheader("📊 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="💰 Monthly Spending",
            value=f"${kpis['total_monthly_spending']:,.2f}",
            delta=None
        )
    
    with col2:
        utilization = kpis['credit_utilization']
        st.metric(
            label="📈 Credit Utilization",
            value=f"{utilization:.1f}%",
            delta=f"{'Good' if utilization < 30 else 'High'}"
        )
    
    with col3:
        st.metric(
            label="💸 Est. Interest Paid",
            value=f"${kpis['interest_paid']:,.2f}",
            delta="Last 12 months"
        )
    
    with col4:
        st.metric(
            label="🎁 Rewards Earned",
            value=f"${kpis['rewards_earned']:,.2f}",
            delta="Last 12 months"
        )
    
    with col5:
        credit_score = kpis['credit_score']
        score_status = "Excellent" if credit_score >= 750 else "Good" if credit_score >= 650 else "Fair"
        st.metric(
            label="🏆 Credit Score",
            value=credit_score,
            delta=score_status
        )

def display_spending_breakdown(spending_data):
    """Display spending breakdown by category with bar chart showing top 10 categories"""
    if not spending_data:
        st.info("No spending data available for this user")
        return
    
    st.subheader("🛍️ Spending by Category (Last 3 Months)")
    
    # Create DataFrame and calculate percentages
    df = pd.DataFrame(spending_data)
    
    if df.empty:
        st.info("No spending data to display")
        return
    
    # Calculate total spending for percentage calculation
    total_spending = df['amount'].sum()
    df['percentage'] = (df['amount'] / total_spending * 100).round(1)
    
    # Get top 10 categories
    top_10_df = df.head(10)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create horizontal bar chart with better clarity
        fig = px.bar(
            top_10_df, 
            x='percentage',
            y='category',
            orientation='h',
            title="Top 10 Spending Categories",
            labels={'percentage': 'Percentage of Total Spending (%)', 'category': 'Category'},
            color='percentage',
            color_continuous_scale='viridis',
            text='percentage'
        )
        
        # Update layout for better readability
        fig.update_traces(
            texttemplate='%{text:.1f}%', 
            textposition='outside',
            hovertemplate="<b>%{y}</b><br>" +
                         "Amount: $%{customdata[0]:,.2f}<br>" +
                         "Percentage: %{x:.1f}%<br>" +
                         "Transactions: %{customdata[1]}<br>" +
                         "<extra></extra>",
            customdata=top_10_df[['amount', 'transactions']].values
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=500,
            showlegend=False,
            xaxis_title="Percentage of Total Spending (%)",
            yaxis_title="Category"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Category Details")
        
        # Display top 5 categories with enhanced metrics - FIXED: Remove nested columns
        for i, (_, row) in enumerate(top_10_df.head(5).iterrows(), 1):
            with st.container():
                st.markdown(f"**#{i} {row['category']}**")
                
                # Instead of nested columns, use simple metrics side by side
                st.metric("Amount", f"${row['amount']:,.2f}")
                st.metric("Share", f"{row['percentage']:.1f}%")
                st.caption(f"{row['transactions']} transactions")
                
                # Progress bar for visual representation
                st.progress(row['percentage'] / 100)
                st.markdown("---")
        
        # Summary statistics
        st.subheader("📈 Summary")
        st.metric("Total Categories", len(df))
        st.metric("Total Spending", f"${total_spending:,.2f}")
        top_5_percentage = top_10_df.head(5)['percentage'].sum()
        st.metric("Top 5 Share", f"{top_5_percentage:.1f}%")

def display_monthly_trend(trend_data):
    """Display monthly spending trend"""
    if not trend_data:
        st.info("No trend data available")
        return
    
    st.subheader("📈 Monthly Spending Trend")
    
    df = pd.DataFrame(trend_data)
    
    if not df.empty:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['month'],
            y=df['spending'],
            mode='lines+markers',
            name='Monthly Spending',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Spending Trend Over Time",
            xaxis_title="Month",
            yaxis_title="Amount ($)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trend data to display")

def display_fraud_risk(fraud_score):
    """Display fraud risk indicator"""
    st.subheader("🔒 Fraud Risk Assessment")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Create gauge chart for fraud risk
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = fraud_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Fraud Risk Score (%)"},
            delta = {'reference': 5},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgreen"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if fraud_score < 5:
            st.success("✅ Low fraud risk - Your account shows normal transaction patterns")
        elif fraud_score < 15:
            st.warning("⚠️ Moderate fraud risk - Monitor your account regularly")
        else:
            st.error("🚨 High fraud risk - Consider reviewing recent transactions")
        
        st.markdown("""
        **Fraud Prevention Tips:**
        - Monitor your accounts regularly
        - Set up transaction alerts
        - Use secure payment methods
        - Report suspicious activity immediately
        """)

def display_overview_page(user_dataframes, selected_user_id):
    """Display the main overview page"""
    
    # Initialize KPI calculator with user-specific data
    kpi_calculator = OptimizedKPICalculator(user_dataframes)
    
    # Get user profile and KPIs
    user_profile = kpi_calculator.get_user_profile(selected_user_id)
    
    if not user_profile:
        st.error(f"No profile found for User {selected_user_id}")
        return
    
    # Display user profile
    display_user_profile(user_profile)
    
    st.markdown("---")
    
    # Get and display KPIs
    kpis = kpi_calculator.get_all_kpis(selected_user_id)
    display_kpi_cards(kpis)
    
    st.markdown("---")
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        display_spending_breakdown(kpis['spending_by_category'])
    
    with col2:
        display_fraud_risk(kpis['fraud_risk_score'])
    
    st.markdown("---")
    
    # Monthly trend
    display_monthly_trend(kpis['monthly_trend'])

def display_merchant_analysis_page(user_dataframes, selected_user_id):
    """Display merchant analysis page"""
    st.title("🏪 Merchant Intelligence Dashboard")
    st.markdown("*Advanced merchant analysis with pattern recognition and loyalty insights*")
    st.markdown("---")
    
    # Initialize merchant analyzer
    merchant_analyzer = MerchantAnalyzer(
        user_dataframes.get('transactions', pd.DataFrame()),
        user_dataframes.get('mcc_codes', pd.DataFrame())
    )
    
    # Time period selection
    time_period_options = {
        "Last Month": "1_month",
        "Last 3 Months": "3_months", 
        "Last 6 Months": "6_months",
        "Last Year": "Last Year"
    }
    
    selected_period_display = st.selectbox(
        "📅 Analysis Period",
        options=list(time_period_options.keys()),
        index=1
    )
    
    



    selected_period = time_period_options[selected_period_display]
    
    # Get merchant insights
    merchant_insights = merchant_analyzer.get_merchant_insights(selected_user_id, selected_period)
    
    if not merchant_insights or not merchant_insights.get('merchant_stats') is not None:
        st.warning("No merchant data available for analysis.")
        return
    
    merchant_stats = merchant_insights['merchant_stats']
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Merchants", merchant_insights['total_merchants'])
    
    with col2:
        concentration = merchant_insights.get('top_merchant_concentration', {})
        st.metric("Top 3 Concentration", f"{concentration.get('top_3_percentage', 0):.1f}%")
    
    with col3:
        volatility = merchant_insights.get('spending_volatility', {})
        st.metric("Spending Volatility", f"{volatility.get('volatility_coefficient', 0):.2f}")
    
    with col4:
        loyalty_ops = merchant_insights.get('loyalty_opportunities', pd.DataFrame())
        st.metric("Loyalty Opportunities", len(loyalty_ops))
    
    # Merchant tabs
    tab1, tab2, tab3 = st.tabs(["📊 Top Merchants", "🔄 Subscriptions", "🎯 Loyalty Opportunities"])
    
    with tab1:
        if not merchant_stats.empty:
            # Top merchants visualization
            top_merchants = merchant_stats.head(10)
            
            fig = px.bar(
                top_merchants,
                x='total_spent',
                y='merchant_name',
                orientation='h',
                title="Top Merchants by Spending",
                color='total_spent',
                color_continuous_scale='viridis'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Merchant details
            st.dataframe(
                top_merchants[['merchant_name', 'total_spent', 'transaction_count', 'consistency_score']],
                use_container_width=True
            )
    
    with tab2:
        
        
        subscriptions = merchant_insights.get('subscription_merchants', pd.DataFrame())
        if subscriptions.get('insufficient_data', False):
            st.subheader("🔄 Subscription Analysis")
            
            # Display the warning message
            st.warning(f"⚠️ {subscriptions.get('message', 'Insufficient data for subscription detection')}")
            
            # Show recommendation
            if subscriptions.get('recommendation'):
                st.info(f"💡 {subscriptions['recommendation']}")
            
            # Show helpful tip
            st.markdown("""
            **Why 1 month isn't enough?**
            - Most subscriptions bill monthly (only 1 transaction)
            - Need multiple transactions to detect patterns
            - Recommended: Use 3+ months for accurate detection
            """)
        else:
            
            if not subscriptions.empty:
                st.subheader("🔄 Detected Subscription Services")
                
                total_monthly = subscriptions['estimated_monthly_cost'].sum()
                st.metric("Total Monthly Subscriptions", f"${total_monthly:.2f}")
                
                for _, sub in subscriptions.iterrows():
                    st.info(f"📍 {sub['merchant_name']}: ~${sub['estimated_monthly_cost']:.2f}/month")
            else:
                st.info("No subscription patterns detected.")
    
    with tab3:
        loyalty_ops = merchant_insights.get('loyalty_opportunities', pd.DataFrame())
        
        if not loyalty_ops.empty:
            st.subheader("🎯 Top Loyalty Opportunities")
            
            for _, merchant in loyalty_ops.head(5).iterrows():
                with st.expander(f"💳 {merchant['merchant_name']} - Score: {merchant['loyalty_score']:.1f}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Spent", f"${merchant['total_spent']:,.2f}")
                    with col2:
                        st.metric("Visits", f"{merchant['transaction_count']}")
                    with col3:
                        potential_rewards = merchant['total_spent'] * 0.02
                        st.metric("Potential Rewards", f"${potential_rewards:.2f}")
        else:
            st.info("No specific loyalty opportunities identified.")

def display_optimization_overview(optimization):
    """Display optimization overview with key metrics + complete rewards analysis"""
    st.subheader("🎯 Portfolio Optimization Overview")
    
    results = optimization['optimization_results']
    optimal_portfolio = optimization['optimal_portfolio']
    
    # Key metrics in prominent cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        improvement = results.get('annual_improvement', 0)
        st.metric(
            "💰 Annual Improvement",
            f"${improvement:.2f}",
            delta=f"{results.get('improvement_percentage', 0):.1f}% increase",
            delta_color="normal"
        )
    
    with col2:
        signup_bonuses = results.get('signup_bonuses', 0)
        st.metric(
            "🎁 Signup Bonuses",
            f"${signup_bonuses:.2f}",
            delta="One-time bonus",
            delta_color="normal"
        )
    
    with col3:
        first_year_benefit = results.get('first_year_benefit', 0)
        st.metric(
            "🚀 First Year Benefit",
            f"${first_year_benefit:.2f}",
            delta="Including bonuses",
            delta_color="normal"
        )
    
    with col4:
        payback_period = results.get('payback_period', 0)
        if payback_period > 0:
            payback_text = f"{payback_period:.1f} years"
            payback_delta = "Fee payback period"
        else:
            payback_text = "Immediate"
            payback_delta = "No annual fees"
        
        st.metric(
            "⏱️ Payback Period",
            payback_text,
            delta=payback_delta
        )
    
    # Optimization summary
    st.markdown("---")
    
    if improvement > 100:
        st.success(f"🎉 **Excellent Optimization Opportunity!** You could earn ${improvement:.2f} more per year by optimizing your card portfolio.")
    elif improvement > 50:
        st.info(f"💡 **Good Optimization Potential** - ${improvement:.2f} additional annual rewards possible.")
    elif improvement > 0:
        st.warning(f"⚖️ **Modest Improvement Available** - ${improvement:.2f} more per year, consider if worth the effort.")
    else:
        st.success("✅ **Your current portfolio is already well-optimized!**")
    
    # Quick insights
    if optimal_portfolio:
        st.subheader("🔍 Quick Insights")
        
        total_fees = optimal_portfolio.get('total_annual_fees', 0)
        if total_fees > 0:
            st.info(f"💳 **Annual Fees**: ${total_fees:.2f} total across recommended cards")
        else:
            st.success("🆓 **No Annual Fees** in recommended portfolio")
        
        recommended_cards = len(optimal_portfolio.get('cards', []))
        st.info(f"🃏 **Portfolio Size**: {recommended_cards} cards recommended for optimal rewards")

    # ==== NEW SECTION: Complete Rewards Analysis ====
    st.markdown("---")
    st.subheader("📊 Complete Rewards Analysis")
    st.markdown("*Detailed breakdown of current vs potential rewards by spending category*")
    
    # Get spending analysis data
    spending_analysis = optimization.get('spending_analysis', {})
    if not spending_analysis or not spending_analysis.get('annual_spending_by_category'):
        st.warning("⚠️ No detailed spending analysis available for rewards breakdown.")
        return
    
    # Initialize rewards optimizer to get detailed analysis
    from enhanced_rewards_optimizer import RewardsOptimizer
    rewards_optimizer = RewardsOptimizer(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    
    # Calculate detailed rewards by category
    spending_by_category = spending_analysis['annual_spending_by_category']
    current_annual_rewards = optimization.get('current_annual_rewards', 0)
    
    # Create detailed analysis similar to spending_analyzer
    detailed_rewards_analysis = []
    
    for category, spending in spending_by_category.items():
        # Current rewards (assume 1% default)
        current_rewards = spending * 0.01
        
        # Find best potential rate from optimal portfolio
        best_rate = 0.01  # Default
        best_card = "Current Card"
        
        if optimal_portfolio and optimal_portfolio.get('card_details'):
            for card_detail in optimal_portfolio['card_details']:
                rate = card_detail['categories'].get(category, card_detail['categories']['default'])
                if rate > best_rate:
                    best_rate = rate
                    best_card = card_detail['name']
        
        potential_rewards = spending * best_rate
        additional_rewards = potential_rewards - current_rewards
        
        detailed_rewards_analysis.append({
            'category': category,
            'spending': spending,
            'current_rewards': current_rewards,
            'potential_rewards': potential_rewards,
            'additional_rewards': additional_rewards,
            'best_card': best_card,
            'best_reward_rate': best_rate * 100,
            'improvement_pct': (additional_rewards / current_rewards * 100) if current_rewards > 0 else 0
        })
    
    rewards_df = pd.DataFrame(detailed_rewards_analysis)
    rewards_df = rewards_df.sort_values('additional_rewards', ascending=False)
    
    if not rewards_df.empty:
        # Overview metrics for rewards analysis
        col1, col2, col3 = st.columns(3)
        
        total_current = rewards_df['current_rewards'].sum()
        total_potential = rewards_df['potential_rewards'].sum()
        total_additional = rewards_df['additional_rewards'].sum()
        
        with col1:
            st.metric(
                "Current Rewards Earned",
                f"${total_current:.2f}",
                "For analyzed period"
            )
        
        with col2:
            st.metric(
                "Potential Additional Rewards", 
                f"${total_additional:.2f}",
                f"{(total_additional/total_current*100):.1f}% increase" if total_current > 0 else "N/A"
            )
        
        with col3:
            st.metric(
                "Optimized Total Rewards",
                f"${total_potential:.2f}",
                f"vs ${total_current:.2f} current"
            )
        
        # Visualization section
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Current vs Potential comparison chart
            comparison_data = rewards_df.head(8).copy()  # Top 8 categories
            
            fig_comparison = go.Figure()
            
            fig_comparison.add_trace(go.Bar(
                name='Current Rewards',
                x=comparison_data['category'],
                y=comparison_data['current_rewards'],
                marker_color='lightblue'
            ))
            
            fig_comparison.add_trace(go.Bar(
                name='Potential Rewards',
                x=comparison_data['category'],
                y=comparison_data['potential_rewards'],
                marker_color='darkgreen'
            ))
            
            fig_comparison.update_layout(
                title="Current vs Potential Rewards by Category",
                xaxis_title="Category",
                yaxis_title="Rewards ($)",
                barmode='group',
                height=400,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        with col2:
            # Top opportunities chart
            top_opportunities = rewards_df.head(6)
            fig_rewards = px.bar(
                top_opportunities,
                x='additional_rewards',
                y='category',
                orientation='h',
                title="Top Rewards Optimization Opportunities",
                color='additional_rewards',
                color_continuous_scale='greens',
                text='additional_rewards'
            )
            fig_rewards.update_traces(
                texttemplate='$%{text:.2f}',
                textposition='inside'
            )
            fig_rewards.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig_rewards, use_container_width=True)
        
        # Actionable insights
        st.subheader("🎯 Key Rewards Insights")
        
        # Filter significant opportunities
        significant_opportunities = rewards_df[rewards_df['additional_rewards'] > 1.0]
        
        if not significant_opportunities.empty:
            # Show top 3 opportunities in a concise format
            st.markdown("**🏆 Top Opportunities:**")
            
            for i, (_, opportunity) in enumerate(significant_opportunities.head(3).iterrows(), 1):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"**#{i}. {opportunity['category']}**")
                    st.caption(f"${opportunity['spending']:,.0f} spending")
                
                with col2:
                    st.metric("Additional Rewards", f"${opportunity['additional_rewards']:.2f}")
                
                with col3:
                    st.metric("Best Card", opportunity['best_card'])
                
                with col4:
                    st.metric("Rate", f"{opportunity['best_reward_rate']:.1f}%")
        else:
            st.success("✅ You're already optimizing your rewards well across all categories!")
        
        # Detailed rewards table (collapsible)
        with st.expander("📋 View Complete Rewards Analysis Table"):
            st.dataframe(
                rewards_df[['category', 'spending', 'current_rewards', 'potential_rewards', 'additional_rewards', 'best_card', 'best_reward_rate']],
                column_config={
                    "category": "Category",
                    "spending": st.column_config.NumberColumn("Spending", format="$%.2f"),
                    "current_rewards": st.column_config.NumberColumn("Current Rewards", format="$%.2f"),
                    "potential_rewards": st.column_config.NumberColumn("Potential Rewards", format="$%.2f"),
                    "additional_rewards": st.column_config.NumberColumn("Additional Rewards", format="$%.2f"),
                    "best_card": "Recommended Card",
                    "best_reward_rate": st.column_config.NumberColumn("Best Rate", format="%.1f%%")
                },
                use_container_width=True
            )
            
            # Export functionality
            col1, col2 = st.columns(2)
            
            with col1:
                csv_rewards = rewards_df.to_csv(index=False)
                st.download_button(
                    label="💰 Download Complete Rewards Analysis",
                    data=csv_rewards,
                    file_name=f"complete_rewards_analysis.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Create summary report
                summary_data = {
                    'Metric': [
                        'Current Total Rewards',
                        'Potential Total Rewards', 
                        'Additional Rewards Possible',
                        'Percentage Improvement',
                        'Best Opportunity Category',
                        'Best Opportunity Value'
                    ],
                    'Value': [
                        f"${total_current:.2f}",
                        f"${total_potential:.2f}",
                        f"${total_additional:.2f}",
                        f"{(total_additional/total_current*100):.1f}%" if total_current > 0 else "N/A",
                        rewards_df.iloc[0]['category'] if len(rewards_df) > 0 else "N/A",
                        f"${rewards_df.iloc[0]['additional_rewards']:.2f}" if len(rewards_df) > 0 else "N/A"
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                csv_summary = summary_df.to_csv(index=False)
                st.download_button(
                    label="📈 Download Summary Report",
                    data=csv_summary,
                    file_name=f"rewards_optimization_summary.csv",
                    mime="text/csv"
                )
    else:
        st.warning("⚠️ No rewards analysis available. This might be due to limited spending data or category information.")
def display_current_vs_optimal(optimization, rewards_optimizer):
    """Display detailed current vs optimal comparison"""
    st.subheader("📊 Current Portfolio vs Optimal Portfolio")
    
    current_portfolio = optimization.get('current_portfolio', [])
    optimal_portfolio = optimization.get('optimal_portfolio')
    current_rewards = optimization.get('current_annual_rewards', 0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Current Portfolio")
        
        if current_portfolio:
            st.metric("Annual Rewards", f"${current_rewards:.2f}")
            
            # Display current cards
            for i, card in enumerate(current_portfolio, 1):
                with st.container():
                    st.markdown(f"**Card {i}:** {card['card_brand']} {card['card_type']}")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.caption(f"Credit Limit: ${card['credit_limit']:,.0f}")
                    with col_b:
                        st.caption(f"Card ID: {card['card_id']}")
                    
                    st.markdown("---")
        else:
            st.info("ℹ️ No current card data available")
            st.caption("Using 1% default cashback assumption")
    
    with col2:
        st.markdown("### 🎯 Optimal Portfolio")
        
        if optimal_portfolio:
            st.metric(
                "Optimized Annual Rewards", 
                f"${optimal_portfolio['net_annual_rewards']:.2f}",
                delta=f"+${optimization['optimization_results']['annual_improvement']:.2f}"
            )
            
            # Display optimal cards
            for i, card_detail in enumerate(optimal_portfolio['card_details'], 1):
                with st.container():
                    st.markdown(f"**Recommended Card {i}:** {card_detail['name']}")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if card_detail['annual_fee'] > 0:
                            st.caption(f"Annual Fee: ${card_detail['annual_fee']}")
                        else:
                            st.caption("✅ No Annual Fee")
                    
                    with col_b:
                        # Show best categories for this card
                        best_categories = []
                        for cat, rate in card_detail['categories'].items():
                            if cat != 'default' and rate > 0.02:  # More than 2%
                                best_categories.append(f"{cat} ({rate*100:.0f}%)")
                        
                        if best_categories:
                            st.caption(f"Best for: {best_categories[0]}")
                        else:
                            st.caption(f"General rewards: {card_detail['categories']['default']*100:.1f}%")
                    
                    st.markdown("---")
    
    # Detailed comparison chart
    if optimal_portfolio:
        st.subheader("📈 Rewards Comparison by Category")
        
        spending_analysis = optimization.get('spending_analysis', {})
        if spending_analysis and spending_analysis.get('annual_spending_by_category'):
            
            # Create comparison data
            categories = []
            current_rewards_by_cat = []
            optimal_rewards_by_cat = []
            
            for category, spending in spending_analysis['annual_spending_by_category'].items():
                categories.append(category)
                
                # Current rewards (assume 1% default)
                current_reward = spending * 0.01
                current_rewards_by_cat.append(current_reward)
                
                # Optimal rewards
                best_rate = 0
                for card_name in optimal_portfolio['cards']:
                    card_info = rewards_optimizer.card_rewards_database[card_name]
                    rate = card_info['categories'].get(category, card_info['categories']['default'])
                    best_rate = max(best_rate, rate)
                
                optimal_reward = spending * best_rate
                optimal_rewards_by_cat.append(optimal_reward)
            
            # Create comparison chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Current Rewards',
                x=categories,
                y=current_rewards_by_cat,
                marker_color='lightblue'
            ))
            
            fig.add_trace(go.Bar(
                name='Optimal Rewards',
                x=categories,
                y=optimal_rewards_by_cat,
                marker_color='darkgreen'
            ))
            
            fig.update_layout(
                title="Current vs Optimal Rewards by Category",
                xaxis_title="Spending Category",
                yaxis_title="Annual Rewards ($)",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

def display_card_comparison_table(optimization, rewards_optimizer):
    """Display comprehensive card comparison table"""
    st.subheader("💰 Complete Card Comparison for Your Spending")
    
    spending_analysis = optimization.get('spending_analysis')
    
    if not spending_analysis:
        st.warning("Unable to generate card comparison due to insufficient spending data.")
        return
    
    # Generate comparison table
    comparison_table = rewards_optimizer.generate_card_comparison_table(spending_analysis)
    
    if not comparison_table.empty:
        # Add ranking
        comparison_table['Rank'] = range(1, len(comparison_table) + 1)
        
        # Reorder columns
        columns_order = ['Rank', 'Card', 'Net Rewards', 'Gross Rewards', 'Annual Fee', 'Signup Bonus', 'First Year Value']
        comparison_table = comparison_table[columns_order]
        
        # Style the dataframe
        st.dataframe(
            comparison_table,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", format="%d"),
                "Card": "Card Name",
                "Annual Fee": st.column_config.NumberColumn("Annual Fee", format="$%d"),
                "Gross Rewards": st.column_config.NumberColumn("Gross Rewards", format="$%.2f"),
                "Net Rewards": st.column_config.NumberColumn("Net Rewards", format="$%.2f"),
                "Signup Bonus": st.column_config.NumberColumn("Signup Bonus", format="$%d"),
                "First Year Value": st.column_config.NumberColumn("First Year Value", format="$%.2f")
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Top 3 recommendations highlight
        st.subheader("🏆 Top 3 Recommendations")
        
        top_3 = comparison_table.head(3)
        
        for idx, (_, card) in enumerate(top_3.iterrows(), 1):
            medal = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉"
            
            with st.expander(f"{medal} **#{idx}: {card['Card']}** - ${card['Net Rewards']:.2f} annual value"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Net Annual Rewards", f"${card['Net Rewards']:.2f}")
                    if card['Annual Fee'] > 0:
                        st.caption(f"After ${card['Annual Fee']} annual fee")
                    else:
                        st.caption("No annual fee")
                
                with col2:
                    st.metric("Signup Bonus", f"${card['Signup Bonus']:.0f}")
                    st.caption("One-time bonus")
                
                with col3:
                    st.metric("First Year Value", f"${card['First Year Value']:.2f}")
                    st.caption("Including signup bonus")
                
                # Show why this card is good for the user
                card_name = card['Card']
                if card_name in rewards_optimizer.card_rewards_database:
                    card_info = rewards_optimizer.card_rewards_database[card_name]
                    best_categories = []
                    
                    for category, rate in card_info['categories'].items():
                        if category != 'default' and rate >= 0.03:  # 3% or higher
                            best_categories.append(f"**{category}**: {rate*100:.0f}% rewards")
                    
                    if best_categories:
                        st.markdown("**🎯 Best Rewards Categories:**")
                        for cat in best_categories[:3]:  # Show top 3
                            st.markdown(f"• {cat}")
    else:
        st.error("Unable to generate card comparison table.")

def display_impact_analysis(optimization):
    """Display detailed impact analysis"""
    st.subheader("📈 Optimization Impact Analysis")
    
    results = optimization.get('optimization_results', {})
    optimal_portfolio = optimization.get('optimal_portfolio')
    
    if not results:
        st.warning("No optimization results to analyze.")
        return
    
    # Impact metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💸 Financial Impact")
        
        # Annual improvement breakdown
        annual_improvement = results.get('annual_improvement', 0)
        monthly_improvement = annual_improvement / 12
        
        st.metric("Monthly Improvement", f"${monthly_improvement:.2f}")
        st.metric("Annual Improvement", f"${annual_improvement:.2f}")
        
        # ROI calculation
        if optimal_portfolio:
            total_fees = optimal_portfolio.get('total_annual_fees', 0)
            if total_fees > 0:
                roi = (annual_improvement / total_fees) * 100
                st.metric("Return on Investment", f"{roi:.1f}%", help="Annual improvement vs annual fees")
            else:
                st.success("♾️ Infinite ROI - No annual fees!")
    
    with col2:
        st.markdown("#### ⏰ Time Analysis")
        
        payback_period = results.get('payback_period', 0)
        if payback_period > 0:
            payback_months = payback_period * 12
            st.metric("Payback Period", f"{payback_months:.0f} months")
            
            if payback_period < 1:
                st.success("✅ Annual fees pay for themselves in less than a year!")
            elif payback_period < 2:
                st.info("👍 Reasonable payback period")
            else:
                st.warning("⚠️ Long payback period - consider carefully")
        else:
            st.success("✅ Immediate benefits - no annual fees to recover")
        
        # Break-even analysis
        break_even = results.get('break_even_spending', {})
        if break_even:
            st.markdown("**Break-even Spending:**")
            for card, spending in break_even.items():
                st.caption(f"{card}: ${spending:,.0f}/year")
    
    # Visual impact analysis
    st.markdown("#### 📊 5-Year Projection")
    
    # Calculate 5-year projection
    years = list(range(1, 6))
    current_rewards_projection = [optimization['current_annual_rewards'] * year for year in years]
    
    if optimal_portfolio:
        optimal_rewards_projection = [optimal_portfolio['net_annual_rewards'] * year for year in years]
        
        # Add signup bonuses to first year
        signup_bonuses = results.get('signup_bonuses', 0)
        optimal_rewards_projection[0] += signup_bonuses
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=years,
            y=current_rewards_projection,
            mode='lines+markers',
            name='Current Portfolio',
            line=dict(color='lightblue', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=years,
            y=optimal_rewards_projection,
            mode='lines+markers',
            name='Optimal Portfolio',
            line=dict(color='green', width=3)
        ))
        
        fig.update_layout(
            title="5-Year Cumulative Rewards Projection",
            xaxis_title="Years",
            yaxis_title="Cumulative Rewards ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show total 5-year benefit
        total_5_year_benefit = optimal_rewards_projection[-1] - current_rewards_projection[-1]
        st.success(f"💰 **5-Year Total Benefit**: ${total_5_year_benefit:,.2f}")

def display_actionable_recommendations(optimization,user_dataframes):
    """Display actionable recommendations with prioritization"""
    st.subheader("🔮 Personalized Action Plan")
    
    recommendations = optimization.get('recommendations', [])
    results = optimization.get('optimization_results', {})
    optimal_portfolio = optimization.get('optimal_portfolio')
    
    if not recommendations:
        st.info("No specific recommendations available.")
        return
    
    # Priority-based recommendations
    high_priority = [r for r in recommendations if r.get('priority') == 'High']
    medium_priority = [r for r in recommendations if r.get('priority') == 'Medium']
    low_priority = [r for r in recommendations if r.get('priority') == 'Low']
    
    # High Priority Actions
    if high_priority:
        st.markdown("### 🔴 High Priority Actions")
        for i, rec in enumerate(high_priority, 1):
            with st.container():
                st.error(f"**Action {i}: {rec['title']}**")
                st.markdown(rec['description'])
                st.markdown(f"**Next Step:** {rec['action']}")
                st.markdown("---")
    
    # Medium Priority Actions  
    if medium_priority:
        st.markdown("### 🟡 Medium Priority Actions")
        for i, rec in enumerate(medium_priority, 1):
            with st.container():
                st.warning(f"**Action {i}: {rec['title']}**")
                st.markdown(rec['description'])
                st.markdown(f"**Next Step:** {rec['action']}")
                st.markdown("---")
    
    # Low Priority Actions
    if low_priority:
        st.markdown("### 🔵 Low Priority Actions")
        for i, rec in enumerate(low_priority, 1):
            with st.container():
                st.info(f"**Action {i}: {rec['title']}**")
                st.markdown(rec['description'])
                st.markdown(f"**Next Step:** {rec['action']}")
                st.markdown("---")
    
    # Implementation Timeline
    st.markdown("### 📅 Suggested Implementation Timeline")
    
    if optimal_portfolio and results.get('annual_improvement', 0) > 50:
        timeline_steps = [
            "**Week 1-2**: Research and compare recommended cards",
            "**Week 3**: Apply for highest-priority card (best signup bonus)",
            "**Month 2**: Once approved, begin using new card for optimal categories",
            "**Month 3**: Apply for second card if recommended",
            "**Month 4**: Optimize spending across all cards",
            "**Month 6**: Review and track rewards earned vs projections"
        ]
        
        for step in timeline_steps:
            st.markdown(f"• {step}")
    else:
        st.info("💡 **Minor Optimization**: Consider implementing changes gradually as your current portfolio is already well-optimized.")
    
    # Quick action buttons (mockup - would integrate with real application systems)
    st.markdown("### 🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📧 Email This Analysis"):
            st.success("Analysis would be emailed to you!")
    
    with col2:
        if st.button("📅 Set Reminder"):
            st.success("Reminder set to review in 3 months!")
    
    with col3:
        if st.button("💾 Save to Profile"):
            st.success("Optimization saved to your profile!")
    
    # Additional insights
    if optimal_portfolio:
        st.markdown("### 💡 Additional Insights")
        
        with st.expander("🔍 See detailed card application tips"):
            st.markdown("""
            **Credit Card Application Tips:**
            
            1. **Credit Score Impact**: Each application may temporarily lower your credit score by 5-10 points
            2. **Timing**: Space applications 2-3 months apart for best approval odds
            3. **Income Requirements**: Ensure you meet minimum income requirements
            4. **Spending Requirements**: Plan to meet signup bonus spending requirements naturally
            5. **Annual Fee Strategy**: Set calendar reminders before annual fees post
            
            **Approval Odds Factors:**
            • Current credit score
            • Income level  
            • Existing relationship with bank
            • Recent credit applications (5/24 rule for Chase)
            • Current debt levels
            """)
        
        with st.expander("📊 See spending strategy for optimal rewards"):
            spending_analysis = optimization.get('spending_analysis', {})
            if spending_analysis and spending_analysis.get('annual_spending_by_category'):
                st.markdown("**Recommended Card Usage by Category:**")
                
                # Import here to avoid circular imports
                from enhanced_rewards_optimizer import RewardsOptimizer
                # temp_optimizer = RewardsOptimizer({},{},{})
                temp_optimizer = RewardsOptimizer(
                    user_dataframes.get('cards', pd.DataFrame()),
                    user_dataframes.get('transactions', pd.DataFrame()),
                    user_dataframes.get('mcc_codes', pd.DataFrame())
                )
                print(temp_optimizer)
                
                for category, spending in spending_analysis['annual_spending_by_category'].items():
                    if spending > 100:  # Only show significant categories
                        best_card = None
                        best_rate = 0
                        
                        for card_name in optimal_portfolio['cards']:
                            card_info = temp_optimizer.card_rewards_database[card_name]
                            rate = card_info['categories'].get(category, card_info['categories']['default'])
                            if rate > best_rate:
                                best_rate = rate
                                best_card = card_name
                        
                        st.markdown(f"• **{category}** (${spending:,.0f}/year): Use {best_card} ({best_rate*100:.1f}% rewards)")



def display_rewards_optimization_page(user_dataframes, selected_user_id):
    """Enhanced rewards optimization page with all portfolio optimization features"""
    st.title("💳 Advanced Rewards & Portfolio Optimization")
    st.markdown("*Comprehensive credit card portfolio optimization with real market data and actionable insights*")
    st.markdown("---")
    
    # Initialize rewards optimizer
    rewards_optimizer = RewardsOptimizer(
        user_dataframes.get('cards', pd.DataFrame()),
        user_dataframes.get('transactions', pd.DataFrame()),
        user_dataframes.get('mcc_codes', pd.DataFrame())
    )
    
    # Enhanced sidebar controls
    st.sidebar.header("🎛️ Optimization Controls")
    
    # Optimization parameters
    time_period_options = {
        "Last 3 Months": "3_months",
        "Last 6 Months": "6_months", 
        "Last Year": "1_year"
    }
    
    selected_period_display = st.sidebar.selectbox(
        "📅 Analysis Period",
        options=list(time_period_options.keys()),
        index=2
    )
    
    time_period = time_period_options[selected_period_display]
    
    max_cards = st.sidebar.slider(
        "🃏 Maximum Cards in Portfolio",
        min_value=1,
        max_value=5,
        value=3,
        help="Maximum number of cards to recommend for optimal portfolio"
    )
    
    # Advanced optimization options
    st.sidebar.subheader("⚙️ Advanced Options")
    include_annual_fees = st.sidebar.checkbox("💳 Include cards with annual fees", value=True)
    prioritize_signup_bonuses = st.sidebar.checkbox("🎁 Prioritize signup bonuses", value=True)
    show_detailed_analysis = st.sidebar.checkbox("📊 Show detailed analysis", value=True)
    
    # Get optimization results
    with st.spinner("🔄 Optimizing your credit card portfolio..."):
        optimization = rewards_optimizer.optimize_card_portfolio(selected_user_id, time_period, max_cards)
    
    if not optimization:
        st.warning("⚠️ Unable to perform portfolio optimization due to limited transaction data.")
        st.info("💡 Try selecting a different time period or ensure you have sufficient transaction history.")
        return
    
    # Main content tabs - Enhanced with more comprehensive analysis
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Optimization Results", 
        "📊 Current vs Optimal", 
        "💰 Card Comparison",
        "📈 Impact Analysis",
        "🔮 Recommendations"
    ])
    
    with tab1:
        display_optimization_overview(optimization)
    
    with tab2:
        display_current_vs_optimal(optimization, rewards_optimizer)
    
    with tab3:
        display_card_comparison_table(optimization, rewards_optimizer)
    
    with tab4:
        display_impact_analysis(optimization)
    
    with tab5:
        display_actionable_recommendations(optimization, user_dataframes)

# Optimized Pandas-based KPI Calculator (same as before)
class OptimizedKPICalculator:
    def __init__(self, dataframes):
        self.cards_df = dataframes.get('cards', pd.DataFrame())
        self.transactions_df = dataframes.get('transactions', pd.DataFrame())
        self.users_df = dataframes.get('users', pd.DataFrame())
        self.mcc_df = dataframes.get('mcc_codes', pd.DataFrame())
        self.fraud_df = dataframes.get('fraud_labels', pd.DataFrame())
    
    def get_user_profile(self, user_id):
        """Get user profile information"""
        if self.users_df.empty:
            return None
            
        user_profile = self.users_df[self.users_df['id'] == user_id]
        
        if user_profile.empty:
            return None
        
        user = user_profile.iloc[0]
        
        # Get user's cards
        if not self.cards_df.empty:
            user_cards = self.cards_df[self.cards_df['client_id'] == user_id]
        else:
            user_cards = pd.DataFrame()
        
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
        
        for _, card in user_cards.iterrows():
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
        if self.transactions_df.empty:
            return 0.0
            
        user_transactions = self.transactions_df[self.transactions_df['client_id'] == user_id]
        
        if user_transactions.empty:
            return 0.0
        
        # Convert date column to datetime if it's not already
        user_transactions = user_transactions.copy()
        user_transactions['date'] = pd.to_datetime(user_transactions['date'])
        
        # Get latest date
        latest_date = user_transactions['date'].max()
        start_date = latest_date - pd.Timedelta(days=30 * months_back)
        
        # Filter for spending in the period
        period_spending = user_transactions[
            (user_transactions['date'] >= start_date) &
            (user_transactions['date'] <= latest_date) &
            (user_transactions['amount'] > 0)
        ]
        
        return period_spending['amount'].sum() if not period_spending.empty else 0.0
    
    def calculate_credit_utilization(self, user_id):
        """Calculate average credit utilization percentage"""
        if self.cards_df.empty:
            return 0.0
            
        user_cards = self.cards_df[self.cards_df['client_id'] == user_id]
        
        if user_cards.empty:
            return 0.0
        
        total_credit_limit = user_cards['credit_limit'].sum()
        
        if total_credit_limit == 0:
            return 0.0
        
        current_spending = self.calculate_monthly_spending(user_id, 1)
        utilization = (current_spending / total_credit_limit) * 100
        
        return min(utilization, 100.0)
    
    def calculate_interest_paid(self, user_id, months_back=12):
        """Estimate interest paid based on debt and typical credit card APR"""
        if self.users_df.empty:
            return 0.0
            
        user_data = self.users_df[self.users_df['id'] == user_id]
        
        if user_data.empty:
            return 0.0
        
        total_debt = user_data.iloc[0]['total_debt']
        
        if pd.isna(total_debt) or total_debt == 0:
            return 0.0
        
        estimated_annual_interest = total_debt * 0.18
        monthly_interest = estimated_annual_interest / 12
        
        return monthly_interest * months_back
    
    def calculate_rewards_earned(self, user_id, months_back=12):
        """Calculate estimated rewards earned (assuming 1% cashback)"""
        total_spending = 0
        for month in range(months_back):
            monthly_spend = self.calculate_monthly_spending(user_id, month + 1)
            total_spending += monthly_spend
        
        rewards = total_spending * 0.01
        return rewards
    
    def get_spending_by_category(self, user_id, months_back=3):
        """Get spending breakdown by merchant category"""
        if self.transactions_df.empty:
            return []
            
        user_transactions = self.transactions_df[self.transactions_df['client_id'] == user_id]
        
        if user_transactions.empty:
            return []
        
        user_transactions = user_transactions.copy()
        user_transactions['date'] = pd.to_datetime(user_transactions['date'])
        latest_date = user_transactions['date'].max()
        start_date = latest_date - pd.Timedelta(days=30 * months_back)
        
        # Filter transactions
        period_transactions = user_transactions[
            (user_transactions['date'] >= start_date) &
            (user_transactions['amount'] > 0)
        ]
        
        if period_transactions.empty:
            return []
        
        # Join with MCC codes if available
        if not self.mcc_df.empty:
            spending_with_categories = period_transactions.merge(
                self.mcc_df, on='mcc', how='left'
            )
            spending_with_categories['category'] = spending_with_categories['category'].fillna('Other')
        else:
            spending_with_categories = period_transactions.copy()
            spending_with_categories['category'] = 'Other'
        
        # Group by category
        category_spending = spending_with_categories.groupby('category').agg({
            'amount': 'sum',
            'id': 'count'
        }).reset_index()
        
        category_spending.columns = ['category', 'amount', 'transactions']
        category_spending = category_spending.sort_values('amount', ascending=False)
        
        return category_spending.to_dict('records')
    
    def get_fraud_risk_score(self, user_id):
        """Calculate fraud risk score based on transaction patterns"""
        if self.transactions_df.empty:
            return 0.0
            
        user_transactions = self.transactions_df[self.transactions_df['client_id'] == user_id]
        
        if user_transactions.empty:
            return 0.0
        
        # Join with fraud labels if available
        if not self.fraud_df.empty:
            fraud_analysis = user_transactions.merge(
                self.fraud_df, left_on='id', right_on='transaction_id', how='left'
            )
            
            total_transactions = len(fraud_analysis)
            fraud_transactions = fraud_analysis['is_fraud'].sum() if not fraud_analysis['is_fraud'].isna().all() else 0
            
            if total_transactions == 0:
                return 0.0
            
            fraud_percentage = (fraud_transactions / total_transactions) * 100
            return fraud_percentage
        else:
            return 0.0
    
    def get_monthly_trend(self, user_id, months=6):
        """Get monthly spending trend"""
        if self.transactions_df.empty:
            return []
            
        user_transactions = self.transactions_df[self.transactions_df['client_id'] == user_id]
        
        if user_transactions.empty:
            return []
        
        user_transactions = user_transactions.copy()
        user_transactions['date'] = pd.to_datetime(user_transactions['date'])
        latest_date = user_transactions['date'].max()
        
        trends = []
        for i in range(months):
            month_start = latest_date - pd.Timedelta(days=30 * (i + 1))
            month_end = latest_date - pd.Timedelta(days=30 * i)
            
            monthly_transactions = user_transactions[
                (user_transactions['date'] >= month_start) &
                (user_transactions['date'] < month_end) &
                (user_transactions['amount'] > 0)
            ]
            
            monthly_spend = monthly_transactions['amount'].sum() if not monthly_transactions.empty else 0.0
            
            trends.append({
                'month': f"{month_start.strftime('%B')} {month_start.year}",
                'spending': monthly_spend
            })
        
        return list(reversed(trends))
    
    def get_all_kpis(self, user_id):
        """Get all KPIs for a user"""
        profile = self.get_user_profile(user_id)
        credit_score = profile['credit_score'] if profile else 0
        
        return {
            'total_monthly_spending': self.calculate_monthly_spending(user_id),
            'credit_utilization': self.calculate_credit_utilization(user_id),
            'interest_paid': self.calculate_interest_paid(user_id),
            'rewards_earned': self.calculate_rewards_earned(user_id),
            'credit_score': credit_score,
            'spending_by_category': self.get_spending_by_category(user_id),
            'fraud_risk_score': self.get_fraud_risk_score(user_id),
            'monthly_trend': self.get_monthly_trend(user_id)
        }

def main():
    """Main dashboard function with complete navigation"""
    
    # Enhanced Navigation with feature badges
    st.sidebar.title("🚀 Navigation")
    
    # Navigation options with feature descriptions
    page_options = {
        "📊 Overview": {
            "description": "Basic KPIs and user profile",
            "features": ["User Profile", "KPI Cards", "Basic Charts"]
        },
        "🛍️ Spending Analysis": {
            "description": "Detailed spending breakdown",  
            "features": ["Category Analysis", "Time Filtering", "Export Data"]
        },
        "🏪 Merchant Intelligence": {
            "description": "Advanced merchant insights",
            "features": ["Subscription Detection", "Loyalty Analysis", "Merchant Patterns"]
        },
        "💳 Rewards Optimization": {
            "description": "Credit card portfolio optimization",
            "features": ["Portfolio Analysis", "Signup Bonuses", "Break-even Calculations"]
        },
        "🔍 Advanced Intelligence": {
            "description": "AI-powered comprehensive analysis",
            "features": ["Predictive Analytics", "Anomaly Detection", "Deep Insights"]
        }
    }
    
    # Display navigation with descriptions
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        options=list(page_options.keys()),
        index=0,
        format_func=lambda x: f"{x}"
    )
    
    # Show page description and features
    page_info = page_options[page]
    st.sidebar.markdown(f"**{page_info['description']}**")
    
    for feature in page_info['features']:
        st.sidebar.markdown(f"• {feature}")
    
    st.sidebar.markdown("---")
    
    # Load user list
    user_list = get_user_list()
    
    if not user_list:
        st.error("Failed to load user list. Please check your data files.")
        return
    
    # User selection
    st.sidebar.header("👤 User Selection")
    
    user_options = {user[1]: user[0] for user in user_list}
    selected_user_display = st.sidebar.selectbox(
        "Select User for Analysis",
        options=list(user_options.keys()),
        index=0
    )
    selected_user_id = user_options[selected_user_display]
    
    # Load user-specific data with progress indicator
    with st.spinner(f"🔄 Loading data for {selected_user_display.split(' - ')[0]}..."):
        user_dataframes = load_user_data(selected_user_id)
    
    if not user_dataframes:
        st.error(f"❌ Failed to load data for {selected_user_display}")
        return
    
    # Success indicator
    st.sidebar.success(f"✅ Data loaded for User {selected_user_id}")
    
    # Display selected page
    if page == "📊 Overview":
        st.title("💳 Credit Card Optimization Dashboard")
        st.markdown("*Comprehensive financial overview and key performance indicators*")
        st.markdown("---")
        display_overview_page(user_dataframes, selected_user_id)
        
        # Sidebar info for overview
        st.sidebar.markdown("---")
        st.sidebar.subheader("📋 Overview Features")
        st.sidebar.info("""
        **Current Page Includes:**
        • User profile and demographics
        • Key financial metrics (KPIs)
        • Spending category breakdown
        • Fraud risk assessment
        • Monthly spending trends
        • Credit utilization tracking
        """)
        
    elif page == "🛍️ Spending Analysis":
        display_spending_analysis_page(user_dataframes, selected_user_id)
        
    elif page == "🏪 Merchant Intelligence":
        display_merchant_analysis_page(user_dataframes, selected_user_id)
        
    elif page == "💳 Rewards Optimization":
        display_rewards_optimization_page(user_dataframes, selected_user_id)
        
    elif page == "🔍 Advanced Intelligence":
        display_advanced_spending_analysis(user_dataframes, selected_user_id)
    
    # Performance metrics in sidebar
    if st.sidebar.checkbox("📊 Show Performance Metrics"):
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚡ Performance Info")
        
        # Data size metrics
        transaction_count = len(user_dataframes.get('transactions', pd.DataFrame()))
        card_count = len(user_dataframes.get('cards', pd.DataFrame()))
        
        st.sidebar.metric("📊 Transactions Loaded", f"{transaction_count:,}")
        st.sidebar.metric("💳 Cards Loaded", f"{card_count}")
        
        # Memory optimization indicator
        if transaction_count > 0:
            st.sidebar.success("🚀 Memory Optimized: Loading only user-specific data")
        else:
            st.sidebar.warning("⚠️ No transaction data found for this user")
    
    # Feature toggles
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Advanced Settings")
    
    # Export options
    if st.sidebar.button("📤 Export All Analysis"):
        st.sidebar.info("Export functionality would download comprehensive analysis reports")
    
    # Data refresh
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()
    
    # Help section
    with st.sidebar.expander("❓ Help & Tips"):
        st.markdown("""
        **Navigation Tips:**
        • Start with Overview for basic insights
        • Use Spending Analysis for detailed breakdowns
        • Try Merchant Intelligence for subscription insights
        • Check Rewards Optimization for card recommendations
        • Explore Advanced Intelligence for AI-powered insights
        
        **Performance:**
        • Data loads only for selected user
        • All analysis runs locally and privately
        • Charts are interactive - hover and click to explore
        """)
    
    # Footer with version info
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("*🔒 Privacy-First Analytics*")
    with col2:
        st.markdown("*⚡ Memory Optimized*")
    with col3:
        st.markdown("*🚀 Advanced Intelligence Built-In*")
    
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <small>
            Dashboard built with Streamlit • PySpark • Advanced Analytics<br>
            Your financial data stays private and secure
        </small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()