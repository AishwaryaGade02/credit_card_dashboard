import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from spending_analyzer import SpendingAnalyzer


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from spending_analyzer import SpendingAnalyzer
# REMOVED: from merchant_analyzer import MerchantAnalyzer

def display_advanced_spending_analysis(user_dataframes, selected_user_id):
    """Display the advanced spending analysis page with enhanced features (Merchant Intelligence moved to separate page)"""
    
    st.title("🔍 Advanced Spending Intelligence")
    st.markdown("*Comprehensive analysis with AI-powered insights and predictive analytics*")
    st.markdown("---")
    
    # Initialize analyzers
    spending_analyzer = SpendingAnalyzer(user_dataframes)
    # REMOVED: merchant_analyzer initialization
    
    # Enhanced sidebar controls
    st.sidebar.header("🎛️ Analysis Controls")
    
    # Time period selection with more options
    time_period_options = {
        "Last 30 Days": "1_month",
        "Last 3 Months": "3_months", 
        "Last 6 Months": "6_months",
        "Last Year": "1_year",
        "Custom Range": "custom"
    }
    
    selected_period_display = st.sidebar.selectbox(
        "📅 Analysis Period",
        options=list(time_period_options.keys()),
        index=1
    )
    
    selected_period = time_period_options[selected_period_display]
    
    # Custom date range
    start_date, end_date = None, None
    if selected_period == "custom":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start Date")
        with col2:
            end_date = st.date_input("End Date")
    
    # Analysis depth selection
    analysis_depth = st.sidebar.selectbox(
        "🔬 Analysis Depth",
        ["Quick Overview", "Standard Analysis", "Deep Dive"],
        index=1
    )
    
    # Feature toggles - REMOVED merchant intelligence
    st.sidebar.subheader("📊 Features")
    show_spending_predictions = st.sidebar.checkbox("📈 Spending Predictions", value=False)
    show_anomaly_detection = st.sidebar.checkbox("🚨 Anomaly Detection", value=False)
    
    # Updated main content tabs - REMOVED Merchant Intelligence
    tab1, tab2, tab3 = st.tabs([
        "📊 Overview Dashboard", 
        "🛍️ Category Deep Dive", 
        "🔮 Predictive Insights"
    ])
    
    with tab1:
        display_overview_dashboard(
            spending_analyzer, selected_user_id, selected_period, start_date, end_date
        )
    
    with tab2:
        display_category_deep_dive(
            spending_analyzer, selected_user_id, selected_period, start_date, end_date
        )
    
    with tab3:
        if show_spending_predictions or show_anomaly_detection:
            display_predictive_insights(
                spending_analyzer, selected_user_id, selected_period,
                show_spending_predictions, show_anomaly_detection
            )
        else:
            st.info("Enable Predictive features in the sidebar to view this analysis.")
    
    # Add helpful note about Merchant Intelligence
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Merchant Intelligence** is available as a dedicated page in the main navigation for comprehensive merchant analysis.")

# Keep all other functions (display_overview_dashboard, display_category_deep_dive, display_predictive_insights) exactly the same
# REMOVE the display_merchant_intelligence function entirely from this file

# Keep all the existing display functions exactly the same
def display_overview_dashboard(analyzer, user_id, time_period, start_date=None, end_date=None):
    """Enhanced overview dashboard with key metrics"""
    
    st.subheader("📊 Spending Overview Dashboard")
    
    # Get comparison metrics
    comparison_metrics = analyzer.get_comparison_metrics(user_id, time_period)
    category_data = analyzer.get_category_breakdown(user_id, time_period, start_date, end_date)
    
    if comparison_metrics:
        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            spending_change = comparison_metrics.get('spending_change_pct', 0)
            st.metric(
                "💰 Total Spending",
                f"${comparison_metrics.get('current_spending', 0):,.2f}",
                delta=f"{spending_change:+.1f}%",
                delta_color="inverse" if spending_change > 0 else "normal"
            )
        
        with col2:
            transaction_change = comparison_metrics.get('transaction_change_pct', 0)
            st.metric(
                "🛒 Transactions",
                f"{comparison_metrics.get('current_transactions', 0):,}",
                delta=f"{transaction_change:+.1f}%"
            )
        
        with col3:
            avg_current = comparison_metrics.get('avg_transaction_current', 0)
            avg_previous = comparison_metrics.get('avg_transaction_previous', 0)
            avg_change = ((avg_current - avg_previous) / avg_previous * 100) if avg_previous > 0 else 0
            st.metric(
                "📊 Avg Transaction",
                f"${avg_current:.2f}",
                delta=f"{avg_change:+.1f}%"
            )
        
        with col4:
            # Calculate spending velocity (transactions per day)
            days_in_period = 30 if time_period == '1_month' else 90
            velocity = comparison_metrics.get('current_transactions', 0) / days_in_period
            st.metric(
                "⚡ Spending Velocity",
                f"{velocity:.1f}/day",
                delta="transactions per day"
            )
        
        with col5:
            # Calculate diversity score (number of categories used)
            diversity_score = len(category_data) if not category_data.empty else 0
            st.metric(
                "🎯 Category Diversity",
                f"{diversity_score}",
                delta="active categories"
            )
    
    st.markdown("---")
    
    # Enhanced visualizations
    if not category_data.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Interactive spending distribution with drill-down
            fig = px.sunburst(
                category_data.head(10),
                names='category',
                values='total_spent',
                title="Interactive Spending Distribution",
                color='total_spent',
                color_continuous_scale='viridis'
            )
            fig.update_traces(
                hovertemplate="<b>%{label}</b><br>" +
                            "Amount: $%{value:,.2f}<br>" +
                            "Percentage: %{percentParent}<br>" +
                            "<extra></extra>"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top categories with enhanced metrics
            st.subheader("🏆 Top Categories")
            for i, row in category_data.head(5).iterrows():
                with st.container():
                    st.markdown(f"**{row['category']}**")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Amount", f"${row['total_spent']:,.2f}")
                    with col_b:
                        st.metric("Share", f"{row['percentage']:.1f}%")
                    st.progress(row['percentage'] / 100)
                    st.markdown("---")
    
    # Spending trends with forecasting
    trends_data = analyzer.get_spending_trends(user_id, "1_year")
    if not trends_data.empty:
        st.subheader("📈 Spending Trends & Patterns")
        
        # Create enhanced trend visualization
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Monthly Spending by Category', 'Total Monthly Spending'),
            specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
        )
        
        # Category trends
        for category in trends_data['category'].unique()[:5]:  # Top 5 categories
            category_data_filtered = trends_data[trends_data['category'] == category]
            fig.add_trace(
                go.Scatter(
                    x=category_data_filtered['month_year_str'],
                    y=category_data_filtered['amount'],
                    name=category,
                    mode='lines+markers'
                ),
                row=1, col=1
            )
        
        # Total spending trend
        monthly_totals = trends_data.groupby('month_year_str')['amount'].sum().reset_index()
        fig.add_trace(
            go.Scatter(
                x=monthly_totals['month_year_str'],
                y=monthly_totals['amount'],
                name='Total Spending',
                mode='lines+markers',
                line=dict(width=3, color='red')
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            title="Comprehensive Spending Analysis",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_category_deep_dive(analyzer, user_id, time_period, start_date=None, end_date=None):
    """Deep dive into category analysis"""
    
    st.subheader("🛍️ Category Deep Dive Analysis")
    
    category_data = analyzer.get_category_breakdown(user_id, time_period, start_date, end_date)
    
    if category_data.empty:
        st.warning("No category data available for the selected period.")
        return
    
    # Category selection for detailed analysis
    selected_category = st.selectbox(
        "Select Category for Detailed Analysis",
        options=category_data['category'].tolist(),
        index=0
    )
    
    # Category-specific insights
    category_info = category_data[category_data['category'] == selected_category].iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Spent", f"${category_info['total_spent']:,.2f}")
    with col2:
        st.metric("Transactions", f"{category_info['transaction_count']:,}")
    with col3:
        st.metric("Avg Transaction", f"${category_info['avg_transaction']:,.2f}")
    with col4:
        st.metric("% of Total", f"{category_info['percentage']:.1f}%")
    
    # Enhanced category visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Category comparison chart
        fig_comparison = px.bar(
            category_data.head(10),
            x='total_spent',
            y='category',
            orientation='h',
            title="Category Spending Comparison",
            color='total_spent',
            color_continuous_scale='plasma',
            text='total_spent'
        )
        fig_comparison.update_traces(
            texttemplate='$%{text:,.0f}',
            textposition='inside'
        )
        fig_comparison.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    with col2:
        # Transaction frequency vs amount scatter
        fig_scatter = px.scatter(
            category_data,
            x='transaction_count',
            y='avg_transaction',
            size='total_spent',
            hover_name='category',
            title="Transaction Patterns by Category",
            labels={
                'transaction_count': 'Number of Transactions',
                'avg_transaction': 'Average Transaction ($)',
                'total_spent': 'Total Spent'
            }
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Category insights and recommendations
    st.subheader(f"💡 Insights for {selected_category}")
    
    # Generate insights based on category data
    insights = []
    
    if category_info['avg_transaction'] > category_data['avg_transaction'].mean():
        insights.append(f"Your average {selected_category} transaction (${category_info['avg_transaction']:.2f}) is above your overall average.")
    
    if category_info['percentage'] > 20:
        insights.append(f"{selected_category} represents a significant portion ({category_info['percentage']:.1f}%) of your spending. Consider if this aligns with your priorities.")
    
    if category_info['transaction_count'] > category_data['transaction_count'].quantile(0.75):
        insights.append(f"You make frequent {selected_category} purchases ({category_info['transaction_count']} transactions). This might be a good category for a specialized rewards card.")
    
    for insight in insights:
        st.info(f"ℹ️ {insight}")

def display_merchant_intelligence(analyzer, user_id, time_period):
    """Display advanced merchant intelligence"""
    
    st.subheader("🏪 Merchant Intelligence Dashboard")
    
    # Get merchant insights
    merchant_insights = analyzer.get_merchant_insights(user_id, time_period)
    
    if not merchant_insights or merchant_insights.get('merchant_stats') is None:
        st.warning("No merchant data available for analysis.")
        return
    
    merchant_stats = merchant_insights['merchant_stats']
    
    # Merchant overview metrics
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
    
    # Merchant analysis tabs
    tab1, tab2, tab3 = st.tabs(["📊 Top Merchants", "🔄 Subscriptions", "🎯 Loyalty Opportunities"])
    
    with tab1:
        # Enhanced merchant visualization
        if not merchant_stats.empty:
            top_merchants = merchant_stats.head(10)
            
            fig = px.treemap(
                top_merchants,
                names='merchant_name',
                values='total_spent',
                title="Merchant Spending Treemap",
                color='consistency_score',
                color_continuous_scale='RdYlGn',
                hover_data=['transaction_count', 'avg_amount']
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed merchant table
            st.subheader("📋 Merchant Details")
            st.dataframe(
                top_merchants[['merchant_name', 'total_spent', 'transaction_count', 
                             'avg_amount', 'consistency_score', 'frequency_score']],
                column_config={
                    "merchant_name": "Merchant",
                    "total_spent": st.column_config.NumberColumn("Total Spent", format="$%.2f"),
                    "transaction_count": "Transactions",
                    "avg_amount": st.column_config.NumberColumn("Avg Amount", format="$%.2f"),
                    "consistency_score": st.column_config.NumberColumn("Consistency", format="%.3f"),
                    "frequency_score": st.column_config.NumberColumn("Frequency", format="%.2f")
                },
                use_container_width=True
            )
    
    with tab2:
        # Subscription analysis
        subscriptions = merchant_insights.get('subscription_merchants', pd.DataFrame())
        
        if not subscriptions.empty:
            st.subheader("🔄 Subscription Services Detected")
            
            total_monthly = subscriptions['estimated_monthly_cost'].sum()
            st.metric("Total Monthly Subscriptions", f"${total_monthly:.2f}")
            
            # Subscription visualization
            fig_subs = px.bar(
                subscriptions,
                x='merchant_name',
                y='estimated_monthly_cost',
                title="Monthly Subscription Costs",
                color='estimated_monthly_cost',
                color_continuous_scale='reds'
            )
            fig_subs.update_xaxes(tickangle=45)
            st.plotly_chart(fig_subs, use_container_width=True)
            
            # Subscription optimization suggestions
            st.subheader("💡 Subscription Optimization")
            high_cost_subs = subscriptions[subscriptions['estimated_monthly_cost'] > 20]
            
            if not high_cost_subs.empty:
                st.warning(f"You have {len(high_cost_subs)} high-cost subscriptions (>${20}/month). Consider reviewing these for potential savings.")
                
                for _, sub in high_cost_subs.iterrows():
                    st.info(f"📍 {sub['merchant_name']}: ~${sub['estimated_monthly_cost']:.2f}/month")
        else:
            st.info("No subscription patterns detected in your spending.")
    
    with tab3:
        # Loyalty opportunities
        loyalty_ops = merchant_insights.get('loyalty_opportunities', pd.DataFrame())
        
        if not loyalty_ops.empty:
            st.subheader("🎯 Loyalty Program Opportunities")
            
            # Loyalty score visualization
            fig_loyalty = px.scatter(
                loyalty_ops.head(15),
                x='frequency_score',
                y='total_spent',
                size='loyalty_score',
                hover_name='merchant_name',
                title="Loyalty Opportunity Matrix",
                labels={
                    'frequency_score': 'Visit Frequency',
                    'total_spent': 'Total Spending ($)',
                    'loyalty_score': 'Loyalty Score'
                },
                color='loyalty_score',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_loyalty, use_container_width=True)
            
            # Top loyalty recommendations
            st.subheader("🏆 Top Loyalty Recommendations")
            for _, merchant in loyalty_ops.head(5).iterrows():
                with st.expander(f"💳 {merchant['merchant_name']} - Loyalty Score: {merchant['loyalty_score']:.1f}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Spent", f"${merchant['total_spent']:,.2f}")
                    with col2:
                        st.metric("Visits", f"{merchant['transaction_count']}")
                    with col3:
                        st.metric("Frequency", f"{merchant['frequency_score']:.1f}/month")
                    
                    potential_rewards = merchant['total_spent'] * 0.02  # Assume 2% rewards
                    st.success(f"💰 Potential annual rewards: ~${potential_rewards:.2f}")
        else:
            st.info("No specific loyalty opportunities identified.")

def display_predictive_insights(analyzer, user_id, time_period, show_predictions, show_anomalies):
    """Display predictive insights and anomaly detection"""
    
    st.subheader("🔮 Predictive Insights & Anomaly Detection")
    
    if show_predictions:
        st.subheader("📈 Spending Predictions")
        
        # Get historical data for prediction
        trends_data = analyzer.get_spending_trends(user_id, "1_year")
        
        if not trends_data.empty:
            # Simple linear trend prediction (you can enhance with more sophisticated models)
            monthly_totals = trends_data.groupby('month_year_str')['amount'].sum().reset_index()
            
            if len(monthly_totals) >= 3:
                # Calculate trend
                x = np.arange(len(monthly_totals))
                y = monthly_totals['amount'].values
                
                # Simple linear regression
                slope, intercept = np.polyfit(x, y, 1)
                
                # Predict next 3 months
                future_months = ['Next Month', 'Month +2', 'Month +3']
                predictions = []
                
                for i in range(3):
                    pred_value = slope * (len(monthly_totals) + i) + intercept
                    predictions.append(max(0, pred_value))  # Ensure non-negative
                
                # Visualization
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=monthly_totals['month_year_str'],
                    y=monthly_totals['amount'],
                    mode='lines+markers',
                    name='Historical Spending',
                    line=dict(color='blue', width=2)
                ))
                
                # Predictions
                fig.add_trace(go.Scatter(
                    x=future_months,
                    y=predictions,
                    mode='lines+markers',
                    name='Predicted Spending',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title="Spending Prediction Based on Historical Trends",
                    xaxis_title="Time Period",
                    yaxis_title="Spending ($)",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Prediction summary
                avg_prediction = np.mean(predictions)
                trend_direction = "increasing" if slope > 0 else "decreasing"
                
                st.info(f"📊 **Prediction Summary**: Your spending trend is {trend_direction}. "
                       f"Expected average monthly spending: ${avg_prediction:.2f}")
                
            else:
                st.warning("Not enough historical data for reliable predictions.")
        else:
            st.warning("No historical data available for predictions.")
    
    if show_anomalies:
        st.subheader("🚨 Anomaly Detection")
        
        # Get transaction data
        if not analyzer.transactions_df.empty:
            user_transactions = analyzer.transactions_df[
                (analyzer.transactions_df['client_id'] == user_id) &
                (analyzer.transactions_df['amount'] > 0)
            ].copy()
            
            if not user_transactions.empty:
                user_transactions['date'] = pd.to_datetime(user_transactions['date'])
                
                # Detect amount anomalies (transactions significantly higher than usual)
                mean_amount = user_transactions['amount'].mean()
                std_amount = user_transactions['amount'].std()
                threshold = mean_amount + (2 * std_amount)  # 2 standard deviations
                
                anomalies = user_transactions[user_transactions['amount'] > threshold]
                
                if not anomalies.empty:
                    st.warning(f"🚨 Detected {len(anomalies)} unusual transactions (significantly above average)")
                    
                    # Show anomalies
                    anomaly_display = anomalies[['date', 'amount', 'merchant_id']].sort_values('amount', ascending=False)
                    anomaly_display['amount_formatted'] = anomaly_display['amount'].apply(lambda x: f"${x:,.2f}")
                    
                    st.dataframe(
                        anomaly_display[['date', 'amount_formatted', 'merchant_id']],
                        column_config={
                            "date": "Date",
                            "amount_formatted": "Amount",
                            "merchant_id": "Merchant ID"
                        },
                        use_container_width=True
                    )
                    
                    # Anomaly visualization
                    fig_anomaly = px.scatter(
                        user_transactions.tail(100),  # Last 100 transactions
                        x='date',
                        y='amount',
                        title="Transaction Amount Timeline with Anomalies",
                        hover_data=['merchant_id']
                    )
                    
                    # Highlight anomalies
                    fig_anomaly.add_hline(
                        y=threshold, 
                        line_dash="dash", 
                        line_color="red",
                        annotation_text="Anomaly Threshold"
                    )
                    
                    st.plotly_chart(fig_anomaly, use_container_width=True)
                    
                else:
                    st.success("✅ No unusual spending patterns detected in your recent transactions.")
            else:
                st.info("No transaction data available for anomaly detection.")
        else:
            st.info("No transaction data available for anomaly detection.")