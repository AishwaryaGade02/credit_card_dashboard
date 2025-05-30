import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from spending_analyzer import SpendingAnalyzer

def display_spending_analysis_page(user_dataframes, selected_user_id):
    """Display the spending analysis page - REMOVED rewards optimization section"""
    
    st.title("📊 Advanced Spending Analysis")
    st.markdown("---")
    
    # Initialize spending analyzer
    analyzer = SpendingAnalyzer(user_dataframes)
    
    # Sidebar controls for filtering
    st.sidebar.header("🔧 Analysis Controls")
    
    # Time period selection
    time_period_options = {
        "Last Month": "1_month",
        "Last 3 Months": "3_months", 
        "Last 6 Months": "6_months",
        "Last Year": "1_year",
        "Custom Range": "custom"
    }
    
    selected_period_display = st.sidebar.selectbox(
        "Select Time Period",
        options=list(time_period_options.keys()),
        index=1  # Default to 3 months
    )
    
    selected_period = time_period_options[selected_period_display]
    
    # Custom date range if selected
    start_date, end_date = None, None
    if selected_period == "custom":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start Date")
        with col2:
            end_date = st.date_input("End Date")
    
    # Comparison metrics at the top
    st.subheader("📈 Period Comparison")
    comparison_metrics = analyzer.get_comparison_metrics(selected_user_id, selected_period)
    
    if comparison_metrics:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            spending_change = comparison_metrics.get('spending_change_pct', 0)
            st.metric(
                "Total Spending",
                f"${comparison_metrics.get('current_spending', 0):,.2f}",
                delta=f"{spending_change:+.1f}%"
            )
        
        with col2:
            transaction_change = comparison_metrics.get('transaction_change_pct', 0)
            st.metric(
                "Transaction Count",
                f"{comparison_metrics.get('current_transactions', 0):,}",
                delta=f"{transaction_change:+.1f}%"
            )
        
        with col3:
            avg_current = comparison_metrics.get('avg_transaction_current', 0)
            avg_previous = comparison_metrics.get('avg_transaction_previous', 0)
            avg_change = ((avg_current - avg_previous) / avg_previous * 100) if avg_previous > 0 else 0
            st.metric(
                "Avg Transaction",
                f"${avg_current:.2f}",
                delta=f"{avg_change:+.1f}%"
            )
        
        with col4:
            st.metric(
                "Analysis Period",
                selected_period_display,
                delta="vs Previous Period"
            )
    
    st.markdown("---")
    
    # Spending Patterns Section
    st.subheader("🕐 Spending Patterns")
    
    patterns = analyzer.get_spending_patterns(selected_user_id, selected_period)
    
    if patterns and patterns.get('total_transactions', 0) > 0:
        # Charts in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            # Spending by day of week
            if not patterns['by_day_of_week'].empty:
                fig_dow = px.bar(
                    patterns['by_day_of_week'],
                    x='day_of_week',
                    y='amount',
                    title="Spending by Day of Week",
                    color='amount',
                    color_continuous_scale='blues'
                )
                # Order days properly
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                fig_dow.update_xaxes(categoryorder='array', categoryarray=day_order)
                st.plotly_chart(fig_dow, use_container_width=True)
        
        with col2:
            # Spending by hour of day
            if not patterns['by_hour'].empty:
                fig_hour = px.line(
                    patterns['by_hour'],
                    x='hour',
                    y='amount',
                    title="Spending by Hour of Day",
                    markers=True
                )
                fig_hour.update_layout(
                    xaxis_title="Hour (24h format)",
                    yaxis_title="Total Spending ($)"
                )
                st.plotly_chart(fig_hour, use_container_width=True)
        
        # Summary stats - Enhanced with largest and smallest transactions
        col3, col4 = st.columns(2)
        
        with col3:
            st.metric("Largest Transaction", f"${patterns['max_transaction']:.2f}")
        with col4:
            st.metric("Smallest Transaction", f"${patterns['min_transaction']:.2f}")
    else:
        st.info("No spending pattern data available for the selected period")
    
    st.markdown("---")
    
    # Get data for remaining sections
    category_data = analyzer.get_category_breakdown(selected_user_id, selected_period, start_date, end_date)
    merchant_data = analyzer.get_top_merchants(selected_user_id, selected_period, start_date, end_date)
    
    # Category Breakdown Section
    st.subheader("🛍️ Category Breakdown")
    
    if not category_data.empty:
        # Create tabs for different views
        tab1, tab2 = st.tabs(["📋 Detailed Table", "📈 Trends"])
        
        with tab1:
            # Detailed table
            st.dataframe(
                category_data[['category', 'total_spent', 'transaction_count', 'avg_transaction', 'percentage']],
                column_config={
                    "category": "Category",
                    "total_spent": st.column_config.NumberColumn(
                        "Total Spent",
                        format="$%.2f"
                    ),
                    "transaction_count": "Transactions",
                    "avg_transaction": st.column_config.NumberColumn(
                        "Avg Transaction",
                        format="$%.2f"
                    ),
                    "percentage": st.column_config.NumberColumn(
                        "Percentage",
                        format="%.1f%%"
                    )
                },
                use_container_width=True
            )
        
        with tab2:
            # Spending trends over time
            trends_data = analyzer.get_spending_trends(selected_user_id, "1_year")
            if not trends_data.empty:
                # Create a line chart for trends
                fig_trends = px.line(
                    trends_data,
                    x='month_year_str',
                    y='amount',
                    color='category',
                    title="Spending Trends by Category Over Time",
                    markers=True
                )
                fig_trends.update_layout(
                    xaxis_title="Month",
                    yaxis_title="Amount ($)",
                    hovermode='x unified'
                )
                st.plotly_chart(fig_trends, use_container_width=True)
            else:
                st.info("No trend data available for the selected period")
    else:
        st.info("No category data available for the selected period")
    
    st.markdown("---")
    
    # Top Merchants Section
    st.subheader("🏪 Top Merchants Analysis")
    
    if not merchant_data.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Horizontal bar chart for top merchants
            fig_merchants = px.bar(
                merchant_data,
                x='total_spent',
                y='merchant_name',
                orientation='h',
                title=f"Top 10 Merchants by Spending - {selected_period_display}",
                color='total_spent',
                color_continuous_scale='plasma',
                text='total_spent'
            )
            fig_merchants.update_traces(
                texttemplate='$%{text:,.0f}',
                textposition='inside'
            )
            fig_merchants.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig_merchants, use_container_width=True)
        
        with col2:
            st.subheader("📊 Merchant Insights")
            total_merchant_spending = merchant_data['total_spent'].sum()
            top_merchant = merchant_data.iloc[0]
            
            st.metric(
                "Top Merchant",
                top_merchant['merchant_name'],
                f"${top_merchant['total_spent']:,.2f}"
            )
            
            st.metric(
                "Top 3 Merchants",
                "% of Total Spending",
                f"{(merchant_data.head(3)['total_spent'].sum() / total_merchant_spending * 100):.1f}%"
            )
            
            st.metric(
                "Most Frequent",
                merchant_data.loc[merchant_data['transaction_count'].idxmax(), 'merchant_name'],
                f"{merchant_data['transaction_count'].max()} transactions"
            )
        
        # Detailed merchant table
        st.subheader("📋 Detailed Merchant Analysis")
        st.dataframe(
            merchant_data[['merchant_name', 'total_spent', 'transaction_count', 'avg_transaction']],
            column_config={
                "merchant_name": "Merchant",
                "total_spent": st.column_config.NumberColumn(
                    "Total Spent",
                    format="$%.2f"
                ),
                "transaction_count": "Transactions",
                "avg_transaction": st.column_config.NumberColumn(
                    "Avg Transaction",
                    format="$%.2f"
                )
            },
            use_container_width=True
        )
    else:
        st.info("No merchant data available for the selected period")
    
    st.markdown("---")
    
    # REMOVED: Rewards Optimization Section completely
    # This section has been moved to the dedicated Rewards Optimization page
    
    # Information note about rewards optimization
    st.info("💡 **Looking for rewards optimization?** Check out the dedicated **Rewards Optimization** page in the navigation menu for comprehensive credit card portfolio analysis and optimization insights.")
    
    st.markdown("---")
    
    # Export functionality
    st.subheader("📤 Export Analysis Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if not category_data.empty:
            csv_categories = category_data.to_csv(index=False)
            st.download_button(
                label="📊 Download Category Data",
                data=csv_categories,
                file_name=f"category_analysis_{selected_user_id}_{selected_period}.csv",
                mime="text/csv"
            )
    
    with col2:
        if not merchant_data.empty:
            csv_merchants = merchant_data.to_csv(index=False)
            st.download_button(
                label="🏪 Download Merchant Data",
                data=csv_merchants,
                file_name=f"merchant_analysis_{selected_user_id}_{selected_period}.csv",
                mime="text/csv"
            )
    
    with col3:
        if patterns:
            # Create a summary export of spending patterns
            patterns_summary = pd.DataFrame([
                {'Metric': 'Total Transactions', 'Value': patterns.get('total_transactions', 0)},
                {'Metric': 'Average Transaction', 'Value': f"${patterns.get('avg_transaction', 0):.2f}"},
                {'Metric': 'Largest Transaction', 'Value': f"${patterns.get('max_transaction', 0):.2f}"},
                {'Metric': 'Smallest Transaction', 'Value': f"${patterns.get('min_transaction', 0):.2f}"}
            ])
            csv_patterns = patterns_summary.to_csv(index=False)
            st.download_button(
                label="📈 Download Patterns Summary",
                data=csv_patterns,
                file_name=f"spending_patterns_{selected_user_id}_{selected_period}.csv",
                mime="text/csv"
            )