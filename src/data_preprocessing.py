import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pandas as pd

class DataPreprocessor:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("CreditCardDashboard") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.driver.memory", "2g") \
            .config("spark.driver.maxResultSize", "1g") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        self.data_path = None
    
    def set_data_path(self, data_path):
        """Set the data directory path"""
        self.data_path = data_path
    
    def clean_amount_column(self, df, amount_col='amount'):
        """Clean amount column by removing $ and , signs and converting to double"""
        return df.withColumn(
            amount_col,
            regexp_replace(col(amount_col), r'[\$,]', '').cast(DoubleType())
        )
    
    def clean_date_columns(self, df, date_cols):
        """Clean date columns and convert them to proper date format"""
        for date_col in date_cols:
            if date_col == 'expires' or date_col == 'acct_open_date':
                # Handle MM/YYYY format
                df = df.withColumn(
                    date_col,
                    to_date(concat(col(date_col), lit("/01")), "MM/yyyy/dd")
                )
            elif date_col == 'date':
                # Handle datetime format
                df = df.withColumn(
                    date_col,
                    to_timestamp(col(date_col), "yyyy-MM-dd HH:mm:ss")
                )
        return df
    
    def get_user_list(self):
        """Get list of unique users for dropdown - lightweight operation"""
        users_file = os.path.join(self.data_path, "users_cards.csv")
        
        # Read only id and address columns to minimize memory usage
        users_df = self.spark.read.option("header", "true").csv(users_file) \
            .select("id", "address") \
            .limit(1000)  # Limit to first 1000 users to prevent memory issues
        
        users_list = users_df.collect()
        return [(int(row['id']), f"User {row['id']} - {row['address']}") for row in users_list]
    
    def load_user_specific_data(self, user_id):
        """Load only data relevant to the specific user"""
        if not self.data_path:
            raise ValueError("Data path not set. Call set_data_path() first.")
        
        # Load user data (single row)
        users_file = os.path.join(self.data_path, "users_cards.csv")
        users_df = self.spark.read.option("header", "true").csv(users_file) \
            .filter(col("id") == user_id)
        
        if users_df.count() == 0:
            raise ValueError(f"User {user_id} not found")
        
        # Clean users data
        users_df = self.clean_amount_column(users_df, 'per_capita_income') \
                   .withColumn('yearly_income', regexp_replace(col('yearly_income'), r'[\$,]', '').cast(DoubleType())) \
                   .withColumn('total_debt', regexp_replace(col('total_debt'), r'[\$,]', '').cast(DoubleType())) \
                   .withColumn("id", col("id").cast(IntegerType())) \
                   .withColumn("current_age", col("current_age").cast(IntegerType())) \
                   .withColumn("retirement_age", col("retirement_age").cast(IntegerType())) \
                   .withColumn("birth_year", col("birth_year").cast(IntegerType())) \
                   .withColumn("birth_month", col("birth_month").cast(IntegerType())) \
                   .withColumn("latitude", col("latitude").cast(DoubleType())) \
                   .withColumn("longitude", col("longitude").cast(DoubleType())) \
                   .withColumn("credit_score", col("credit_score").cast(IntegerType())) \
                   .withColumn("num_credit_cards", col("num_credit_cards").cast(IntegerType()))
        
        # Load user's cards only
        cards_file = os.path.join(self.data_path, "cards_data.csv")
        cards_df = self.spark.read.option("header", "true").csv(cards_file) \
            .filter(col("client_id") == user_id)
        
        # Clean cards data
        cards_df = self.clean_amount_column(cards_df, 'credit_limit') \
                  .withColumn("id", col("id").cast(IntegerType())) \
                  .withColumn("client_id", col("client_id").cast(IntegerType())) \
                  .withColumn("cvv", col("cvv").cast(IntegerType())) \
                  .withColumn("num_cards_issued", col("num_cards_issued").cast(IntegerType())) \
                  .withColumn("year_pin_last_changed", col("year_pin_last_changed").cast(IntegerType()))
        
        cards_df = self.clean_date_columns(cards_df, ['expires', 'acct_open_date'])
        
        # Load user's transactions only
        transactions_file = os.path.join(self.data_path, "transaction_data.csv")
        transactions_df = self.spark.read.option("header", "true").csv(transactions_file) \
            .filter(col("client_id") == user_id)
        
        # Clean transactions data
        transactions_df = self.clean_amount_column(transactions_df, 'amount') \
                         .withColumn("id", col("id").cast(IntegerType())) \
                         .withColumn("client_id", col("client_id").cast(IntegerType())) \
                         .withColumn("card_id", col("card_id").cast(IntegerType())) \
                         .withColumn("merchant_id", col("merchant_id").cast(IntegerType())) \
                         .withColumn("zip", col("zip").cast(DoubleType())) \
                         .withColumn("mcc", col("mcc").cast(IntegerType()))
        
        transactions_df = self.clean_date_columns(transactions_df, ['date'])
        
        # Add derived columns
        transactions_df = transactions_df.withColumn("year", year(col("date"))) \
                                       .withColumn("month", month(col("date"))) \
                                       .withColumn("day", dayofmonth(col("date")))
        
        # Load MCC codes (small reference table)
        mcc_file = os.path.join(self.data_path, "mcc_codes.csv")
        mcc_df = self.spark.read.option("header", "true").csv(mcc_file) \
                .withColumn("mcc", col("mcc").cast(IntegerType()))
        
        # Load fraud labels for user's transactions only
        fraud_file = os.path.join(self.data_path, "train_fraud_labels.csv")
        
        # Get user's transaction IDs first
        user_transaction_ids = transactions_df.select("id").rdd.map(lambda row: row[0]).collect()
        
        if user_transaction_ids:
            fraud_df = self.spark.read.option("header", "true").csv(fraud_file) \
                      .filter(col("transaction_id").isin(user_transaction_ids)) \
                      .withColumn("transaction_id", col("transaction_id").cast(IntegerType())) \
                      .withColumn("is_fraud", col("is_fraud").cast(BooleanType()))
        else:
            # Create empty DataFrame with correct schema if no transactions
            schema = StructType([
                StructField("transaction_id", IntegerType(), True),
                StructField("is_fraud", BooleanType(), True)
            ])
            fraud_df = self.spark.createDataFrame([], schema)
        
        return {
            'users': users_df,
            'cards': cards_df,
            'transactions': transactions_df,
            'mcc_codes': mcc_df,
            'fraud_labels': fraud_df
        }
    
    def close_spark(self):
        """Close Spark session"""
        self.spark.stop()