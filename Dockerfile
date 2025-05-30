# Use Python 3.9 with Java pre-installed for PySpark
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    openjdk-17-jdk \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for Spark
RUN mkdir -p /tmp/spark-warehouse

# Expose Streamlit port
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "streamlit/dashboard.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true"]