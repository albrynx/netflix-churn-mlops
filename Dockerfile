# Upgrade to Python 3.11 to support scikit-learn 1.7.2
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the local files to the container
COPY requirements.txt .
COPY app.py .
COPY churn_model_balanced.pkl .
COPY label_encoders.pkl .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit runs on
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]