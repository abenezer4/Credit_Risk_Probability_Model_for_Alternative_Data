# Base image
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/

# Copy MLflow data (Optional but recommended for standalone execution)
# If your model is stored in sqlite (mlflow.db) or mlruns, copy them so the container has the model.
# Based on your image:
COPY notebooks/mlruns /app/notebooks/mlruns
COPY mlflow.db /app/notebooks/mlflow.db

# Set Environment variable so MLflow knows where to look inside container
# Adjust this based on whether you use the DB or the mlruns folder
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# Expose port
EXPOSE 8000

# Run command
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]