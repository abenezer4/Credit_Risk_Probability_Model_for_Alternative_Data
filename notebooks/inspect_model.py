# inspect_model.py
import mlflow.sklearn
import os

# Point to your database
os.environ['MLFLOW_TRACKING_URI'] = "sqlite:///mlflow.db"

# Load the model
model = mlflow.sklearn.load_model("models:/CreditRiskModel/Latest")

# Print the expected feature names
print("--- REQUIRED FEATURES ---")
if hasattr(model, "feature_names_in_"):
    print(list(model.feature_names_in_))
else:
    print("Could not retrieve feature names automatically. Please check your X_train columns.")