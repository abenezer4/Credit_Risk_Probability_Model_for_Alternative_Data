import pandas as pd
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from src.api.pydantic_models import CreditRiskInput, CreditRiskOutput
import os

# Define the exact order required by the model (copy-pasted from your output)
REQUIRED_COLUMNS = [
    'total_amount', 'avg_amount', 'std_amount', 'txn_count', 
    'Amount_min', 'Amount_max', 'Value_sum', 'Value_mean', 
    'avg_txn_hour', 'std_txn_hour', 
    'ProductCategory_airtime_ratio', 'ProductCategory_data_bundles_ratio', 
    'ProductCategory_financial_services_ratio', 'ProductCategory_movies_ratio', 
    'ProductCategory_other_ratio', 'ProductCategory_ticket_ratio', 
    'ProductCategory_transport_ratio', 'ProductCategory_tv_ratio', 
    'ProductCategory_utility_bill_ratio', 
    'ChannelId_ChannelId_1_ratio', 'ChannelId_ChannelId_2_ratio', 
    'ChannelId_ChannelId_3_ratio', 'ChannelId_ChannelId_5_ratio'
]

model = None
MODEL_NAME = "CreditRiskModel"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        # Point to notebooks DB as discussed previously
        os.environ['MLFLOW_TRACKING_URI'] = "sqlite:///notebooks/mlflow.db"
        print(f"🔄 Loading model: models:/{MODEL_NAME}/Latest")
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Latest")
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
    
    yield
    model = None

app = FastAPI(title="Bati Bank Credit Risk API", lifespan=lifespan)

@app.get("/")
def health_check():
    return {"status": "running", "model_loaded": model is not None}

@app.post("/predict", response_model=CreditRiskOutput)
def predict_risk(input_data: CreditRiskInput, customer_id: str = "Unknown"):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # 1. Convert Pydantic object to Dictionary
        data_dict = input_data.dict()
        
        # 2. Create DataFrame
        df_input = pd.DataFrame([data_dict])

        # 3. Reorder columns to match training data exactly
        # This prevents the "Feature names seen at fit time, yet now missing" error
        df_input = df_input[REQUIRED_COLUMNS]
        
        # 4. Predict
        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1]

        return {
            "customer_id": customer_id,
            "is_high_risk": int(prediction),
            "risk_probability": round(float(probability), 4)
        }

    except KeyError as e:
        # This catches if a column is missing from the reordering step
        raise HTTPException(status_code=400, detail=f"Missing feature column: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")