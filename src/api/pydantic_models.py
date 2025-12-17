from pydantic import BaseModel, Field

class CreditRiskInput(BaseModel):
    # --- Numeric Aggregate Features (Required) ---
    total_amount: float = Field(..., description="Total transaction amount (Scaled)")
    avg_amount: float = Field(..., description="Average transaction amount (Scaled)")
    std_amount: float = Field(..., description="Standard deviation of amount (Scaled)")
    txn_count: float = Field(..., description="Total transaction count (Scaled)")
    
    # Note: These kept the aggregation suffix in your training data
    Amount_min: float = Field(..., description="Minimum transaction amount (Scaled)")
    Amount_max: float = Field(..., description="Maximum transaction amount (Scaled)")
    Value_sum: float = Field(..., description="Sum of absolute values (Scaled)")
    Value_mean: float = Field(..., description="Mean of absolute values (Scaled)")
    
    # --- Time Features ---
    avg_txn_hour: float = Field(default=0.0, description="Average hour of transaction")
    std_txn_hour: float = Field(default=0.0, description="Standard deviation of transaction hour")

    # --- Categorical Ratios (Optional/Default 0.0) ---
    # These represent the behavior profile. 
    # Example: If user only buys Airtime, set airtime_ratio=1.0 and others=0.0
    ProductCategory_airtime_ratio: float = 0.0
    ProductCategory_data_bundles_ratio: float = 0.0
    ProductCategory_financial_services_ratio: float = 0.0
    ProductCategory_movies_ratio: float = 0.0
    ProductCategory_other_ratio: float = 0.0
    ProductCategory_ticket_ratio: float = 0.0
    ProductCategory_transport_ratio: float = 0.0
    ProductCategory_tv_ratio: float = 0.0
    ProductCategory_utility_bill_ratio: float = 0.0

    # --- Channel Ratios ---
    ChannelId_ChannelId_1_ratio: float = 0.0
    ChannelId_ChannelId_2_ratio: float = 0.0
    ChannelId_ChannelId_3_ratio: float = 0.0
    ChannelId_ChannelId_5_ratio: float = 0.0

    class Config:
        schema_extra = {
            "example": {
                "total_amount": 0.5,
                "avg_amount": 0.1,
                "std_amount": 0.2,
                "txn_count": 1.0,
                "Amount_min": -0.5,
                "Amount_max": 0.5,
                "Value_sum": 0.5,
                "Value_mean": 0.1,
                "avg_txn_hour": 12.0,
                "std_txn_hour": 2.5,
                "ProductCategory_airtime_ratio": 0.8,
                "ProductCategory_financial_services_ratio": 0.2,
                "ChannelId_ChannelId_3_ratio": 1.0
            }
        }

class CreditRiskOutput(BaseModel):
    customer_id: str
    is_high_risk: int
    risk_probability: float