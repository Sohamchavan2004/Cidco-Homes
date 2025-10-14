from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Initialize FastAPI app
app = FastAPI(title="Property Price Prediction API")

# Globals
model = None
preprocessor = None
target_variable = 'price_per_sqft'

# Columns used during training
cat_cols = ['distance_category', 'location_type']
num_cols = ['Carpet_Area_sqft', 'No_of_Towers', 'location_popularity']

# Load model and preprocessor on startup
@app.on_event("startup")
def load_assets():
    global model, preprocessor

    try:
        model_path = "xgboost_regressor_model.pkl"
        preprocessor_path = "preprocessor.pkl"

        # Check files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")

        # Load trained model and preprocessor
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)

        print("✅ Model and preprocessor loaded successfully.")

    except Exception as e:
        print(f"❌ Error loading model or preprocessor: {e}")
        model = None
        preprocessor = None


# Input schema
class PropertyRequest(BaseModel):
    Carpet_Area_sqft: float
    No_of_Towers: int
    distance_category: str
    location_popularity: float
    location_type: str


# Prediction endpoint
@app.post("/predict")
def predict_price(request: PropertyRequest):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model or preprocessor not loaded. Please restart the server.")

    try:
        # Convert input JSON to DataFrame
        input_df = pd.DataFrame([request.dict()])

        # Apply the same preprocessor used during training
        input_processed = preprocessor.transform(input_df)

        # Predict
        prediction = model.predict(input_processed)

        return {target_variable: round(float(prediction[0]), 2)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

