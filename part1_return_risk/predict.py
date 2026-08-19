import os
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

_MODEL_CACHE = None


def load_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Please run `python part1_return_risk/train.py` first."
            )
        _MODEL_CACHE = joblib.load(MODEL_PATH)
    return _MODEL_CACHE


def get_risk_level(prob: float) -> str:
    if prob < 0.35:
        return "LOW"
    elif prob < 0.65:
        return "MEDIUM"
    else:
        return "HIGH"


def predict_return_risk(order_data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
    model_pkg = load_model()
    pipeline = model_pkg["pipeline"]
    feature_cols = model_pkg["feature_cols"]
    optimal_threshold = model_pkg.get("optimal_threshold", 0.5)

    if isinstance(order_data, dict):
        df_input = pd.DataFrame([order_data])
    elif isinstance(order_data, pd.DataFrame):
        df_input = order_data.copy()
    else:
        raise ValueError("order_data must be a dict or a pandas DataFrame")

    for col in feature_cols:
        if col not in df_input.columns:
            df_input[col] = None

    X_eval = df_input[feature_cols]

    prob_return = float(pipeline.predict_proba(X_eval)[0, 1])
    prediction = int(prob_return >= optimal_threshold)
    risk_level = get_risk_level(prob_return)

    return {
        "prediction": prediction,
        "probability": round(prob_return, 4),
        "risk_level": risk_level
    }


if __name__ == "__main__":
    sample_order = {
        "product_category": "Footwear",
        "price_inr": 2499,
        "discount_pct": 20,
        "payment_method": "COD",
        "customer_tenure_days": 300,
        "num_previous_orders": 8,
        "num_previous_returns": 3,
        "delivery_distance_km": 150,
        "delivery_days": 6,
        "is_weekend_order": 1,
        "rating_given": 3
    }
    print("Testing Part 1 Return Risk Prediction with sample order:")
    print(sample_order)
    try:
        result = predict_return_risk(sample_order)
        print("\nPrediction Result:")
        print(result)
    except FileNotFoundError as e:
        print(f"\n{e}")
