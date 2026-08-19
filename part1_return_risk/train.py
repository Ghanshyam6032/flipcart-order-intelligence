import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "orders_dataset.csv"
MODEL_PATH = BASE_DIR / "model.pkl"


def generate_dataset(n_samples: int = 6000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    N = n_samples

    categories = ["Apparel", "Electronics", "Home", "Footwear", "Beauty"]
    cat_probs = [0.32, 0.22, 0.18, 0.18, 0.10]

    payment_methods = ["COD", "Prepaid_Card", "Prepaid_UPI", "Wallet"]
    pay_probs = [0.42, 0.24, 0.24, 0.10]

    product_category = rng.choice(categories, size=N, p=cat_probs)
    payment_method = rng.choice(payment_methods, size=N, p=pay_probs)

    base_price = {
        "Apparel": (400, 2200),
        "Electronics": (1200, 45000),
        "Home": (300, 8000),
        "Footwear": (500, 4500),
        "Beauty": (150, 2500),
    }

    price_inr = np.round(
        np.array([rng.uniform(*base_price[c]) for c in product_category]),
        0
    )

    discount_pct = np.clip(rng.normal(22, 15, N), 0, 75)

    customer_tenure_days = np.clip(rng.exponential(380, N), 1, 2500).round(0)

    num_previous_orders = np.clip(
        (customer_tenure_days / 45) + rng.normal(0, 2, N),
        0,
        None
    ).round(0)

    base_return_rate = np.clip(rng.beta(1.5, 9, N), 0, 1)

    num_previous_returns = np.round(
        base_return_rate * num_previous_orders
    ).clip(0, num_previous_orders)

    delivery_distance_km = np.clip(rng.gamma(3, 90, N), 2, 2200).round(1)
    delivery_days = np.clip(rng.normal(4.5, 2.2, N), 1, 21).round(0)
    is_weekend_order = rng.integers(0, 2, N)

    rating_given = rng.integers(1, 6, N).astype(float)
    missing_mask = (
        rng.random(N) < np.where(payment_method == "COD", 0.22, 0.06)
    )
    rating_given[missing_mask] = np.nan

    fit_risk_cat = np.isin(product_category, ["Apparel", "Footwear"]).astype(float)

    prev_return_ratio = np.where(
        num_previous_orders > 0,
        num_previous_returns / np.maximum(num_previous_orders, 1),
        0
    )

    electronics_max_price = base_price["Electronics"][1]

    z = (
        -2.2
        + 1.9 * prev_return_ratio
        + 0.55 * fit_risk_cat
        + 0.014 * (discount_pct - 20) / 10
        + 0.9 * (payment_method == "COD").astype(float)
        + 0.10 * (delivery_days - 4.5) / 2
        + 0.30 * (price_inr / electronics_max_price)
        + 0.05 * is_weekend_order
        - 0.15 * np.tanh(customer_tenure_days / 500)
    )

    prob_return = 1 / (1 + np.exp(-z))
    returned = (rng.random(N) < prob_return).astype(int)

    df = pd.DataFrame({
        "order_id": np.arange(1, N + 1),
        "product_category": product_category,
        "price_inr": price_inr,
        "discount_pct": np.round(discount_pct, 1),
        "payment_method": payment_method,
        "customer_tenure_days": customer_tenure_days.astype(int),
        "num_previous_orders": num_previous_orders.astype(int),
        "num_previous_returns": num_previous_returns.astype(int),
        "delivery_distance_km": delivery_distance_km,
        "delivery_days": delivery_days.astype(int),
        "is_weekend_order": is_weekend_order,
        "rating_given": rating_given,
        "returned": returned,
    })

    return df


def train_model():
    print("=" * 60)
    print("PART 1: FLIPKART RETURN RISK MODEL TRAINING")
    print("=" * 60)

    if not DATASET_PATH.exists():
        print(f"Generating 6,000 synthetic orders -> {DATASET_PATH.name}...")
        df = generate_dataset(n_samples=6000, random_state=42)
        df.to_csv(DATASET_PATH, index=False)
        print(f"Dataset successfully saved with shape: {df.shape}")
    else:
        print(f"Loading existing dataset -> {DATASET_PATH.name}...")
        df = pd.read_csv(DATASET_PATH)
        print(f"Loaded dataset with shape: {df.shape}")

    return_rate = df["returned"].mean()
    print(f"Dataset Return Rate: {return_rate:.2%} ({df['returned'].sum()} returns / {len(df)} total)")

    cat_cols = ["product_category", "payment_method"]
    num_cols = [
        "price_inr",
        "discount_pct",
        "customer_tenure_days",
        "num_previous_orders",
        "num_previous_returns",
        "delivery_distance_km",
        "delivery_days",
        "is_weekend_order",
        "rating_given"
    ]
    feature_cols = cat_cols + num_cols

    X = df[feature_cols]
    y = df["returned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")

    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ])

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=8,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])

    print("\nTraining RandomForestClassifier...")
    pipeline.fit(X_train, y_train)

    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    y_test_pred_default = (y_test_proba >= 0.5).astype(int)
    roc_auc = roc_auc_score(y_test, y_test_proba)

    print("\n" + "-" * 50)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("\nClassification Report (Default Threshold = 0.5):")
    print(classification_report(y_test, y_test_pred_default, target_names=["Not Returned (0)", "Returned (1)"]))

    print("-" * 50)
    print("Performing Threshold Tuning for returned = 1 detection...")

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_test_proba)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = precisions[:-1] + recalls[:-1]
        f1_scores = np.where(
            denom > 0,
            2 * (precisions[:-1] * recalls[:-1]) / denom,
            0.0
        )
    best_idx = np.argmax(f1_scores)
    optimal_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])

    print(f"Optimal Decision Threshold: {optimal_threshold:.4f} (Max F1-Score: {best_f1:.4f})")

    y_test_pred_tuned = (y_test_proba >= optimal_threshold).astype(int)
    print("\nClassification Report (Tuned Threshold):")
    print(classification_report(y_test, y_test_pred_tuned, target_names=["Not Returned (0)", "Returned (1)"]))

    cm = confusion_matrix(y_test, y_test_pred_tuned)
    print("Confusion Matrix (Tuned Threshold):")
    print(f"TN: {cm[0,0]} | FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]} | TP: {cm[1,1]}")

    model_package = {
        "pipeline": pipeline,
        "feature_cols": feature_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "optimal_threshold": optimal_threshold,
        "metrics": {
            "roc_auc": float(roc_auc),
            "best_f1": float(best_f1),
            "precision": float(precision_score(y_test, y_test_pred_tuned)),
            "recall": float(recall_score(y_test, y_test_pred_tuned)),
            "default_precision": float(precision_score(y_test, y_test_pred_default)),
            "default_recall": float(recall_score(y_test, y_test_pred_default)),
            "dataset_return_rate": float(return_rate)
        }
    }

    joblib.dump(model_package, MODEL_PATH)
    print(f"\nModel pipeline and threshold package successfully saved -> {MODEL_PATH.name}")
    print("=" * 60)


if __name__ == "__main__":
    train_model()
