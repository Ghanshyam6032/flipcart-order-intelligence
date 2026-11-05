import os
import sys
from pathlib import Path
from typing import Optional
import modal

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# 1. Modal Image & Dependencies + Directories Configuration
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi>=0.110.0",
        "uvicorn>=0.28.0",
        "mistralai>=1.0.0",
        "python-dotenv>=1.0.0",
        "scikit-learn>=1.4.0",
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "Pillow>=10.2.0",
        "joblib>=1.3.2",
        "python-multipart>=0.0.9",
        "requests>=2.31.0",
        "ai-edge-litert>=2.0.0"
    )
    .add_local_dir("part1_return_risk", remote_path="/root/part1_return_risk")
    .add_local_dir("part2_image_classifier", remote_path="/root/part2_image_classifier")
    .add_local_dir("part3_support_agent", remote_path="/root/part3_support_agent")
)

app = modal.App("flipcart-order-intelligence-backend", image=image)

# 2. FastAPI App Setup
web_app = FastAPI(
    title="Flipkart Order Intelligence API",
    description="Multi-modal AI Customer Support with Return Risk Analysis and TFLite Vision Classification",
    version="2.0.0"
)

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@web_app.get("/health")
def health_check():
    return {"status": "ok", "message": "API live on Modal"}

@web_app.post("/support")
@web_app.post("/api/support")
async def support_endpoint(
    question: str = Form(...),
    price_inr: float = Form(2499.0),
    discount_pct: float = Form(15.0),
    payment_method: str = Form("COD"),
    customer_tenure_days: int = Form(365),
    num_previous_orders: int = Form(6),
    num_previous_returns: int = Form(1),
    delivery_distance_km: float = Form(150.0),
    delivery_days: int = Form(4),
    is_weekend_order: int = Form(0),
    rating_given: Optional[float] = Form(None),
    product_category: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    try:
        # Lazy imports for Modal runtime container paths
        ROOT_DIR = Path("/root")
        if str(ROOT_DIR) not in sys.path:
            sys.path.insert(0, str(ROOT_DIR))

        from part1_return_risk.predict import predict_return_risk
        from part2_image_classifier.predict import predict_image
        from part3_support_agent.agent import generate_support_response

        detected_category = product_category or "Apparel"
        image_confidence = 0.95

        if image is not None and image.filename:
            image_bytes = await image.read()
            if len(image_bytes) > 0:
                try:
                    vision_result = predict_image(image_bytes)
                    detected_category = vision_result["category"]
                    image_confidence = float(vision_result["confidence"])
                except Exception as img_err:
                    print(f"Warning: TFLite image classification notice: {img_err}")
                    detected_category = product_category or "Apparel"

        final_order_cat = product_category or detected_category
        valid_categories = ["Electronics", "Apparel", "Footwear", "Home", "Beauty"]
        if final_order_cat not in valid_categories:
            final_order_cat = "Apparel"

        order_data = {
            "product_category": final_order_cat,
            "price_inr": float(price_inr),
            "discount_pct": float(discount_pct),
            "payment_method": payment_method,
            "customer_tenure_days": int(customer_tenure_days),
            "num_previous_orders": int(num_previous_orders),
            "num_previous_returns": int(num_previous_returns),
            "delivery_distance_km": float(delivery_distance_km),
            "delivery_days": int(delivery_days),
            "is_weekend_order": int(is_weekend_order),
            "rating_given": float(rating_given) if rating_given is not None else None
        }

        risk_result = predict_return_risk(order_data)
        return_risk = risk_result["risk_level"]
        return_probability = float(risk_result["probability"])

        agent_reply = generate_support_response(
            question=question,
            order_data=order_data,
            return_risk=return_risk,
            return_probability=return_probability,
            product_category=detected_category,
            image_confidence=image_confidence
        )

        return {
            "response": agent_reply,
            "return_risk": return_risk,
            "return_probability": return_probability,
            "product_category": detected_category,
            "image_confidence": image_confidence
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Support processing error: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the support request. Please try again."
        )

# 3. Modal ASGI Endpoint Deployment with Mistral API Secret
@app.function(
    min_containers=1,
    secrets=[modal.Secret.from_name("my-mistral-secret")]
)
@modal.asgi_app()
def fastapi_app():
    return web_app