import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from part1_return_risk.predict import predict_return_risk
from part2_image_classifier.predict import predict_image
from part3_support_agent.agent import generate_support_response

app = FastAPI(
    title="Flipkart Order Intelligence API",
    description="Multi-modal AI Customer Support with Return Risk Analysis and TFLite Vision Classification",
    version="2.0.0"
)

allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "*")
allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if "*" not in allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/support")
@app.post("/api/support")
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


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("part3_support_agent.app:app", host=host, port=port, reload=False)
