# Flipkart Order Intelligence & AI Support System

An end-to-end multi-modal machine learning, computer vision, and AI agent platform designed for e-commerce order intelligence. The system predicts return risk using an ensemble ML pipeline, classifies product images using a trained **TensorFlow Lite (TFLite)** model, and delivers contextual customer support resolutions via the Mistral AI API and FastAPI.

---

## 📌 Architecture Overview

```text
Customer Order Telemetry + Product Image + Customer Inquiry
                          │
         ┌────────────────┴────────────────┐
         ▼                                 ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│  Part 1: Return Risk ML   │   │ Part 2: TensorFlow Lite   │
│  Random Forest Pipeline   │   │ Product Image Classifier  │
│ (Threshold Tuned, joblib) │   │ (model_quant.tflite)      │
└─────────────┬─────────────┘   └─────────────┬─────────────┘
              │                               │
              ▼                               ▼
       Return Risk Tier               Detected Category
       & Probability                  & Confidence Score
              │                               │
              └───────────────┬───────────────┘
                              ▼
                ┌───────────────────────────┐
                │   Part 3: Support Agent   │
                │  FastAPI + Mistral AI LLM │
                └─────────────┬─────────────┘
                              ▼
                ┌───────────────────────────┐
                │    Frontend Dashboard     │
                │  Responsive Flipkart UI   │
                └───────────────────────────┘
```

---

## 🚀 Key Features

1. **Part 1 — Return Risk Prediction (ML Pipeline)**:
   - Automated 6,000 synthetic order dataset generator replicating real-world e-commerce dynamics.
   - Comprehensive data preprocessing (Median Imputation, One-Hot Encoding, StandardScaler) avoiding data leakage.
   - **RandomForestClassifier** with class-weight balancing and decision threshold tuning to maximize $F_1$-score on positive returns.
   - Outputs Return Probability and Risk Level (`LOW`, `MEDIUM`, `HIGH`).

2. **Part 2 — Product Image Classification (TensorFlow Lite)**:
   - Inference powered by Google Teachable Machine exported **TensorFlow Lite** model (`model_quant.tflite`).
   - Categorizes product photos directly into Flipkart categories:
     - `Electronics`
     - `Apparel`
     - `Footwear`
     - `Home`
     - `Beauty`
   - Handles RGB/RGBA/Grayscale images with automatic resizing, normalization, and Softmax confidence score calculation.

3. **Part 3 — FastAPI Backend & Mistral AI Support Agent**:
   - High-throughput asynchronous FastAPI endpoints (`/health` and `/api/support`).
   - Integration with the official **Mistral API** using the `mistralai` SDK and `.env` configuration.
   - Multipart form-data handling with safe in-memory image processing.

4. **Frontend Dashboard**:
   - Single-file responsive interface (`frontend/index.html`) crafted in modern Flipkart aesthetics (`#2874f0`, `#ff9f00`).
   - Image drag-and-drop zone with instant preview.
   - Real-time telemetry cards and quick test scenario loaders.

---

## 📁 Exact Project Structure

```text
Flipkart-Project/
│
├── part1_return_risk/
│   ├── train.py              # Synthetic data generator + Random Forest training & threshold tuning
│   ├── predict.py            # Reusable return risk prediction module
│   ├── orders_dataset.csv    # 6,000 sample dataset (generated)
│   └── model.pkl             # Serialized pipeline & threshold artifact (generated)
│
├── part2_image_classifier/
│   ├── model_quant.tflite    # Trained TensorFlow Lite product classifier model
│   ├── labels.txt            # Category mapping (Electronics, Apparel, Footwear, Home, Beauty)
│   └── predict.py            # Reusable TensorFlow Lite vision inference module
│
├── part3_support_agent/
│   ├── app.py                # FastAPI REST API with multipart support & CORS
│   ├── agent.py              # Mistral AI agent integration & prompt engineering
│   └── requirements.txt      # Lightweight Python 3.12 dependencies
│
├── frontend/
│   └── index.html            # Complete single-file responsive Flipkart UI
│
├── .env                      # Environment variable configuration (MISTRAL_API_KEY)
└── README.md                 # Complete system documentation
```

---

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| **Runtime** | Python 3.12.x |
| **ML Engine** | NumPy, Pandas, Scikit-Learn, RandomForest, Joblib |
| **Vision Engine** | TensorFlow Lite (`ai-edge-litert` / `tflite-runtime`), Pillow |
| **API Backend** | FastAPI, Uvicorn, Python-Multipart |
| **LLM Agent** | Mistral API (`mistralai`), python-dotenv |
| **Frontend** | Vanilla HTML5, Modern CSS3, JavaScript (Fetch API, FormData) |

---

## ⚙️ Setup & Installation (Windows)

### 1. Prerequisites
Ensure **Python 3.12** is installed:
```powershell
py -3.12 --version
```

### 2. Create and Activate Virtual Environment
Open PowerShell inside the project directory:
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r part3_support_agent/requirements.txt
```

### 4. Configure Environment Variables
Edit the `.env` file in the root directory:
```env
MISTRAL_API_KEY=your_actual_mistral_api_key_here
```
*(Note: If left as the placeholder, the system will seamlessly provide structured contextual responses while remaining 100% operational).*

---

## 🚀 Running the Application

### 1. Start the FastAPI Server
```powershell
uvicorn part3_support_agent.app:app --host 127.0.0.1 --port 8000 --reload
```
The server will be available at:
- **API Root**: `http://127.0.0.1:8000`
- **Health Check**: `http://127.0.0.1:8000/health`
- **Interactive OpenAPI Docs**: `http://127.0.0.1:8000/docs`

### 2. Launch the Frontend
Open `frontend/index.html` directly in your browser:
```powershell
start frontend/index.html
```

---

## 📊 API Reference

### Health Check
- **Endpoint**: `GET /health`
- **Response**:
```json
{
  "status": "ok"
}
```

### Support & Order Analysis
- **Endpoint**: `POST /api/support`
- **Content-Type**: `multipart/form-data`

#### Form Parameters:
| Field | Type | Description | Example |
|---|---|---|---|
| `question` | string | Customer message/query | `"The size is too tight, how do I exchange?"` |
| `price_inr` | float | Item price in INR | `2499` |
| `discount_pct` | float | Discount percentage | `20.0` |
| `payment_method` | string | COD / Prepaid_Card / Prepaid_UPI / Wallet | `"COD"` |
| `customer_tenure_days` | int | Customer account age in days | `300` |
| `num_previous_orders` | int | Total historical orders | `8` |
| `num_previous_returns` | int | Total historical returns | `3` |
| `delivery_distance_km` | float | Transit distance in km | `150.0` |
| `delivery_days` | int | Delivery timeframe in days | `6` |
| `is_weekend_order` | int | 1 (Weekend) or 0 (Weekday) | `1` |
| `rating_given` | float | Optional 1-5 rating | `3.0` |
| `product_category` | string | Optional product category override | `"Footwear"` |
| `image` | file | Product image (JPG/PNG/WEBP) | `product_sample.png` |

#### Example Response:
```json
{
  "response": "Hello! I understand you are inquiring about returning or exchanging your Footwear (Rs. 2,499, COD). As this item is within our 7-day return window, we have authorized a hassle-free doorstep exchange or return pickup. Our courier partner will inspect the original packaging upon collection.",
  "return_risk": "HIGH",
  "return_probability": 0.7147,
  "product_category": "Footwear",
  "image_confidence": 0.9421
}
```

---

## 🔍 Troubleshooting

- **TFLite Interpreter Selection**: The application automatically loads `ai-edge-litert`, `tflite-runtime`, or `tensorflow.lite` based on what is installed.
- **Port Conflict on 8000**: If port 8000 is occupied, start the backend on `--port 8001` and update `const API_URL = "http://127.0.0.1:8001";` in `frontend/index.html`.
- **Mistral API Rate Limits**: The support agent contains built-in exception handling and fallback generation to ensure continuous uptime.
