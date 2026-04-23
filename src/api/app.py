"""
NexTune — Bluetooth Headphones Price Prediction Web App
FastAPI + Jinja2 HTML form → predicted price + suggested range
"""

import os, json
import numpy as np
import joblib
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
TMPL_DIR   = os.path.join(os.path.dirname(__file__), "templates")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# ── Load model artifacts ──────────────────────────────────────────────────────
model      = joblib.load(os.path.join(MODEL_DIR, "price_predictor.pkl"))
scaler     = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
le_dict    = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
FEATURES   = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))
brand_avg  = joblib.load(os.path.join(MODEL_DIR, "brand_avg.pkl"))
BRANDS     = json.load(open(os.path.join(MODEL_DIR, "brands.json")))

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="NexTune Price Predictor")
templates = Jinja2Templates(directory=TMPL_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def predict(brand, category, rating, bt_version, driver_mm,
            has_anc, has_enc, has_hi_res, has_spatial, has_dual,
            has_codec, has_low_lat, ipx_level, anc_db, review_count):

    # Brand tier
    avg = brand_avg.get(brand, 0)
    tier = "premium" if avg >= 10000 else ("mid" if avg >= 3000 else "budget")

    def enc(col, val):
        le = le_dict.get(col)
        return int(le.transform([val])[0]) if le and val in le.classes_ else 0

    row = {
        "brand_tier_enc":    enc("brand_tier", tier),
        "price_per_hour":    0,          # unknown for new product
        "has_anc":           int(has_anc),
        "rating":            float(rating),
        "has_hi_res_audio":  int(has_hi_res),
        "high_rating":       int(float(rating) >= 4.0),
        "has_spatial_audio": int(has_spatial),
        "bluetooth_version": float(bt_version),
        "brand_enc":         enc("brand", brand),
        "has_dual_pairing":  int(has_dual),
        "has_premium_codec": int(has_codec),
        "bt_major":          int(float(bt_version)),
        "anc_db":            float(anc_db),
        "has_enc":           int(has_enc),
        "review_count":      float(review_count),
        "ipx_level":         float(ipx_level),
        "driver_size_mm":    float(driver_mm),
        "has_low_latency":   int(has_low_lat),
    }

    vec = np.array([[row.get(f, 0) for f in FEATURES]])
    vec_sc = scaler.transform(vec)
    log_pred = model.predict(vec_sc)[0]
    price = float(np.expm1(log_pred))
    return round(price), round(price * 0.9), round(price * 1.1), tier


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "brands": BRANDS,
        "result": None,
    })


@app.post("/predict", response_class=HTMLResponse)
async def predict_price(
    request: Request,
    brand:         str   = Form(...),
    category:      str   = Form(...),
    rating:        float = Form(...),
    bt_version:    float = Form(...),
    driver_mm:     float = Form(...),
    has_anc:       int   = Form(0),
    has_enc:       int   = Form(0),
    has_hi_res:    int   = Form(0),
    has_spatial:   int   = Form(0),
    has_dual:      int   = Form(0),
    has_codec:     int   = Form(0),
    has_low_lat:   int   = Form(0),
    ipx_level:     float = Form(0),
    anc_db:        float = Form(0),
    review_count:  float = Form(0),
):
    price, low, high, tier = predict(
        brand, category, rating, bt_version, driver_mm,
        has_anc, has_enc, has_hi_res, has_spatial, has_dual,
        has_codec, has_low_lat, ipx_level, anc_db, review_count
    )

    return templates.TemplateResponse("index.html", {
        "request":  request,
        "brands":   BRANDS,
        "result": {
            "price":    f"₹{price:,}",
            "low":      f"₹{low:,}",
            "high":     f"₹{high:,}",
            "brand":    brand,
            "tier":     tier,
            "category": category,
            "has_anc":  bool(has_anc),
        },
        # Keep form values
        "form": {
            "brand": brand, "category": category, "rating": rating,
            "bt_version": bt_version, "driver_mm": driver_mm,
            "has_anc": has_anc, "has_enc": has_enc, "has_hi_res": has_hi_res,
            "has_spatial": has_spatial, "has_dual": has_dual, "has_codec": has_codec,
            "has_low_lat": has_low_lat, "ipx_level": ipx_level, "anc_db": anc_db,
            "review_count": review_count,
        }
    })
