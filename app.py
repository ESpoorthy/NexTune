"""NexTune – Headphone Price Predictor (Full 18-feature model)"""

import streamlit as st
import numpy as np
import joblib, os, json, base64

st.set_page_config(page_title="NexTune", page_icon="🎧", layout="wide",
                   initial_sidebar_state="expanded")

# ── helpers ───────────────────────────────────────────────────────────────────
def get_b64(path):
    try:
        with open(path, "rb") as f: return base64.b64encode(f.read()).decode()
    except: return None

@st.cache_resource(show_spinner=False)
def load_artifacts():
    paths = {
        "model":    ["models/price_predictor.pkl", "model.pkl"],
        "scaler":   ["models/scaler.pkl",          "scaler.pkl"],
        "encoders": ["models/label_encoders.pkl",  "encoder.pkl"],
        "features": ["models/features.pkl"],
        "brand_avg":["models/brand_avg.pkl"],
    }
    def first(lst):
        for p in lst:
            if os.path.exists(p): return p
        return None

    mp = first(paths["model"])
    sp = first(paths["scaler"])
    ep = first(paths["encoders"])
    fp = first(paths["features"])
    bp = first(paths["brand_avg"])

    if not mp: return None, None, None, None, None, "model file not found. Run model_training.ipynb first."
    try:
        model    = joblib.load(mp)
        scaler   = joblib.load(sp) if sp else None
        encoders = joblib.load(ep) if ep else None
        features = joblib.load(fp) if fp else None
        brand_avg= joblib.load(bp) if bp else {}
        return model, scaler, encoders, features, brand_avg, None
    except Exception as e:
        return None, None, None, None, None, str(e)

@st.cache_data(show_spinner=False)
def load_brands():
    for p in ["models/brands.json", "data/earbuds_by_company.json"]:
        if os.path.exists(p):
            try:
                data = json.load(open(p, encoding="utf-8"))
                if isinstance(data, list) and data and isinstance(data[0], str):
                    return data
                return sorted({x["brand"] for x in data if "brand" in x})
            except: pass
    return ["Boat","Sony","JBL","Noise","Realme","OnePlus","Samsung","Apple",
            "boAt","GOBOULT","pTron","truke","Portronics","Fire-Boltt","Mivi"]

def predict_price(model, scaler, encoders, features, brand_avg,
                  brand, rating, reviews, has_anc, bt_version, driver_mm,
                  has_enc, has_hi_res, has_spatial, has_dual, has_codec,
                  has_low_lat, ipx_level, anc_db):
    try:
        avg  = brand_avg.get(brand, 0) if brand_avg else 0
        tier = "premium" if avg >= 10000 else ("mid" if avg >= 3000 else "budget")

        def enc(col, val):
            if not encoders: return 0
            le = encoders.get(col)
            return int(le.transform([val])[0]) if le and val in le.classes_ else 0

        row = {
            "brand_tier_enc":    enc("brand_tier", tier),
            "price_per_hour":    0,
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
            "review_count":      float(reviews),
            "ipx_level":         float(ipx_level),
            "driver_size_mm":    float(driver_mm),
            "has_low_latency":   int(has_low_lat),
        }

        feat_list = features if features else list(row.keys())
        vec = np.array([[row.get(f, 0) for f in feat_list]])

        if scaler:
            vec = scaler.transform(vec)

        log_pred = model.predict(vec)[0]
        price = float(np.expm1(log_pred))
        return price, tier, None
    except Exception as e:
        return None, None, str(e)

def price_segment(p):
    if p < 1000:   return "💚 Budget",    "#22c55e", "#052e16"
    elif p < 3000: return "💙 Mid-Range", "#3b82f6", "#0c1a3a"
    elif p < 8000: return "💜 Premium",   "#a78bfa", "#1e1040"
    else:          return "🏆 Flagship",  "#f59e0b", "#2d1a00"

# ── load ──────────────────────────────────────────────────────────────────────
bg = get_b64("assets/bg.jpg")
model, scaler, encoders, features, brand_avg, load_err = load_artifacts()
brands = sorted(list(encoders["brand"].classes_)) if encoders and "brand" in encoders else load_brands()

# ── session state ─────────────────────────────────────────────────────────────
defaults = {
    "theme": "dark", "brand": brands[0], "rating": 4.0, "reviews": 500,
    "anc": False, "bt_version": 5.3, "driver_mm": 10, "has_enc": False,
    "has_hi_res": False, "has_spatial": False, "has_dual": False,
    "has_codec": False, "has_low_lat": False, "ipx_level": 0, "anc_db": 0,
    "has_touch": False, "has_voice": False, "has_fast_charge": False,
    "has_foldable": False, "battery_hrs": 30,
    "result": None, "tier": None, "err": None,
}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

# ── theme ─────────────────────────────────────────────────────────────────────
is_dark = st.session_state["theme"] == "dark"

APP_BG      = "#0f172a"            if is_dark else "#eef2f7"
OVERLAY     = "rgba(8,10,28,.83)"  if is_dark else "rgba(220,228,240,.80)"
CARD        = "rgba(30,41,59,.85)" if is_dark else "rgba(255,255,255,.90)"
CARD_BORDER = "#334155"            if is_dark else "#c7d2fe"
SB_BG       = "#0c1525"            if is_dark else "#f8fafc"
SB_BORDER   = "#1e293b"            if is_dark else "#c7d2fe"
TEXT        = "#e2e8f0"            if is_dark else "#0f172a"
MUTED       = "#94a3b8"            if is_dark else "#334155"
FAINT       = "#475569"            if is_dark else "#64748b"
INPUT_BG    = "#0d1526"            if is_dark else "#f8fafc"
INPUT_BORDER= "#334155"            if is_dark else "#94a3b8"
ACCENT      = "#6366f1"            if is_dark else "#4f46e5"
ACCENT2     = "#4f46e5"            if is_dark else "#4338ca"
PRICE_COL   = "#60a5fa"            if is_dark else "#2563eb"
HERO_BG     = "linear-gradient(135deg,#13132b,#1e1050,#13132b)" if is_dark else "linear-gradient(135deg,rgba(237,233,254,.95),rgba(219,234,254,.95),rgba(237,233,254,.95))"
HERO_BORDER = "#312e81"            if is_dark else "#a5b4fc"
DIVIDER     = "#1e293b"            if is_dark else "#cbd5e1"
STAT_BG     = "rgba(13,21,38,.7)"  if is_dark else "rgba(241,245,249,.9)"
FOOTER_COL  = "#334155"            if is_dark else "#64748b"
IDLE_P      = "#475569"            if is_dark else "#64748b"

bg_css = f"background-image:url('data:image/jpg;base64,{bg}');background-size:cover;background-position:center top;background-attachment:scroll;" if bg else ""

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
*{{font-family:'Inter',sans-serif;box-sizing:border-box;}}
.stApp{{background-color:{APP_BG};color:{TEXT};{bg_css}}}
.stApp::before{{content:'';position:fixed;inset:0;background:{OVERLAY};z-index:0;pointer-events:none;}}
.stApp>*{{position:relative;z-index:1;}}
[data-testid="stSidebar"]{{position:relative;z-index:100!important;}}
header[data-testid="stHeader"],footer{{display:none!important;}}
.block-container{{padding:1.8rem 2rem 1rem!important;max-width:100%!important;}}
.hero{{background:{HERO_BG};border:1px solid {HERO_BORDER};border-radius:16px;padding:2rem;text-align:center;margin-bottom:1.5rem;box-shadow:0 0 40px rgba(99,102,241,.15);}}
.hero h1{{font-size:2.4rem;font-weight:800;background:linear-gradient(90deg,#818cf8,#60a5fa,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 .3rem;}}
.hero p{{color:{MUTED};font-size:1rem;margin:0;}}
.nt-card{{background:{CARD};border:1px solid {CARD_BORDER};border-radius:14px;padding:22px 20px;backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);box-shadow:0 6px 28px rgba(0,0,0,.18);}}
.nt-card-title{{font-size:.95rem;font-weight:700;color:{TEXT};padding-bottom:.7rem;margin-bottom:1rem;border-bottom:1px solid {CARD_BORDER};}}
.sec-label{{font-size:.72rem;font-weight:700;color:{ACCENT};text-transform:uppercase;letter-spacing:.1em;margin:1rem 0 .4rem;}}
[data-testid="stWidgetLabel"] p{{color:{MUTED}!important;font-size:.78rem!important;font-weight:600!important;text-transform:uppercase!important;letter-spacing:.07em!important;}}
.stSelectbox>div>div{{background:{INPUT_BG}!important;border:1px solid {INPUT_BORDER}!important;border-radius:9px!important;color:{TEXT}!important;}}
.stSelectbox>div>div>div{{color:{TEXT}!important;}}
.stSelectbox svg{{fill:{MUTED}!important;}}
[role="option"]{{background:{INPUT_BG}!important;color:{TEXT}!important;}}
[role="option"]:hover{{background:{CARD_BORDER}!important;}}
.stNumberInput input{{background:{INPUT_BG}!important;border:1px solid {INPUT_BORDER}!important;border-radius:9px!important;color:{TEXT}!important;}}
.stNumberInput button{{background:{INPUT_BG}!important;border-color:{INPUT_BORDER}!important;color:{TEXT}!important;}}
[data-testid="stToggle"] p,[data-testid="stToggle"] span{{color:{TEXT}!important;}}
[data-testid="stRadio"] label p,[data-testid="stRadio"] label span{{color:{TEXT}!important;font-size:.88rem!important;}}
[data-testid="stMain"] div[data-testid="stButton"]>button{{background:linear-gradient(135deg,{ACCENT2},{ACCENT})!important;color:#fff!important;border:none!important;border-radius:9px!important;padding:.6rem 1rem!important;font-size:.95rem!important;font-weight:700!important;width:100%!important;box-shadow:0 4px 14px rgba(99,102,241,.4)!important;transition:all .2s!important;}}
[data-testid="stMain"] div[data-testid="stButton"]>button:hover{{opacity:.88!important;transform:translateY(-1px)!important;}}
[data-testid="stSidebar"]{{background:{SB_BG}!important;border-right:1px solid {SB_BORDER}!important;position:relative;z-index:100!important;}}
[data-testid="stSidebar"] *{{color:{TEXT}!important;}}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{{color:{ACCENT}!important;font-weight:700!important;margin-bottom:.4rem!important;}}
[data-testid="stSidebar"] hr{{border-color:{DIVIDER}!important;opacity:1!important;}}
[data-testid="stSidebar"] div[data-testid="stButton"]>button{{background:transparent!important;border:1px solid {CARD_BORDER}!important;color:{TEXT}!important;box-shadow:none!important;font-size:.85rem!important;padding:.4rem .8rem!important;width:100%!important;}}
[data-testid="stSidebar"] div[data-testid="stButton"]>button:hover{{background:{CARD_BORDER}!important;transform:none!important;}}
[data-testid="stAlert"]{{background:{CARD}!important;color:{TEXT}!important;border-radius:9px!important;}}
.nt-footer{{text-align:center;color:{FOOTER_COL};font-size:.78rem;padding:1.2rem 0 .5rem;border-top:1px solid {DIVIDER};margin-top:1.5rem;}}
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎧 NexTune")
    st.markdown("---")
    st.markdown("### 🎨 Theme")
    theme_choice = st.radio("theme_radio", options=["🌙 Dark", "☀️ Light"],
                            index=0 if is_dark else 1, horizontal=True,
                            label_visibility="collapsed")
    new_theme = "dark" if theme_choice == "🌙 Dark" else "light"
    if new_theme != st.session_state["theme"]:
        st.session_state["theme"] = new_theme
        st.rerun()

    st.markdown("---")
    st.markdown("### 📌 About")
    st.markdown("NexTune uses a **Gradient Boosting** model trained on 219 Amazon India products to predict headphone prices.")
    st.markdown("### 🤖 Model")
    if model:
        st.markdown(f"- **Type:** {type(model).__name__}\n- **Features:** 18 scraped + engineered\n- **MAPE:** ~17.6%\n- **R²:** 0.80")
    else:
        st.markdown("_Not loaded yet._")
    st.markdown("### 📖 How to Use")
    st.markdown("1. Pick a **brand** & **category**\n2. Set **rating** & **reviews**\n3. Choose **Bluetooth version**\n4. Toggle **features**\n5. Click **Predict Price**")
    st.markdown("---")
    if st.button("🎲 Sample Input", use_container_width=True):
        st.session_state.update({"brand": "Boat" if "Boat" in brands else brands[0],
                                  "rating": 4.2, "reviews": 5000, "anc": True,
                                  "bt_version": 5.3, "driver_mm": 10,
                                  "has_enc": True, "has_low_lat": True,
                                  "ipx_level": 4, "anc_db": 30,
                                  "result": None, "err": None})
    if st.button("🔄 Reset", use_container_width=True):
        for k, v in defaults.items():
            st.session_state[k] = v

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🎧 NexTune Price Predictor</h1>
  <p>Predict headphone market prices using machine learning — powered by 18 scraped features</p>
</div>""", unsafe_allow_html=True)

if load_err:
    st.warning(f"⚠️ {load_err}")

# ── MAIN COLUMNS ──────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2, gap="large")

# ════════════════════════════════
#  LEFT — Specifications
# ════════════════════════════════
with col_left:
    st.markdown(f'<div class="nt-card"><div class="nt-card-title">📋 Specifications</div>', unsafe_allow_html=True)

    brand = st.selectbox("🏷️ Brand", options=brands,
                         index=brands.index(st.session_state["brand"])
                               if st.session_state["brand"] in brands else 0)

    # Rating slider
    r = st.session_state["rating"]
    rc = "#ef4444" if r <= 2.0 else ("#f59e0b" if r < 4.0 else "#22c55e")
    st.markdown(f"""<div style="display:flex;justify-content:space-between;align-items:center;margin-top:.7rem;margin-bottom:.15rem;">
  <span style="font-size:.78rem;font-weight:600;color:{MUTED};text-transform:uppercase;letter-spacing:.07em;">⭐ Rating</span>
  <span style="font-size:.9rem;font-weight:700;color:{rc};">{r:.1f} / 5.0</span>
</div>""", unsafe_allow_html=True)
    rating = st.slider("rating_sl", 1.0, 5.0, value=float(st.session_state["rating"]),
                       step=0.1, label_visibility="collapsed")
    st.session_state["rating"] = rating

    reviews = st.number_input("💬 Reviews Count", min_value=0, max_value=500_000,
                               value=int(st.session_state["reviews"]), step=100)

    bt_version = st.selectbox("📶 Bluetooth Version",
                               options=[5.0, 5.1, 5.2, 5.3, 5.4, 6.0],
                               index=3,
                               format_func=lambda x: f"v{x}")

    driver_mm = st.number_input("🔊 Driver Size (mm)", min_value=6, max_value=50,
                                 value=int(st.session_state["driver_mm"]), step=1)

    ipx_level = st.selectbox("💧 IPX Water Resistance",
                              options=[0, 4, 5, 6, 7],
                              format_func=lambda x: "None" if x == 0 else f"IPX{x}")

    battery_hrs = st.slider("🔋 Battery Life (hours)", min_value=5, max_value=200,
                             value=int(st.session_state.get("battery_hrs", 30)), step=5)

    # Features section
    st.markdown(f'<div class="sec-label">Features</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        has_anc     = st.toggle("🔇 ANC",                value=bool(st.session_state["anc"]))
        has_enc     = st.toggle("🎙️ ENC",                value=bool(st.session_state["has_enc"]))
        has_hi_res  = st.toggle("🎵 Hi-Res Audio",       value=bool(st.session_state["has_hi_res"]))
        has_spatial = st.toggle("🌐 Spatial Audio",      value=bool(st.session_state["has_spatial"]))
        has_voice   = st.toggle("🗣️ Voice Assistant",    value=bool(st.session_state.get("has_voice", False)))
    with c2:
        has_dual        = st.toggle("📱 Dual Pairing",       value=bool(st.session_state["has_dual"]))
        has_codec       = st.toggle("⚡ Premium Codec",      value=bool(st.session_state["has_codec"]))
        has_low_lat     = st.toggle("🎮 Low Latency",        value=bool(st.session_state["has_low_lat"]))
        has_touch       = st.toggle("👆 Smart Touch",        value=bool(st.session_state.get("has_touch", False)))
        has_fast_charge = st.toggle("🔌 Fast Charging",      value=bool(st.session_state.get("has_fast_charge", False)))
        has_foldable    = st.toggle("📦 Foldable Design",    value=bool(st.session_state.get("has_foldable", False)))

    anc_db = 0
    if has_anc:
        anc_db = st.number_input("ANC Level (dB)", min_value=0, max_value=60,
                                  value=int(st.session_state["anc_db"]), step=5)

    st.markdown(f'<div style="border-top:1px solid {DIVIDER};margin:1rem 0;"></div>', unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict Price", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════
#  RIGHT — Result
# ════════════════════════════════
with col_right:
    if predict_btn:
        st.session_state.update({
            "brand": brand, "rating": rating, "reviews": reviews,
            "anc": has_anc, "bt_version": bt_version, "driver_mm": driver_mm,
            "has_enc": has_enc, "has_hi_res": has_hi_res, "has_spatial": has_spatial,
            "has_dual": has_dual, "has_codec": has_codec, "has_low_lat": has_low_lat,
            "ipx_level": ipx_level, "anc_db": anc_db, "battery_hrs": battery_hrs,
            "has_touch": has_touch, "has_voice": has_voice,
            "has_fast_charge": has_fast_charge, "has_foldable": has_foldable,
        })
        if model:
            with st.spinner("Analyzing..."):
                p, tier, e = predict_price(
                    model, scaler, encoders, features, brand_avg,
                    brand, rating, reviews, has_anc, bt_version, driver_mm,
                    has_enc, has_hi_res, has_spatial, has_dual, has_codec,
                    has_low_lat, ipx_level, anc_db
                )
            st.session_state["result"] = p
            st.session_state["tier"]   = tier
            st.session_state["err"]    = e
            st.session_state["has_touch"] = has_touch
        else:
            st.session_state["result"] = None
            st.session_state["err"]    = "Model not loaded. Run model_training.ipynb first."

    st.markdown(f'<div class="nt-card"><div class="nt-card-title">🎯 Prediction Result</div>', unsafe_allow_html=True)

    if st.session_state["err"]:
        st.error(st.session_state["err"])

    elif st.session_state["result"] is not None:
        price = st.session_state["result"]
        lo    = max(0, price * 0.9)
        hi    = price * 1.1
        seg, sc, sbg = price_segment(price)
        anc_lbl = "✅ Yes" if st.session_state["anc"] else "❌ No"
        tier_lbl = st.session_state.get("tier", "budget")

        st.markdown(f"""
<div style="text-align:center;padding:1.6rem 0 1rem;">
  <div style="font-size:.68rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:{ACCENT};margin-bottom:.5rem;">Recommended Market Price</div>
  <div style="font-size:3.8rem;font-weight:800;color:{PRICE_COL};line-height:1;">₹{price:,.0f}</div>
  <div style="font-size:.9rem;color:{MUTED};margin-top:.5rem;">Suggested Range: ₹{lo:,.0f} &nbsp;–&nbsp; ₹{hi:,.0f}</div>
  <div style="margin-top:.8rem;">
    <span style="display:inline-block;padding:.3rem 1rem;border-radius:999px;font-size:.8rem;font-weight:600;background:{sbg};color:{sc};border:1px solid {sc};">{seg}</span>
    &nbsp;
    <span style="display:inline-block;padding:.3rem 1rem;border-radius:999px;font-size:.8rem;font-weight:600;background:rgba(99,102,241,.15);color:{ACCENT};border:1px solid {ACCENT};">{tier_lbl} brand</span>
  </div>
</div>
<div style="display:flex;gap:.65rem;margin-top:.4rem;flex-wrap:wrap;">
  <div style="flex:1;min-width:80px;background:{STAT_BG};border:1px solid {CARD_BORDER};border-radius:9px;padding:.7rem .4rem;text-align:center;">
    <div style="font-size:.62rem;color:{FAINT};text-transform:uppercase;letter-spacing:.07em;">Predicted</div>
    <div style="font-size:.95rem;font-weight:700;color:{TEXT};margin-top:.15rem;">₹{price:,.0f}</div>
  </div>
  <div style="flex:1;min-width:80px;background:{STAT_BG};border:1px solid {CARD_BORDER};border-radius:9px;padding:.7rem .4rem;text-align:center;">
    <div style="font-size:.62rem;color:{FAINT};text-transform:uppercase;letter-spacing:.07em;">Min (−10%)</div>
    <div style="font-size:.95rem;font-weight:700;color:{TEXT};margin-top:.15rem;">₹{lo:,.0f}</div>
  </div>
  <div style="flex:1;min-width:80px;background:{STAT_BG};border:1px solid {CARD_BORDER};border-radius:9px;padding:.7rem .4rem;text-align:center;">
    <div style="font-size:.62rem;color:{FAINT};text-transform:uppercase;letter-spacing:.07em;">Max (+10%)</div>
    <div style="font-size:.95rem;font-weight:700;color:{TEXT};margin-top:.15rem;">₹{hi:,.0f}</div>
  </div>
  <div style="flex:1;min-width:80px;background:{STAT_BG};border:1px solid {CARD_BORDER};border-radius:9px;padding:.7rem .4rem;text-align:center;">
    <div style="font-size:.62rem;color:{FAINT};text-transform:uppercase;letter-spacing:.07em;">ANC</div>
    <div style="font-size:.85rem;font-weight:700;color:{TEXT};margin-top:.15rem;">{anc_lbl}</div>
  </div>
  <div style="flex:1;min-width:80px;background:{STAT_BG};border:1px solid {CARD_BORDER};border-radius:9px;padding:.7rem .4rem;text-align:center;">
    <div style="font-size:.62rem;color:{FAINT};text-transform:uppercase;letter-spacing:.07em;">Touch</div>
    <div style="font-size:.85rem;font-weight:700;color:{TEXT};margin-top:.15rem;">{"✅ Yes" if st.session_state.get("has_touch") else "❌ No"}</div>
  </div>
</div>""", unsafe_allow_html=True)

        # ── Battery description ──────────────────────────────────────────────
        batt = st.session_state.get("battery_hrs", 30)
        ipx  = st.session_state.get("ipx_level", 0)
        fc   = st.session_state.get("has_fast_charge", False)

        if batt <= 15:
            batt_icon, batt_label, batt_color = "🔴", "Low Battery Life", "#ef4444"
            batt_desc = f"Only {batt}h playtime — suitable for short commutes. Charge frequently."
        elif batt <= 30:
            batt_icon, batt_label, batt_color = "🟡", "Average Battery Life", "#f59e0b"
            batt_desc = f"{batt}h playtime — good for daily use. Charges every 1–2 days."
        elif batt <= 60:
            batt_icon, batt_label, batt_color = "🟢", "Good Battery Life", "#22c55e"
            batt_desc = f"{batt}h playtime — excellent for all-day use. Charges every 2–3 days."
        elif batt <= 100:
            batt_icon, batt_label, batt_color = "💚", "Long Battery Life", "#10b981"
            batt_desc = f"{batt}h playtime — great for travel & extended sessions. Rarely needs charging."
        else:
            batt_icon, batt_label, batt_color = "⚡", "Ultra-Long Battery Life", "#6366f1"
            batt_desc = f"{batt}h playtime — exceptional endurance. Ideal for frequent travellers."

        fc_note = " Fast charging supported — 10 min charge = ~2h playback." if fc else ""
        waterproof_note = f" IPX{ipx} rated — {'splash proof' if ipx == 4 else 'sweat & rain resistant' if ipx == 5 else 'water resistant' if ipx == 6 else 'waterproof' if ipx >= 7 else ''}." if ipx > 0 else ""

        st.markdown(f"""
<div style="margin-top:1rem;background:{STAT_BG};border:1px solid {batt_color}44;border-radius:12px;padding:14px 16px;">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
    <span style="font-size:1.2rem;">{batt_icon}</span>
    <span style="font-size:.8rem;font-weight:700;color:{batt_color};text-transform:uppercase;letter-spacing:.08em;">{batt_label}</span>
    <span style="margin-left:auto;font-size:.85rem;font-weight:700;color:{batt_color};">{batt}h</span>
  </div>
  <div style="font-size:.82rem;color:{MUTED};line-height:1.6;">{batt_desc}{fc_note}{waterproof_note}</div>
</div>""", unsafe_allow_html=True)

    else:
        st.markdown(f"""
<div style="text-align:center;padding:4rem 1rem;">
  <div style="font-size:3.2rem;">🎯</div>
  <div style="font-size:.92rem;color:{IDLE_P};margin-top:.9rem;line-height:1.7;">
    Fill in the inputs and click<br>
    <strong style="color:{ACCENT};">Predict Price</strong> to see results
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown('<div class="nt-footer">Built using ML + Streamlit &nbsp;·&nbsp; NexTune © 2025</div>',
            unsafe_allow_html=True)
