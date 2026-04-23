"""NexTune – Headphone Price Predictor"""

import streamlit as st
import numpy as np
import pickle, os, json, base64

st.set_page_config(page_title="NexTune", page_icon="🎧", layout="wide", initial_sidebar_state="expanded")

# ── helpers ──────────────────────────────────────────────────────────────────────
def get_b64(path):
    try:
        with open(path,"rb") as f: return base64.b64encode(f.read()).decode()
    except: return None

@st.cache_resource(show_spinner=False)
def load_artifacts():
    if not os.path.exists("model.pkl") or not os.path.exists("encoder.pkl"):
        return None, None, "model.pkl / encoder.pkl not found."
    try:
        with open("model.pkl","rb") as f: model = pickle.load(f)
        with open("encoder.pkl","rb") as f: encoder = pickle.load(f)
        return model, encoder, None
    except Exception as e:
        return None, None, str(e)

@st.cache_data(show_spinner=False)
def load_brands():
    try:
        with open("data/earbuds_by_company.json","r",encoding="utf-8") as f:
            return sorted({x["brand"] for x in json.load(f) if "brand" in x})
    except:
        return ["Boat","Sony","JBL","Noise","Realme","OnePlus","Samsung","Apple"]

def do_predict(model, encoder, brand, rating, reviews, anc):
    try:
        enc = encoder.transform([brand])[0]
        return float(model.predict(np.array([[enc, rating, reviews, anc]]))[0]), None
    except ValueError:
        return None, f"Brand '{brand}' not in encoder."
    except Exception as e:
        return None, str(e)

def price_segment(p):
    if p < 1000:   return "💚 Budget",    "#22c55e","#052e16"
    elif p < 3000: return "💙 Mid-Range", "#3b82f6","#0c1a3a"
    elif p < 8000: return "💜 Premium",   "#a78bfa","#1e1040"
    else:          return "🏆 Flagship",  "#f59e0b","#2d1a00"

# ── load ─────────────────────────────────────────────────────────────────────────
bg      = get_b64("assets/bg.jpg")
model, encoder, load_err = load_artifacts()
brands  = sorted(list(encoder.classes_)) if encoder else load_brands()

# ── session state ─────────────────────────────────────────────────────────────────
for k,v in {"theme":"dark","brand":brands[0],"rating":4.0,"reviews":500,
            "anc":False,"result":None,"err":None}.items():
    if k not in st.session_state: st.session_state[k] = v

# ── theme values ─────────────────────────────────────────────────────────────────
is_dark = st.session_state["theme"] == "dark"

APP_BG       = "#0f172a"          if is_dark else "#eef2f7"
OVERLAY      = "rgba(8,10,28,.83)"if is_dark else "rgba(220,228,240,.80)"
CARD         = "rgba(30,41,59,.85)"if is_dark else "rgba(255,255,255,.90)"
CARD_BORDER  = "#334155"          if is_dark else "#c7d2fe"
SB_BG        = "#0c1525"           if is_dark else "#f8fafc"
SB_BORDER    = "#1e293b"          if is_dark else "#c7d2fe"
TEXT         = "#e2e8f0"          if is_dark else "#0f172a"
MUTED        = "#94a3b8"          if is_dark else "#334155"
FAINT        = "#475569"          if is_dark else "#64748b"
INPUT_BG     = "#0d1526"          if is_dark else "#f8fafc"
INPUT_BORDER = "#334155"          if is_dark else "#94a3b8"
ACCENT       = "#6366f1"          if is_dark else "#4f46e5"
ACCENT2      = "#4f46e5"          if is_dark else "#4338ca"
PRICE_COL    = "#60a5fa"          if is_dark else "#2563eb"
HERO_BG      = "linear-gradient(135deg,#13132b,#1e1050,#13132b)" if is_dark else "linear-gradient(135deg,rgba(237,233,254,.95),rgba(219,234,254,.95),rgba(237,233,254,.95))"
HERO_BORDER  = "#312e81"          if is_dark else "#a5b4fc"
DIVIDER      = "#1e293b"          if is_dark else "#cbd5e1"
STAT_BG      = "rgba(13,21,38,.7)"if is_dark else "rgba(241,245,249,.9)"
FOOTER_COL   = "#334155"          if is_dark else "#64748b"
IDLE_P       = "#475569"          if is_dark else "#64748b"

# ── inject CSS ───────────────────────────────────────────────────────────────────
bg_css = f"background-image:url('data:image/jpg;base64,{bg}');background-size:cover;background-position:center;background-attachment:fixed;" if bg else ""

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
*{{font-family:'Inter',sans-serif;box-sizing:border-box;}}
.stApp{{background-color:{APP_BG};color:{TEXT};{bg_css}}}
.stApp::before{{content:'';position:fixed;inset:0;background:{OVERLAY};z-index:0;pointer-events:none;}}
.stApp>*{{position:relative;z-index:1;}}
/* sidebar must sit above the overlay */
[data-testid="stSidebar"]{{position:relative;z-index:100!important;}}
header[data-testid="stHeader"],footer{{display:none!important;}}
.block-container{{padding:1.8rem 2rem 1rem!important;max-width:100%!important;}}

/* hero */
.hero{{background:{HERO_BG};border:1px solid {HERO_BORDER};border-radius:16px;
       padding:2rem;text-align:center;margin-bottom:1.5rem;
       box-shadow:0 0 40px rgba(99,102,241,.15);}}
.hero h1{{font-size:2.4rem;font-weight:800;
          background:linear-gradient(90deg,#818cf8,#60a5fa,#a78bfa);
          -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 .3rem;}}
.hero p{{color:{MUTED};font-size:1rem;margin:0;}}

/* cards */
.nt-card{{background:{CARD};border:1px solid {CARD_BORDER};border-radius:14px;
          padding:22px 20px;backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
          box-shadow:0 6px 28px rgba(0,0,0,.18);}}
.nt-card-title{{font-size:.95rem;font-weight:700;color:{TEXT};
                padding-bottom:.7rem;margin-bottom:1rem;
                border-bottom:1px solid {CARD_BORDER};}}

/* widget labels */
[data-testid="stWidgetLabel"] p{{color:{MUTED}!important;font-size:.78rem!important;
    font-weight:600!important;text-transform:uppercase!important;letter-spacing:.07em!important;}}

/* selectbox */
.stSelectbox>div>div{{background:{INPUT_BG}!important;border:1px solid {INPUT_BORDER}!important;
    border-radius:9px!important;color:{TEXT}!important;}}
.stSelectbox>div>div>div{{color:{TEXT}!important;}}
.stSelectbox svg{{fill:{MUTED}!important;}}
[role="option"]{{background:{INPUT_BG}!important;color:{TEXT}!important;}}
[role="option"]:hover{{background:{CARD_BORDER}!important;}}

/* number input */
.stNumberInput input{{background:{INPUT_BG}!important;border:1px solid {INPUT_BORDER}!important;
    border-radius:9px!important;color:{TEXT}!important;}}
.stNumberInput button{{background:{INPUT_BG}!important;border-color:{INPUT_BORDER}!important;color:{TEXT}!important;}}

/* slider */
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"]{{color:{MUTED}!important;}}
[data-testid="stSlider"] div[data-baseweb="slider"] div[data-testid="stThumbValue"]{{
    color:{TEXT}!important;background:{INPUT_BG}!important;}}

/* toggle */
[data-testid="stToggle"] p,[data-testid="stToggle"] span{{color:{TEXT}!important;}}

/* radio */
[data-testid="stRadio"] label p,[data-testid="stRadio"] label span{{color:{TEXT}!important;font-size:.88rem!important;}}
[data-testid="stRadio"]>div{{background:transparent!important;}}

/* predict button */
[data-testid="stMain"] div[data-testid="stButton"]>button{{
    background:linear-gradient(135deg,{ACCENT2},{ACCENT})!important;
    color:#fff!important;border:none!important;border-radius:9px!important;
    padding:.6rem 1rem!important;font-size:.95rem!important;font-weight:700!important;
    width:100%!important;box-shadow:0 4px 14px rgba(99,102,241,.4)!important;transition:all .2s!important;}}
[data-testid="stMain"] div[data-testid="stButton"]>button:hover{{opacity:.88!important;transform:translateY(-1px)!important;}}

/* sidebar */
[data-testid="stSidebar"]{{
    background:{SB_BG}!important;
    border-right:1px solid {SB_BORDER}!important;
    position:relative;
    z-index:100!important;
}}
/* all text inside sidebar */
[data-testid="stSidebar"] *{{
    color:{TEXT}!important;
}}
/* headings get accent color */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3{{
    color:{ACCENT}!important;
    font-weight:700!important;
    margin-bottom:.4rem!important;
}}
[data-testid="stSidebar"] hr{{
    border-color:{DIVIDER}!important;
    opacity:1!important;
}}
/* sidebar radio */
[data-testid="stSidebar"] [data-testid="stRadio"] label p,
[data-testid="stSidebar"] [data-testid="stRadio"] label span{{
    color:{TEXT}!important;font-size:.9rem!important;font-weight:600!important;}}
/* sidebar buttons */
[data-testid="stSidebar"] div[data-testid="stButton"]>button{{
    background:transparent!important;
    border:1px solid {CARD_BORDER}!important;
    color:{TEXT}!important;
    box-shadow:none!important;
    font-size:.85rem!important;
    padding:.4rem .8rem!important;
    width:100%!important;
}}
[data-testid="stSidebar"] div[data-testid="stButton"]>button:hover{{
    background:{CARD_BORDER}!important;
    transform:none!important;
}}

/* alert */
[data-testid="stAlert"]{{background:{CARD}!important;color:{TEXT}!important;border-radius:9px!important;}}

/* footer */
.nt-footer{{text-align:center;color:{FOOTER_COL};font-size:.78rem;
            padding:1.2rem 0 .5rem;border-top:1px solid {DIVIDER};margin-top:1.5rem;}}
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎧 NexTune")
    st.markdown("---")

    st.markdown("### 🎨 Theme")
    theme_choice = st.radio(
        "theme_radio",
        options=["🌙 Dark", "☀️ Light"],
        index=0 if is_dark else 1,
        horizontal=True,
        label_visibility="collapsed",
    )
    new_theme = "dark" if theme_choice == "🌙 Dark" else "light"
    if new_theme != st.session_state["theme"]:
        st.session_state["theme"] = new_theme
        st.rerun()

    st.markdown("---")
    st.markdown("### 📌 About")
    st.markdown("NexTune uses a trained **ML model** to predict headphone prices based on brand, rating, reviews, and ANC.")

    st.markdown("### 🤖 Model")
    if model:
        st.markdown(f"- **Type:** {type(model).__name__}\n- **Features:** Brand, Rating, Reviews, ANC\n- **Output:** ₹ INR")
    else:
        st.markdown("_Not loaded yet._")

    st.markdown("### 📖 How to Use")
    st.markdown("1. Pick a **brand**\n2. Set **rating**\n3. Enter **reviews**\n4. Toggle **ANC**\n5. Click **Predict Price**")

    st.markdown("---")
    if st.button("🎲 Sample Input", use_container_width=True):
        st.session_state.update({"brand":"Boat" if "Boat" in brands else brands[0],
                                  "rating":4.3,"reviews":1200,"anc":True,
                                  "result":None,"err":None})
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.update({"brand":brands[0],"rating":4.0,"reviews":500,
                                  "anc":False,"result":None,"err":None})

# ── HERO ──────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🎧 NexTune Price Predictor</h1>
  <p>Predict headphone prices using machine learning</p>
</div>""", unsafe_allow_html=True)

if load_err:
    st.warning(f"⚠️ {load_err}")

# ── MAIN COLUMNS ─────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2, gap="large")

# ════════════════════════════════
#  LEFT — Specifications
# ════════════════════════════════
with col_left:
    st.markdown(f'<div class="nt-card"><div class="nt-card-title">📋 Specifications</div>', unsafe_allow_html=True)

    brand = st.selectbox("🏷️ Brand", options=brands,
                         index=brands.index(st.session_state["brand"])
                               if st.session_state["brand"] in brands else 0)

    # rating label row
    r = st.session_state["rating"]
    rc = "#ef4444" if r <= 2.0 else ("#f59e0b" if r < 4.0 else "#22c55e")
    rh = "Low rating" if r <= 2.0 else ("Average rating" if r < 4.0 else "High rating")
    st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
            margin-top:.7rem;margin-bottom:.15rem;">
  <span style="font-size:.78rem;font-weight:600;color:{MUTED};
               text-transform:uppercase;letter-spacing:.07em;">⭐ Rating</span>
  <span style="font-size:.9rem;font-weight:700;color:{rc};">{r:.1f} / 5.0</span>
</div>""", unsafe_allow_html=True)

    st.markdown(f"""<style>
[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"]{{
    background:{rc}!important;border-color:{rc}!important;box-shadow:0 0 8px {rc}88!important;}}
</style>""", unsafe_allow_html=True)

    rating = st.slider("rating_sl", 1.0, 5.0,
                       value=float(st.session_state["rating"]),
                       step=0.1, label_visibility="collapsed")
    st.session_state["rating"] = rating

    rc2 = "#ef4444" if rating<=2.0 else ("#f59e0b" if rating<4.0 else "#22c55e")
    rh2 = "Low rating" if rating<=2.0 else ("Average rating" if rating<4.0 else "High rating")
    st.markdown(f'<div style="margin-top:-.3rem;margin-bottom:.9rem;"><span style="font-size:.73rem;color:{rc2};font-weight:600;">● &nbsp;{rh2}</span></div>', unsafe_allow_html=True)

    reviews = st.number_input("💬 Reviews", min_value=0, max_value=500_000,
                               value=int(st.session_state["reviews"]), step=100)
    anc = st.toggle("🔇 Active Noise Cancellation (ANC)", value=bool(st.session_state["anc"]))

    st.markdown(f'<div style="border-top:1px solid {DIVIDER};margin:1rem 0;"></div>', unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict Price", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)   # close nt-card

# ════════════════════════════════
#  RIGHT — Prediction Result
# ════════════════════════════════
with col_right:
    # run prediction
    if predict_btn:
        st.session_state.update({"brand":brand,"rating":rating,"reviews":reviews,"anc":anc})
        if model and encoder:
            with st.spinner("Analyzing..."):
                p, e = do_predict(model, encoder, brand, rating, int(reviews), int(anc))
            st.session_state["result"] = p
            st.session_state["err"]    = e
        else:
            st.session_state["result"] = None
            st.session_state["err"]    = "Model not loaded."

    st.markdown(f'<div class="nt-card"><div class="nt-card-title">🎯 Prediction Result</div>', unsafe_allow_html=True)

    if st.session_state["err"]:
        st.error(st.session_state["err"])

    elif st.session_state["result"] is not None:
        price = st.session_state["result"]
        lo, hi = max(0, price-300), price+300
        seg, sc, sbg = price_segment(price)
        anc_lbl = "✅ Yes" if st.session_state["anc"] else "❌ No"

        st.markdown(f"""
<div style="text-align:center;padding:1.6rem 0 1rem;">
  <div style="font-size:.68rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;
              color:{ACCENT};margin-bottom:.5rem;">Estimated Price</div>
  <div style="font-size:3.8rem;font-weight:800;color:{PRICE_COL};line-height:1;">₹{price:,.0f}</div>
  <div style="font-size:.9rem;color:{MUTED};margin-top:.5rem;">₹{lo:,.0f} &nbsp;–&nbsp; ₹{hi:,.0f}</div>
  <div style="margin-top:.8rem;">
    <span style="display:inline-block;padding:.3rem 1rem;border-radius:999px;font-size:.8rem;
                 font-weight:600;background:{sbg};color:{sc};border:1px solid {sc};">{seg}</span>
  </div>
</div>
<div style="display:flex;gap:.65rem;margin-top:.4rem;">
  <div style="flex:1;background:{STAT_BG};border:1px solid {CARD_BORDER};border-radius:9px;padding:.7rem .4rem;text-align:center;">
    <div style="font-size:.62rem;color:{FAINT};text-transform:uppercase;letter-spacing:.07em;">Predicted</div>
    <div style="font-size:.95rem;font-weight:700;color:{TEXT};margin-top:.15rem;">₹{price:,.0f}</div>
  </div>
  <div style="flex:1;background:{STAT_BG};border:1px solid {CARD_BORDER};border-radius:9px;padding:.7rem .4rem;text-align:center;">
    <div style="font-size:.62rem;color:{FAINT};text-transform:uppercase;letter-spacing:.07em;">Min</div>
    <div style="font-size:.95rem;font-weight:700;color:{TEXT};margin-top:.15rem;">₹{lo:,.0f}</div>
  </div>
  <div style="flex:1;background:{STAT_BG};border:1px solid {CARD_BORDER};border-radius:9px;padding:.7rem .4rem;text-align:center;">
    <div style="font-size:.62rem;color:{FAINT};text-transform:uppercase;letter-spacing:.07em;">Max</div>
    <div style="font-size:.95rem;font-weight:700;color:{TEXT};margin-top:.15rem;">₹{hi:,.0f}</div>
  </div>
  <div style="flex:1;background:{STAT_BG};border:1px solid {CARD_BORDER};border-radius:9px;padding:.7rem .4rem;text-align:center;">
    <div style="font-size:.62rem;color:{FAINT};text-transform:uppercase;letter-spacing:.07em;">ANC</div>
    <div style="font-size:.85rem;font-weight:700;color:{TEXT};margin-top:.15rem;">{anc_lbl}</div>
  </div>
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

    st.markdown("</div>", unsafe_allow_html=True)   # close nt-card

# ── FOOTER ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="nt-footer">Built using ML + Streamlit &nbsp;·&nbsp; NexTune © 2025</div>',
            unsafe_allow_html=True)
