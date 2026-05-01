"""
NexTune — Retrain model without price_per_hour (target leakage fix)
Run: /Users/saispoorthyeturu/NexTune-1/.venv/bin/python3 scripts/retrain_model.py
"""
import pandas as pd
import numpy as np
import joblib, os, json, warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv('data/final-merged-dataset.csv')
print(f"Loaded: {df.shape}")

# ── Fix numeric types ─────────────────────────────────────────────────────────
num_cols = ['price_inr','rating','review_count','battery_life_hrs','bluetooth_version',
            'driver_size_mm','ipx_level','latency_ms','anc_db']
bin_cols = ['has_noise_cancellation','has_enc','has_usb_c','has_premium_codec',
            'has_touch_control','has_voice_assistant','has_fast_charge','has_dual_pairing',
            'has_gaming_mode','has_hi_res_audio','has_spatial_audio','has_low_latency']

for c in num_cols + bin_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
for c in bin_cols:
    if c in df.columns:
        df[c] = df[c].fillna(0).astype(int)

df = df.dropna(subset=['price_inr','brand'])
df = df[df['price_inr'] > 0].reset_index(drop=True)
df['log_price'] = np.log1p(df['price_inr'])

# ── Normalize brand names ─────────────────────────────────────────────────────
BRAND_MAP = {
    'boat': 'boAt', 'boAt': 'boAt', 'Boat': 'boAt',
    'sony': 'Sony', 'Sony WH': 'Sony', 'Sony WF': 'Sony', 'Sony WI': 'Sony',
    'jbl': 'JBL', 'JBL Tune': 'JBL', 'JBL Live': 'JBL', 'JBL Vibe': 'JBL',
    'noise': 'Noise', 'Noise Airwave': 'Noise', 'Noise Buds': 'Noise',
    'realme': 'Realme', 'Realme Buds': 'Realme',
    'oneplus': 'OnePlus', 'OnePlus Bullets': 'OnePlus', 'OnePlus Nord': 'OnePlus',
    'samsung': 'Samsung', 'apple': 'Apple',
    'sennheiser': 'Sennheiser',
    'marshall': 'Marshall', 'Marshall Monitor': 'Marshall',
    'skullcandy': 'Skullcandy', 'Skullcandy Hesh': 'Skullcandy',
    'edifier': 'Edifier', 'Edifier W': 'Edifier',
    'jlab': 'JLab', 'JLab JBuds': 'JLab',
    'mivi': 'Mivi',
    'ptron': 'pTron', 'PTron': 'pTron', 'pTron Bassbuds': 'pTron', 'PTron Studio': 'pTron',
    'portronics': 'Portronics', 'Portronics Muffs': 'Portronics', 'Portronics Twins': 'Portronics',
    'goboult': 'GOBOULT', 'GOBOult': 'GOBOULT',
    'fire-boltt': 'Fire-Boltt', 'Fire-BolttAero': 'Fire-Boltt',
    'zebronics': 'ZEBRONICS', 'Zebronics Duke': 'ZEBRONICS',
    'hammer': 'HAMMER', 'HAmmer': 'HAMMER',
    'truke': 'truke', 'iqoo': 'iQOO', 'iQOO TWS': 'iQOO',
    'soundcore': 'soundcore',
    'boult': 'Boult Audio', 'Boult Audio AirBass': 'Boult Audio', 'Boult Audio Z': 'Boult Audio',
    'philips': 'Philips', 'PHILIPS': 'Philips', 'hp': 'HP',
}

def normalize_brand(b):
    if pd.isna(b): return 'Unknown'
    b = str(b).strip()
    if b in BRAND_MAP: return BRAND_MAP[b]
    b_lower = b.lower()
    for k, v in BRAND_MAP.items():
        if b_lower.startswith(k.lower()): return v
    return b.split()[0] if b else 'Unknown'

df['brand'] = df['brand'].apply(normalize_brand)

# ── Brand tier ────────────────────────────────────────────────────────────────
brand_avg = df.groupby('brand')['price_inr'].mean()
df['brand_tier'] = df['brand'].apply(
    lambda b: 'premium' if brand_avg.get(b,0)>=10000
              else ('mid' if brand_avg.get(b,0)>=3000 else 'budget')
)

# ── Encode categoricals ───────────────────────────────────────────────────────
le_dict = {}
for col in ['category','brand','brand_tier']:
    if col in df.columns:
        le = LabelEncoder()
        df[col+'_enc'] = le.fit_transform(df[col].fillna('Unknown').astype(str))
        le_dict[col] = le

# ── Engineered features (NO price_per_hour — it leaks the target) ─────────────
df['has_anc']     = df['has_noise_cancellation'].fillna(0).astype(int)
df['bt_major']    = df['bluetooth_version'].apply(lambda x: int(x) if pd.notna(x) else 5)
df['high_rating'] = (df['rating'] >= 4.0).astype(int)

print(f"Clean dataset : {len(df)} products")
print(f"Price range   : ₹{df.price_inr.min():.0f} – ₹{df.price_inr.max():.0f}")
print(f"Unique brands : {df.brand.nunique()}: {sorted(df.brand.unique())}")

# ── Feature selection (price_per_hour REMOVED) ────────────────────────────────
FEATURES = [
    'brand_tier_enc',    # strongest signal
    'brand_enc',
    'has_anc',
    'anc_db',
    'rating',
    'high_rating',
    'has_hi_res_audio',
    'has_spatial_audio',
    'bluetooth_version',
    'bt_major',
    'has_dual_pairing',
    'has_premium_codec',
    'has_enc',
    'review_count',
    'ipx_level',
    'driver_size_mm',
    'has_low_latency',
]
FEATURES = [f for f in FEATURES if f in df.columns]
TARGET   = 'log_price'

X = df[FEATURES].fillna(df[FEATURES].median())
y = df[TARGET]
print(f"\nFeatures ({len(FEATURES)}): {FEATURES}")

# ── Train/test split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
_, _, _, y_test_orig = train_test_split(X, df['price_inr'], test_size=0.2, random_state=42)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Train models ──────────────────────────────────────────────────────────────
models = {
    'Ridge':            Ridge(alpha=10.0),
    'Random Forest':    RandomForestRegressor(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1),
    'Gradient Boosting':GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42),
}

results = {}
for name, m in models.items():
    m.fit(X_train_sc, y_train)
    y_pred = np.expm1(m.predict(X_test_sc))
    y_true = y_test_orig.values
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    results[name] = {'R2': r2, 'RMSE': rmse, 'MAPE': mape, 'model': m}
    print(f"{name:22} R²={r2:.4f}  RMSE=₹{rmse:.0f}  MAPE={mape:.1f}%")

# ── Pick best ─────────────────────────────────────────────────────────────────
best_name  = max(results, key=lambda k: results[k]['R2'])
best_model = results[best_name]['model']
print(f"\n🏆 Best: {best_name}  R²={results[best_name]['R2']:.4f}  MAPE={results[best_name]['MAPE']:.1f}%")

# ── Feature importance ────────────────────────────────────────────────────────
if hasattr(best_model, 'feature_importances_'):
    fi = sorted(zip(FEATURES, best_model.feature_importances_), key=lambda x: -x[1])
    print("\nFeature importances:")
    for f, imp in fi:
        print(f"  {f:<25} {imp:.4f}")

# ── Save artifacts ────────────────────────────────────────────────────────────
SAVE_DIR = 'models'
os.makedirs(SAVE_DIR, exist_ok=True)

joblib.dump(best_model,           f'{SAVE_DIR}/price_predictor.pkl')
joblib.dump(scaler,               f'{SAVE_DIR}/scaler.pkl')
joblib.dump(le_dict,              f'{SAVE_DIR}/label_encoders.pkl')
joblib.dump(FEATURES,             f'{SAVE_DIR}/features.pkl')
joblib.dump(brand_avg.to_dict(),  f'{SAVE_DIR}/brand_avg.pkl')

# Save clean brand list
brands_list = sorted(df['brand'].unique().tolist())
with open(f'{SAVE_DIR}/brands.json', 'w') as f:
    json.dump(brands_list, f)

print(f"\n✅ Saved to {SAVE_DIR}/")
print(f"   Brands: {brands_list}")

# ── Quick sanity check ────────────────────────────────────────────────────────
print("\n── Sanity checks ──")
def quick_predict(brand, has_anc, bt_version, driver_mm, rating, reviews, anc_db=0):
    avg  = brand_avg.get(brand, brand_avg.mean())
    tier = 'premium' if avg >= 10000 else ('mid' if avg >= 3000 else 'budget')
    def enc(col, val):
        le = le_dict.get(col)
        return int(le.transform([val])[0]) if le and val in le.classes_ else 0
    row = {
        'brand_tier_enc': enc('brand_tier', tier), 'brand_enc': enc('brand', brand),
        'has_anc': int(has_anc), 'anc_db': float(anc_db),
        'rating': float(rating), 'high_rating': int(rating >= 4.0),
        'has_hi_res_audio': 0, 'has_spatial_audio': 0,
        'bluetooth_version': float(bt_version), 'bt_major': int(bt_version),
        'has_dual_pairing': 0, 'has_premium_codec': 0, 'has_enc': 0,
        'review_count': float(reviews), 'ipx_level': 0,
        'driver_size_mm': float(driver_mm), 'has_low_latency': 0,
    }
    vec = np.array([[row.get(f, 0) for f in FEATURES]])
    vec_sc = scaler.transform(vec)
    price = float(np.expm1(best_model.predict(vec_sc)[0]))
    print(f"  {brand:12} ANC={has_anc} BT={bt_version} → ₹{price:,.0f}  (tier={tier})")

quick_predict('Sony',       has_anc=True,  bt_version=5.2, driver_mm=40, rating=4.5, reviews=5000, anc_db=30)
quick_predict('boAt',       has_anc=False, bt_version=5.3, driver_mm=10, rating=4.0, reviews=10000)
quick_predict('Sennheiser', has_anc=True,  bt_version=5.2, driver_mm=40, rating=4.3, reviews=2000, anc_db=25)
quick_predict('Noise',      has_anc=False, bt_version=5.0, driver_mm=10, rating=4.1, reviews=3000)
quick_predict('JBL',        has_anc=True,  bt_version=5.3, driver_mm=40, rating=4.2, reviews=8000, anc_db=20)
