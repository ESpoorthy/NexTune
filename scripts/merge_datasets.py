"""
Dataset Merger — NexTune Price Prediction
Combines all scraped sources into one unified dataset with 22 features.
"""

import json, re, os, sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.scrapers.enhanced_scraper import EnhancedScraper

SCHEMA = {
    'product_name': None, 'brand': None, 'country_of_origin': None,
    'price_inr': None, 'rating': None, 'review_count': 0,
    'category': 'unknown', 'source': 'unknown',
    'battery_life_hrs': None, 'bluetooth_version': None, 'driver_size_mm': None,
    'mic_count': None, 'anc_db': None, 'ipx_level': None, 'ipx_rating': None,
    'charging_time_hrs': None, 'latency_ms': None, 'range_m': None, 'eq_modes': None,
    'has_noise_cancellation': 0, 'has_enc': 0, 'has_usb_c': 0, 'has_premium_codec': 0,
    'has_touch_control': 0, 'has_voice_assistant': 0, 'has_fast_charge': 0,
    'has_dual_pairing': 0, 'has_gaming_mode': 0, 'has_hi_res_audio': 0,
    'has_spatial_audio': 0, 'has_low_latency': 0,
    'active_noise_cancellation': None,
}

def _schema(df, source):
    for col, default in SCHEMA.items():
        if col not in df.columns:
            df[col] = default
    df['source'] = source
    return df

def _num(val):
    if not val or (isinstance(val, float) and pd.isna(val)): return None
    m = re.search(r'(\d+(?:\.\d+)?)', str(val))
    return float(m.group(1)) if m else None

def _bt(val):
    if not val or (isinstance(val, float) and pd.isna(val)): return None
    m = re.search(r'([0-9]\.[0-9])', str(val))
    return float(m.group(1)) if m else None

def load_all():
    datasets = []

    # 1. Enhanced dataset
    if os.path.exists('data/enhanced-headphones-dataset.csv'):
        df = pd.read_csv('data/enhanced-headphones-dataset.csv')
        datasets.append(_schema(df, 'enhanced_scrape'))
        print(f"  ✓ enhanced-headphones-dataset.csv : {len(df)}")

    # 2. NexTune cleaned
    if os.path.exists('data/nexttune-cleaned-data.csv'):
        raw = pd.read_csv('data/nexttune-cleaned-data.csv')
        df  = pd.DataFrame({'product_name': raw['Name'], 'brand': raw['Brand'],
                             'price_inr': raw['Price'], 'rating': raw['Rating']})
        df  = df[df['price_inr'] > 0]
        datasets.append(_schema(df, 'nexttune_cleaned'))
        print(f"  ✓ nexttune-cleaned-data.csv       : {len(df)}")

    # 3. earbuds_by_company.json (teammate's file — 148 products + 4 new features)
    if os.path.exists('data/earbuds_by_company.json'):
        data = json.load(open('data/earbuds_by_company.json'))
        rows = []
        for b in data:
            for p in b['products']:
                f = p.get('features', {})
                rows.append({
                    'product_name': p.get('name'), 'brand': b['brand'],
                    'country_of_origin': b['country_of_origin'],
                    'price_inr': p.get('price_inr'), 'rating': p.get('rating'),
                    'review_count': p.get('review_count', 0),
                    'battery_life_hrs': _num(f.get('battery_life')),
                    'bluetooth_version': _bt(f.get('bluetooth_version')),
                    'driver_size_mm': _num(f.get('driver_size')),
                    'mic_count': f.get('microphones'),
                    'has_noise_cancellation': 1 if f.get('noise_cancellation') else 0,
                    'has_enc':           1 if f.get('enc') else 0,
                    'has_touch_control': 1 if f.get('touch_controls') else 0,
                    'has_voice_assistant':1 if f.get('voice_assistant') else 0,
                    'has_fast_charge':   1 if f.get('fast_charging') else 0,
                    'has_dual_pairing':  1 if f.get('multipoint_connection') else 0,
                    'has_gaming_mode':   1 if f.get('gaming_mode') else 0,
                    'has_hi_res_audio':  1 if f.get('hi_res_audio') else 0,
                    'has_spatial_audio': 1 if f.get('spatial_audio') else 0,
                    'has_low_latency':   1 if f.get('low_latency') else 0,
                    'has_ipx':           1 if f.get('water_resistance') else 0,
                })
        df = pd.DataFrame(rows)
        datasets.append(_schema(df, 'earbuds_by_company'))
        print(f"  ✓ earbuds_by_company.json         : {len(df)}")

    # 4. Raw Amazon scrape (if exists)
    if os.path.exists('data/raw_amazon_scrape.csv'):
        df = pd.read_csv('data/raw_amazon_scrape.csv')
        datasets.append(_schema(df, 'amazon_scrape'))
        print(f"  ✓ raw_amazon_scrape.csv           : {len(df)}")

    return datasets

def enhance_new(df):
    scraper = EnhancedScraper()
    rows = []
    for _, row in df.iterrows():
        row   = row.copy()
        feats = scraper.extract_with_prompt_engineering(str(row.get('product_name', '')))
        for k, v in feats.items():
            cur = row.get(k)
            if cur is None or (isinstance(cur, float) and pd.isna(cur)) or cur == 'unknown' or cur == 0:
                row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    print("=" * 65)
    print("DATASET MERGER — NexTune")
    print("=" * 65)
    print("\nLoading sources...")
    datasets = load_all()

    for i in range(1, len(datasets)):
        datasets[i] = enhance_new(datasets[i])

    df = pd.concat(datasets, ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(subset=['product_name'], keep='first')
    print(f"\n  Merged → {len(df)} unique records ({before-len(df)} duplicates removed)")

    # Summary
    print(f"\n  Price  : ₹{df.price_inr.min():.0f}–₹{df.price_inr.max():.0f}  (median ₹{df.price_inr.median():.0f})")
    print(f"  Brands : {df.brand.nunique()}")
    print(f"  Sources: {df.source.value_counts().to_dict()}")
    if 'country_of_origin' in df.columns:
        print(f"  Countries: {df.country_of_origin.value_counts().head(5).to_dict()}")

    df.to_csv('data/final-merged-dataset.csv', index=False)
    df.to_csv('data/enhanced-headphones-dataset.csv', index=False)
    print("\n✅ Saved → data/final-merged-dataset.csv")
    print("✅ Updated → data/enhanced-headphones-dataset.csv")

if __name__ == "__main__":
    main()
