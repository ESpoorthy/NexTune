"""
Data Preparation — NexTune Price Prediction
28 model features: 22 scraped + 6 engineered (including country_of_origin).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_FEATURES = [
    # Categorical encoded
    'category_enc', 'brand_enc', 'brand_tier_enc', 'country_enc',
    # Core scraped numerical
    'rating', 'review_count',
    'battery_life_hrs', 'driver_size_mm',
    'bluetooth_version', 'bt_major',
    'mic_count', 'anc_db', 'ipx_level',
    'charging_time_hrs', 'latency_ms',
    # Binary scraped (original 7)
    'has_noise_cancellation', 'has_usb_c', 'has_premium_codec',
    'has_touch_control', 'has_voice_assistant',
    'has_fast_charge', 'has_dual_pairing',
    # Binary scraped (4 new from earbuds_by_company.json)
    'has_enc', 'has_gaming_mode', 'has_hi_res_audio',
    'has_spatial_audio', 'has_low_latency',
    # Engineered
    'has_ipx', 'high_rating', 'price_per_hour',
]

CATEGORICAL = ['category', 'brand', 'brand_tier', 'country_of_origin']


class DataPreparation:
    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.feature_columns = MODEL_FEATURES
        self.target_column   = 'price_inr'

    def load_data(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} cols from {filepath}")
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        before = len(df)
        df = df.dropna(subset=['price_inr', 'brand'])
        df = df[df['price_inr'] > 0]
        logger.info(f"Dropped {before-len(df)} rows (missing price/brand)")

        for col in ['battery_life_hrs', 'driver_size_mm']:
            if col in df.columns:
                df[col] = df.groupby('category')[col].transform(lambda x: x.fillna(x.median()))

        if 'bluetooth_version' in df.columns:
            mode = df['bluetooth_version'].mode()
            df['bluetooth_version'] = df['bluetooth_version'].fillna(mode[0] if len(mode) else 5.0)

        for col in ['mic_count','anc_db','ipx_level','charging_time_hrs','latency_ms','range_m','eq_modes']:
            if col in df.columns: df[col] = df[col].fillna(0)

        if 'rating'       in df.columns: df['rating']       = df['rating'].fillna(df['rating'].median())
        if 'review_count' in df.columns: df['review_count'] = df['review_count'].fillna(0)

        binary_cols = ['has_noise_cancellation','has_enc','has_usb_c','has_premium_codec',
                       'has_touch_control','has_voice_assistant','has_fast_charge','has_dual_pairing',
                       'has_gaming_mode','has_hi_res_audio','has_spatial_audio','has_low_latency',
                       'active_noise_cancellation']
        for col in binary_cols:
            if col in df.columns: df[col] = df[col].fillna(0)

        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(subset=['product_name'], keep='first')
        logger.info(f"Removed {before-len(df)} duplicates")
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # has_anc — unify both ANC columns
        anc_cols = [c for c in ['has_noise_cancellation','active_noise_cancellation'] if c in df.columns]
        df['has_anc'] = df[anc_cols].max(axis=1).fillna(0).astype(int) if anc_cols else 0

        df['has_ipx']        = (df['ipx_level'].fillna(0) > 0).astype(int) if 'ipx_level' in df.columns else 0
        df['price_per_hour'] = df['price_inr'] / (df['battery_life_hrs'].fillna(0) + 1)
        df['bt_major']       = df['bluetooth_version'].apply(lambda x: int(x) if pd.notna(x) else 5) \
                               if 'bluetooth_version' in df.columns else 5
        df['high_rating']    = (df['rating'] >= 4.0).astype(int) if 'rating' in df.columns else 0

        brand_avg = df.groupby('brand')['price_inr'].mean()
        df['brand_tier'] = df['brand'].apply(
            lambda b: 'premium' if brand_avg.get(b,0) >= 10000 else ('mid' if brand_avg.get(b,0) >= 3000 else 'budget')
        )

        # Fill new binary columns
        for col in ['has_enc','has_gaming_mode','has_hi_res_audio','has_spatial_audio','has_low_latency']:
            if col not in df.columns: df[col] = 0
            else: df[col] = df[col].fillna(0).astype(int)

        return df

    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        df = df.copy()
        col_map = {
            'category':         'category_enc',
            'brand':            'brand_enc',
            'brand_tier':       'brand_tier_enc',
            'country_of_origin':'country_enc',
        }
        for col, enc_col in col_map.items():
            if col not in df.columns:
                df[enc_col] = 0
                continue
            if fit:
                le = LabelEncoder()
                df[enc_col] = le.fit_transform(df[col].fillna('Unknown').astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    df[enc_col] = df[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else 0
                    )
        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        for col in MODEL_FEATURES:
            if col not in df.columns: df[col] = 0
        X = df[MODEL_FEATURES].fillna(0)
        y = df[self.target_column]
        logger.info(f"Feature matrix: {X.shape}")
        return X, y

    def full_pipeline(self, filepath: str, test_size: float = 0.2):
        df = self.load_data(filepath)
        df = self.handle_missing_values(df)
        df = self.remove_duplicates(df)
        df = self.engineer_features(df)
        df = self.encode_categorical(df, fit=True)
        X, y = self.prepare_features(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        X_train_sc = self.scaler.fit_transform(X_train)
        X_test_sc  = self.scaler.transform(X_test)
        logger.info(f"Train: {len(X_train)}  Test: {len(X_test)}")
        return X_train_sc, X_test_sc, y_train.values, y_test.values


if __name__ == "__main__":
    prep = DataPreparation()
    X_tr, X_te, y_tr, y_te = prep.full_pipeline('data/enhanced-headphones-dataset.csv')
    print(f"X_train: {X_tr.shape}  X_test: {X_te.shape}")
    print(f"Features ({len(MODEL_FEATURES)}): {MODEL_FEATURES}")
