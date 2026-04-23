"""
Enhanced Web Scraper with Prompt Engineering
Extracts 22 structured features from product titles using regex patterns.

Numerical (10): battery_life_hrs, bluetooth_version, driver_size_mm, mic_count,
                anc_db, ipx_level, charging_time_hrs, latency_ms, range_m, eq_modes
Binary   (12): has_noise_cancellation, has_enc, has_usb_c, has_premium_codec,
               has_touch_control, has_voice_assistant, has_fast_charge,
               has_dual_pairing, has_gaming_mode, has_hi_res_audio,
               has_spatial_audio, has_low_latency
+ category (form-factor detection)
"""

import re, os
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedScraper:
    EXTRACTION_PATTERNS = {
        # Numerical
        'battery_life':    r'(\d+)\s*(?:hours?|hrs?|h|hr)\s*(?:battery|playtime|playback|music)?',
        'bluetooth_ver':   r'bluetooth\s*(?:version|v)?\s*([0-9]\.[0-9])',
        'driver_size':     r'(\d+(?:\.\d+)?)\s*mm\s*(?:driver|drivers?)',
        'anc_db':          r'(\d+)\s*db\s*(?:anc|noise|hybrid)',
        'mic_count':       r'(\d+)\s*mic(?:s|rophone)?',
        'ipx_rating':      r'ipx\s*(\d+)',
        'charging_time':   r'(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\s*(?:fast\s*)?charg',
        'latency_ms':      r'(\d+)\s*ms\s*(?:low\s*)?latency',
        'range_m':         r'(\d+)\s*m(?:eter|etre)?\s*(?:range|pairing|wireless)',
        'eq_modes':        r'(\d+)\s*(?:custom\s*)?eq\s*(?:modes?|profiles?)',
        # Binary
        'usb_c':           r'usb[\s\-]?(?:type[\s\-]?c|c\b)',
        'codec':           r'\b(aptx|aac|ldac|sbc|aptx[\s\-]?hd|aptx[\s\-]?adaptive)\b',
        'touch_control':   r'\b(touch\s*control|touch\s*sensor|gesture\s*control)\b',
        'voice_assistant': r'\b(alexa|google\s*assistant|siri|voice\s*assistant)\b',
        'fast_charge':     r'\b(fast\s*charg|quick\s*charg|rapid\s*charg)\b',
        'dual_pairing':    r'\b(dual\s*pair|multi[\s\-]?point|dual\s*device)\b',
        'enc':             r'\b(enc|environmental\s*noise\s*cancel)',
        'gaming_mode':     r'\b(gaming\s*mode|low[\s\-]?latency\s*gaming)',
        'hi_res_audio':    r'\b(hi[\s\-]?res(?:\s*audio)?|high[\s\-]?resolution\s*audio)',
        'spatial_audio':   r'\b(spatial\s*audio|3d\s*(?:audio|sound)|dolby\s*atmos)',
    }

    ANC_KW     = ['anc', 'active noise', 'noise cancell', 'hybrid anc']
    TWS_KW     = ['tws', 'true wireless', 'earbuds', 'ear buds', 'in-ear']
    OVEREAR_KW = ['over ear', 'over-ear', 'overhead', 'on-ear', 'on ear', 'headphone']
    NECKBAND_KW= ['neckband', 'neck band']

    def __init__(self):
        self.stats = {'total_processed': 0, 'features_extracted': 0, 'normalization_applied': 0}

    def extract_with_prompt_engineering(self, text: str) -> Dict:
        if not text or not isinstance(text, str):
            return {}
        t  = text.lower()
        ex = {}

        def _num(key):
            m = re.search(self.EXTRACTION_PATTERNS[key], t, re.IGNORECASE)
            return float(m.group(1)) if m else None

        ex['battery_life_hrs']  = _num('battery_life')
        ex['bluetooth_version'] = _num('bluetooth_ver')
        ex['driver_size_mm']    = _num('driver_size')
        ex['anc_db']            = _num('anc_db')
        ex['mic_count']         = _num('mic_count')
        ex['charging_time_hrs'] = _num('charging_time')
        ex['latency_ms']        = _num('latency_ms')
        ex['range_m']           = _num('range_m')
        ex['eq_modes']          = _num('eq_modes')

        m = re.search(self.EXTRACTION_PATTERNS['ipx_rating'], t, re.IGNORECASE)
        ex['ipx_rating'] = f"IPX{m.group(1)}" if m else None
        ex['ipx_level']  = int(m.group(1)) if m else None

        def _flag(key): return 1 if re.search(self.EXTRACTION_PATTERNS[key], t, re.IGNORECASE) else 0

        ex['has_noise_cancellation'] = 1 if any(kw in t for kw in self.ANC_KW) else 0
        ex['has_enc']            = _flag('enc')
        ex['has_usb_c']          = _flag('usb_c')
        ex['has_premium_codec']  = _flag('codec')
        ex['has_touch_control']  = _flag('touch_control')
        ex['has_voice_assistant']= _flag('voice_assistant')
        ex['has_fast_charge']    = _flag('fast_charge')
        ex['has_dual_pairing']   = _flag('dual_pairing')
        ex['has_gaming_mode']    = _flag('gaming_mode')
        ex['has_hi_res_audio']   = _flag('hi_res_audio')
        ex['has_spatial_audio']  = _flag('spatial_audio')
        ex['has_low_latency']    = 1 if (ex['latency_ms'] or ex['has_gaming_mode']) else 0

        if any(kw in t for kw in self.TWS_KW):
            ex['category'] = 'true wireless earbuds'
        elif any(kw in t for kw in self.OVEREAR_KW):
            ex['category'] = 'over-ear headphone'
        elif any(kw in t for kw in self.NECKBAND_KW):
            ex['category'] = 'neckband'
        else:
            ex['category'] = 'unknown'

        self.stats['features_extracted'] += sum(1 for v in ex.values() if v)
        return ex

    def normalize_units(self, value, field: str) -> Optional[float]:
        if not value or pd.isna(value): return None
        s = str(value).lower().strip()
        try:
            if field == 'battery_life':
                if 'min' in s: return float(re.search(r'(\d+)', s).group(1)) / 60
                return float(re.search(r'(\d+)', s).group(1))
            elif field == 'bluetooth_version':
                m = re.search(r'([0-9]\.[0-9])', s)
                return float(m.group(1)) if m else None
            elif field == 'driver_size':
                if 'cm' in s: return float(re.search(r'(\d+\.?\d*)', s).group(1)) * 10
                return float(re.search(r'(\d+\.?\d*)', s).group(1))
            elif field == 'price':
                return float(re.sub(r'[₹,\s]', '', s))
            else:
                return float(s)
        except (ValueError, AttributeError):
            return None

    def enhance_dataset(self, input_file: str, output_file: str) -> pd.DataFrame:
        logger.info(f"Loading {input_file}")
        df = pd.read_csv(input_file)
        logger.info(f"Processing {len(df)} records...")

        rows = []
        for idx, row in df.iterrows():
            name  = str(row.get('product_name', '') or row.get('name', ''))
            feats = self.extract_with_prompt_engineering(name)
            for col, val in feats.items():
                if col not in df.columns:
                    row[col] = val
                elif pd.isna(row[col]) or row[col] == 'unknown' or row[col] == 0:
                    row[col] = val
            rows.append(row)
            self.stats['total_processed'] += 1
            if (idx + 1) % 50 == 0:
                logger.info(f"  {idx+1}/{len(df)}")

        df = pd.DataFrame(rows)
        for col, field in [('battery_life_hrs','battery_life'),('bluetooth_version','bluetooth_version'),
                            ('driver_size_mm','driver_size'),('price_inr','price')]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: self.normalize_units(x, field) if pd.notna(x) else x)

        df['enhanced_at'] = datetime.now().isoformat()
        df.to_csv(output_file, index=False)
        logger.info(f"Saved → {output_file}")
        self._print_stats(df)
        return df

    def _print_stats(self, df):
        logger.info("=" * 60)
        for col in ['battery_life_hrs','bluetooth_version','driver_size_mm','has_noise_cancellation',
                    'has_enc','has_gaming_mode','has_hi_res_audio','has_spatial_audio','has_low_latency']:
            if col in df.columns:
                logger.info(f"  {col:30s}: {df[col].notna().mean()*100:.1f}%")
        logger.info("=" * 60)


def main():
    scraper = EnhancedScraper()
    src = 'data/final-merged-dataset.csv' if os.path.exists('data/final-merged-dataset.csv') \
          else 'data/combined-headphones-dataset.csv'
    df = scraper.enhance_dataset(src, 'data/enhanced-headphones-dataset.csv')
    logger.info(f"Done — {len(df)} records")

if __name__ == "__main__":
    main()
