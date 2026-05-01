# 🎧 NexTune — Bluetooth Headphones Price Predictor

> An ML-powered system for predicting optimal prices of Bluetooth headphones in the Indian market, built with Streamlit and FastAPI.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Data Pipeline](#data-pipeline)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

NexTune helps price new Bluetooth headphones in the competitive Indian e-commerce market. By scraping product data from Amazon India, analyzing market trends, and training a Gradient Boosting model on 18 engineered features, the system provides data-driven price recommendations based on product specifications.

**Key Capabilities:**
- 🕷️ Automated web scraping from Amazon India (BeautifulSoup + Selenium)
- 📊 Comprehensive exploratory data analysis
- 🤖 Gradient Boosting price prediction — R² ≈ 0.80, MAPE ≈ 17.6%
- 🎨 Interactive Streamlit web app with dark/light theme
- ✅ Property-based testing with Hypothesis

---

## 💡 Problem Statement

Your company is launching new wireless Bluetooth headphones in the Indian market. The data science team needs to recommend a suitable price based on:

1. **Product Specifications**: Bluetooth version, ANC/ENC, driver size, IPX rating, codec support, etc.
2. **Market Demand**: Competitor pricing and customer preferences from Amazon India

**Challenge**: Determine optimal pricing that balances competitiveness with profitability while considering brand tier and market positioning.

---

## ✨ Features

### 🔍 Data Collection
- **Basic Scraper**: Fast extraction of static HTML using BeautifulSoup
- **Enhanced Scraper**: JavaScript-rendered content with Selenium + undetected-chromedriver
- **Smart Extraction**: Regex-based parsing for unstructured spec data
- **Unit Normalization**: Automatic conversion of various formats (e.g., "30h" → 30)

### 📈 Data Analysis
- **Automated EDA**: Statistical summaries and visualizations
- **Feature Engineering**: `price_per_hour`, `brand_tier`, `bt_major`, `high_rating`, etc.
- **Missing Value Handling**: Intelligent imputation strategies
- **Data Quality**: Deduplication and validation across multiple datasets

### 🧠 Machine Learning
- **18 Features**: Brand tier, ANC, ENC, Hi-Res Audio, Spatial Audio, Dual Pairing, Premium Codec, Low Latency, IPX level, ANC dB, driver size, Bluetooth version, rating, review count, and more
- **Model**: Gradient Boosting Regressor (log-transformed target)
- **Artifacts**: `price_predictor.pkl`, `scaler.pkl`, `label_encoders.pkl`, `features.pkl`, `brand_avg.pkl`, `brands.json`

### 🌐 Web App
- **Streamlit App** (`app.py`): Full-featured interactive UI with dark/light theme, brand selector, feature toggles, battery life analysis, and price segment badges

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     AMAZON INDIA                                │
│              (Bluetooth Headphones / Earbuds)                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA COLLECTION LAYER                        │
│  ┌──────────────────┐         ┌──────────────────┐            │
│  │  Basic Scraper   │         │ Enhanced Scraper │            │
│  │ (BeautifulSoup)  │         │   (Selenium)     │            │
│  └──────────────────┘         └──────────────────┘            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATA STORAGE                               │
│   headphones-raw.csv → enhanced → combined → final-merged       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ANALYSIS & TRAINING LAYER                      │
│  ┌──────────────────┐         ┌──────────────────┐            │
│  │   EDA Notebook   │    →    │  Model Training  │            │
│  │  (Jupyter/Colab) │         │  (Scikit-learn)  │            │
│  └──────────────────┘         └──────────────────┘            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT LAYER                             │
│         Streamlit App · app.py · localhost:8501                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Scraping** | BeautifulSoup 4, Selenium, undetected-chromedriver | Data extraction from Amazon India |
| **Data Processing** | Pandas, NumPy | Data manipulation and feature engineering |
| **Visualization** | Matplotlib, Seaborn | EDA and insights generation |
| **Machine Learning** | Scikit-learn (GradientBoostingRegressor) | Model training and evaluation |
| **Streamlit App** | Streamlit | Interactive web UI with theming |
| **Testing** | Pytest, Hypothesis | Unit and property-based testing |
| **Serialization** | Joblib | Model artifact persistence |

---

## 📁 Project Structure

```
NexTune/
├── app.py                              # Streamlit web app (main UI)
├── requirements.txt                    # Python dependencies
├── assets/
│   └── bg.jpg                          # Background image for Streamlit app
├── data/
│   ├── headphones-raw.csv              # Raw scraped data
│   ├── amazon-earphones-market-data.csv
│   ├── indian-wireless-headphones-scraped-data.csv
│   ├── enhanced-headphones-dataset.csv
│   ├── combined-headphones-dataset.csv
│   ├── final-merged-dataset.csv        # Final training dataset
│   ├── nexttune-cleaned-data.csv
│   └── earbuds_by_company.csv / .json
├── models/
│   ├── price_predictor.pkl             # Trained GradientBoosting model
│   ├── scaler.pkl                      # StandardScaler
│   ├── label_encoders.pkl              # LabelEncoders for categorical features
│   ├── features.pkl                    # Ordered feature list
│   ├── brand_avg.pkl                   # Brand average prices
│   └── brands.json                     # List of known brands
├── notebooks/
│   ├── NexTune_Price_Prediction.ipynb  # Main prediction notebook
│   ├── eda_notebook.ipynb              # Exploratory data analysis
│   └── model_training.ipynb           # Model training pipeline
├── scripts/
│   └── merge_datasets.py               # Dataset merging utility
├── src/
│   ├── data/
│   │   └── preparation.py              # Data cleaning & preprocessing
│   └── scrapers/
│       ├── amazon_scraper.py           # Basic BeautifulSoup scraper
│       └── enhanced_scraper.py         # Selenium-based scraper
├── tests/                              # Test suite
└── .kiro/specs/                        # Project specifications
```

---

## 🚀 Installation

### Prerequisites

- **Python 3.8 or higher** (tested with Python 3.14)
- **VS Code** (recommended) with Python extension
- Chrome/Chromium browser (for Selenium scraper)
- ChromeDriver matching your Chrome version

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/ESpoorthy/NexTune.git
cd NexTune
```

2. **Open in VS Code**
```bash
code .
```

3. **Create virtual environment**
```bash
python3 -m venv .venv
```

4. **Activate virtual environment**
```bash
# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

5. **Install dependencies**
```bash
pip install -r requirements.txt
```

6. **Retrain the model** (required for first-time setup or after scikit-learn version changes)
```bash
python3 scripts/retrain_model.py
```

This will generate all required model files in the `models/` directory:
- `price_predictor.pkl` - Trained Gradient Boosting model
- `scaler.pkl` - Feature scaler
- `label_encoders.pkl` - Categorical encoders
- `features.pkl` - Feature list
- `brand_avg.pkl` - Brand average prices
- `brands.json` - Available brands

7. **Install ChromeDriver** (optional - only needed for enhanced scraper)
```bash
# macOS
brew install chromedriver

# Ubuntu/Debian
sudo apt-get install chromium-chromedriver
```

---

## 📖 Usage

### Run the App from VS Code

#### Option 1: Using VS Code Terminal (Recommended)

1. **Open the project in VS Code**
```bash
code .
```

2. **Open the integrated terminal** (`` Ctrl+` `` or `View > Terminal`)

3. **Activate virtual environment** (if not already activated)
```bash
# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

4. **Run the Streamlit app**
```bash
streamlit run app.py
```

5. **Access the app** - VS Code will show a popup with the URL, or manually open:
   - Local: `http://localhost:8501`
   - Network: `http://192.168.x.x:8501`

#### Option 2: Using VS Code Run Configuration

1. Create `.vscode/launch.json` in your project root:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Run Streamlit App",
            "type": "python",
            "request": "launch",
            "module": "streamlit",
            "args": ["run", "app.py"],
            "console": "integratedTerminal"
        }
    ]
}
```

2. Press `F5` or go to `Run > Start Debugging`

#### What the App Does:

- Select a brand and headphone category
- Set rating, review count, Bluetooth version, driver size
- Toggle features: ANC, ENC, Hi-Res Audio, Spatial Audio, Dual Pairing, Premium Codec, Low Latency, IPX rating, Fast Charging, etc.
- Get an instant predicted price with a ±10% suggested range and segment badge
- **Fully scrollable interface** - all inputs are accessible with smooth scrolling

### Data Collection

```bash
# Basic scraping
python src/scrapers/amazon_scraper.py

# Enhanced scraping (requires ChromeDriver)
python src/scrapers/enhanced_scraper.py
```

### Model Training

Open and run `notebooks/model_training.ipynb` in Jupyter or Google Colab. This trains the Gradient Boosting model and saves all artifacts to `models/`.

### Exploratory Data Analysis

```bash
jupyter notebook notebooks/eda_notebook.ipynb
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Algorithm** | GradientBoostingRegressor |
| **R² Score** | ~0.80 |
| **MAPE** | ~17.6% |
| **Features** | 18 scraped + engineered |
| **Training Data** | 219 Amazon India products |
| **Target** | log(price) → expm1 to recover ₹ |

### Top Features

```
brand_enc / brand_tier    ████████████████████████████████ ~35%
has_anc / anc_db          ████████████████████ ~20%
rating / high_rating      ████████████████ ~16%
bluetooth_version         ████████████ ~12%
driver_size_mm            ████████ ~8%
review_count              ████ ~5%
ipx_level / has_enc       ███ ~4%
```

### Price Segments

| Segment | Range |
|---------|-------|
| 💚 Budget | < ₹1,000 |
| 💙 Mid-Range | ₹1,000 – ₹3,000 |
| 💜 Premium | ₹3,000 – ₹8,000 |
| 🏆 Flagship | > ₹8,000 |

---

## 🔄 Data Pipeline

1. **Scraping** — Extract 200+ products from Amazon India (name, brand, price, rating, reviews, specs)
2. **Merging** — Combine multiple scraped datasets via `scripts/merge_datasets.py`
3. **Cleaning** — Remove duplicates, handle missing values, normalize units
4. **Feature Engineering** — `brand_tier`, `bt_major`, `high_rating`, `has_*` binary flags, `anc_db`, `ipx_level`
5. **Training** — Log-transform price, scale features, train GradientBoostingRegressor
6. **Serialization** — Save model + all artifacts to `models/` with Joblib

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Property-based tests only
pytest tests/ -v -k "property"
```

Property-based tests (Hypothesis) validate:
- Data serialization round-trips
- Unit normalization equivalence
- Missing value handling after preprocessing
- Deduplication idempotence
- Train/test split proportions
- Model prediction validity (₹500 – ₹50,000 range)
- Model serialization consistency

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please follow PEP 8, add tests for new features, and ensure all tests pass before submitting.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Data sourced from Amazon India
- Built with open-source tools and libraries
- Inspired by real-world pricing challenges in Indian e-commerce

---

**Made with ❤️ for data-driven pricing decisions**
