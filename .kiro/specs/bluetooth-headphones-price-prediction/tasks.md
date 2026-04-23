# Implementation Plan: Bluetooth Headphones Price Prediction System

## Overview

This implementation plan breaks down the Bluetooth Headphones Price Prediction System into discrete coding tasks. The system will be built in Python using BeautifulSoup/Selenium for scraping, Pandas/Scikit-learn for ML, and FastAPI for deployment. Tasks are ordered to enable incremental validation, with property-based tests integrated throughout to catch errors early.

## Tasks

 1. Set up project structure and dependencies
  - Create directory structure: `src/`, `tests/`, `data/`, `models/`, `notebooks/`
  - Create `requirements.txt` with all dependencies (beautifulsoup4, selenium, pandas, scikit-learn, fastapi, uvicorn, hypothesis, pytest)
  - Create `README.md` with project overview and setup instructions
  - Initialize Python package structure with `__init__.py` files
  - _Requirements: 1.2, 2.2, 3.1, 4.6, 5.1_

- [ ] 2. Implement basic web scraper
  - [ ] 2.1 Create BasicScraper class with XPath selectors
    - Implement `BasicScraper` class in `src/scrapers/basic_scraper.py`
    - Add methods: `fetch_page()`, `extract_product_data()`, `scrape_products()`, `save_to_csv()`
    - Define XPath selectors for product_name, price, brand, rating, review_count, specifications
    - Implement error handling with retry logic and exponential backoff
    - Add logging for failed URLs to `scraping_errors.log`
    - _Requirements: 1.1, 1.2, 1.3_

  - [ ]* 2.2 Write property test for CSV serialization round-trip
    - **Property 1: Data Serialization Round-Trip**
    - **Validates: Requirements 1.2**
    - Generate random product data, serialize to CSV, deserialize, verify all fields preserved
    - Use hypothesis to generate ScrapedProduct instances
    - _Requirements: 1.2_

  - [ ]* 2.3 Write unit tests for BasicScraper
    - Test XPath extraction with mock HTML
    - Test error handling with simulated network failures
    - Test CSV writing and reading with sample data
    - _Requirements: 1.1, 1.2, 1.3_

  - [ ] 2.4 Create scraping script to collect initial dataset
    - Create `scripts/scrape_basic.py` to scrape from existing data/headphones-raw.csv format
    - Generate summary report showing total products scraped and errors
    - Verify at least 100 products collected
    - _Requirements: 1.4, 1.5_

- [ ] 3. Implement enhanced scraper with Selenium
  - [ ] 3.1 Create EnhancedScraper class with browser automation
    - Implement `EnhancedScraper` class in `src/scrapers/enhanced_scraper.py`
    - Set up Selenium WebDriver with Chrome options (headless mode)
    - Add methods: `wait_for_element()`, `extract_with_prompt_engineering()`, `normalize_units()`, `scrape_with_retry()`
    - Define regex patterns for battery_life, bluetooth_version, driver_size, frequency_response
    - _Requirements: 2.1, 2.2, 2.4_

  - [ ]* 3.2 Write property test for unit normalization equivalence
    - **Property 2: Unit Normalization Equivalence**
    - **Validates: Requirements 2.4**
    - Generate equivalent values in different formats (e.g., "30h", "30 hours", "1800 minutes")
    - Verify normalization produces identical output
    - _Requirements: 2.4_

  - [ ]* 3.3 Write unit tests for EnhancedScraper
    - Test regex pattern extraction with various formats
    - Test unit normalization for battery_life, bluetooth_version, driver_size
    - Test error handling and retry logic
    - _Requirements: 2.2, 2.4_

  - [ ] 3.4 Enhance existing dataset with additional features
    - Run enhanced scraper to extract additional features (ratings, review counts, detailed specs)
    - Merge with basic scraper output
    - Verify dataset completeness ≥95% for core features
    - _Requirements: 2.3, 2.5_

- [ ] 4. Checkpoint - Verify scraping components
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Implement data preparation and EDA
  - [ ] 5.1 Create DataPreparation class
    - Implement `DataPreparation` class in `src/data/preparation.py`
    - Add methods: `load_data()`, `handle_missing_values()`, `remove_duplicates()`, `encode_categorical()`, `normalize_numerical()`, `create_train_test_split()`
    - Implement missing value strategy: drop rows for brand/price, impute battery_life/bluetooth_version, fill noise_cancellation with "No"
    - _Requirements: 3.3, 3.4, 3.5, 3.6, 3.7_

  - [ ]* 5.2 Write property test for missing value handling completeness
    - **Property 3: Missing Value Handling Completeness**
    - **Validates: Requirements 3.3**
    - Generate datasets with random missing patterns
    - Apply handling strategy, verify no missing values in core features
    - _Requirements: 3.3_

  - [ ]* 5.3 Write property test for deduplication idempotence
    - **Property 4: Deduplication Idempotence**
    - **Validates: Requirements 3.4**
    - Generate datasets with random duplicates
    - Apply deduplication multiple times, verify idempotence
    - _Requirements: 3.4_

  - [ ]* 5.4 Write property test for train-test split proportions
    - **Property 5: Train-Test Split Proportions**
    - **Validates: Requirements 3.5**
    - Generate datasets of various sizes
    - Create splits, verify 80/20 proportions (±1%)
    - _Requirements: 3.5_

  - [ ]* 5.5 Write property test for categorical encoding information preservation
    - **Property 6: Categorical Encoding Information Preservation**
    - **Validates: Requirements 3.6**
    - Generate random categorical data
    - Encode then decode, verify information preservation
    - _Requirements: 3.6_

  - [ ]* 5.6 Write property test for numerical normalization statistical properties
    - **Property 7: Numerical Normalization Statistical Properties**
    - **Validates: Requirements 3.7**
    - Generate random numerical arrays
    - Apply StandardScaler, verify mean≈0, std≈1
    - _Requirements: 3.7_

  - [ ]* 5.7 Write unit tests for DataPreparation
    - Test missing value imputation strategies
    - Test deduplication with known duplicates
    - Test categorical encoding/decoding
    - _Requirements: 3.3, 3.4, 3.6, 3.7_

  - [ ] 5.8 Create EDA notebook in Google Colab
    - Create `notebooks/eda_analysis.ipynb`
    - Load scraped dataset and display summary statistics
    - Create visualizations: price distribution, price vs battery_life, brand-wise box plots, correlation heatmap, missing value heatmap
    - Perform feature engineering: `has_noise_cancellation`, `price_per_hour`, `brand_tier`, `bluetooth_major_version`, `high_rating`
    - Apply data preparation pipeline and save processed dataset to `data/processed_data.csv`
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

- [ ] 6. Checkpoint - Verify data preparation
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Implement price prediction model
  - [ ] 7.1 Create PricePredictionModel class
    - Implement `PricePredictionModel` class in `src/models/price_model.py`
    - Add methods: `train_models()`, `evaluate_models()`, `select_best_model()`, `get_feature_importance()`, `save_model()`
    - Define model candidates: LinearRegression, Ridge, Lasso, RandomForestRegressor, GradientBoostingRegressor
    - Implement evaluation metrics: R², RMSE, MAE, MAPE
    - _Requirements: 4.1, 4.2, 4.5, 4.6_

  - [ ]* 7.2 Write property test for model prediction validity
    - **Property 8: Model Prediction Validity**
    - **Validates: Requirements 4.4**
    - Generate random valid feature sets
    - Get predictions, verify positive values in range [500, 50000] INR
    - _Requirements: 4.4_

  - [ ]* 7.3 Write property test for model serialization round-trip
    - **Property 9: Model Serialization Round-Trip**
    - **Validates: Requirements 4.6**
    - Train models with random data
    - Serialize, deserialize, predict, verify identical predictions (within 1e-6 tolerance)
    - _Requirements: 4.6_

  - [ ]* 7.4 Write unit tests for PricePredictionModel
    - Test model training with small synthetic dataset
    - Test feature importance extraction
    - Test model serialization/deserialization
    - _Requirements: 4.1, 4.5, 4.6_

  - [ ] 7.5 Create model training script
    - Create `scripts/train_model.py`
    - Load processed dataset from `data/processed_data.csv`
    - Train all candidate models and evaluate performance
    - Select best model with R² ≥ 0.70
    - Report top 5 feature importances
    - Save trained model to `models/price_predictor.pkl`
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [ ] 8. Checkpoint - Verify model training
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Implement FastAPI deployment service
  - [ ] 9.1 Create API models and service class
    - Create `src/api/models.py` with Pydantic models: `HeadphoneFeatures`, `PricePrediction`
    - Create `src/api/service.py` with `ModelService` class
    - Implement methods: `__init__()`, `preprocess_features()`, `predict()`, `reload_model()`
    - Add model loading with error handling
    - _Requirements: 5.1, 5.5_

  - [ ] 9.2 Create FastAPI application with endpoints
    - Create `src/api/app.py` with FastAPI application
    - Implement POST `/predict` endpoint with input validation
    - Implement GET `/health` endpoint with model status
    - Implement GET `/model/info` endpoint with metadata and feature importance
    - Add error handling: 422 for validation errors, 500 for prediction failures, 503 for model not loaded
    - Add request logging with timestamps and input features
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.6_

  - [ ]* 9.3 Write property test for API response time bounds
    - **Property 10: API Response Time Bounds**
    - **Validates: Requirements 5.2**
    - Generate random valid inputs
    - Measure response time, verify <2000ms
    - _Requirements: 5.2_

  - [ ]* 9.4 Write property test for API input validation error handling
    - **Property 11: API Input Validation Error Handling**
    - **Validates: Requirements 5.3**
    - Generate random invalid inputs (missing fields, out-of-range values, wrong types)
    - Submit to API, verify 422 errors with descriptive messages
    - _Requirements: 5.3_

  - [ ]* 9.5 Write unit tests for API endpoints
    - Test endpoint routing and request handling
    - Test input validation with invalid inputs
    - Test error response formats
    - Test health check endpoint
    - Use FastAPI TestClient for integration tests
    - _Requirements: 5.1, 5.2, 5.3, 5.6_

  - [ ] 9.6 Create API startup script
    - Create `scripts/run_api.py` to start Uvicorn server
    - Load model from `models/price_predictor.pkl`
    - Configure logging and error handling
    - _Requirements: 5.1, 5.5_

- [ ] 10. Integration and final validation
  - [ ] 10.1 Create end-to-end integration tests
    - Test complete scraping workflow with mock HTML pages
    - Test model training pipeline with small real dataset (50-100 products)
    - Test API with real model and concurrent requests
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 4.1, 4.2, 4.3, 5.1, 5.2_

  - [ ] 10.2 Create documentation and usage examples
    - Update `README.md` with setup instructions, usage examples, API documentation
    - Add example API requests with curl/Python
    - Document scraping configuration and XPath selectors
    - Document model training process and hyperparameters
    - _Requirements: 1.1, 2.1, 4.1, 5.1_

  - [ ] 10.3 Verify all requirements are met
    - Verify scraper extracts ≥100 products
    - Verify dataset completeness ≥95% for core features
    - Verify model achieves R² ≥ 0.70
    - Verify API response time <2 seconds
    - Verify all error handling and logging works correctly
    - _Requirements: 1.4, 2.5, 4.3, 5.2_

- [ ] 11. Final checkpoint - Complete system validation
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- Checkpoints ensure incremental validation at major milestones
- The existing `data/headphones-raw.csv` file can be used as the initial dataset
- All code examples in tasks use Python as specified in the design document
