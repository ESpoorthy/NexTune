"""Basic tests for NexTune application."""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_imports():
    """Test that required packages can be imported."""
    import pandas as pd
    import numpy as np
    import streamlit as st
    import joblib
    assert True


def test_data_directory_exists():
    """Test that data directory exists."""
    assert os.path.exists('data')


def test_models_directory_exists():
    """Test that models directory exists."""
    assert os.path.exists('models')


def test_app_file_exists():
    """Test that main app file exists."""
    assert os.path.exists('app.py')


def test_requirements_file_exists():
    """Test that requirements file exists."""
    assert os.path.exists('requirements.txt')


def test_model_artifacts_exist():
    """Test that model artifacts exist."""
    assert os.path.exists('models/price_predictor.pkl')
    assert os.path.exists('models/scaler.pkl')
    assert os.path.exists('models/label_encoders.pkl')
    assert os.path.exists('models/features.pkl')
    assert os.path.exists('models/brand_avg.pkl')
    assert os.path.exists('models/brands.json')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
