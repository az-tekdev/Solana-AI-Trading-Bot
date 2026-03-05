"""Tests for AI model module."""

import pytest
import pandas as pd
import numpy as np
from src.ai_model import AIModel


def test_ai_model_initialization():
    """Test AI model initialization."""
    model = AIModel(model_type="lstm")
    assert model.model_type == "lstm"
    assert not model.is_trained


def test_prepare_features():
    """Test feature preparation."""
    model = AIModel(model_type="lstm")
    
    # Create sample data
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": np.random.rand(100) * 0.01,
        "high": np.random.rand(100) * 0.01,
        "low": np.random.rand(100) * 0.01,
        "close": np.random.rand(100) * 0.01,
        "volume": np.random.randint(1000, 100000, 100)
    })
    
    X, y = model.prepare_features(df)
    
    assert len(X) > 0
    assert len(y) > 0
    assert X.shape[1] == model.sequence_length
    assert X.shape[2] == 5  # 5 features


def test_train_lstm():
    """Test LSTM model training."""
    model = AIModel(model_type="lstm")
    
    # Create sample data
    dates = pd.date_range(start="2024-01-01", periods=200, freq="1h")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": np.random.rand(200) * 0.01,
        "high": np.random.rand(200) * 0.01,
        "low": np.random.rand(200) * 0.01,
        "close": np.random.rand(200) * 0.01,
        "volume": np.random.randint(1000, 100000, 200)
    })
    
    results = model.train(df)
    
    assert model.is_trained
    assert "train_loss" in results or "test_loss" in results


def test_predict():
    """Test model prediction."""
    model = AIModel(model_type="lstm")
    
    # Create sample data
    dates = pd.date_range(start="2024-01-01", periods=200, freq="1h")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": np.random.rand(200) * 0.01,
        "high": np.random.rand(200) * 0.01,
        "low": np.random.rand(200) * 0.01,
        "close": np.random.rand(200) * 0.01,
        "volume": np.random.randint(1000, 100000, 200)
    })
    
    # Train first
    model.train(df)
    
    # Predict
    prediction = model.predict(df)
    
    assert "signal" in prediction
    assert "confidence" in prediction
    assert prediction["signal"] in ["buy", "sell", "hold"]
