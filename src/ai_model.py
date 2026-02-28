"""AI/ML models for trading decision making."""

import os
import pickle
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

from .config import settings
from .utils import setup_logger

logger = setup_logger(__name__)


class LSTMPredictor(nn.Module):
    """LSTM model for price prediction."""
    
    def __init__(self, input_size: int = 5, hidden_size: int = 64, num_layers: int = 2):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out[:, -1, :])  # Take last output
        output = self.fc(lstm_out)
        return output


class AIModel:
    """AI model wrapper for trading decisions."""
    
    def __init__(self, model_type: str = "lstm"):
        self.model_type = model_type
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        self.sequence_length = 60  # 60 time steps for LSTM
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features from historical data."""
        # Select features
        features = ['open', 'high', 'low', 'close', 'volume']
        data = df[features].values
        
        # Normalize
        data_scaled = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(data_scaled)):
            X.append(data_scaled[i - self.sequence_length:i])
            y.append(data_scaled[i, 3])  # Close price
        
        return np.array(X), np.array(y)
    
    def train(self, df: pd.DataFrame) -> Dict:
        """Train the AI model on historical data."""
        try:
            logger.info(f"Training {self.model_type} model on {len(df)} samples")
            
            if self.model_type == "lstm":
                return self._train_lstm(df)
            elif self.model_type == "random_forest":
                return self._train_random_forest(df)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
                
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise
    
    def _train_lstm(self, df: pd.DataFrame) -> Dict:
        """Train LSTM model."""
        X, y = self.prepare_features(df)
        
        # Split data
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train).unsqueeze(1)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test).unsqueeze(1)
        
        # Initialize model
        self.model = LSTMPredictor(input_size=5, hidden_size=64, num_layers=2)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        epochs = 50
        batch_size = 32
        train_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / (len(X_train) // batch_size + 1)
            train_losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # Evaluate
        self.model.eval()
        with torch.no_grad():
            test_pred = self.model(X_test)
            test_loss = criterion(test_pred, y_test).item()
        
        self.is_trained = True
        logger.info(f"Training complete. Test loss: {test_loss:.6f}")
        
        return {
            "train_loss": train_losses[-1],
            "test_loss": test_loss,
            "epochs": epochs
        }
    
    def _train_random_forest(self, df: pd.DataFrame) -> Dict:
        """Train Random Forest classifier."""
        # Prepare features
        df['price_change'] = df['close'].pct_change()
        df['ma_5'] = df['close'].rolling(5).mean()
        df['ma_20'] = df['close'].rolling(20).mean()
        df['volatility'] = df['close'].rolling(20).std()
        
        # Create labels (1 for buy, 0 for hold, -1 for sell)
        df['signal'] = 0
        df.loc[df['price_change'] > 0.02, 'signal'] = 1
        df.loc[df['price_change'] < -0.02, 'signal'] = -1
        
        # Prepare data
        features = ['close', 'volume', 'ma_5', 'ma_20', 'volatility']
        df_clean = df[features + ['signal']].dropna()
        
        X = df_clean[features].values
        y = df_clean['signal'].values
        
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        self.is_trained = True
        logger.info(f"Random Forest trained. Train score: {train_score:.4f}, Test score: {test_score:.4f}")
        
        return {
            "train_score": train_score,
            "test_score": test_score
        }
    
    def predict(self, recent_data: pd.DataFrame) -> Dict:
        """Make prediction on recent data."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            if self.model_type == "lstm":
                return self._predict_lstm(recent_data)
            elif self.model_type == "random_forest":
                return self._predict_random_forest(recent_data)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"signal": "hold", "confidence": 0.0, "predicted_price": None}
    
    def _predict_lstm(self, recent_data: pd.DataFrame) -> Dict:
        """LSTM prediction."""
        # Prepare last sequence
        features = ['open', 'high', 'low', 'close', 'volume']
        data = recent_data[features].tail(self.sequence_length).values
        data_scaled = self.scaler.transform(data)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(data_scaled).unsqueeze(0)
            pred = self.model(X).item()
        
        # Inverse transform
        dummy = np.zeros((1, 5))
        dummy[0, 3] = pred
        pred_price = self.scaler.inverse_transform(dummy)[0, 3]
        
        current_price = recent_data['close'].iloc[-1]
        price_change_pct = ((pred_price - current_price) / current_price) * 100
        
        # Generate signal
        if price_change_pct > 2.0:
            signal = "buy"
            confidence = min(abs(price_change_pct) / 10.0, 1.0)
        elif price_change_pct < -2.0:
            signal = "sell"
            confidence = min(abs(price_change_pct) / 10.0, 1.0)
        else:
            signal = "hold"
            confidence = 0.3
        
        return {
            "signal": signal,
            "confidence": confidence,
            "predicted_price": pred_price,
            "current_price": current_price,
            "expected_change_pct": price_change_pct
        }
    
    def _predict_random_forest(self, recent_data: pd.DataFrame) -> Dict:
        """Random Forest prediction."""
        # Prepare features
        recent_data = recent_data.copy()
        recent_data['ma_5'] = recent_data['close'].rolling(5).mean()
        recent_data['ma_20'] = recent_data['close'].rolling(20).mean()
        recent_data['volatility'] = recent_data['close'].rolling(20).std()
        
        features = ['close', 'volume', 'ma_5', 'ma_20', 'volatility']
        last_row = recent_data[features].tail(1).fillna(0).values
        
        pred = self.model.predict(last_row)[0]
        proba = self.model.predict_proba(last_row)[0]
        confidence = max(proba)
        
        signal_map = {1: "buy", 0: "hold", -1: "sell"}
        
        return {
            "signal": signal_map.get(pred, "hold"),
            "confidence": float(confidence),
            "predicted_price": None,
            "current_price": recent_data['close'].iloc[-1]
        }
    
    def save(self, path: str):
        """Save model to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if self.model_type == "lstm":
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
                'sequence_length': self.sequence_length
            }, path)
        else:
            with open(path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler
                }, f)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from file."""
        if not os.path.exists(path):
            logger.warning(f"Model file not found: {path}")
            return False
        
        try:
            if self.model_type == "lstm":
                checkpoint = torch.load(path)
                self.model = LSTMPredictor(input_size=5, hidden_size=64, num_layers=2)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.scaler = checkpoint['scaler']
                self.sequence_length = checkpoint['sequence_length']
            else:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.scaler = data['scaler']
            
            self.is_trained = True
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
