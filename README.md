# Try-Quant: Bitcoin Trading Signal Prediction

A machine learning project for predicting Bitcoin buy/sell signals using XGBoost with advanced feature engineering and class balancing techniques.

## Project Overview

This project uses historical Bitcoin price data to train a binary classification model that predicts whether to buy (1) or sell (0) based on technical indicators and price patterns. The model achieves 81.33% accuracy on test data with a threshold-based ensemble approach.

## Dataset

### Data Sources
- **Time Period**: 2017-2023
- **Cryptocurrency**: Bitcoin (BTC)
- **Total Records**: 93,864 rows (after processing)
- **Timeframes**: Multiple intervals (3m, 5m, 15m, 30m, 1h, 2h, 4h, 1d)

### Data Structure
```
data/
├── 2018-22/          # Historical data from 2018-2022
├── 2023/             # 2023 data
└── merged/           # Merged datasets by timeframe
```

### Raw Data Files
- 15-minute, 30-minute, 1-hour, 2-hour, 4-hour, and daily intervals
- OHLCV format (Open, High, Low, Close, Volume)

## Features

### Technical Indicators (25 features)
1. **Price Features**: open, high, low, close
2. **Volume Indicators**: volume, volume_SMA_20, OBV, Volume_ratio
3. **Trend Indicators**: SMA_20, trend
4. **Momentum Indicators**: 
   - RSI_14 (Relative Strength Index)
   - RSI_lag_1
   - ROC_10 (Rate of Change)
   - momentum divergence (mom_div)
5. **Volatility Indicators**:
   - ATR_14 (Average True Range)
   - ATR_ratio
   - volatility (20-period rolling std)
6. **MACD Indicators**: MACD, MACD_signal
7. **Price Patterns**:
   - high_low_spread
   - open_close_diff
   - price_delta
8. **Derived Features**:
   - price_vol_ratio
   - hour, dayofweek (temporal features)

## Target Definition

**Buy Signal (1)**: When the next close price is greater than current close × 1.005 (0.5% profit threshold)

**Sell Signal (0)**: Otherwise

## Data Processing Pipeline

### 1. Class Distribution
- **Original Dataset**: 
  - Class 0 (Sell): 83,371 samples
  - Class 1 (Buy): 10,459 samples
  - **Imbalance Ratio**: ~8:1

### 2. Undersampling
Applied RandomUnderSampler to reduce majority class:
- **After Undersampling**:
  - Class 0: 60,000 samples
  - Class 1: 10,459 samples
  - **New Ratio**: ~5.7:1

### 3. Train-Validation-Test Split
- **Training Set**: 60% (42,275 samples before SMOTE)
  - Class 0: 36,044
  - Class 1: 6,231
- **Validation Set**: 20% (14,092 samples)
  - Class 0: 11,959
  - Class 1: 2,133
- **Test Set**: 20% (14,092 samples)
  - Class 0: 11,997
  - Class 1: 2,095

### 4. SMOTE (Synthetic Minority Over-sampling Technique)
Applied only to training data:
- **After SMOTE**:
  - Class 0: 36,044 samples
  - Class 1: 30,000 samples (synthetically generated)
  - **Final Training Size**: 66,044 samples
  - **Balanced Ratio**: ~1.2:1

### 5. Feature Scaling
StandardScaler applied to normalize all features to zero mean and unit variance.

## Model Architecture

### XGBoost Classifier
Optimized using Bayesian Optimization with the following search space:

**Hyperparameters**:
- `max_depth`: 3-8
- `learning_rate`: 0.05-0.5
- `colsample_bytree`: 0.6-1.0
- `subsample`: 0.6-1.0
- `gamma`: 0-5
- `min_child_weight`: 5-10
- `num_boost_round`: 50-500
- `scale_pos_weight`: 4.0 (fixed)

**Best Parameters** (after 30 iterations):
```python
{
    'max_depth': 7,
    'learning_rate': 0.318,
    'colsample_bytree': 0.834,
    'subsample': 0.978,
    'gamma': 0.597,
    'min_child_weight': 5,
    'num_boost_round': 185,
    'scale_pos_weight': 4.0
}
```

### Ensemble Approach
Final predictions use a weighted ensemble:
- **60% XGBoost** predictions
- **40% Logistic Regression** predictions (L1 penalty)
- **Decision Threshold**: 0.6 (increased for precision)

## Model Performance

### Test Set Results (Threshold = 0.6)
```
Accuracy: 81.33%

              precision    recall  f1-score   support
Class 0 (Sell)    0.88      0.90      0.89     11,997
Class 1 (Buy)     0.36      0.32      0.34      2,095

Macro Avg         0.62      0.61      0.61     14,092
Weighted Avg      0.80      0.81      0.81     14,092
```

### Feature Importance (Top 10)
1. **hour** (878) - Time of day
2. **mom_div** (750) - Momentum divergence
3. **ATR_ratio** (698) - Volatility ratio
4. **Volume_ratio** (686) - Volume relative to average
5. **volatility** (640) - Price volatility
6. **ROC_10** (636) - Rate of change
7. **OBV** (633) - On-Balance Volume
8. **high_low_spread** (613) - Price range
9. **volume** (544) - Trading volume
10. **RSI_lag_1** (544) - Previous RSI value

## Installation

### Requirements
```bash
pip install -r requirements.txt
```

### Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- pandas-ta
- xgboost
- bayesian-optimization
- imbalanced-learn (for SMOTE)

## Usage

### Training the Model
```python
# Run the Jupyter notebook
jupyter notebook quant.ipynb
```

### Model Files
- `optimized_xgboost.joblib` - Trained XGBoost model
- `optimized_xgboost_model.joblib` - Alternative model checkpoint
- `optimized_model.joblib` - Final ensemble model

### Flask API (main.py)
Deploy the model as a REST API:
```python
python main.py
```

**Endpoint**: POST `/predict`

**Request Format**:
```json
{
  "open": 45000.0,
  "high": 45500.0,
  "low": 44800.0,
  "close": 45200.0,
  "volume": 1000000,
  "SMA_20": 45100.0,
  "RSI_14": 55.0,
  ...
}
```

**Response**:
```json
{
  "prediction": "Buy"  // or "Sell"
}
```

## Key Techniques

1. **Class Imbalance Handling**:
   - RandomUnderSampler for majority class
   - SMOTE for minority class oversampling
   - scale_pos_weight parameter in XGBoost

2. **Hyperparameter Optimization**:
   - Bayesian Optimization for efficient search
   - 30 iterations with early stopping

3. **Model Ensemble**:
   - XGBoost + Logistic Regression
   - Weighted voting with custom threshold

4. **Feature Engineering**:
   - 25 technical indicators
   - Interaction features (spreads, ratios)
   - Temporal features (hour, day of week)
   - Lag features for time series

## Project Structure
```
try-quant/
├── data/
│   ├── 2018-22/              # Historical BTC data
│   ├── 2023/                 # Recent BTC data
│   └── merged/               # Merged timeframe data
├── models/
│   └── optimized_model.joblib
├── main.py                   # Flask API
├── quant.ipynb              # Training notebook
├── requirements.txt         # Dependencies
├── processed_data_version_7.csv  # Processed dataset
└── optimized_xgboost.joblib # Trained model

```

## Future Improvements

- Add more cryptocurrencies for diversification
- Implement walk-forward optimization
- Add risk management features (stop-loss, take-profit)
- Real-time data streaming integration
- Backtesting framework with transaction costs
- Deep learning models (LSTM, Transformer)

## License

This project is for educational and research purposes only. Not financial advice.

## Disclaimer

⚠️ **Trading cryptocurrencies carries significant risk. This model is for educational purposes only and should not be used for actual trading without proper risk management and additional validation.**
