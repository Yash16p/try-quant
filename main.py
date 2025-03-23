from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open("final_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the expected features
EXPECTED_FEATURES = [
    "open", "high", "low", "close", "volume", "SMA_20", "RSI_14", "MACD", 
    "ATR_14", "volume_SMA_20", "OBV", "ROC_10", "price_delta", "volatility", 
    "close_lag_1", "RSI_lag_1", "hour", "dayofweek", "minute", "ATR_ratio", 
    "Volume_ratio", "MACD_signal"
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input
        data = request.get_json()

        # Convert input into DataFrame
        df = pd.DataFrame([data])

        # Check for missing features
        missing_features = [f for f in EXPECTED_FEATURES if f not in df.columns]
        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"}), 400

        # Ensure correct feature order
        df = df[EXPECTED_FEATURES]

        # Make prediction
        prediction = model.predict(df)[0] 

        # Convert 0/1 to "Sell"/"Buy"
        result = "Buy" if prediction == 1 else "Sell"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
