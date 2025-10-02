from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import numpy as np
from flask_cors import CORS

# --- Initialization ---
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load the trained XGBoost model
try:
    xgb_clf = joblib.load('xgb_fraud_model.joblib')
    print("✅ XGBoost model loaded successfully.")
except FileNotFoundError:
    print("❌ ERROR: Model file 'xgb_fraud_model.joblib' not found.")
    xgb_clf = None

@app.route("/form")
def serve_frontend():
    return send_from_directory('.', 'index.html')  # Serve your HTML page

@app.route("/predict", methods=["POST"])
def predict():
    if xgb_clf is None:
        return jsonify({"error": "Model not loaded. Cannot make predictions."}), 500

    try:
        data = request.json
    except Exception as e:
        return jsonify({"error": "Invalid JSON format", "details": str(e)}), 400

    # Required fields
    required_fields = [
        "amount", "orig_balance_ratio", "hour", 
        "transaction_type", "exact_drain_flag", 
        "large_transfer_flag", "is_flagged"
    ]
    
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400

    # --- Feature Engineering ---
    amount = float(data["amount"])
    orig_balance_ratio = float(data["orig_balance_ratio"])
    hour = int(data["hour"])
    transaction_type = data["transaction_type"]
    exact_drain_flag = int(data["exact_drain_flag"])
    large_transfer_flag = int(data["large_transfer_flag"])
    is_flagged = int(data["is_flagged"])

    log_amount = np.log(amount + 1)
    deltaOrig = 1 - orig_balance_ratio
    log_amount_mean = np.log(40000 + 1)  # placeholder
    amount_count = 5
    amount_mean = 50000
    amount_std = 20000

    # One-hot encode transaction type
    type_CASH_IN = int(transaction_type == 'CASH_IN')
    type_CASH_OUT = int(transaction_type == 'CASH_OUT')
    type_DEBIT = int(transaction_type == 'DEBIT')
    type_PAYMENT = int(transaction_type == 'PAYMENT')
    type_TRANSFER = int(transaction_type == 'TRANSFER')

    # Build DataFrame matching model features
    transaction_df = pd.DataFrame({
        'log_amount': [log_amount],
        'deltaOrig': [deltaOrig],
        'orig_balance_ratio': [orig_balance_ratio],
        'exact_drain_flag': [exact_drain_flag],
        'large_transfer_flag': [large_transfer_flag],
        'is_flagged': [is_flagged],
        'hour': [hour],
        'type_CASH_IN': [type_CASH_IN],
        'type_CASH_OUT': [type_CASH_OUT],
        'type_DEBIT': [type_DEBIT],
        'type_PAYMENT': [type_PAYMENT],
        'type_TRANSFER': [type_TRANSFER],
        'amount_count': [amount_count],
        'amount_mean': [amount_mean],
        'amount_std': [amount_std],
        'log_amount_mean': [log_amount_mean]
    })

    # Make prediction
    pred = xgb_clf.predict(transaction_df)[0]
    prob = xgb_clf.predict_proba(transaction_df)[:,1][0]

    result = {
        "prediction": "Fraud" if pred == 1 else "Not Fraud",
        "probability": float(prob)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
