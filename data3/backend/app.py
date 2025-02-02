from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Check if the model file exists before trying to load
model_path = '/backend/model.pkl'
label_encoder_path = '/backend/label_encoder.pkl'
scaler_path = '/backend/scaler.pkl'

# Ensure the model, label encoder, and scaler files exist
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found! Please make sure '{model_path}' exists.")
if not os.path.exists(label_encoder_path):
    raise FileNotFoundError(f"Label encoder file not found! Please make sure '{label_encoder_path}' exists.")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler file not found! Please make sure '{scaler_path}' exists.")

# Try to load the model, label encoder, and scaler
try:
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)
    scaler = joblib.load(scaler_path)
    print("Model, label encoder, and scaler loaded successfully!")
except Exception as e:
    print(f"Error loading model or related files: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    customer_name = data.get('customer_name')

    if not customer_name:
        return jsonify({'error': 'Customer name is required'}), 400

    try:
        # You can implement your own logic here to prepare the customer data
        # For example, we can use the encoded version of the customer_name:
        customer_name_encoded = label_encoder.transform([customer_name])[0]

        # Here, you need to add logic for transforming customer features (this is a simplified example):
        customer_features = [customer_name_encoded]  # Add all relevant features

        # Make the prediction
        predicted_spend = model.predict([customer_features])[0]

        # Return the prediction
        return jsonify({
            'customer_name': customer_name,
            'predicted_spend': round(float(predicted_spend), 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
