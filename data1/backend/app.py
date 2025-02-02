from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import pandas as pd
import joblib

app = Flask(__name__)

# Enable CORS for all routes (this allows cross-origin requests from any origin)
CORS(app)

# Load the trained model, label encoder, and scaler
model = joblib.load('model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Load the dataset
df = pd.read_csv('customer_data.csv')

def prepare_customer_data(customer_name):
    # Get the customer's latest transaction
    customer_data = df[df['Customer_Name'] == customer_name].sort_values(by='Date', ascending=False).iloc[0]

    # Extract relevant features for prediction
    customer_features = customer_data[['Total_Items', 'Total_Cost', 'Payment_Method', 'City',
                                       'Discount_Applied', 'Customer_Category', 'Season', 'Promotion', 'Street']].copy()

    # Label encoding for categorical variables
    categorical_columns = ['Payment_Method', 'City', 'Customer_Category', 'Season', 'Promotion', 'Street']
    for col in categorical_columns:
        if customer_features[col] in label_encoder.classes_:
            customer_features[col] = label_encoder.transform([customer_features[col]])[0]
        else:
            customer_features[col] = len(label_encoder.classes_)

    # Standardize numerical features
    numerical_columns = ['Total_Items', 'Total_Cost']
    customer_features[numerical_columns] = scaler.transform([customer_features[numerical_columns]])[0]

    return customer_features

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    customer_name = data.get('customer_name')

    if not customer_name:
        return jsonify({'error': 'Customer name is required'}), 400

    try:
        # Prepare customer data
        customer_features = prepare_customer_data(customer_name)

        # Make the prediction
        predicted_spend = model.predict([customer_features])[0] * 5.3

        # Return the prediction
        return jsonify({
            'customer_name': customer_name,
            'predicted_spend': round(float(predicted_spend), 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
