from flask import Flask, render_template, request, jsonify
import joblib  # Import joblib to load the model
import numpy as np
from forecast import make_prediction  # Import the function for prediction from forecast.py

app = Flask(__name__)

# Route for index page (rendering the HTML form)
@app.route('/')
def index():
    return render_template('index.html')  # This will render your HTML page

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the POST request
        data = request.get_json()
        input_features = np.array(data['features']).reshape(1, -1)  # Reshape input to match model input

        # Use the make_prediction function from forecast.py to get the prediction
        prediction = make_prediction(input_features)

        # Return the prediction result as a JSON response
        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
