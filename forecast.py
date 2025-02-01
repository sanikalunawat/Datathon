import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np

# Function to load and preprocess the data
def load_and_preprocess_data():
    df = pd.read_csv("dataset.csv")
    df = df.drop(columns=['Customer_Name', 'Customer_ID'])

    # Impute missing values for 'Age'
    imputer = SimpleImputer(strategy='most_frequent')
    df['Age'] = imputer.fit_transform(df[['Age']])

    # Label encode categorical columns
    label_encoder = LabelEncoder()
    categorical_columns = ['Gender', 'Income_Level', 'Visit_Frequency', 'Customer_Category',
                           'Loyalty_Member', 'Payment_Method', 'City', 'Region', 'Store_Type',
                           'Season', 'Day_of_Week']
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])

    # Date and Time features
    df['Date'] = pd.to_datetime(df['Date'])
    df['Time_of_Purchase'] = pd.to_datetime(df['Time_of_Purchase'], format='%H:%M:%S').dt.time
    df['Month'] = df['Date'].dt.month
    df['Day_of_Year'] = df['Date'].dt.dayofyear

    # Normalize 'Total_Cost'
    scaler = MinMaxScaler()
    df['Total_Cost'] = scaler.fit_transform(df[['Total_Cost']])

    # Features and target
    X = df[['Age', 'Gender', 'Income_Level', 'Visit_Frequency', 'Customer_Category',
            'Loyalty_Member', 'Total_Items', 'Discount_Applied', 'City', 'Region',
            'Month', 'Day_of_Year']]
    y = df['Total_Cost']

    return X, y

# Function to train the model and save it
def train_and_save_model():
    X, y = load_and_preprocess_data()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model using joblib
    joblib.dump(model, 'model/model1.pkl')
    print("Model saved to 'model/model1.pkl'")

# Function to load the saved model
def load_model():
    model = joblib.load('model/model1.pkl')
    return model

# Function to make predictions using the loaded model
def make_prediction(features):
    model = load_model()
    prediction = model.predict([features])
    return prediction[0]
