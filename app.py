import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
st.title("Retail Outlet Sales Forecasting & Customer Behavior Analysis")
st.subheader("📊 Predicting Customer Spending Trends")

df = pd.read_csv("retail_transactions_india_updated.csv")

# Data Preprocessing
df = df.drop(columns=['Customer_Name', 'Customer_ID'])

# Handle missing values
imputer = SimpleImputer(strategy='most_frequent')
df['Age'] = imputer.fit_transform(df[['Age']])

# Label Encoding
label_encoder = LabelEncoder()
categorical_columns = ['Gender', 'Income_Level', 'Visit_Frequency', 'Customer_Category',
                       'Loyalty_Member', 'Payment_Method', 'City', 'Region', 'Store_Type',
                       'Season', 'Day_of_Week']
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Day_of_Year'] = df['Date'].dt.dayofyear

scaler = MinMaxScaler()
df['Total_Cost'] = scaler.fit_transform(df[['Total_Cost']])

# Define Features and Target
X = df[['Age', 'Gender', 'Income_Level', 'Visit_Frequency', 'Customer_Category',
        'Loyalty_Member', 'Total_Items', 'Discount_Applied', 'City', 'Region', 'Store_Type',
        'Month', 'Day_of_Year']]
y = df['Total_Cost']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Performance Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

st.subheader("📈 Model Performance Metrics")
st.write(f"✔️ **Mean Absolute Error (MAE):** {mae:.4f}")
st.write(f"✔️ **Mean Squared Error (MSE):** {mse:.4f}")
st.write(f"✔️ **Root Mean Squared Error (RMSE):** {rmse:.4f}")

# Display Sample Predictions
st.subheader("🔍 Sample Predictions vs. Actual Values")
predictions_df = pd.DataFrame({"Actual": y_test[:10].values, "Predicted": y_pred[:10]})
st.dataframe(predictions_df)

# Interactive Input for User Prediction
st.subheader("🎯 Predict Customer Spending")
age = st.slider("Select Age", 18, 65, 30)
total_items = st.slider("Select Total Items Purchased", 1, 10, 3)
discount_applied = st.slider("Select Discount Applied (%)", 0, 30, 10)

# Encoding categorical inputs
gender = label_encoder.transform([st.selectbox("Select Gender", ["Male", "Female"])])[0]
income_level = label_encoder.transform([st.selectbox("Select Income Level", ["Low", "Middle", "High"])])[0]
city = label_encoder.transform([st.selectbox("Select City", df['City'].unique())])[0]

# Prepare input for prediction
input_data = np.array([[age, gender, income_level, 1, 1, 1, total_items, discount_applied, city, 1, 1, 5, 100]])
predicted_spending = model.predict(input_data)[0]

st.subheader("💰 Estimated Spending")
st.write(f"🛒 **Predicted Total Cost:** ₹{predicted_spending:.2f}")

