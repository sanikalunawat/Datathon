import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load your dataset (use your actual dataset path here)
df = pd.read_csv('customer_data.csv')

# Preprocess the data (handle missing values, encode categorical data, etc.)
label_encoder = LabelEncoder()
scaler = StandardScaler()

# Encode categorical variables
df['City_Encoded'] = label_encoder.fit_transform(df['City'])
df['Customer_Name_Encoded'] = label_encoder.fit_transform(df['Customer_Name'])
df['Store_Type_Encoded'] = label_encoder.fit_transform(df['Store_Type'])

# Select features and target variable
X = df[['Customer_Name_Encoded', 'City_Encoded', 'Store_Type_Encoded', 'Total_Items', 'Discount_Applied']]
y = df['Total_Cost']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model, label encoder, and scaler to files
joblib.dump(model, 'model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model, Label Encoder, and Scaler saved successfully!")
