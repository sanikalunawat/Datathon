# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # Example model
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load your dataset
try:
    df = pd.read_csv('backend/customer_data.csv')  # Update the path if needed
except FileNotFoundError:
    print("Error: File 'customer_data.csv' not found. Check the file path.")
    exit()

# Check if the dataset is loaded correctly
print("Dataset Columns:", df.columns)
print("Dataset Shape:", df.shape)
print(df.head())

# Verify that the required columns exist
required_columns = ['Total_Items', 'Total_Cost', 'Payment_Method', 'City',
                    'Discount_Applied', 'Customer_Category', 'Season', 'Promotion', 'Street']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    print(f"Error: The following columns are missing in the dataset: {missing_columns}")
    exit()

# Prepare features and target
X = df[required_columns]
y = df['Total_Cost']  # Target column

# Label encoding for categorical variables
label_encoder = LabelEncoder()
categorical_columns = ['Payment_Method', 'City', 'Customer_Category', 'Season', 'Promotion', 'Street']
for col in categorical_columns:
    X[col] = label_encoder.fit_transform(X[col])

# Standardize numerical features
scaler = StandardScaler()
numerical_columns = ['Total_Items', 'Total_Cost']
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model (example: RandomForestRegressor)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model (optional)
score = model.score(X_test, y_test)
print(f"Model R^2 Score: {score}")

# Save the model, label encoder, and scaler
joblib.dump(model, 'backend/model.pkl')
joblib.dump(label_encoder, 'backend/label_encoder.pkl')
joblib.dump(scaler, 'backend/scaler.pkl')

print("Model, label encoder, and scaler saved successfully!")