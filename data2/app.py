from flask import Flask, render_template, request
import pandas as pd
import ast
from collections import Counter

app = Flask(__name__)

# Load the dataset
def load_data():
    data = pd.read_csv('r.csv')
    data['Product'] = data['Product'].apply(ast.literal_eval)  # Convert 'Product' column to list
    return data

# Recommendation function
def recommend_products(customer_name, data):
    # Get the customer's past purchases
    customer_data = data[data['Customer_Name'] == customer_name]

    # Collect all the products purchased by the customer
    all_products = []
    for products in customer_data['Product']:
        all_products.extend(products)

    # Find the most common products purchased by the customer
    product_counter = Counter(all_products)
    most_common_products = product_counter.most_common(3)  # Top 3 products

    # Get other customers who bought similar products
    recommendations = []
    for product, _ in most_common_products:
        similar_customers = data[data['Product'].apply(lambda x: product in x)]

        # Recommend products bought by similar customers that this customer has not bought yet
        for _, row in similar_customers.iterrows():
            for p in row['Product']:
                if p not in all_products and p not in recommendations:
                    recommendations.append(p)

    # Return the recommendations
    return recommendations[:5]  # Limit to top 5 recommendations

# Home route
@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    customer_name = ""
    error = ""
    if request.method == 'POST':
        customer_name = request.form['customer_name']
        data = load_data()
        if customer_name in data['Customer_Name'].values:
            recommendations = recommend_products(customer_name, data)
        else:
            error = "Customer not found. Please enter a valid customer name."
    return render_template('index.html', recommendations=recommendations, customer_name=customer_name, error=error)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)