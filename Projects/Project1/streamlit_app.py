import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset
file_path = 'house_price_prediction_dataset.csv'  # Adjust this path if needed
df = pd.read_csv(file_path)

# Assume the model is trained using the same code provided earlier
# Feature selection
features = ['num_bedrooms', 'num_bathrooms', 'square_footage', 'age_of_house']
X = df[features]
y = df['house_price']

# Train the model
model = LinearRegression()
model.fit(X, y)

# UI Components
st.title("House Price Prediction")

# Input fields for user data
bedrooms = st.number_input("Enter the number of bedrooms:", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Enter the number of bathrooms:", min_value=1, max_value=10, value=2)
sqft_living = st.number_input("Enter the square footage:", min_value=500, max_value=10000, value=1500)
age = st.number_input("Enter the age of the house:", min_value=0, max_value=100, value=20)

# Prediction button
if st.button("Predict House Price"):
    # Create input array for prediction
    input_data = np.array([[bedrooms, bathrooms, sqft_living, age]])
    
    # Perform prediction
    predicted_price = model.predict(input_data)
    
    # Display the prediction
    st.write(f"Predicted House Price: ${predicted_price[0]:,.2f}")

# To run the app, save this script as app.py and execute the following command in the terminal:
# streamlit run app.py
