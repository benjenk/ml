import streamlit as st
import joblib
import numpy as np

# Load the model and scaler
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

# Function to preprocess input
def preprocess_input(year):
    scaled_year = scaler.transform(np.array([[year]]))

    return scaled_year

# Function to predict based on input
def predict(year):
    processed_input = preprocess_input(year)
    prediction = model.predict(processed_input)
    return prediction

# Streamlit app
def main():
    st.title('Prediksi menggunakan model Machine Learning')

    # Input for 'tahun'
    tahun = st.number_input('Masukkan tahun', min_value=0, max_value=3000, value=2023)

    # When the user clicks the 'Predict' button
    if st.button('Prediksi'):
        prediction = predict(tahun)
        st.write(f'Prediksi untuk tahun {tahun} => {prediction[0]}')

if __name__ == '__main__':
    main()
