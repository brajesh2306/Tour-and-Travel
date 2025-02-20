import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the model and pipeline from the pickle file
with open('tourist_attractions_model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    pipeline = data['pipeline']

# Load the dataset
data = pd.read_csv('Top Indian Places to Visit 1.csv')

# Function to recommend top 5 places
def recommend_top_places(city, entrance_fee, data, model, pipeline, top_n=5):
    # Filter data for the given city
    city_data = data[data['City'].str.lower() == city.lower()]

    # Generate other features with default or random values
    city_data['Entrance Fee in INR'] = entrance_fee
    city_data['Number of google review in lakhs'] = np.random.uniform(0.1, 2.0, len(city_data))  # Random number of Google reviews in lakhs

    # Preprocess the city data
    X_city = pipeline.transform(city_data[['City', 'Name', 'Type', 'Google review rating', 'Entrance Fee in INR', 'Weekly Off', 'Significance', 'Number of google review in lakhs', 'Best Time to visit']])

    # Predict the ratings
    city_data['Predicted Rating'] = model.predict(X_city)

    # Sort by predicted rating and get the top places
    top_places = city_data.sort_values(by='Predicted Rating', ascending=False).head(top_n)

    return top_places[['Name', 'Predicted Rating']]

# Streamlit app layout
st.title('Tourist Attraction Recommender')
st.write('Enter the city and entrance fee to get the top 5 places to visit.')

# User inputs
city = st.text_input('City')
entrance_fee = st.number_input('Entrance Fee in INR', min_value=0)

# Recommend top places when the button is clicked
if st.button('Recommend'):
    if city and entrance_fee >= 0:
        top_places = recommend_top_places(city, entrance_fee, data, model, pipeline)
        st.write(f'Top 5 places to visit in {city}:')
        st.dataframe(top_places)
    else:
        st.write('Please enter valid inputs.')
