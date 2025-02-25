import streamlit as st
import pandas as pd
import requests
import joblib

def fetch_weather_data():
    api_key = '0c874012f39542739e4f812b3de29c53'
    location = 'Mumbai'
    url = f"http://api.weatherbit.io/v2.0/current?city={location}&key={api_key}&units=M"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if 'data' in data:
            weather = data['data'][0]
            outdoor_temp = weather.get('temp', 0)
            humidity = weather.get('rh', 0)
            wind_speed = weather.get('wind_spd', 0)
            precipitation = weather.get('precip', 0)
        else:
            st.error("Weather data could not be retrieved.")
            outdoor_temp, humidity, wind_speed, precipitation = 0, 0, 0, 0
        
        solar_radiation = 800
        return outdoor_temp, humidity, wind_speed, solar_radiation, precipitation
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching weather data: {e}")
        return 0, 0, 0, 800, 0

def predict_energy_consumption(model, outdoor_temp, humidity, building_size, solar_radiation, wind_speed, precipitation):
    try:
        temp_humidity_interaction = outdoor_temp * humidity
        input_data = pd.DataFrame({
            'Outdoor Temperature (°C)': [outdoor_temp],
            'Humidity (%)': [humidity],
            'Building Size (m²)': [building_size],
            'Solar Radiation (W/m²)': [solar_radiation],
            'Wind Speed (m/s)': [wind_speed],
            'Precipitation (mm)': [precipitation],
            'Temp_Humidity_Interaction': [temp_humidity_interaction]
        })
        predicted_energy_consumption = model.predict(input_data)
        return predicted_energy_consumption[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

def calculate():
    st.write("Performing calculation...")

    model = load_model('energy_consumption_model.pkl')
    if model is None:
        st.write("Model is not available. Returning to main page.")
        return

    outdoor_temp, humidity, wind_speed, solar_radiation, precipitation = fetch_weather_data()
    
    building_size = 50000  # Default value or replace with user input

    prediction = predict_energy_consumption(model, outdoor_temp, humidity, building_size, solar_radiation, wind_speed, precipitation)
    if prediction is not None:
        st.session_state.result = f'Estimated Total Energy Consumption for Tomorrow: {prediction:.2f} kWh'
    else:
        st.session_state.result = "Prediction could not be made."

    st.session_state.page = "Home"
    st.experimental_set_query_params(page="Home")  # Set query params for navigation
    st.write("Redirecting to home page...")

def load_model(model_path):
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

if st.session_state.get('page') == "Calculate":
    calculate()
else:
    st.write("Redirecting to home page...")
    st.experimental_set_query_params(page="Home")
