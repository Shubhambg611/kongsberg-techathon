import streamlit as st
import pandas as pd
import joblib
import requests
from datetime import datetime

@st.cache_data
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def fetch_weather_data():
    api_key = '0c874012f39542739e4f812b3de29c53'
    location = 'Mumbai'
    url = f"https://api.weatherbit.io/v2.0/current?city={location}&key={api_key}&units=M"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if 'data' in data and len(data['data']) > 0:
            weather = data['data'][0]
            outdoor_temp = weather.get('temp', 0)
            humidity = weather.get('rh', 0)
            wind_speed = weather.get('wind_spd', 0)
            precipitation = weather.get('precip', 0)
            solar_radiation = weather.get('solar_rad', 800)  # Fetch solar radiation if available
            return outdoor_temp, humidity, wind_speed, solar_radiation, precipitation
        else:
            st.error("Weather data could not be retrieved. Please check the API response.")
            return 0, 0, 0, 800, 0
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
        st.error(f"Error making prediction: {e}")
        return None

def predict_monthly_consumption():
    model = load_model('xgboost_model.pkl')
    if model is None:
        return "Model is not available."

    try:
        next_month_year = datetime.now().year
        next_month_number = datetime.now().month + 1

        if next_month_number > 12:
            next_month_number = 1
            next_month_year += 1

        next_month_name = datetime(next_month_year, next_month_number, 1).strftime('%B')

        last_known_value = 1000  # Replace with actual last known value
        next_month = pd.DataFrame({
            'Year': [next_month_year],
            'Month': [next_month_number],
            'Prev_Total_Energy_Consumption': [last_known_value]
        })

        next_month_prediction = model.predict(next_month)
        return f"Predicted Total Energy Consumption for {next_month_name} {next_month_year}: {int(next_month_prediction[0])} kWh"
    except Exception as e:
        st.error(f"Error predicting monthly consumption: {e}")
        return "Prediction could not be made."

def predict_solar_energy(outdoor_temp, solar_rad):
    model_path = 'D:/Kongsburg/solar_generation_model.pkl'
    rf_model = load_model(model_path)
    if rf_model is None:
        return "Solar generation model is not available."

    try:
        new_data = pd.DataFrame({
            'Outdoor Temperature (°C)': [outdoor_temp],
            'Solar Radiation (W/m²)': [solar_rad]
        })
        predicted_generation = rf_model.predict(new_data)
        return int(predicted_generation[0])
    except Exception as e:
        st.error(f"Error making solar energy prediction: {e}")
        return None

def perform_daily_calculation():
    st.write("Performing daily energy consumption calculation...")

    # Set a fixed building size value
    building_size = 50000  # Set the building size to the desired fixed value (e.g., 50000 m²)

    outdoor_temp, humidity, wind_speed, solar_radiation, precipitation = fetch_weather_data()

    st.subheader("Fetched Weather Data")
    st.write(f"Temperature: {outdoor_temp}°C")
    st.write(f"Humidity: {humidity}%")
    st.write(f"Wind Speed: {wind_speed} m/s")
    st.write(f"Solar Radiation: {solar_radiation} W/m²")
    st.write(f"Precipitation: {precipitation} mm")

    energy_model = load_model('energy_consumption_model.pkl')
    if energy_model is None:
        st.write("Energy consumption model is not available. Returning to main page.")
        return

    solar_model = load_model('D:/Kongsburg/solar_generation_model.pkl')
    if solar_model is None:
        st.write("Solar generation model is not available. Returning to main page.")
        return

    # Predict energy consumption
    energy_consumption = predict_energy_consumption(energy_model, outdoor_temp, humidity, building_size, solar_radiation, wind_speed, precipitation)
    
    # Predict solar energy generation
    solar_generation = predict_solar_energy(outdoor_temp, solar_radiation)
    
    if energy_consumption is not None and solar_generation is not None:
        st.session_state.result = (f'Estimated Total Energy Consumption for Tomorrow: {energy_consumption:.2f} kWh\n'
                                   f'Predicted Daily Solar Energy Generation: {solar_generation} kWh')
    else:
        st.session_state.result = "Prediction could not be made."

    st.session_state.page = "Home"

def optimize_thermostat():
    st.title("Thermostat Optimization")

    outdoor_temp = st.number_input('Outdoor Temperature (°C)', value=22)
    solar_radiation = st.number_input('Solar Radiation (W/m²)', value=150)

    if st.button("Optimize Thermostat Setting"):
        model_path = 'thermostat_model.pkl'
        thermostat_model = load_model(model_path)
        if thermostat_model is None:
            st.write("Thermostat optimization model is not available.")
            return

        try:
            new_data = pd.DataFrame({
                'Outdoor Temperature (°C)': [outdoor_temp],
                'Solar Radiation (W/m²)': [solar_radiation],
                'Insulation Quality_average': [1],
                'Insulation Quality_good': [0],
                'Insulation Quality_poor': [0],
                'Total Energy Consumption (kWh)': [2100]  # Fixed value
            })
            predicted_temperature = thermostat_model.predict(new_data)
            st.session_state.result = f"Predicted Thermostat Setting: {predicted_temperature[0]:.2f}°C"
        except Exception as e:
            st.error(f"Error making thermostat prediction: {e}")
            st.session_state.result = "Prediction could not be made."

    if st.session_state.result:
        st.subheader("Result")
        st.write(st.session_state.result)
        st.session_state.result = None

def show_predictive_maintenance():
    st.title("Machine Predictive Maintenance Classification")

    # Load the predictive maintenance model
    rfc = load_model('model.joblib')
    if rfc is None:
        st.write("Predictive maintenance model is not available.")
        return

    # Getting the input data from the user
    col1, col2 = st.columns(2)

    with col1:
        selected_type = st.selectbox('Select a Type', ['Low', 'Medium', 'High'])
        type_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
        selected_type = type_mapping[selected_type]

    with col2:
        air_temperature = st.text_input('Air temperature [C]')

    with col1:
        process_temperature = st.text_input('Process temperature [C]')

    with col2:
        rotational_speed = st.text_input('Rotational speed [rpm]')

    with col1:
        torque = st.text_input('Torque [Nm]')

    with col2:
        tool_wear = st.text_input('Tool wear [min]')

    # Code for Prediction
    failure_pred = ''

    if st.button('Predict Failure'):
        try:
            failure_pred = rfc.predict([[selected_type, air_temperature, 
                                         process_temperature, rotational_speed,
                                         torque, tool_wear]])
            failure_pred = 'Failure' if failure_pred[0] == 1 else 'No Failure'
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            failure_pred = "Prediction could not be made."

    st.success(failure_pred)

def main():
    if 'page' not in st.session_state:
        st.session_state.page = "Home"
    if 'result' not in st.session_state:
        st.session_state.result = None

    st.title("Energy Management Dashboard")

    st.sidebar.title("Menu")
    page = st.sidebar.radio("Select a Page", ["Home", "Daily Consumption", "Monthly Prediction", "Solar Prediction", "Thermostat Optimization", "Predictive Maintenance"])

    st.session_state.page = page

    if st.session_state.page == "Home":
        st.subheader("Home")
        st.write("Welcome to the Energy Management Dashboard. Use the sidebar to navigate.")

    elif st.session_state.page == "Daily Consumption":
        st.subheader("Daily Consumption Prediction")
        if st.button("Calculate Daily Consumption"):
            perform_daily_calculation()

    elif st.session_state.page == "Monthly Prediction":
        st.subheader("Monthly Energy Prediction")
        if st.button("Predict Monthly Energy Consumption"):
            result = predict_monthly_consumption()
            st.session_state.result = result

    elif st.session_state.page == "Solar Prediction":
        st.subheader("Daily Solar Energy Prediction")
        outdoor_temp, _, _, solar_radiation, _ = fetch_weather_data()

        st.write("Fetched Weather Data")
        st.write(f"Temperature: {outdoor_temp}°C")
        st.write(f"Solar Radiation: {solar_radiation} W/m²")

        if st.button("Predict Daily Solar Energy Generation"):
            prediction = predict_solar_energy(outdoor_temp, solar_radiation)
            if prediction is not None:
                st.session_state.result = f'Predicted Daily Solar Energy Generation: {prediction} kWh'
            else:
                st.session_state.result = "Prediction could not be made."

    elif st.session_state.page == "Thermostat Optimization":
        optimize_thermostat()

    elif st.session_state.page == "Predictive Maintenance":
        show_predictive_maintenance()

    if st.session_state.result:
        st.subheader("Result")
        st.write(st.session_state.result)
        st.session_state.result = None

if __name__== "__main__":
    main()
