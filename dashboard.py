import streamlit as st
import pandas as pd
import joblib
import requests
from datetime import datetime
import os
from google.cloud import aiplatform
import google.generativeai as genai

# Configure the Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
genai.configure(api_key=" ")
# Create the model configuration
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "max_output_tokens": 2048,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.0-pro",
    generation_config=generation_config,
)

def load_model(model_path):
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def fetch_weather_data():
    api_key = '1f11388f3eba4d47828231e9877c4c7f'
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
            solar_radiation = weather.get('solar_rad', 800)
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
    model = load_model('/mnt/A42AC5272AC4F778/Kongsberg/xgboost_model.pkl')
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
    model_path = '/mnt/A42AC5272AC4F778/Kongsberg/solar_generation_model.pkl'
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

    energy_model = load_model('/mnt/A42AC5272AC4F778/Kongsberg/energy_consumption_model.pkl')
    if energy_model is None:
        st.write("Energy consumption model is not available. Returning to main page.")
        return

    solar_model = load_model('/mnt/A42AC5272AC4F778/Kongsberg/solar_generation_model.pkl')
    if solar_model is None:
        st.write("Solar generation model is not available. Returning to main page.")
        return

    # Predict energy consumption
    energy_consumption = predict_energy_consumption(energy_model, outdoor_temp, humidity, building_size, solar_radiation, wind_speed, precipitation)
    
    # Predict solar energy generation
    solar_generation = predict_solar_energy(outdoor_temp, solar_radiation)
    
    if energy_consumption is not None and solar_generation is not None:
        st.session_state.result = (f'Estimated Total Energy Consumption for Tomorrow: {energy_consumption:.2f} kWh\n'
                                   f'\nPredicted Daily Solar Energy Generation: {solar_generation} kWh')
    else:
        st.session_state.result = "Prediction could not be made."

    st.session_state.page = "Home"

def optimize_thermostat():
    st.title("Thermostat Optimization")

    outdoor_temp = st.number_input('Outdoor Temperature (°C)', value=22)
    solar_radiation = st.number_input('Solar Radiation (W/m²)', value=150)

    if st.button("Optimize Thermostat Setting"):
        model_path = '/mnt/A42AC5272AC4F778/Kongsberg/thermostat_model.pkl'
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
    rfc = load_model('/mnt/A42AC5272AC4F778/Kongsberg/Predictive maintainance/model.joblib')
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

    st.subheader("Prediction Result")
    st.write(f"Predicted Failure: {failure_pred}")

def display_energy_saving_recommendations():
    st.title("Energy Saving Recommendations")

    # Create the chat session
    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    "Energy Saving Recommendations",
                ],
            },
            {
                "role": "model",
                "parts": [
                    "*Lighting:\n\n Install energy-efficient LED or CFL bulbs to replace incandescent bulbs.\n* Use natural sunlight whenever possible by opening curtains and blinds.\n* Install motion sensors or timers in areas that receive minimal use.\n* Replace old light fixtures with energy-star rated models.\n\n*Heating and Cooling:\n\n Set the thermostat to a reasonable temperature (78°F in summer, 68°F in winter).\n* Use a programmable thermostat to adjust the temperature automatically when not home.\n* Seal air leaks around windows, doors, and vents.\n* Consider installing a heat pump or geothermal system for efficient heating and cooling.\n\n*Appliances:\n\n Choose energy-star rated appliances when replacing old ones.\n* Unplug unused appliances and electronics to prevent standby power consumption.\n* Wash clothes in cold water and air-dry.\n* Replace old refrigerators with energy-efficient models.\n\n*Water Heating:\n\n Install a low-flow showerhead and faucet aerators to reduce water usage.\n* Insulate your water heater and pipes to minimize heat loss.\n* Consider installing a solar water heater.\n\n*Other Measures:\n\n Install ceiling fans to circulate air and reduce cooling costs.\n* Use dehumidifiers to reduce moisture in the air and make your home feel cooler.\n* Plant trees around your home to provide shade and reduce cooling demand.\n* Regularly inspect and maintain your HVAC system to ensure optimal efficiency.\n* Consider using renewable energy sources such as solar panels or wind turbines.\n\n*Habits and Mindset:\n\n Turn off lights when leaving a room.\n* Unplug chargers and devices when not in use.\n* Use public transportation, walk, or bike instead of driving whenever possible.\n* Be conscious of your energy consumption and make small changes to reduce it.\n\n*Additional Tips:\n\n Conduct an energy audit to identify areas for improvement.\n* Take advantage of rebates and incentives offered by utilities.\n* Consider hiring a professional to install energy-efficient upgrades.\n* Stay informed about new technologies and advancements in energy efficiency.",
                ],
            },
        ]
    )

    # Display the initial recommendations
    st.write("*Energy Saving Recommendations*")
    st.write("Lighting:\n\n Install energy-efficient LED or CFL bulbs to replace incandescent bulbs.\n* Use natural sunlight whenever possible by opening curtains and blinds.\n* Install motion sensors or timers in areas that receive minimal use.\n* Replace old light fixtures with energy-star rated models.\n\n*Heating and Cooling:\n\n Set the thermostat to a reasonable temperature (78°F in summer, 68°F in winter).\n* Use a programmable thermostat to adjust the temperature automatically when not home.\n* Seal air leaks around windows, doors, and vents.\n* Consider installing a heat pump or geothermal system for efficient heating and cooling.\n\n*Appliances:\n\n Choose energy-star rated appliances when replacing old ones.\n* Unplug unused appliances and electronics to prevent standby power consumption.\n* Wash clothes in cold water and air-dry.\n* Replace old refrigerators with energy-efficient models.\n\n*Water Heating:\n\n Install a low-flow showerhead and faucet aerators to reduce water usage.\n* Insulate your water heater and pipes to minimize heat loss.\n* Consider installing a solar water heater.\n\n*Other Measures:\n\n Install ceiling fans to circulate air and reduce cooling costs.\n* Use dehumidifiers to reduce moisture in the air and make your home feel cooler.\n* Plant trees around your home to provide shade and reduce cooling demand.\n* Regularly inspect and maintain your HVAC system to ensure optimal efficiency.\n* Consider using renewable energy sources such as solar panels or wind turbines.\n\n*Habits and Mindset:\n\n Turn off lights when leaving a room.\n* Unplug chargers and devices when not in use.\n* Use public transportation, walk, or bike instead of driving whenever possible.\n* Be conscious of your energy consumption and make small changes to reduce it.\n\n*Additional Tips:\n\n Conduct an energy audit to identify areas for improvement.\n* Take advantage of rebates and incentives offered by utilities.\n* Consider hiring a professional to install energy-efficient upgrades.\n* Stay informed about new technologies and advancements in energy efficiency.")
    st.write("Feel free to ask any questions about energy saving tips.")

    # Text input for user questions
    user_input = st.text_input("Ask a question about energy saving:")

    if st.button("Send"):
        if user_input:
            # Send the user's message and get the response
            response = chat_session.send_message(user_input)
            st.write("*Response:*")
            st.write(response.text)
        else:
            st.write("Please enter a question.")
# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", ["Home", "Daily Consumption", "Monthly Prediction", "Solar Prediction", "Thermostat Optimization", "Predictive Maintenance", "Energy Saving Recommendations"])

st.session_state.page = page

if st.session_state.page == "Home":
    st.subheader("Home")
    st.write("Welcome to the Energy Management Dashboard. Use the sidebar to navigate.")
    st.image("/mnt/A42AC5272AC4F778/Kongsberg/img.jpg", use_column_width=False,)
    st.write("""
        This project aims to provide an integrated solution for managing energy consumption in buildings.
        The dashboard offers several functionalities, including predicting daily and monthly energy usage, optimizing thermostat settings,
        forecasting solar energy generation, and providing energy-saving recommendations. It leverages various data inputs, such as weather
        conditions and building specifications, to help you make informed decisions on energy management and sustainability efforts.
        """)

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
    
elif st.session_state.page == "Energy Saving Recommendations":
    display_energy_saving_recommendations()

if st.session_state.result:
    st.subheader("Result")
    st.write(st.session_state.result)
    st.session_state.result = None
