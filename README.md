Air Quality Prediction and Classification App

This Python application predicts the Air Quality Index (AQI) based on pollutant levels (PM2.5, PM10, NO2, CO) and weather conditions (temperature, humidity). It uses machine learning models such as Linear Regression, Polynomial Regression, Logistic Regression, and Random Forest to predict AQI and classify air quality as safe or unsafe. The app also provides a visual comparison of model performance.

Deployment
The application is deployed on https://huggingface.co/spaces/fayazam33/Air_Quality_Predictor_App_by_fayaz . You can try the live demo there and interact with the models for real-time AQI predictions.

Models Used

Linear Regression

Polynomial Regression

Logistic Regression (for air quality classification)

Random Forest (for air quality classification)

Gradio

This app utilizes Gradio, a Python library that allows the creation of user-friendly, interactive web interfaces for machine learning models. 
With Gradio, users can interact with the AQI prediction models by adjusting the input sliders for pollutants and weather conditions.
The interface is designed to be simple to use, providing a seamless experience to predict AQI and classify air quality in real-time. 
Gradio also generates a shareable link for easy deployment and sharing.
