import gradio as gr
import pandas as pd 
import joblib 
import matplotlib.pyplot as plt 
import seaborn as sns 

# Load the dataset 
scalar = joblib.load("Data/scaler.pkl")
linear_model = joblib.load("Data/linear_model.pkl")
polynomial_model = joblib.load("Data/polynomial_model.pkl")
logistic = joblib.load("Data/Logistic_model.pkl")
rf_classifier = joblib.load("Data/RandomForest_model.pkl")


def predict_aqi(pm25 , pm10 , no2 , co , temp , humidity) :
    input_data  = pd.DataFrame([[pm25 , pm10 , no2 , co , temp , humidity]] , columns=["PM2.5" , "PM10" , "NO2" , "CO" , "Temperature" , "Humidity"])
    input_scaled = scalar.transform(input_data)
    
    linear = linear_model.predict(input_scaled)[0]
    poly = polynomial_model.predict(input_scaled)[0]
    lgs = logistic.predict(input_scaled)[0]
    rf = rf_classifier.predict(input_scaled)[0]
    
    #creating a performance plot
    models = ["linear" , "poly"]
    predictions = [linear , poly]
    plt.figure(figsize=(8 , 4))
    sns.barplot(x = models , y = predictions)
    plt.title("AQI Predictions by Model")
    plt.ylabel("Predicted AQI")
    plt.savefig("aqi_plot.png")
    plt.close()
    
    linear_color = "green" if linear < 50 else "yellow" if linear < 100 else "red"
    poly_color = "green" if poly < 50 else "yellow" if poly < 100 else "red"
    lgs_color = "green" if lgs == 0 else "red"
    rf_color = "green" if rf == 0 else "red"

    output_text = f"""
        <div style="font-family: Arial, sans-serif; line-height: 1.6;">
            <h3 style="color: #333;">AQI Predictions</h3>
            <p style="color: {linear_color};">Linear Regression AQI:<strong>{linear:.2f}</strong></p>
            <p style="color: {poly_color};">Polynomial Regression AQI: <strong>{poly:.2f}</strong></p>
            <p style="color: {lgs_color};">Logistic Classification: <strong>{'Safe' if lgs == 0 else 'Unsafe'}</strong></p>
            <p style="color: {rf_color};">Random Forest Classification: <strong>{'Safe' if rf == 0 else 'Unsafe'}</strong></p>
        </div>
            """
    return output_text , "aqi_plot.png" 
    
    
    
    
    
    
        
if __name__ == "__main__":
        iface = gr.Interface(
            fn = predict_aqi , 
            inputs=[
                gr.Slider(minimum=0, maximum=200, label="PM2.5 (µg/m³)", value=50),
                gr.Slider(minimum=0, maximum=300, label="PM10 (µg/m³)", value=80),
                gr.Slider(minimum=0, maximum=100, label="NO2 (µg/m³)", value=20),
                gr.Slider(minimum=0, maximum=10, label="CO (mg/m³)", value=1),
                gr.Slider(minimum=-10, maximum=40, label="Temperature (°C)", value=20),
                gr.Slider(minimum=0, maximum=100, label="Humidity (%)", value=50)
            ],
            outputs=[
                gr.HTML(label = "Predictions"),
                gr.Image(label = "Model Comparison Plot")
                
            ],
            title = "Air Quality Prediction and Classification ",
            description="""
            Enter pollutant levels and Weather Condition To predict AQI and Classify air Quality.
            <br><br>
            <div style="font-size: 14px; text-align: right; color: #333; position: absolute; right: 10px; bottom: 10px;">
                Developed by <strong>Fayaz Ali Muktadir</strong>
            </div>
            """,
        )
        iface.launch()