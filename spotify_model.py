# spotify_model.py

import pandas as pd
import joblib

def predict_cluster(features_dict):
    scaler = joblib.load("models/scaler.pkl")
    model = joblib.load("models/kmeans_model.pkl")

    input_df = pd.DataFrame([features_dict])
    scaled = scaler.transform(input_df)
    prediction = model.predict(scaled)
    return int(prediction[0])
