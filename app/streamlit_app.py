"""Streamlit demo wired to the saved Bike Rental Demand model."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import streamlit as st

# Streamlit places the app directory, rather than the repository root, first
# on sys.path when this file is launched directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modeling.predict import load_model, predict_one


def main() -> None:
    st.set_page_config(page_title="Bike Rental Forecast", page_icon="B", layout="centered")
    st.title("Bike Rental Demand Forecast")
    st.caption("Hourly demand forecast using the team's XGBoost model.")

    try:
        load_model()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    timestamp = st.datetime_input(
        "Date and time",
        value=datetime(2012, 6, 15, 8),
        min_value=datetime(2011, 1, 1),
        max_value=datetime(2012, 12, 19, 23),
        help="Choose a time covered by the training data.",
    )
    left, right = st.columns(2)
    with left:
        weather = st.selectbox(
            "Weather",
            options=[1, 2, 3, 4],
            format_func=lambda value: {
                1: "Clear / few clouds",
                2: "Mist / cloudy",
                3: "Light rain or snow",
                4: "Heavy rain or snow",
            }[value],
        )
        holiday = st.checkbox("Holiday")
    with right:
        temp = st.number_input("Temperature (C)", min_value=-10.0, max_value=50.0, value=24.0)
        atemp = st.number_input("Feels-like temperature (C)", min_value=-10.0, max_value=60.0, value=26.0)
        humidity = st.slider("Humidity", 0, 100, 60)
        windspeed = st.number_input("Wind speed", min_value=0.0, max_value=100.0, value=12.0)

    if st.button("Forecast rentals", type="primary"):
        try:
            prediction = predict_one(
                when=timestamp,
                temp=temp,
                atemp=atemp,
                humidity=humidity,
                windspeed=windspeed,
                weather=weather,
                holiday=int(holiday),
            )
        except (ValueError, TypeError) as error:
            st.error(f"Unable to create a forecast: {error}")
        else:
            st.metric("Predicted rentals", f"{prediction:,}")
            st.caption("Season and working-day status are derived automatically from the chosen date.")

    with st.expander("Validation result"):
        st.write("Best saved model: XGBoost")
        metrics = st.columns(3)
        metrics[0].metric("RMSLE", "0.2999")
        metrics[1].metric("RMSE", "57.16")
        metrics[2].metric("R2", "0.9310")


if __name__ == "__main__":
    main()
