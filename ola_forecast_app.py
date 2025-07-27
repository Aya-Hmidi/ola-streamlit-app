
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.title("🚲 Ola Bike Ride Demand Forecasting")

@st.cache_data
def load_data():
    df = pd.read_csv("ola_ride_requests.csv")
    df.rename(columns={'request_longitutude': 'request_longitude'}, inplace=True)
    df['request_Time'] = pd.to_datetime(df['request_Time'], errors='coerce')
    df = df.dropna(subset=['request_Time'])
    df['hour'] = df['request_Time'].dt.hour
    df['day_of_week'] = df['request_Time'].dt.dayofweek
    return df

df = load_data()
st.subheader("📊 Raw Dataset Sample")
st.write(df.head())

df['cluster'] = KMeans(n_clusters=5, random_state=42).fit_predict(df[['request_latitude', 'request_longitude']])

df['request_hour'] = df['request_Time'].dt.hour
df['request_count'] = df.groupby('request_hour')['request_hour'].transform('count')

X = df[['request_latitude', 'request_longitude', 'pickup_lat', 'pickup_long',
        'drop_lat', 'drop_long', 'request_hour']]
y = df['request_count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("🧠 Model Evaluation")
st.write(f"Mean Squared Error (MSE): `{mse:.2f}`")
st.write(f"R-squared (R²): `{r2:.2f}`")

st.subheader("📌 Feature Importance")
importances = model.feature_importances_
features = X.columns
fig, ax = plt.subplots()
sns.barplot(x=importances, y=features, ax=ax)
ax.set_title("Feature Importance")
st.pyplot(fig)

st.subheader("📈 Try Predicting Ride Demand")

lat = st.slider("Request Latitude", float(df['request_latitude'].min()), float(df['request_latitude'].max()))
lon = st.slider("Request Longitude", float(df['request_longitude'].min()), float(df['request_longitude'].max()))
pickup_lat = st.slider("Pickup Latitude", float(df['pickup_lat'].min()), float(df['pickup_lat'].max()))
pickup_long = st.slider("Pickup Longitude", float(df['pickup_long'].min()), float(df['pickup_long'].max()))
drop_lat = st.slider("Drop Latitude", float(df['drop_lat'].min()), float(df['drop_lat'].max()))
drop_long = st.slider("Drop Longitude", float(df['drop_long'].min()), float(df['drop_long'].max()))
hour = st.slider("Hour of Request", 0, 23)

input_data = pd.DataFrame([[lat, lon, pickup_lat, pickup_long, drop_lat, drop_long, hour]],
                          columns=X.columns)

prediction = model.predict(input_data)[0]
st.success(f"📌 Predicted Ride Requests for this hour: **{round(prediction)}**")
