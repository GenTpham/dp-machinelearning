import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('🎁 Machine Learning App')

st.info('This is app builds a machine learning model!')

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/GenTpham/RandomForest/refs/heads/main/weather.csv')
  df

  st.write('**X**')
  df = df.drop('date', axis = 1)
  X_raw = df.drop('weather', axis = 1)
  X_raw

  st.write('**y**')
  y_raw = df.weather
  y_raw 
with st.expander('Data Visualization'):
  st.scatter_chart(data = df, x = 'precipitation', y = 'wind', color = 'weather')

# Data preparations
with st.sidebar:
  st.header('Input features')
  # precipitation,temp_max,temp_min,wind
  precipitation = st.slider('precipitation (mm)', 0.0, 60.0, 5.5)
  temp_max = st.slider('temp_max (℃)', -2.0, 40.0, 10.0)
  temp_min = st.slider('temp_min (℃)',-10.0, 20.0, 10.0)
  wind = st.slider('wind (km/h)', 0.0, 20.0, 10.0)

  # Create a DataFrame for the input features
  data = {'precipitation': precipitation,
          'temp_max': temp_max,
          'temp_min': temp_min,
          'wind': wind}
  input_df = pd.DataFrame(data, index =[0])
  input_weather = pd.concat([input_df, X_raw], axis = 0)
  
with st.expander('Input features'):
  st.write('**Input weather**')
  input_df
  st.write('**Combined weather data**')
  input_weather

X = input_weather[1:]
input_row = input_weather[:1]
# Encode Y
target_mapper = {'drizzle': 0,
                  'fog': 1,
                  'rain': 2,
                  'snow': 3,
                  'sun': 4}
def target_encode(val):
  return target_mapper[val]

y = y_raw.apply(target_encode)

with st.expander('Data preparation'):
  st.write('**Encoded weather**')
  y
  
# Model Training
clf = RandomForestClassifier()
clf.fit(X, y)

# Make Predictions
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ['drizzle', 'fog', 'rain', 'snow', 'sun']
df_prediction_proba.rename(columns = {'drizzle': 0,
                                        'fog': 1,
                                        'rain': 2,
                                        'snow': 3,
                                        'sun': 4})

st.subheader('Predicted Weather'):
weather_types = np.array(['drizzle', 'fog', 'rain', 'snow', 'sun'])
st.success(str(weather_types[prediction][0]))



