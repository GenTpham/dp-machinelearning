import streamlit as st
import pandas as pd

st.title('🎁 Machine Learning App')

st.info('This is app builds a machine learning model!')

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/GenTpham/RandomForest/refs/heads/main/weather.csv')
  df

  st.write('**X**')
  df = df.drop('date', axis = 1)
  X = df.drop('weather', axis = 1)
  X

  st.write('**y**')
  y = df.weather
  y 
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
  input_weather = pd.concat([input_df, X], axis = 0)

with st.expander('Input features')
  st.write('**Input weather**')
  input_df
  st.write('**Combined weather data**')
  input_weather




