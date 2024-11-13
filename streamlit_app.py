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
  precipitation = st.slider('precipitation (mm)', 0.0, 5.0, 60.0)
  temp_max = st.slider('temp_max (℃)', -2, 10, 40)
  temp_min = st.slider('temp_min (℃)',-10, 10, 20)
  wind = st.slider('wind (km/h)', 0, 10, 20)
