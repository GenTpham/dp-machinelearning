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
