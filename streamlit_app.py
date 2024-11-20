import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cấu hình trang
st.set_page_config(
    page_title="Dự báo Thời tiết",
    page_icon="⛅",
    layout="wide"
)

# Tiêu đề và giới thiệu
st.title('⛅ Dự báo Thời tiết')
st.info('Ứng dụng sử dụng mô hình máy học Random Forest để dự báo thời tiết dựa trên các thông số đầu vào.')

# Load và xử lý dữ liệu
@st.cache_data
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/GenTpham/RandomForest/refs/heads/main/weather.csv')
    return df

df = load_data()

# Tab cho các phần khác nhau của ứng dụng
tab1, tab2, tab3 = st.tabs(["📊 Dữ liệu", "📈 Phân tích", "🎯 Dự báo"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.write('**Dữ liệu gốc**')
        st.dataframe(df, height=300)
    
    with col2:
        df_processed = df.drop('date', axis=1)
        X_raw = df_processed.drop('weather', axis=1)
        y_raw = df_processed.weather
        st.write('**Thống kê dữ liệu**')
        st.write(df.describe())

with tab2:
    st.subheader("Trực quan hóa dữ liệu")
    col1, col2 = st.columns(2)
    
    with col1:
        st.scatter_chart(
            data=df,
            x='precipitation',
            y='wind',
            color='weather',
            height=400
        )
    
    with col2:
        st.scatter_chart(
            data=df,
            x='temp_max',
            y='temp_min',
            color='weather',
            height=400
        )

# Sidebar cho input
with st.sidebar:
    st.header('🎛️ Thông số đầu vào')
    st.markdown('---')
    
    precipitation = st.slider(
        'Lượng mưa (mm)', 
        min_value=0.0,
        max_value=60.0,
        value=5.5,
        help="Lượng mưa trong ngày"
    )
    
    temp_max = st.slider(
        'Nhiệt độ cao nhất (℃)',
        min_value=-2.0,
        max_value=40.0,
        value=10.0,
        help="Nhiệt độ cao nhất trong ngày"
    )
    
    temp_min = st.slider(
        'Nhiệt độ thấp nhất (℃)',
        min_value=-10.0,
        max_value=30.0,
        value=10.0,
        help="Nhiệt độ thấp nhất trong ngày"
    )
    
    wind = st.slider(
        'Tốc độ gió (km/h)',
        min_value=0.0,
        max_value=20.0,
        value=10.0,
        help="Tốc độ gió trung bình"
    )

# Chuẩn bị dữ liệu
data = {
    'precipitation': precipitation,
    'temp_max': temp_max,
    'temp_min': temp_min,
    'wind': wind
}
input_df = pd.DataFrame(data, index=[0])
input_weather = pd.concat([input_df, X_raw], axis=0)

# Mã hóa biến mục tiêu
target_mapper = {
    'drizzle': 0,
    'fog': 1,
    'rain': 2,
    'snow': 3,
    'sun': 4
}

def target_encode(val):
    return target_mapper[val]

y = y_raw.apply(target_encode)

# Huấn luyện mô hình
X = input_weather[1:]
input_row = input_weather[:1]

# Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Đánh giá mô hình
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Dự đoán
with tab3:
    st.subheader('🎯 Kết quả dự báo')
    
    # Dự đoán và xác suất
    prediction = clf.predict(input_row)
    prediction_proba = clf.predict_proba(input_row)
    
    # Hiển thị kết quả
    weather_types = np.array(['drizzle', 'fog', 'rain', 'snow', 'sun'])
    weather_icons = {
        'drizzle': '🌧️',
        'fog': '🌫️',
        'rain': '⛈️',
        'snow': '❄️',
        'sun': '☀️'
    }
    
    predicted_weather = weather_types[prediction][0]
    
    # Hiển thị dự báo chính
    st.markdown(f"""
    ### Dự báo: {weather_icons[predicted_weather]} {predicted_weather.title()}
    #### Độ chính xác tổng thể của mô hình: {accuracy:.2%}
    """)
    
    # Hiển thị xác suất dự báo
    st.write("**Xác suất cho từng loại thời tiết:**")
    
    # Tạo DataFrame cho xác suất
    proba_df = pd.DataFrame(
        prediction_proba,
        columns=[f"{weather_icons[w]} {w.title()}" for w in weather_types]
    )
    
    # Hiển thị xác suất dạng bảng
    st.dataframe(
        proba_df.applymap(lambda x: f"{x:.2%}"),
        hide_index=True
    )
    
    # Hiển thị biểu đồ xác suất
    st.bar_chart(proba_df.T)
