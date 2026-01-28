import streamlit as st
import numpy as np
import joblib

st.title("🚗 BMW Sales Predictor")

try:
    model = joblib.load('saved_models/ada_model.pkl')
    st.success("✅ Модель загружена")
except Exception as e:
    st.error(f"❌ Ошибка: {e}")
    st.stop()


st.write("Нажмите кнопку для предсказания")

if st.button("Сделать предсказание"):

    test_data = np.random.rand(1, 31)
    

    prediction = model.predict(test_data)[0]
    proba = model.predict_proba(test_data)[0]
    
    if prediction == 1:
        st.success(f"✅ HIGH SALES (вероятность: {proba[1]:.1%})")
    else:
        st.error(f"❌ LOW SALES (вероятность: {proba[0]:.1%})")

