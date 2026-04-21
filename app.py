import streamlit as st
import pickle
import numpy as np
import sklearn # تأكيد وجود المكتبة

# تحميل الملفات بأسماء جهازك
try:
    with open('final_diabetes_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('final_scaler (1).pkl', 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error("ارفعي ملفات الـ .pkl اللي على جهازك لـ GitHub")

st.title('Diabetes Prediction System')

inputs = [st.number_input(l, value=0.0) for l in ['Pregnancies', 'Glucose', 'BP', 'Skin', 'Insulin', 'BMI', 'DPF', 'Age']]

if st.button('Predict'):
    features = np.array(inputs).reshape(1, -1)
    prediction = model.predict(scaler.transform(features))
    st.write(f"النتيجة: {'Positive' if prediction[0] == 1 else 'Negative'}")
