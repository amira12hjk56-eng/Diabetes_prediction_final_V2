import streamlit as st
import pickle
import numpy as np
import os

# دالة للبحث عن ملف الموديل بأي اسم متاح
def load_file(possible_names):
    for name in possible_names:
        if os.path.exists(name):
            return pickle.load(open(name, 'rb'))
    return None

st.title('Diabetes Prediction System')

# محاولة تحميل الملفات بكل الأسامي اللي ظهرت في الصور
model = load_file(['diabetes_model (1).pkl', 'diabetes_model.pkl', 'final_diabetes_model.pkl'])
scaler = load_file(['final_scaler (1).pkl', 'scaler.pkl', 'final_scaler.pkl'])

if model is None or scaler is None:
    st.error("⚠️ مشكلة في ملفات الموديل: تأكدي إن ملفات الـ .pkl مرفوعة على GitHub في نفس الفولدر مع app.py")
else:
    # المدخلات
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input('Pregnancies', min_value=0)
        glucose = st.number_input('Glucose', min_value=0)
        bp = st.number_input('Blood Pressure', min_value=0)
        skin = st.number_input('Skin Thickness', min_value=0)
    with col2:
        insulin = st.number_input('Insulin', min_value=0)
        bmi = st.number_input('BMI', min_value=0.0)
        dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0)
        age = st.number_input('Age', min_value=0)

    if st.button('Predict'):
        # الحل التقني لمشكلة الـ ValueError (Reshape)
        input_data = np.array([pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]).reshape(1, -1)
        
        # عمل التحجيم والتوقع
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        
        if prediction[0] == 1:
            st.error('النتيجة: الشخص مصاب بالسكري')
        else:
            st.success('النتيجة: الشخص سليم')