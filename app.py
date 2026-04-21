
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. تحميل الموديل والـ Scaler
# تأكدي أن الأسماء مطابقة للملفات اللي حفظناها
model = joblib.load('final_diabetes_model.pkl')
scaler = joblib.load('final_scaler.pkl')

st.set_page_config(page_title="Diabetes Risk Detector", layout="centered")

st.title("🩺 Smart Detection of Diabetes Risk")
st.write("Please enter the patient's data to predict the risk level:")

# 2. إنشاء خانات الإدخال (الترتيب مهم جداً بناءً على الأعمدة في X)
# ملحوظة: الترتيب هنا لازم يطابق ترتيب الأعمدة اللي الموديل اتدرب عليها
age = st.number_input("Age", min_value=1, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
fasting_glucose = st.number_input("Fasting Glucose Level", value=100.0)
hba1c = st.number_input("HbA1c Level", value=5.5)
blood_pressure = st.number_input("Blood Pressure", value=120)
waist = st.number_input("Waist Circumference (cm)", value=80.0)
diabetes_score = st.number_input("Diabetes Risk Score", value=50.0)

# 3. زرار التوقع
if st.button("Predict Risk Level"):
    # تجميع البيانات في مصفوفة (Array) بنفس ترتيب الأعمدة في ملف الإكسيل
    # تأكدي من إضافة باقي الأعمدة لو كانت موجودة في الـ X بتاعك
    input_data = np.array([[age, bmi, blood_pressure, fasting_glucose, hba1c, waist, diabetes_score]])
    
    # عمل Scaling للبيانات الجديدة
    input_scaled = scaler.transform(input_data)
    
    # التوقع
    prediction = model.predict(input_scaled)
    
    # عرض النتيجة
    st.subheader(f"The predicted category is: {prediction[0]}")
    
    if "High" in str(prediction[0]):
        st.error("Warning: High Risk detected. Please consult a doctor.")
    else:
        st.success("Safe: Low Risk or Prediabetes detected.")
