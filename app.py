import streamlit as st
import pickle
import numpy as np
import pandas as pd

# 1. تحميل الموديل والـ Scaler
# تأكدي إن الأسماء دي هي نفس أسماء الملفات اللي رفعتيها على GitHub
model = pickle.load(open('diabetes_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# عنوان الصفحة
st.title('Diabetes Prediction App')

# 2. إنشاء خانات الإدخال (تأكدي من ترتيبهم حسب تدريب الموديل)
pregnancies = st.number_input('Pregnancies', min_value=0)
glucose = st.number_input('Glucose', min_value=0)
blood_pressure = st.number_input('Blood Pressure', min_value=0)
skin_thickness = st.number_input('Skin Thickness', min_value=0)
insulin = st.number_input('Insulin', min_value=0)
bmi = st.number_input('BMI', min_value=0.0)
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0)
age = st.number_input('Age', min_value=0)

# زرار التوقع
if st.button('Predict'):
    # 3. تجميع المدخلات في قائمة
    features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
    
    # 4. تحويل المدخلات لشكل مصفوفة (الخطوة اللي كانت ناقصة)
    features_array = np.array(features).reshape(1, -1)
    
    # 5. عمل Scaling للمدخلات
    scaled_features = scaler.transform(features_array)
    
    # 6. التوقع
    prediction = model.predict(scaled_features)
    
    # عرض النتيجة
    if prediction[0] == 1:
        st.error('The person is likely to have diabetes.')
    else:
        st.success('The person is likely to be healthy.')