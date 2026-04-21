import streamlit as st
import pickle
import numpy as np

# تحميل الموديل والـ Scaler
# ملحوظة: تأكدي أن الأسماء دي بالظبط هي اللي موجودة في ملفاتك على غيت هاب
model = pickle.load(open('diabetes_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title('Diabetes Prediction System')

# مدخلات المستخدم
pregnancies = st.number_input('Pregnancies')
glucose = st.number_input('Glucose')
blood_pressure = st.number_input('Blood Pressure')
skin_thickness = st.number_input('Skin Thickness')
insulin = st.number_input('Insulin')
bmi = st.number_input('BMI')
dpf = st.number_input('Diabetes Pedigree Function')
age = st.number_input('Age')

if st.button('Predict'):
    # تجميع البيانات في مصفوفة وعمل Reshape (هذا هو الحل)
    input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]).reshape(1, -1)
    
    # تحويل البيانات باستخدام الـ Scaler
    input_scaled = scaler.transform(input_data)
    
    # التوقع
    prediction = model.predict(input_scaled)
    
    if prediction[0] == 1:
        st.error('Positive: The person might have diabetes.')
    else:
        st.success('Negative: The person is likely healthy.')
