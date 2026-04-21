import streamlit as st
import pickle
import numpy as np

# الكود ده متعدل بالأسماء اللي في صورتك الأولى بالظبط
try:
    # بنجرب نفتح الملف بالاسم اللي فيه (1)
    model = pickle.load(open('diabetes_model (1).pkl', 'rb'))
    scaler = pickle.load(open('final_scaler (1).pkl', 'rb'))
except Exception as e:
    st.error(f"Error: {e}")
    st.write("تأكدي إن الأسماء دي موجودة في GitHub: 'diabetes_model (1).pkl' و 'final_scaler (1).pkl'")

st.title('Diabetes Prediction System')

pregnancies = st.number_input('Pregnancies', value=0)
glucose = st.number_input('Glucose', value=0)
blood_pressure = st.number_input('Blood Pressure', value=0)
skin_thickness = st.number_input('Skin Thickness', value=0)
insulin = st.number_input('Insulin', value=0)
bmi = st.number_input('BMI', value=0.0)
dpf = st.number_input('Diabetes Pedigree Function', value=0.0)
age = st.number_input('Age', value=0)

if st.button('Predict'):
    input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    
    if prediction[0] == 1:
        st.error('مصاب بالسكري')
    else:
        st.success('سليم')
