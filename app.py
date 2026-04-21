import streamlit as st
import pickle
import numpy as np
import os

# الدالة دي بتدور على أي ملف موديل أو سكيلر موجود في الفولدر عندك
def load_res(patterns):
    for file in os.listdir('.'):
        for p in patterns:
            if p in file and file.endswith('.pkl'):
                return pickle.load(open(file, 'rb'))
    return None

st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title('🏥 Diabetes Prediction System')

# الكود هيدور بنفسه على الملفات اللي آخرها pkl
model = load_res(['model', 'diabetes'])
scaler = load_res(['scaler', 'final_scaler'])

if model is None or scaler is None:
    st.error("❌ مشكلة في الملفات: تأكدي إنك رفعتي ملف الموديل والـ scaler بصيغة .pkl")
else:
    st.info("✅ تم تحميل الموديل والـ Scaler بنجاح")
    
    # تقسيم المدخلات بشكل منظم
    col1, col2 = st.columns(2)
    with col1:
        preg = st.number_input('Pregnancies', min_value=0)
        glu = st.number_input('Glucose', min_value=0)
        bp = st.number_input('Blood Pressure', min_value=0)
        skin = st.number_input('Skin Thickness', min_value=0)
    with col2:
        ins = st.number_input('Insulin', min_value=0)
        bmi = st.number_input('BMI', min_value=0.0)
        dpf = st.number_input('DPF', min_value=0.0)
        age = st.number_input('Age', min_value=0)

    if st.button('Predict Now / ابدأ الفحص'):
        # تحويل المدخلات لمصفوفة وعمل Reshape عشان نمنع الـ ValueError
        features = np.array([preg, glu, bp, skin, ins, bmi, dpf, age]).reshape(1, -1)
        
        # تنفيذ التحجيم والتوقع
        try:
            scaled_data = scaler.transform(features)
            prediction = model.predict(scaled_data)
            
            st.markdown("---")
            if prediction[0] == 1:
                st.error('⚠️ النتيجة: الشخص قد يكون مصاباً بالسكري')
            else:
                st.success('✨ النتيجة: الشخص سليم والحمد لله')
        except Exception as e:
            st.error(f"حصلت مشكلة أثناء الحساب: {e}")
