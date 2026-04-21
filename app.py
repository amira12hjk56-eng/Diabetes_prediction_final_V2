import streamlit as st
import joblib

# 1. تحميل الموديل
model = joblib.load('diabetes_model.pkl')

# 2. قاموس الترجمة الشامل
result_map = {
    'Normal': 'طبيعي (سليم) ✅',
    'Prediabetics': 'مرحلة ما قبل السكري ⚠️',
    'Prediabetes': 'مرحلة ما قبل السكري ⚠️',
    'Diabetics': 'مصاب بالسكري 🩺',
    'Diabetic': 'مصاب بالسكري 🩺'
}

st.set_page_config(page_title="توقع السكري", layout="wide")
st.title("نظام التنبؤ بمخاطر السكري 🩺")
st.markdown("---")

# 3. تنظيم الـ 17 خانة في 3 أعمدة
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("العمر", min_value=1, value=25)
    gender = st.selectbox("النوع", options=['Male', 'Female'], format_func=lambda x: 'ذكر' if x == 'Male' else 'أنثى')
    bmi = st.number_input("مؤشر كتلة الجسم (BMI)", value=25.0)
    blood_pressure = st.number_input("ضغط الدم", value=120)
    glucose = st.number_input("مستوى الجلوكوز", value=100)
    insulin = st.number_input("مستوى الإنسولين", value=15)

with col2:
    hba1c = st.number_input("مستوى HbA1c", value=5.5)
    chol = st.number_input("الكوليسترول", value=180)
    trig = st.number_input("الدهون الثلاثية", value=130)
    activity = st.selectbox("مستوى النشاط البدني (0-3)", options=[0, 1, 2, 3])
    calories = st.number_input("السعرات اليومية", value=2000)
    sugar = st.number_input("كمية السكر اليومية", value=30)

with col3:
    sleep = st.number_input("ساعات النوم", value=7)
    stress = st.number_input("مستوى التوتر (1-10)", value=5)
    family = st.selectbox("تاريخ العائلة مع السكري", options=['Yes', 'No'], format_func=lambda x: 'نعم' if x == 'Yes' else 'لا')
    waist = st.number_input("محيط الخصر (سم)", value=85)
    score = st.number_input("درجة خطر سابقة", value=50)

st.markdown("---")

if st.button("توقع النتيجة الآن"):
    g_enc = 1 if gender == 'Male' else 0
    f_enc = 1 if family == 'Yes' else 0
    
    # إرسال الـ 17 قيمة للموديل
    data = [[age, g_enc, bmi, blood_pressure, glucose, insulin, hba1c, chol, trig, 
             activity, calories, sugar, sleep, stress, f_enc, waist, score]]
    
    pred = model.predict(data)[0]
    res = result_map.get(pred, pred)
    
    st.info("نتيجة التحليل:")
    st.subheader(f"التوقع هو: {res}")