import streamlit as st
import joblib
import numpy as np

# تحميل الموديل والـ Scaler
# تأكدي من مطابقة الأسماء للملفات المرفوعة على GitHub
model = joblib.load('final_diabetes_model.pkl')
scaler = joblib.load('final_scaler (1).pkl')

# إعدادات الصفحة
st.set_page_config(page_title="نظام التنبؤ بمخاطر السكري", layout="wide")

# إضافة تنسيق CSS لتغيير الألوان والخلفية
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        background-color: #007bff;
        color: white;
        height: 3em;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_safe=True)

st.title("🩺 نظام التنبؤ الذكي بمخاطر السكري")
st.write("أدخلي بيانات المريض بدقة للحصول على تقييم المخاطر:")

# تقسيم المدخلات لعمودين عشان الشكل يبقى أرتب
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("العمر", min_value=1, max_value=100, value=25)
    gender = st.selectbox("النوع", ["ذكر", "أنثى"])
    bmi = st.number_input("(BMI) مؤشر كتلة الجسم", value=25.0)
    blood_pressure = st.number_input("ضغط الدم", value=120)
    glucose = st.number_input("مستوى الجلوكوز", value=100)

with col2:
    hba1c = st.number_input("HbA1c مستوى", value=5.5)
    cholesterol = st.number_input("الكوليسترول", value=180)
    triglycerides = st.number_input("الدهون الثلاثية", value=130)
    activity = st.slider("مستوى النشاط البدني (0-3)", 0, 3, 1)
    calories = st.number_input("السعرات اليومية", value=2000)

st.divider()

# زر التوقع
if st.button("تحليل الحالة الآن"):
    # تحويل النوع لرقم (مثلاً ذكر=1، أنثى=0) حسب تدريب الموديل عندك
    gender_numeric = 1 if gender == "ذكر" else 0
    
    # تجميع البيانات (الترتيب لازم يطابق ترتيب الأعمدة في ملف الإكسيل الأصلي)
    input_data = np.array([[age, gender_numeric, bmi, blood_pressure, glucose, hba1c, cholesterol, triglycerides, activity, calories]])
    
    # عمل Scaling
    input_scaled = scaler.transform(input_data)
    
    # التوقع
    prediction = model.predict(input_scaled)
    
    # عرض النتيجة بشكل جمالي
    st.subheader("النتيجة التحليلية:")
    if "High" in str(prediction[0]):
        st.error(f"⚠️ تنبيه: مستوى المخاطر هو ({prediction[0]}) - يرجى مراجعة الطبيب.")
    else:
        st.success(f"✅ مطمئن: مستوى المخاطر هو ({prediction[0]}) - استمر في نمط حياة صحي.")