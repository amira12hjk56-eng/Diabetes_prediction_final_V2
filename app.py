import streamlit as st
import joblib
import numpy as np
import pandas as pd

# تحميل الموديل والـ Scaler والـ Encoders
# تأكدي من مطابقة الأسماء للملفات المرفوعة على GitHub
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('final_diabetes_model.pkl')
        scaler = joblib.load('final_scaler (1).pkl')
        # سنحتاج لتحميل الـ categorical encoder إذا تم استخدامه أثناء التدريب
        # categorical_encoder = joblib.load('categorical_encoder.pkl') 
        return model, scaler #, categorical_encoder
    except Exception as e:
        st.error(f"خطأ في تحميل ملفات الموديل. تأكدي من أسماء الملفات على GitHub. الخطأ: {e}")
        return None, None #, None

model, scaler = load_assets()

# إعدادات الصفحة
st.set_page_config(page_title="نظام التنبؤ بمخاطر السكري - نورهان", layout="wide")

# إضافة تنسيق CSS لتغيير الألوان والخلفية
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
        direction: rtl; /* اتجاه الكتابة عربي */
        text-align: right;
    }
    .stButton>button {
        width: 100%;
        background-color: #2563eb;
        color: white;
        height: 3.5em;
        border-radius: 12px;
        font-weight: bold;
        font-size: 1.1em;
    }
    h1 {
        color: #1e3a8a;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# الجزء الجانبي (Sidebar) - إضافة اسم نورهان
with st.sidebar:
    st.markdown("## 💻 تطوير المشروع")
    st.markdown("### نورهان عماد الدين محمد")
    st.divider()
    st.info("هذا النظام يستخدم الذكاء الاصطناعي للتنبؤ بمخاطر السكري بناءً على 17 متغير طبي.")

# العنوان الرئيسي
st.title("🩺 نظام التنبؤ الذكي بمخاطر السكري")
st.write("أدخلي بيانات المريض بدقة (جميع الـ 17 متغير) للحصول على تقييم المخاطر:")

st.divider()

# تقسيم المدخلات لـ 3 أعمدة (17 خانة)
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("العمر (سنة)", min_value=1, max_value=120, value=30)
    gender = st.selectbox("النوع", ["ذكر", "أنثى"])
    polyuria = st.selectbox("تعدد البول (Excessive urination)", ["نعم", "لا"])
    polydipsia = st.selectbox("العطش الشديد (Excessive thirst)", ["نعم", "لا"])
    sudden_weight_loss = st.selectbox("فقدان الوزن المفاجئ", ["نعم", "لا"])
    weakness = st.selectbox("الضعف العام", ["نعم", "لا"])

with col2:
    polyphagia = st.selectbox("تعدد الأكل (Excessive hunger)", ["نعم", "لا"])
    genital_thrush = st.selectbox("عدوى المهبل/القلاع", ["نعم", "لا"])
    visual_blurring = st.selectbox("تشوش الرؤية", ["نعم", "لا"])
    itching = st.selectbox("الحكة", ["نعم", "لا"])
    irritability = st.selectbox("التهيج/العصبية", ["نعم", "لا"])
    delayed_healing = st.selectbox("تأخر التئام الجروح", ["نعم", "لا"])

with col3:
    partial_paresis = st.selectbox("الشلل الجزئي", ["نعم", "لا"])
    muscle_stiffness = st.selectbox("تصلب العضلات", ["نعم", "لا"])
    alopecia = st.selectbox("الثعلبة/فقدان الشعر", ["نعم", "لا"])
    obesity = st.selectbox("السمنة", ["نعم", "لا"])
    # لإضافة خانة سكر الجلوكوز العشوائي إذا كانت موجودة في الداتا (مثال)
    # random_glucose = st.number_input("مستوى الجلوكوز العشوائي (mg/dL)", value=100) 

st.divider()

# زر التوقع
if st.button("تحليل الحالة الآن"):
    if model is None or scaler is None:
        st.error("الموديل غير جاهز. يرجى التحقق من الملفات المرفوعة.")
    else:
        # تحويل البيانات النصية لأرقام (نعم=1، لا=0، ذكر=1، أنثى=0) 
        # هذا مثال، يجب مطابقة طريقة الـ encoding المستخدمة أثناء تدريب الموديل.
        gender_num = 1 if gender == "ذكر" else 0
        polyuria_num = 1 if polyuria == "نعم" else 0
        polydipsia_num = 1 if polydipsia == "نعم" else 0
        weight_loss_num = 1 if sudden_weight_loss == "نعم" else 0
        weakness_num = 1 if weakness == "نعم" else 0
        polyphagia_num = 1 if polyphagia == "نعم" else 0
        genital_thrush_num = 1 if genital_thrush == "نعم" else 0
        visual_blurring_num = 1 if visual_blurring == "نعم" else 0
        itching_num = 1 if itching == "نعم" else 0
        irritability_num = 1 if irritability == "نعم" else 0
        delayed_healing_num = 1 if delayed_healing == "نعم" else 0
        partial_paresis_num = 1 if partial_paresis == "نعم" else 0
        muscle_stiffness_num = 1 if muscle_stiffness == "نعم" else 0
        alopecia_num = 1 if alopecia == "نعم" else 0
        obesity_num = 1 if obesity == "نعم" else 0

        # تجميع البيانات الـ 17 في مصفوفة (الترتيب مهم جداً!)
        input_features = np.array([[
            age, gender_num, polyuria_num, polydipsia_num, weight_loss_num, 
            weakness_num, polyphagia_num, genital_thrush_num, visual_blurring_num,
            itching_num, irritability_num, delayed_healing_num, partial_paresis_num,
            muscle_stiffness_num, alopecia_num, obesity_num #, random_glucose
        ]])

        # عمل Scaling
        input_scaled = scaler.transform(input_features)

        # التوقع
        prediction = model.predict(input_scaled)

        # عرض النتيجة بشكل جمالي
        st.subheader("النتيجة التحليلية:")
        # إذا كانت النتيجة رقمية (0/1)، نحولها لنص
        result_text = "مرتفعة" if "High" in str(prediction[0]) or str(prediction[0]) == "1" else "منخفضة"
        
        if result_text == "مرتفعة":
            st.error(f"⚠️ تنبيه: مستوى مخاطر السكري ({result_text}) - يرجى مراجعة الطبيب للفحص.")
        else:
            st.success(f"✅ مطمئن: مستوى مخاطر السكري ({result_text}) - استمر في نمط حياة صحي.")