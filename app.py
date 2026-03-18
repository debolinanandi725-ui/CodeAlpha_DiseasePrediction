import streamlit as st
import pickle
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from datetime import datetime

# ================= UI =================
st.set_page_config(page_title="AI Medical Report", layout="centered")

st.title("🩺 Disease Prediction System")
st.write("""
This app predicts the likelihood of diseases using Machine Learning models 
(Logistic Regression, SVM, Random Forest, XGBoost).
""")

# ================= LOAD MODELS =================
@st.cache_resource
def load_file(file):
    return pickle.load(open(file, "rb"))

heart_model = load_file("heart_model.pkl")
heart_scaler = load_file("heart_scaler.pkl")
diabetes_model = load_file("diabetes_model.pkl")
diabetes_scaler = load_file("diabetes_scaler.pkl")

# ================= PDF =================
def generate_report(name, disease, result, data):
    safe_name = name.replace(" ", "_")
    file_name = f"{safe_name}_{disease.replace(' ', '_')}_report.pdf"

    doc = SimpleDocTemplate(file_name)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("🏥 My Health Clinic", styles['Title']))
    content.append(Spacer(1, 10))
    content.append(Paragraph("AI-Based Medical Prediction Report", styles['Heading2']))
    content.append(Spacer(1, 20))

    content.append(Paragraph(f"Patient Name: {name}", styles['Normal']))
    content.append(Paragraph(f"Disease: {disease}", styles['Normal']))
    content.append(Paragraph(f"Result: {result}", styles['Normal']))
    content.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    content.append(Spacer(1, 15))

    table_data = [["Parameter", "Value"]]
    for k, v in data.items():
        table_data.append([k, str(v)])

    table = Table(table_data, colWidths=[150, 150])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightblue),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))

    content.append(table)
    content.append(Spacer(1, 20))

    if disease == "Heart Disease":
        advice = "⚠️ Maintain a healthy lifestyle and consult a cardiologist."
    else:
        advice = "⚠️ Monitor sugar levels and consult a specialist."

    content.append(Paragraph(advice, styles['Italic']))
    content.append(Spacer(1, 10))
    content.append(Paragraph("⚠️ AI-generated report. Consult a doctor.", styles['Italic']))

    doc.build(content)
    return file_name

# ================= SELECT =================
option = st.selectbox("Select Disease", ["Heart Disease", "Diabetes"])

# ================= HEART =================
if option == "Heart Disease":
    st.header("❤️ Heart Disease Prediction")

    st.write(f"🤖 Model Used: {type(heart_model).__name__}")

    st.write("📝 Enter name to fetch report")

    name = st.text_input("Patient Name")

    age = st.number_input("Age", 1, 120)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.number_input("Chest Pain (0-3)")
    trestbps = st.number_input("Blood Pressure")
    chol = st.number_input("Cholesterol")
    fbs = st.selectbox("Fasting Sugar", [0, 1])
    restecg = st.number_input("Rest ECG")
    thalach = st.number_input("Max Heart Rate")
    exang = st.selectbox("Exercise Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak")
    slope = st.number_input("Slope")
    ca = st.number_input("CA")
    thal = st.number_input("Thal")

    if st.button("Predict Heart"):
        if chol <= 0 or trestbps <= 0:
            st.error("⚠️ Invalid input values")
        else:
            with st.spinner("Analyzing..."):
                data = np.array([[age, 1 if sex=="Male" else 0, cp, trestbps, chol, fbs,
                                  restecg, thalach, exang, oldpeak, slope, ca, thal]])

                data_scaled = heart_scaler.transform(data)
                pred = heart_model.predict(data_scaled)

                result = "High Risk" if pred[0]==1 else "Low Risk"

            if pred[0] == 1:
                st.error(f"⚠️ {result}")
            else:
                st.success(f"✅ {result}")

            st.caption("Prediction based on AI model")

           

            details = {
                "Age": age, "Sex": sex, "Chest Pain": cp,
                "Blood Pressure": trestbps, "Cholesterol": chol,
                "Fasting Sugar": fbs, "Rest ECG": restecg,
                "Max Heart Rate": thalach, "Exercise Angina": exang,
                "Oldpeak": oldpeak, "Slope": slope,
                "CA": ca, "Thal": thal
            }

            if name:
                file = generate_report(name, "Heart Disease", result, details)
                with open(file, "rb") as f:
                    st.download_button("📥 Download Report", f, file_name=file)

# ================= DIABETES =================
if option == "Diabetes":
    st.header("🩸 Diabetes Prediction")

    st.write(f"🤖 Model Used: {type(diabetes_model).__name__}")

    st.write("📝 Enter name to fetch report")

    name = st.text_input("Patient Name")

    Pregnancies = st.number_input("Pregnancies", 0)
    Glucose = st.number_input("Glucose")
    BP = st.number_input("Blood Pressure")
    Skin = st.number_input("Skin Thickness")
    Insulin = st.number_input("Insulin")
    BMI = st.number_input("BMI")
    DPF = st.number_input("DPF")
    Age = st.number_input("Age", 1, 120)

    if st.button("Predict Diabetes"):
        if Glucose <= 0 or BMI <= 0:
            st.error("⚠️ Invalid input values")
        else:
            with st.spinner("Analyzing..."):
                data = np.array([[Pregnancies, Glucose, BP, Skin, Insulin, BMI, DPF, Age]])

                data_scaled = diabetes_scaler.transform(data)
                pred = diabetes_model.predict(data_scaled)

                result = "Diabetic" if pred[0]==1 else "Not Diabetic"

            if pred[0] == 1:
                st.error(f"⚠️ {result}")
            else:
                st.success(f"✅ {result}")

            st.caption("Prediction based on AI model")

            

            details = {
                "Pregnancies": Pregnancies, "Glucose": Glucose,
                "Blood Pressure": BP, "Skin Thickness": Skin,
                "Insulin": Insulin, "BMI": BMI,
                "DPF": DPF, "Age": Age
            }

            if name:
                file = generate_report(name, "Diabetes", result, details)
                with open(file, "rb") as f:
                    st.download_button("📥 Download Report", f, file_name=file)