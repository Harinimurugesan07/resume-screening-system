import streamlit as st
import joblib
import PyPDF2
import docx2txt
from sklearn.metrics.pairwise import cosine_similarity
import pytesseract
from PIL import Image
import requests


# ---------------- API FUNCTION ---------------- #

def analyze_resume_api(resume_text):

    url = "http://127.0.0.1:8000/predict"

    try:
        response = requests.post(url, json={"resume": resume_text})
        return response.json()

    except Exception as e:
        return {"error": str(e)}


# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Screening Dashboard")
st.markdown("Upload a resume and analyze candidate profile using AI.")

st.divider()


# ---------------- TESSERACT PATH ---------------- #

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ---------------- SKILL LIST ---------------- #

skills_list = [
    "python","machine learning","data science","deep learning","sql",
    "pandas","numpy","tensorflow","keras","java","c++","javascript",
    "react","node","html","css","power bi","tableau","aws","docker","kubernetes"
]


# ---------------- LOAD MODEL ---------------- #

model = joblib.load("resume_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


# ---------------- SKILL EXTRACTION ---------------- #

def extract_skills(text):

    found_skills = []
    text = text.lower()

    for skill in skills_list:
        if skill in text:
            found_skills.append(skill)

    return found_skills


# ---------------- RESUME TEXT EXTRACTION ---------------- #

def extract_text_from_file(uploaded_file):

    file_type = uploaded_file.name.split(".")[-1].lower()
    text = ""

    try:

        if file_type == "pdf":

            reader = PyPDF2.PdfReader(uploaded_file)

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text


        elif file_type == "docx":

            text = docx2txt.process(uploaded_file)


        elif file_type == "txt":

            text = uploaded_file.read().decode("utf-8")


        elif file_type in ["jpg","jpeg","png"]:

            image = Image.open(uploaded_file)
            image = image.convert("L")

            text = pytesseract.image_to_string(image)


        return text.strip()

    except Exception as e:

        st.error(f"Error extracting text: {e}")
        return ""


# ---------------- SIDEBAR ---------------- #

st.sidebar.title("📊 Resume Screening System")

st.sidebar.info(
"""
AI powered system that helps HR teams:

✔ Screen resumes  
✔ Detect candidate skills  
✔ Predict job category  
✔ Calculate resume-job match score
"""
)

st.sidebar.markdown("Developed using **Machine Learning + NLP**")


# ---------------- LAYOUT ---------------- #

col1, col2 = st.columns(2)

with col1:

    uploaded_file = st.file_uploader(
        "📂 Upload Resume",
        type=["pdf","docx","txt","jpg","jpeg","png"]
    )

with col2:

    job_description = st.text_area(
        "📝 Paste Job Description",
        height=200
    )


# ---------------- PROCESS RESUME ---------------- #

if uploaded_file is not None:

    st.divider()

    st.subheader("📁 Uploaded Resume")
    st.write(uploaded_file.name)

    resume_text = extract_text_from_file(uploaded_file)

    if resume_text:

        # ---------------- SKILLS ---------------- #

        skills = extract_skills(resume_text)

        st.subheader("🧾 Detected Skills")

        if skills:

            skill_cols = st.columns(4)

            for i, skill in enumerate(skills):
                skill_cols[i % 4].success(skill)

        else:

            st.warning("No skills detected")


       

        # ---------------- JOB CATEGORY ---------------- #

        if st.button("🔍 Analyze Resume"):
            st.subheader("🤖 Predicted Job Category")
            result = analyze_resume_api(resume_text)
            if "prediction" in result:
                st.success(result["prediction"])
            else:
                st.error("API Error")
                st.write(result)

        # ---------------- MATCH SCORE ---------------- #

        if job_description:

            jd_vector = vectorizer.transform([job_description])
            resume_vector = vectorizer.transform([resume_text])

            similarity = cosine_similarity(resume_vector, jd_vector)

            match_score = round(similarity[0][0] * 100, 2)

            st.subheader("📊 ATS Resume Match Score")

            st.metric("Match Score", f"{match_score}%")

            st.progress(int(match_score))


            if match_score > 70:

                st.success("✅ Strong Match - Candidate Highly Suitable")

            elif match_score > 40:

                st.warning("⚠ Moderate Match - Candidate May Fit")

            else:

                st.error("❌ Low Match - Candidate Not Suitable")