import streamlit as st
import joblib
import PyPDF2
import docx2txt
from sklearn.metrics.pairwise import cosine_similarity
import pytesseract
from PIL import Image
import pandas as pd
import re
import os

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="AI ATS Resume Screening System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI ATS Resume Screening Dashboard")
st.markdown("Smart candidate ranking system for recruiters")

st.divider()

# ---------------- LOAD MODEL ---------------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "resume_classifier_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))

# ---------------- TESSERACT ---------------- #

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------- SKILL LIST ---------------- #

skills_list = [
    "python","machine learning","data science","deep learning","sql",
    "pandas","numpy","tensorflow","keras","java","c++","javascript",
    "react","node","html","css","power bi","tableau","aws","docker","kubernetes"
]

# ---------------- CLEAN FUNCTION ---------------- #

def clean_resume(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub('[^A-Za-z ]+', ' ', text)
    return text.lower()

# ---------------- SKILL EXTRACTION ---------------- #

def extract_skills(text):
    text = text.lower()
    return [skill for skill in skills_list if skill in text]

# ---------------- TEXT EXTRACTION ---------------- #

def extract_text(file):

    file_type = file.name.split(".")[-1].lower()
    text = ""

    if file_type == "pdf":
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()

    elif file_type == "docx":
        text = docx2txt.process(file)

    elif file_type == "txt":
        text = file.read().decode("utf-8")

    elif file_type in ["jpg","jpeg","png"]:
        image = Image.open(file).convert("L")
        text = pytesseract.image_to_string(image)

    return text

# ---------------- SIDEBAR ---------------- #

st.sidebar.header("⚙ Recruiter Settings")

must_have = st.sidebar.text_input("Must-Have Skills (comma separated)")
nice_to_have = st.sidebar.text_input("Nice-to-Have Skills")
min_exp = st.sidebar.slider("Minimum Experience (Years)", 0, 10)
weight = st.sidebar.slider("Skills Weight (%)", 0, 100, 70)

bias_mode = st.sidebar.checkbox("🛡 Enable Bias Shield Mode")

# Convert skills
must_have_list = [s.strip().lower() for s in must_have.split(",") if s]
nice_to_have_list = [s.strip().lower() for s in nice_to_have.split(",") if s]

# ---------------- FILE UPLOAD ---------------- #

uploaded_files = st.file_uploader(
    "📂 Upload Multiple Resumes",
    type=["pdf","docx","txt","jpg","jpeg","png"],
    accept_multiple_files=True
)

job_description = st.text_area("📝 Paste Job Description")

# ---------------- PROCESS ---------------- #

if uploaded_files:

    results = []

    for file in uploaded_files:

        resume_text = extract_text(file)

        if not resume_text:
            continue

        cleaned = clean_resume(resume_text)
        skills = extract_skills(cleaned)

        # ---------------- HARD FILTER ---------------- #
        if must_have_list:
            if not all(skill in cleaned for skill in must_have_list):
                continue

        # ---------------- MATCH SCORE ---------------- #
        resume_vec = vectorizer.transform([cleaned])

        if job_description:
            jd_vec = vectorizer.transform([job_description])
            similarity = cosine_similarity(resume_vec, jd_vec)[0][0]
        else:
            similarity = 0

        # ---------------- EXPERIENCE (simple logic) ---------------- #
        exp_score = 1 if min_exp == 0 else 0.5

        # ---------------- FINAL SCORE ---------------- #
        final_score = (similarity * (weight/100)) + (exp_score * (1 - weight/100))
        final_score = round(final_score * 100, 2)

        # ---------------- PREDICTION ---------------- #
        prediction = model.predict(resume_vec)[0]

        # ---------------- SKILL GAP ---------------- #
        missing_skills = [s for s in must_have_list if s not in skills]
        strength_skills = [s for s in skills if s in must_have_list or s in nice_to_have_list]
        # ---------------- BIAS MODE ---------------- #
        display_name = file.name
        if bias_mode:
            display_name = f"Candidate_{len(results)+1}"

        results.append({
            "Candidate": display_name,
            "Score": final_score,
            "Category": prediction,
            "Skills": ", ".join(skills),
            "Strengths": ", ".join(strength_skills),
            "Missing Skills": ", ".join(missing_skills)
        })

    # ---------------- DISPLAY RESULTS ---------------- #

    if results:

        df = pd.DataFrame(results)
        df = df.sort_values(by="Score", ascending=False)

        st.subheader("🏆 Candidate Ranking")
        st.dataframe(df, use_container_width=True)

        # Highlight Top Candidate
        top = df.iloc[0]

        st.subheader("🥇 Top Candidate")
        st.success(f"{top['Candidate']} - {top['Score']}% Match")

        st.metric("Score", f"{top['Score']}%")

        st.progress(int(top['Score']))

    else:
        st.warning("No candidates matched the criteria")

else:
    st.info("Upload resumes to begin analysis")