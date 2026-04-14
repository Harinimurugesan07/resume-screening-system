from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re

# ---------------- FASTAPI APP ---------------- #

app = FastAPI(
    title="AI Resume Screening API",
    description="API for predicting resume job category",
    version="1.0"
)

# ---------------- LOAD MODEL ---------------- #

model = joblib.load("resume_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ---------------- REQUEST MODEL ---------------- #

class ResumeRequest(BaseModel):
    resume: str

# ---------------- CLEAN RESUME ---------------- #

def clean_resume(text):

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"#\S+", "", text)
    text = re.sub(r"[^A-Za-z ]", " ", text)
    text = text.lower()

    return text

# ---------------- HOME ROUTE ---------------- #

@app.get("/resume")
def home():

    return {"message": "AI Resume Screening API Running"}

# ---------------- PREDICT ROUTE ---------------- #

@app.post("/predict")
def predict(data: ResumeRequest):

    resume = data.resume

    cleaned = clean_resume(resume)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)

    return {
        "prediction": prediction[0]
    }