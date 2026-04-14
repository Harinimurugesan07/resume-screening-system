import pandas as pd
df = pd.read_csv(r'C:\Users\HARINI\OneDrive\Documents\Downloads\archive\UpdatedResumeDataSet.CSV')
print("Sample Data:")
print(df.head())

import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def clean_resume(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

#  This line actually creates the 'cleaned_resume' column
df['cleaned_resume'] = df['Resume'].apply(clean_resume)

print(df[['Resume', 'cleaned_resume']].head())


from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=500)

# Fit and transform the cleaned resumes
X = tfidf.fit_transform(df['cleaned_resume']).toarray()

# Labels (what we want to predict)
y = df['Category']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Initialize the model
model = MultinomialNB()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Sample resume text to test
new_resume = """
I am a skilled Python developer with experience in machine learning, data analysis, and building RESTful APIs using Django.
"""

# 1. Clean the new resume
cleaned_new = clean_resume(new_resume)

# 2. Convert to TF-IDF vector using the same vectorizer
new_vec = tfidf.transform([cleaned_new]).toarray()

# 3. Predict the category
predicted_category = model.predict(new_vec)

# 3. Predict probabilities
probs = model.predict_proba(new_vec)


# 4. Show result
print("Predicted Category:", predicted_category[0])







import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('UpdatedResumeDataSet.csv')  # Make sure this CSV is in the same folder

# Clean text
def clean_resume(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub('[^A-Za-z ]+', ' ', text)
    return text.lower()

df['cleaned'] = df['Resume'].apply(clean_resume)

# Vectorize
tfidf = TfidfVectorizer(max_features=500)
X = tfidf.fit_transform(df['cleaned']).toarray()
y = df['Category']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Save the model and vectorizer
joblib.dump(clf, 'resume_classifier_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print("Model and vectorizer saved successfully!")
