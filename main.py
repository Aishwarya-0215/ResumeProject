import os
import PyPDF2
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords (only first time)
nltk.download('stopwords')

# Function to extract text from PDF
def extract_text(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to clean text
def preprocess(text):
    words = text.lower().split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

# ✅ UPDATED JOB DESCRIPTION (Based on your resume)
job_description = """
Looking for a Computer Science Engineering student with strong knowledge in Python, Java, C, and SQL.
Candidate should have experience in Machine Learning, Natural Language Processing (NLP), and Generative AI.

Hands-on project experience in AI-based applications such as chatbot development, resume screening systems,
and AR/VR applications is preferred.

Knowledge in data analysis, problem-solving, and software development is required.
Familiarity with HTML, CSS, and basic web technologies is an added advantage.

Internship experience in Artificial Intelligence, Data Science, and Generative AI is highly valued.
"""

# Clean job description
job_clean = preprocess(job_description)

# Folder path
folder = "resumes"

print("\n📊 Resume Screening Results:\n")

best_score = 0
best_resume = ""

# Loop through resumes
for file in os.listdir(folder):
    path = os.path.join(folder, file)

    try:
        # Extract & clean resume
        resume_text = extract_text(path)
        resume_clean = preprocess(resume_text)

        # Convert to vectors
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([resume_clean, job_clean])

        # Calculate similarity
        score = cosine_similarity(vectors[0], vectors[1])[0][0] * 100

        print(file, "->", round(score, 2), "%")

        # Find best candidate
        if score > best_score:
            best_score = score
            best_resume = file

    except:
        print(file, "-> Error reading file")

# Final result
print("\n🏆 Top Candidate:", best_resume)
print("🔥 Highest Match Score:", round(best_score, 2), "%")