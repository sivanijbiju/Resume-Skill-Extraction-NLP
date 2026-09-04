from flask import Flask, render_template, request
import re
import nltk
from pypdf import PdfReader
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOPWORDS = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#.\-/ ]", " ", text)
    return " ".join(w for w in text.split() if w not in STOPWORDS)

def load_skills():
    with open("skills_vocabulary.txt", encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip() and not x.startswith("#")]

def extract_skills(text, vocabulary):
    clean = preprocess_text(text)
    found = []
    for skill in vocabulary:
        s = preprocess_text(skill)
        if s and re.search(r"(?<!\w)" + re.escape(s) + r"(?!\w)", clean):
            found.append(skill)
    return list(dict.fromkeys(found))

def pdf_text(file):
    reader = PdfReader(file)
    return "\n".join((p.extract_text() or "") for p in reader.pages)

def similarity(resume, job):
    a, b = preprocess_text(resume), preprocess_text(job)
    if not a or not b:
        return 0.0
    v = TfidfVectorizer()
    m = v.fit_transform([a, b])
    return round(float(cosine_similarity(m[0:1], m[1:2])[0][0] * 100), 2)

@app.route("/", methods=["GET", "POST"])
def index():
    result, error = None, None
    if request.method == "POST":
        f = request.files.get("resume")
        title = request.form.get("job_title", "").strip()
        job = request.form.get("job_description", "").strip()

        if not f or not f.filename:
            error = "Please upload a resume PDF."
        elif not f.filename.lower().endswith(".pdf"):
            error = "Please upload a PDF file only."
        elif not title or not job:
            error = "Please enter both the job title and job description."
        else:
            try:
                resume = pdf_text(f)
                vocab = load_skills()
                resume_skills = extract_skills(resume, vocab)
                job_skills = extract_skills(job, vocab)

                rset = {preprocess_text(x) for x in resume_skills}
                matched = [x for x in job_skills if preprocess_text(x) in rset]
                missing = [x for x in job_skills if preprocess_text(x) not in rset]
                coverage = round(100 * len(matched) / len(job_skills), 2) if job_skills else 0

                result = {
                    "title": title, "filename": f.filename,
                    "score": similarity(resume, job), "coverage": coverage,
                    "resume_skills": resume_skills, "matched": matched,
                    "missing": missing
                }
            except Exception as e:
                error = f"Could not analyze the resume: {e}"

    return render_template("index.html", result=result, error=error)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
