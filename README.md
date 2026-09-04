# Resume Skill Extraction & Job Matching – Web Interface

This is the web interface for the Resume Skill Extraction & Job Matching System.

## Features

- Upload a resume in PDF format
- Enter a job title and job description
- Extract technical skills from the resume
- Calculate resume-job similarity using TF-IDF and cosine similarity
- Display matched and missing skills
- Calculate skill coverage
- Display the extracted resume skills

## Technologies Used

- Python
- Flask
- NLTK
- Scikit-learn
- PyPDF
- HTML
- CSS

## How It Works

1. Upload a PDF resume.
2. Enter the job title.
3. Enter the job description.
4. Click **Analyze Resume**.
5. The system extracts skills from the resume.
6. TF-IDF and cosine similarity are used to calculate the match score.
7. The system displays matched skills, missing skills, skill coverage, and extracted skills.

## Deployment

The web interface is deployed using Render.

## Project Structure

```text
Web_Interface/
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── app.py
├── requirements.txt
├── Procfile
└── skills_vocabulary.txt
