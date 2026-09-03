"""
NLP-Based Resume Skill Extraction and Job Description Matching System
Using TF-IDF and Cosine Similarity

Author: [Your Name]
Course Outcomes: CO3, CO4

This program:
1. Loads resumes and job descriptions from CSV files
2. Preprocesses the text
3. Extracts skills using a predefined skill vocabulary
4. Converts resume and job-description text into TF-IDF vectors
5. Calculates cosine similarity
6. Ranks resumes according to their match score
7. Displays matched and missing skills
"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Download NLTK stopwords
nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))


# ---------------------------------------------------------------------------
# STEP 1: Load data
# ---------------------------------------------------------------------------
def load_data(resumes_path, jobs_path):
    """Load resumes and job descriptions from CSV files."""
    resumes_df = pd.read_csv(resumes_path)
    jobs_df = pd.read_csv(jobs_path)

    return resumes_df, jobs_df


# ---------------------------------------------------------------------------
# STEP 2: Preprocess text
# ---------------------------------------------------------------------------
def preprocess_text(text):
    """
    Clean text for NLP processing:
    - Convert to lowercase
    - Remove punctuation and numbers
    - Remove English stopwords
    """

    text = str(text).lower()

    # Keep only alphabets and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Split into individual words
    words = text.split()

    # Remove stopwords
    words = [word for word in words if word not in STOP_WORDS]

    return " ".join(words)


# ---------------------------------------------------------------------------
# STEP 3: Load skill vocabulary
# ---------------------------------------------------------------------------
def load_skills_vocabulary(path):
    """Load known skills from the vocabulary file."""

    with open(path, "r", encoding="utf-8") as file:
        content = file.read().replace("\n", " ")

    skills = [skill.strip().lower()
              for skill in content.split(",")
              if skill.strip()]

    return skills


# ---------------------------------------------------------------------------
# STEP 4: Extract skills
# ---------------------------------------------------------------------------
def extract_skills(raw_text, skills_list):
    """
    Extract known skills from the given text.

    Word-boundary matching is used so that a skill is matched
    as a complete phrase rather than as part of another word.
    """

    text = str(raw_text).lower()

    found_skills = []

    for skill in skills_list:

        # Escape special characters such as +, ., /, etc.
        escaped_skill = re.escape(skill)

        # Match the complete skill/phrase
        pattern = r"(?<![a-z])" + escaped_skill + r"(?![a-z])"

        if re.search(pattern, text):
            found_skills.append(skill)

    return found_skills


# ---------------------------------------------------------------------------
# STEP 5: Calculate matched and missing skills
# ---------------------------------------------------------------------------
def compare_skills(resume_skills, job_skills):
    """Find skills present in both resume and job description."""

    resume_set = set(resume_skills)
    job_set = set(job_skills)

    matched = sorted(resume_set.intersection(job_set))
    missing = sorted(job_set.difference(resume_set))

    return matched, missing


# ---------------------------------------------------------------------------
# STEP 6: TF-IDF + Cosine Similarity
# ---------------------------------------------------------------------------
def match_resumes_to_job(resumes_df, job_text, top_n=3):
    """
    Calculate TF-IDF cosine similarity between the job description
    and every resume.
    """

    cleaned_resumes = (
        resumes_df["resume_text"]
        .apply(preprocess_text)
        .tolist()
    )

    cleaned_job = preprocess_text(job_text)

    # Combine job description and resumes
    corpus = [cleaned_job] + cleaned_resumes

    # Convert text into TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # First row = job description
    job_vector = tfidf_matrix[0:1]

    # Remaining rows = resumes
    resume_vectors = tfidf_matrix[1:]

    # Calculate cosine similarity
    scores = cosine_similarity(
        resume_vectors,
        job_vector
    ).flatten()

    # Create result table
    results = resumes_df.copy()
    results["match_score"] = scores

    # Sort from highest to lowest
    results = results.sort_values(
        by="match_score",
        ascending=False
    )

    return results.head(top_n)


# ---------------------------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------------------------
def main():

    # Dataset paths
    resumes_path = "Dataset/resumes.csv"
    jobs_path = "Dataset/job_descriptions.csv"
    skills_path = "Dataset/skills_vocabulary.txt"

    # Load datasets
    resumes_df, jobs_df = load_data(
        resumes_path,
        jobs_path
    )

    # Load skills vocabulary
    skills_vocab = load_skills_vocabulary(skills_path)


    # =======================================================================
    # STEP A: Skill extraction
    # =======================================================================

    print("=" * 75)
    print("STEP A: SKILL EXTRACTION FROM RESUMES")
    print("=" * 75)

    for _, row in resumes_df.iterrows():

        skills = extract_skills(
            row["resume_text"],
            skills_vocab
        )

        print(
            f"\n{row['candidate_name']} ({row['resume_id']})"
        )

        if skills:
            print("Skills:", ", ".join(skills))
        else:
            print("Skills: None detected")


    # =======================================================================
    # STEP B: Resume matching
    # =======================================================================

    print("\n")
    print("=" * 75)
    print("STEP B: RESUME - JOB DESCRIPTION MATCHING")
    print("=" * 75)


    for _, job in jobs_df.iterrows():

        job_title = job["job_title"]
        job_id = job["jd_id"]
        job_text = job["job_text"]


        print("\n")
        print("-" * 75)
        print(f"JOB: {job_title} ({job_id})")
        print("-" * 75)


        # Extract required skills
        job_skills = extract_skills(
            job_text,
            skills_vocab
        )

        print("\nRequired Skills:")
        print(", ".join(job_skills))


        # Get top matching resumes
        ranked = match_resumes_to_job(
            resumes_df,
            job_text,
            top_n=3
        )


        print("\nTOP 3 MATCHING RESUMES")
        print("-" * 75)


        for rank, (_, candidate) in enumerate(
            ranked.iterrows(),
            start=1
        ):

            candidate_skills = extract_skills(
                candidate["resume_text"],
                skills_vocab
            )

            matched, missing = compare_skills(
                candidate_skills,
                job_skills
            )

            # Convert similarity to percentage
            score_percentage = (
                candidate["match_score"] * 100
            )

            print(
                f"\n{rank}. {candidate['candidate_name']} "
                f"({candidate['resume_id']})"
            )

            print(
                f"   Match Score: "
                f"{score_percentage:.2f}%"
            )


            print("   Matched Skills:")

            if matched:
                print(
                    "   ✓ " +
                    ", ".join(matched)
                )
            else:
                print("   None")


            print("   Missing Skills:")

            if missing:
                print(
                    "   ✗ " +
                    ", ".join(missing)
                )
            else:
                print("   None")


        print("\n" + "=" * 75)


# ---------------------------------------------------------------------------
# Program execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()