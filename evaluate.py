import pandas as pd
import re


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
resumes_df = pd.read_csv("Dataset/resumes.csv")

with open(
    "Dataset/skills_vocabulary.txt",
    "r",
    encoding="utf-8"
) as file:
    content = file.read().replace("\n", " ")

skills_vocab = [
    skill.strip().lower()
    for skill in content.split(",")
    if skill.strip()
]


# ---------------------------------------------------------
# Skill extraction function
# ---------------------------------------------------------
def extract_skills(text, skills_list):

    text = str(text).lower()

    found_skills = []

    for skill in skills_list:

        pattern = (
            r"(?<![a-z])"
            + re.escape(skill)
            + r"(?![a-z])"
        )

        if re.search(pattern, text):
            found_skills.append(skill)

    return found_skills


# ---------------------------------------------------------
# Manual ground truth
# ---------------------------------------------------------
# These are the skills that are actually present in
# each resume, verified from the resume text.

ground_truth = {

    "R001": [
        "python",
        "sql",
        "machine learning",
        "deep learning",
        "tensorflow",
        "scikit-learn",
        "pandas",
        "numpy",
        "natural language processing",
        "tf-idf",
        "statistics",
        "data visualization",
        "matplotlib",
        "seaborn"
    ],

    "R002": [
        "java",
        "sql",
        "mysql",
        "spring boot",
        "docker",
        "kubernetes",
        "jenkins",
        "git",
        "ci/cd",
        "rest api",
        "microservices",
        "agile"
    ],

    "R003": [
        "javascript",
        "react",
        "html",
        "css",
        "rest api",
        "redux",
        "typescript",
        "node.js",
        "git"
    ],

    "R004": [
        "python",
        "docker",
        "aws",
        "machine learning",
        "deep learning",
        "tensorflow",
        "pytorch",
        "natural language processing",
        "bert",
        "named entity recognition",
        "statistics",
        "linear algebra"
    ],

    "R005": [
        "excel"
    ],

    "R006": [
        "python",
        "sql",
        "pandas",
        "numpy",
        "statistics",
        "power bi",
        "tableau",
        "excel",
        "a/b testing"
    ],

    "R007": [
        "python",
        "javascript",
        "sql",
        "postgresql",
        "react",
        "django",
        "docker",
        "git",
        "aws",
        "machine learning",
        "rest api"
    ],

    "R008": [
        "python",
        "flask",
        "nlp",
        "spacy",
        "nltk",
        "transformers",
        "bert",
        "lstm",
        "tf-idf",
        "cosine similarity",
        "word embeddings"
    ]
}


# ---------------------------------------------------------
# Calculate Precision, Recall and F1
# ---------------------------------------------------------

total_true_positive = 0
total_false_positive = 0
total_false_negative = 0


print("=" * 70)
print("SKILL EXTRACTION EVALUATION")
print("=" * 70)


for _, row in resumes_df.iterrows():

    resume_id = row["resume_id"]
    candidate = row["candidate_name"]
    resume_text = row["resume_text"]

    actual_skills = set(
        skill.lower()
        for skill in ground_truth[resume_id]
    )

    predicted_skills = set(
        extract_skills(resume_text, skills_vocab)
    )

    true_positive = len(
        actual_skills.intersection(predicted_skills)
    )

    false_positive = len(
        predicted_skills.difference(actual_skills)
    )

    false_negative = len(
        actual_skills.difference(predicted_skills)
    )

    total_true_positive += true_positive
    total_false_positive += false_positive
    total_false_negative += false_negative

    if true_positive + false_positive > 0:
        precision = true_positive / (
            true_positive + false_positive
        )
    else:
        precision = 0

    if true_positive + false_negative > 0:
        recall = true_positive / (
            true_positive + false_negative
        )
    else:
        recall = 0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (
            precision + recall
        )
    else:
        f1 = 0

    print()
    print(f"Candidate: {candidate} ({resume_id})")
    print(f"Actual Skills:    {len(actual_skills)}")
    print(f"Extracted Skills: {len(predicted_skills)}")
    print(f"True Positives:   {true_positive}")
    print(f"False Positives:  {false_positive}")
    print(f"False Negatives:  {false_negative}")
    print(f"Precision:        {precision * 100:.2f}%")
    print(f"Recall:           {recall * 100:.2f}%")
    print(f"F1-Score:         {f1 * 100:.2f}%")


# ---------------------------------------------------------
# Overall evaluation
# ---------------------------------------------------------

overall_precision = total_true_positive / (
    total_true_positive + total_false_positive
)

overall_recall = total_true_positive / (
    total_true_positive + total_false_negative
)

overall_f1 = 2 * overall_precision * overall_recall / (
    overall_precision + overall_recall
)


print()
print("=" * 70)
print("OVERALL EVALUATION")
print("=" * 70)

print(f"Total True Positives:  {total_true_positive}")
print(f"Total False Positives: {total_false_positive}")
print(f"Total False Negatives: {total_false_negative}")

print()
print(f"Overall Precision: {overall_precision * 100:.2f}%")
print(f"Overall Recall:    {overall_recall * 100:.2f}%")
print(f"Overall F1-Score:  {overall_f1 * 100:.2f}%")

print("=" * 70)
print("Evaluation completed successfully.")
print("=" * 70)