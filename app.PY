from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
import docx2txt
import io
import re
from pdfminer.high_level import extract_text
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from typing import List, Tuple
import logging

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

nlp = spacy.load("en_core_web_sm")

class SpacyFeaturesExtractor(BaseEstimator, TransformerMixin):
    # same as earlier, no change needed here
    def __init__(self):
        self.keywords = ['python', 'machine learning', 'data analysis', 'nlp', 'flask', 'streamlit', 'javascript', 'sql', 'aws', 'deep learning']

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        features = []
        for doc_text in X:
            doc = nlp(doc_text.lower())
            kw_counts = [doc_text.lower().count(k) for k in self.keywords]
            person_count = len([ent for ent in doc.ents if ent.label_ == "PERSON"])
            org_count = len([ent for ent in doc.ents if ent.label_ == "ORG"])
            gpe_count = len([ent for ent in doc.ents if ent.label_ == "GPE"])
            doc_length = len(doc_text.split())
            noun_chunks = len(list(doc.noun_chunks))
            features.append(kw_counts + [person_count, org_count, gpe_count, doc_length, noun_chunks])
        return np.array(features)

def get_dummy_training_data() -> Tuple[List[str], List[int]]:
    # same dataset as before
    texts = [
        "Experienced Python developer skilled in machine learning and data analysis.",
        "Entry-level candidate with knowledge of JavaScript and HTML.",
        "Senior data scientist with expertise in NLP, deep learning, and AWS cloud.",
        "New graduate eager to learn Flask and Streamlit for web development.",
        "Marketing specialist with no technical skills but strong communication.",
        "DevOps engineer proficient in cloud infrastructure and automation.",
        "Software developer with experience in SQL, Python, and machine learning.",
        "Graphic designer with Photoshop and Illustrator skills.",
        "Data analyst skilled in Excel, Python, and Tableau.",
        "Junior developer familiar with Flask and database management."
    ]
    labels = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
    return texts, labels

def build_model_pipeline() -> Pipeline:
    return Pipeline([
        ('features', SpacyFeaturesExtractor()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

def train_model() -> Pipeline:
    texts, labels = get_dummy_training_data()
    X_train, _, y_train, _ = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
    model = build_model_pipeline()
    model.fit(X_train, y_train)
    return model

model = train_model()

def extract_text_from_pdf(file_bytes: bytes) -> str:
    with io.BytesIO(file_bytes) as f:
        return extract_text(f)

def extract_text_from_docx(file_bytes: bytes) -> str:
    with io.BytesIO(file_bytes) as f:
        return docx2txt.process(f)

def extract_resume_keywords(text: str, top_n: int = 10) -> List[Tuple[str, int]]:
    doc = nlp(text.lower())
    candidates = [chunk.text.strip() for chunk in doc.noun_chunks] + [ent.text.strip() for ent in doc.ents]
    freq = {}
    for cand in candidates:
        if len(cand) > 1 and not cand.isspace():
            freq[cand] = freq.get(cand, 0) + 1
    sorted_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return sorted_keywords[:top_n]

@app.route('/api/upload_resume', methods=['POST'])
def upload_resume():
    if 'resume' not in request.files:
        return jsonify({'error': 'No resume file part'}), 400
    file = request.files['resume']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_bytes = file.read()
    filename = file.filename.lower()

    try:
        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(file_bytes)
        elif filename.endswith('.docx'):
            text = extract_text_from_docx(file_bytes)
        else:
            return jsonify({'error': 'Unsupported file type'}), 400
    except Exception as e:
        logging.error(f"Text extraction failed: {str(e)}")
        return jsonify({'error': f'Failed to extract text: {str(e)}'}), 500

    if not text.strip():
        return jsonify({'error': 'No text extracted from resume'}), 400

    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0][pred]
    label_text = "Good Fit Resume (Technical)" if pred == 1 else "Not a Good Fit Resume"

    keywords = extract_resume_keywords(text)
    keywords_list = [{'keyword': k, 'count': c} for k, c in keywords]

    return jsonify({
        'text': text,
        'prediction': int(pred),
        'confidence': float(proba),
        'label': label_text,
        'keywords': keywords_list
    })

if __name__ == "__main__":
    app.run(debug=True)

