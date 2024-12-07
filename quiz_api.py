import os
import random
import re
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer

DEBUG = True

app = Flask(__name__)
CORS(app)


def debug_print(message):
    if DEBUG:
        print(message)


# Load and preprocess the text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = re.sub(r'[^a-zA-Z0-9.,!? ]+', '', text)  # Remove special characters
    text = text.lower()
    return text


# Calculate TF-IDF
def calculate_tfidf(text):
    if not text.strip():
        return {}

    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform([text])
        tfidf_scores = dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0]))
        debug_print(f"TF-IDF Terms: {tfidf_scores.keys()}")
        return tfidf_scores
    except ValueError as e:
        debug_print(f"Error calculating TF-IDF: {e}")
        return {}


# Select key terms for quiz with varied ranking
def select_quiz_terms(tfidf_scores, n_terms=5, variation=10):
    sorted_terms = sorted(tfidf_scores.items(), key=lambda item: item[1], reverse=True)
    selected_terms = []
    used_terms = set()

    i = 0
    while len(selected_terms) < n_terms and i < len(sorted_terms):
        term, score = sorted_terms[i]
        if term not in used_terms:
            selected_terms.append(term)
            used_terms.add(term)
        i += random.randint(1, variation)

    debug_print(f"Selected Quiz Terms: {selected_terms}")
    return selected_terms


# Generate True/False statements
def generate_true_false_statements(text, quiz_terms, num_questions):
    sentences = text.split('. ')
    statements = []
    used_sentences = set()

    for term in quiz_terms:
        random.shuffle(sentences)
        for sentence in sentences:
            if term in sentence and sentence not in used_sentences:
                used_sentences.add(sentence)

                # Generate true statement
                true_statement = sentence.strip()
                statements.append((true_statement, True))

                # Generate false statement
                alternate_terms = [t for t in quiz_terms if t != term]
                if alternate_terms:
                    random_term = random.choice(alternate_terms)
                    false_statement = sentence.replace(term, random_term.upper())
                    statements.append((false_statement.strip(), False))

                if len(statements) >= num_questions:
                    break

    return statements[:num_questions]


@app.route('/generate-quiz', methods=['POST'])
def generate_quiz_api():
    """
    API endpoint to generate a quiz from an uploaded file.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    file_content = file.read().decode('utf-8', errors='ignore')

    # Preprocess the text
    text = preprocess_text(file_content)
    if not text.strip():
        return jsonify({"error": "File is empty or invalid"}), 400

    # Generate quiz
    tfidf_scores = calculate_tfidf(text)
    quiz_terms = select_quiz_terms(tfidf_scores, n_terms=10, variation=5)
    statements = generate_true_false_statements(text, quiz_terms, num_questions=10)

    # Return as JSON
    quiz = [{"statement": stmt, "is_true": is_true} for stmt, is_true in statements]
    return jsonify(quiz), 200


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
