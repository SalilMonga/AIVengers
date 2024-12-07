from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import re

app = Flask(__name__)
CORS(app)

# Load and preprocess the text
def load_text(file_path):
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'ISO-8859-1']
    text = None

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                text = file.read()
            break
        except UnicodeDecodeError:
            continue

    if text is None:
        raise ValueError("Unable to read the file with the tried encodings.")

    # Clean up text
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9.,!? ]+', '', text)
    text = text.lower()

    return text

# Calculate TF-IDF
def calculate_tfidf(text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    tfidf_scores = dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0]))
    return tfidf_scores

# Select key terms for quiz with varied ranking to avoid repetition
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
        i += random.randint(1, variation)  # Skip ahead by a random number for variety

    return selected_terms

# Generate True/False statements, avoiding duplicates
def generate_true_false_statements(text, quiz_terms, num_questions):
    sentences = text.split('. ')
    statements = []
    used_sentences = set()

    for term in quiz_terms:
        random.shuffle(sentences)  # Randomize sentence selection for each term
        for sentence in sentences:
            if term in sentence and sentence not in used_sentences:
                # Track used sentences to avoid duplication
                used_sentences.add(sentence)

                # True statement
                true_statement = sentence.strip()
                statements.append((true_statement, True))

                # False statement: vary replacement term and method
                random_term = random.choice([t for t in quiz_terms if t != term])
                modified_sentence = sentence.replace(term, random_term.upper())
                statements.append((modified_sentence.strip(), False))
                if len(statements) >= num_questions:
                    return statements  # Stop if we reach the desired number of questions
                break

    # Ensure we have the exact number of unique statements if possible
    unique_statements = []
    seen = set()
    for statement, is_true in statements:
        if statement not in seen:
            seen.add(statement)
            unique_statements.append((statement, is_true))
            if len(unique_statements) == num_questions:
                break

    return unique_statements

@app.route('/generate-quiz', methods=['POST'])
def generate_quiz():
    if 'file' not in request.files:
        print("No file part in request")
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        print("No selected file")
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file
    file_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(file_path)
    print(f"File saved at {file_path}")  # Debug print

    # Load text and generate quiz
    try:
        text = load_text(file_path)
        print(f"Loaded text from file: {text[:100]}...")  # Debug print to show part of the loaded text
    except ValueError as e:
        print(f"Error loading text: {e}")  # Debug print in case of an error
        return jsonify({"error": str(e)}), 400

    tfidf_scores = calculate_tfidf(text)
    print(f"TF-IDF scores: {list(tfidf_scores.items())[:10]}")  # Debug print of first 10 TF-IDF scores

    quiz_terms = select_quiz_terms(tfidf_scores, n_terms=10, variation=5)
    print(f"Selected quiz terms: {quiz_terms}")  # Debug print to show selected quiz terms

    statements = generate_true_false_statements(text, quiz_terms, num_questions=10)
    print(f"Generated quiz statements: {statements}")  # Debug print to show generated statements

    # Prepare the response
    quiz = [{"statement": statement, "is_true": is_true} for statement, is_true in statements]
    return jsonify(quiz)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)

