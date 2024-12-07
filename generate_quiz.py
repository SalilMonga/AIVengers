import os
import random
import re

from flask import Flask, request, jsonify
from flask_cors import CORS

from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer

DEBUG = True  # Set to True to enable all debugging messages


app = Flask(__name__)
CORS(app)

def debug_print(message):
    """
    Prints a debugging message if DEBUG is enabled.

    Args:
        message (str): The message to print.
    """
    if DEBUG:
        print(message)


# Load and preprocess the text
def load_text(file_path):
    """
    Loads and preprocesses text from a file by trying multiple encodings and cleaning up the text.

    Args:
        file_path (str): Path to the text file.

    Returns:
        str: Cleaned text from the file.

    Raises:
        ValueError: If the file cannot be read with the specified encodings.
    """
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

    preprocess_text_for_matching(text)
    return text


# Clean up text
def preprocess_text_for_matching(text):
    """
    Preprocess text for consistent term matching.

    Args:
        text (str): The input text.

    Returns:
        set: A set of tokenized terms from the text.
    """
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9 ]+', '', text)
    text = text.lower()
    return set(text.split())


# Calculate TF-IDF
def calculate_tfidf(text):
    """
    Calculates TF-IDF scores for each term in the provided text.

    Args:
        text (str): The input text to analyze.

    Returns:
        dict: A dictionary where keys are terms and values are TF-IDF scores.
    """
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


# Select key terms for quiz with varied ranking to avoid repetition
def select_quiz_terms(tfidf_scores, n_terms=5, variation=10, guaranteed_terms=None):
    """
    Selects key terms for quiz questions by varying the ranking of terms to increase variety.
    Args:
        tfidf_scores (dict): Dictionary of terms with their TF-IDF scores.
        n_terms (int): Number of terms to select for the quiz.
        variation (int): The variation in term selection to prevent repetition.
        guaranteed_terms (list): Terms that must be included if present in the TF-IDF scores.
    Returns:
        list: A list of selected terms for generating quiz questions.
    """
    sorted_terms = sorted(
        tfidf_scores.items(), key=lambda item: item[1], reverse=True
    )
    selected_terms = []
    used_terms = set()

    # Ensure guaranteed terms are included
    if guaranteed_terms:
        for term in guaranteed_terms:
            if term in tfidf_scores and term not in used_terms:
                selected_terms.append(term)
                used_terms.add(term)

    # Randomly select additional terms
    i = 0
    while len(selected_terms) < n_terms and i < len(sorted_terms):
        term, score = sorted_terms[i]
        if term not in used_terms:
            selected_terms.append(term)
            used_terms.add(term)
        i += random.randint(1, variation)  # Skip ahead by a random number for variety

    debug_print(f"Selected Quiz Terms: {selected_terms}")
    return selected_terms


# Generate True/False statements, avoiding duplicates
def generate_true_false_statements(text, quiz_terms, num_questions, max_attempts=100):
    sentences = text.split('. ')
    if not isinstance(sentences, list) or not sentences:
        debug_print("Invalid or empty sentences. Returning no statements.")
        return []

    max_possible_statements = len(sentences) * len(quiz_terms)
    num_questions = min(num_questions, max_possible_statements)  # Cap questions to maximum possible
    debug_print(f"Max possible statements: {max_possible_statements}")

    statements = []
    used_combinations = set()
    used_statements = set()
    attempts = 0

    while len(statements) < num_questions and attempts < max_attempts:
        random.shuffle(quiz_terms)
        for term in quiz_terms:
            random.shuffle(sentences)
            for sentence in sentences:
                if term in sentence and (sentence, term) not in used_combinations:
                    used_combinations.add((sentence, term))

                    if random.choice([True, False]):
                        true_statement = sentence.strip()
                        if true_statement not in used_statements:
                            statements.append((true_statement, True))
                            used_statements.add(true_statement)
                    else:
                        alternate_terms = [t for t in quiz_terms if t != term]
                        if alternate_terms:
                            random_term = random.choice(alternate_terms)
                            modified_sentence = sentence.replace(term, random_term.upper())
                            if modified_sentence not in used_statements:
                                statements.append((modified_sentence.strip(), False))
                                used_statements.add(modified_sentence)

                    if len(statements) >= num_questions:
                        return statements
                    break
        attempts += 1

    debug_print(f"Generated {len(statements)} statements after {attempts} attempts.")

    return statements

# Main function to handle multiple text files
def generate_quizzes_for_files(file_paths, num_questions):
    """
    Generates quizzes for multiple text files.

    Args:
        file_paths (list): List of file paths to process.
        num_questions (int): Number of questions to generate for each file.

    Returns:
        dict: A dictionary where keys are file names and values are lists of True/False statements.
    """
    quizzes = {}

    for file_path in file_paths:
        text = load_text(file_path)
        tfidf_scores = calculate_tfidf(text)
        quiz_terms = select_quiz_terms(tfidf_scores, n_terms=10, variation=5)
        statements = generate_true_false_statements(text, quiz_terms, num_questions)

        # Store the quiz statements for each file
        file_name = os.path.basename(file_path)
        quizzes[file_name] = statements

    return quizzes


def export_quizzes_to_files(quizzes, output_dir="Quizzes"):
    """
    Generates quizzes for multiple text files.
    Exports quizzes to text files: one with answers and one without.

    Args:
        quizzes (dict): A dictionary where keys are file names and values are lists of (statement, answer) tuples.
        output_dir (str): Directory where output files will be saved.

    Returns:
        tuple: Paths of the two output files.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    with_answers_file = os.path.join(output_dir, f"quiz-with-answers-{timestamp}.txt")
    without_answers_file = os.path.join(output_dir, f"quiz-without-answers-{timestamp}.txt")

    with open(with_answers_file, "w") as answers_file, open(without_answers_file, "w") as questions_file:
        for file_name, statements in quizzes.items():
            answers_file.write(f"Quiz for {file_name}:\n")
            questions_file.write(f"Quiz for {file_name}:\n")
            for i, (statement, is_true) in enumerate(statements, 1):
                answers_file.write(f"Statement {i}: {statement.strip()}\nAnswer: {'True' if is_true else 'False'}\n\n")
                questions_file.write(f"Statement {i}: {statement.strip()}\n\n")
            answers_file.write("=" * 40 + "\n\n")
            questions_file.write("=" * 40 + "\n\n")

    return with_answers_file, without_answers_file


# Base directory for text bank
base_dir = os.path.join(os.path.dirname(__file__), 'TestBank')

# List of text files to process
file_paths = [
    os.path.join(base_dir, 'ArrayBasics.txt'),
    os.path.join(base_dir, 'IntroToDS.txt'),
    # Add additional files
]

# Set number of questions per file
num_questions = 5

# Generate quizzes
quizzes = generate_quizzes_for_files(file_paths, num_questions)

# Export quizzes to the "Quizzes" directory
output_dir = os.path.join(os.path.dirname(__file__), 'Quizzes')
with_answers_file, without_answers_file = export_quizzes_to_files(quizzes, output_dir)

print(f"Quizzes exported:")
print(f"With answers: {with_answers_file}")
print(f"Without answers: {without_answers_file}")

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)