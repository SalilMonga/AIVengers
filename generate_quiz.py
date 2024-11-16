import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import re


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

    # Clean up text
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9.,!? ]+', '', text)
    text = text.lower()

    return text


# Calculate TF-IDF
def calculate_tfidf(text):
    """
    Calculates TF-IDF scores for each term in the provided text.

    Args:
        text (str): The input text to analyze.

    Returns:
        dict: A dictionary where keys are terms and values are TF-IDF scores.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    tfidf_scores = dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0]))
    return tfidf_scores


# Select key terms for quiz with varied ranking to avoid repetition
def select_quiz_terms(tfidf_scores, n_terms=5, variation=10):
    """
    Selects key terms for quiz questions by varying the ranking of terms to increase variety.

    Args:
        tfidf_scores (dict): Dictionary of terms with their TF-IDF scores.
        n_terms (int): Number of terms to select for the quiz.
        variation (int): The variation in term selection to prevent repetition.

    Returns:
        list: A list of selected terms for generating quiz questions.
    """
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
    """
    Generates True/False quiz statements based on key terms in the text.

    Args:
        text (str): The full text content to extract sentences from.
        quiz_terms (list): List of selected key terms for creating statements.
        num_questions (int): Number of questions to generate.

    Returns:
        list: A list of tuples, each containing a statement (str) and a boolean indicating if it’s true.
    """
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


# Base directory for text bank
base_dir = os.path.join(os.path.dirname(__file__), 'Test Bank')

# List of text files to process, using the new base directory
file_paths = [
    os.path.join(base_dir, 'Array Basics.txt'),
    # Add additional files
]

# Set number of questions per file
num_questions = 10

# Generate quizzes
quizzes = generate_quizzes_for_files(file_paths, num_questions)

# Display True/False statements for each file
for file_name, statements in quizzes.items():
    print(f"Quiz for {file_name}:\n")
    for i, (statement, is_true) in enumerate(statements, 1):
        print(f"Statement {i}: {statement} (True/False)")
    print("\n" + "=" * 40 + "\n")
