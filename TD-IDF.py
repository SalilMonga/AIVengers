import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Step 1: Load, clean, and preprocess the text
def load_text(file_path):
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'ISO-8859-1']  # List of encodings to try
    text = None

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                text = file.read()
            print(f"Successfully read file with {encoding} encoding")
            break  # Exit loop if reading was successful
        except UnicodeDecodeError:
            print(f"Failed with encoding: {encoding}")
            continue  # Try the next encoding if there's a UnicodeDecodeError

    if text is None:
        raise ValueError("Unable to read the file with tried encodings.")

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove non-alphanumeric characters except punctuation (.,!?)
    text = re.sub(r'[^a-zA-Z0-9.,!? ]+', '', text)

    # Convert to lowercase for uniformity
    text = text.lower()

    return text

# Step 2: Calculate TF-IDF
def calculate_tfidf(text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    tfidf_scores = dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0]))
    return tfidf_scores

# Step 3: Select key terms for quiz
def select_quiz_terms(tfidf_scores, n_terms=5):
    sorted_terms = sorted(tfidf_scores.items(), key=lambda item: item[1], reverse=True)
    return [term[0] for term in sorted_terms[:n_terms]]

# Step 4: Generate True/False statements
def generate_true_false_statements(text, quiz_terms):
    sentences = text.split('. ')
    statements = []

    for term in quiz_terms:
        for sentence in sentences:
            if term in sentence:
                # Create a True statement
                true_statement = sentence.strip()
                statements.append((true_statement, True))

                # Create a False statement by modifying the key term
                false_statement = sentence.replace(term, random.choice(quiz_terms).upper())
                statements.append((false_statement.strip(), False))
                break
    return statements

# Specify the path to the file at the base level of your project
file_path = os.path.join(os.path.dirname(__file__), 'Test Bank/Array Basics.txt')

# Putting it all together
text = load_text(file_path)
tfidf_scores = calculate_tfidf(text)
quiz_terms = select_quiz_terms(tfidf_scores)
statements = generate_true_false_statements(text, quiz_terms)

# Display True/False statements
for i, (statement, is_true) in enumerate(statements, 1):
    print(f"Statement {i}: {statement} (True/False)")
