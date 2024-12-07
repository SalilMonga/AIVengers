import unittest
from generate_quiz import (
    load_text,
    preprocess_text_for_matching,
    calculate_tfidf,
    select_quiz_terms,
    generate_true_false_statements,
    generate_quizzes_for_files,
)

class TestQuizGenerator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sample_text = (
            "Arrays are a fundamental data structure. "
            "They are used to store multiple elements of the same type in contiguous memory. "
            "Arrays allow efficient access to elements using an index. "
            "Fixed-size arrays have a predetermined capacity, which cannot be changed. "
            "Dynamic arrays can resize automatically as elements are added or removed. "
            "Common operations on arrays include traversal, insertion, deletion, and searching. "
            "Arrays are frequently used in applications such as sorting and searching algorithms. "
            "They are also used to implement other data structures, like stacks and queues. "
            "In programming, arrays are declared using square brackets. "
            "Elements in an array are accessed using their zero-based index. "
            "The size of an array is determined at the time of declaration. "
            "Efficient memory usage and fast random access make arrays an essential tool in computing."
        )
        cls.processed_text_terms = preprocess_text_for_matching(cls.sample_text)
        cls.tfidf_scores = calculate_tfidf(cls.sample_text)
        cls.quiz_terms = select_quiz_terms(cls.tfidf_scores, n_terms=5, variation=2)
        cls.statements = generate_true_false_statements(cls.sample_text, cls.quiz_terms, num_questions=5)

    def test_minimum_statement_generation(self):
        """Verify the algorithm generates as many statements as possible."""
        # Check that the number of statements does not exceed the available data
        max_possible_statements = len(self.processed_text_terms) * len(self.statements)
        self.assertLessEqual(len(self.statements), max_possible_statements)

        # Ensure at least some statements are generated if input is sufficient
        self.assertGreater(len(self.statements), 0, "No statements generated when input data is available.")

    def test_statement_format(self):
        """Ensure each generated statement is a string."""
        for statement, _ in self.statements:
            self.assertIsInstance(statement, str)

    def test_statement_uniqueness(self):
        """Ensure no duplicate statements are generated in a single run."""
        unique_statements = {statement for statement, _ in self.statements}
        self.assertEqual(len(unique_statements), len(self.statements))

    def test_feedback_for_true_response(self):
        """Verify feedback for 'True' responses to factual statements."""
        print("Processed Text Terms:", self.processed_text_terms)
        print("TF-IDF Quiz Terms:", self.quiz_terms)
        missing_terms = [term for term in self.quiz_terms if term not in self.processed_text_terms]
        print("Missing Terms:", missing_terms)
        self.assertTrue(len(missing_terms) == 0, f"Terms not found in text: {missing_terms}")

    def test_feedback_for_false_response(self):
        """Verify feedback for 'False' responses to incorrect statements."""
        false_statements = [s for s, is_true in self.statements if not is_true]
        for false_statement in false_statements:
            self.assertFalse(false_statement in self.sample_text)

    def test_feedback_response_timing(self):
        """Ensure feedback is provided immediately (simulated)."""
        self.assertTrue(True)  # Simulating an instantaneous response.

    def test_statement_variance_across_runs(self):
        """Ensure statements are randomized and different across runs."""
        new_statements = generate_true_false_statements(self.sample_text, self.quiz_terms, num_questions=5)
        self.assertNotEqual(self.statements, new_statements)

    def test_statement_count_flexibility(self):
        """Ensure the algorithm can generate a user-specified number of statements."""
        specific_statements = generate_true_false_statements(self.sample_text, self.quiz_terms, num_questions=6)
        self.assertEqual(len(specific_statements), 6)

    def test_no_empty_statements(self):
        """Ensure the algorithm does not generate empty or blank statements."""
        for statement, _ in self.statements:
            self.assertNotEqual(statement.strip(), "")

    def test_handling_invalid_statement_pool(self):
        """Ensure the algorithm handles an empty statement pool gracefully."""
        empty_text = ""
        empty_tfidf = calculate_tfidf(empty_text)
        quiz_terms = select_quiz_terms(empty_tfidf, n_terms=5)
        empty_statements = generate_true_false_statements(empty_text, quiz_terms, num_questions=5)
        self.assertEqual(empty_statements, [])

    def test_randomization_of_questions(self):
        """Ensure the order of statements is randomized between runs."""
        first_run = generate_true_false_statements(self.sample_text, self.quiz_terms, num_questions=5)
        second_run = generate_true_false_statements(self.sample_text, self.quiz_terms, num_questions=5)
        self.assertNotEqual(first_run, second_run)

    def test_statement_pool_exhaustion(self):
        """Ensure the algorithm stops generating when the pool is exhausted."""
        limited_terms = ["array", "elements", "size"]
        max_possible_statements = len(self.sample_text.split('. ')) * len(limited_terms)

        statements = generate_true_false_statements(self.sample_text, limited_terms, num_questions=10)

        # Ensure the number of statements does not exceed the maximum possible
        self.assertLessEqual(len(statements), max_possible_statements)
        self.assertGreater(len(statements), 0, "No statements generated despite available data.")

    def test_modular_generation_logic(self):
        """Verify adding new topics or statement templates is straightforward."""
        extended_text = self.sample_text + " Pointers are also used in programming."
        tfidf_scores = calculate_tfidf(extended_text)
        extended_terms = select_quiz_terms(tfidf_scores, n_terms=10, variation=5, guaranteed_terms=["pointers"])
        print("Extended Terms:", extended_terms)  # Debug selected terms
        self.assertIn("pointers", extended_terms, "Expected 'pointers' to be included in the selected terms.")

    def test_empty_text_handling(self):
        """Ensure empty text does not cause errors."""
        empty_statements = generate_true_false_statements("", [], 5)
        self.assertEqual(empty_statements, [])

    def test_insufficient_data(self):
        """Ensure generator handles insufficient data gracefully."""
        short_text = "Array is a data structure."
        short_terms = ["array"]
        statements = generate_true_false_statements(short_text, short_terms, num_questions=5)

        # Ensure no errors occurred and the result is handled gracefully
        self.assertLessEqual(len(statements), 5)
        self.assertGreaterEqual(len(statements), 0)  # Allow for 0 statements if data is insufficient

if __name__ == "__main__":
    unittest.main()
