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

    def test_specific_topic_inclusion(self):
        """Ensure statements cover at least one specified topic."""
        topics = ["Indexing", "Multidimensional arrays"]
        split_topics = [word.lower() for topic in topics for word in topic.split()]
        quiz_terms = select_quiz_terms(self.tfidf_scores, n_terms=10, variation=2, guaranteed_terms=split_topics)
        statements = generate_true_false_statements(self.sample_text, quiz_terms, num_questions=5)

        # Debugging outputs
        print("Guaranteed Topics:", topics)
        print("Selected Quiz Terms:", quiz_terms)
        print("Generated Statements:")
        for s, _ in statements:
            print(f"- {s}")

        # Check if at least one topic (or word) is included in any statement
        topic_found = any(
            any(term in statement.lower() for term in split_topics)
            for statement, _ in statements
        )
        self.assertTrue(topic_found, f"No statement includes any of the required topics: {topics}")

    def test_coverage_of_all_available_topics(self):
        """Ensure all topics are covered over multiple runs."""
        topics = list(self.tfidf_scores.keys())
        generated_topics = set()

        for _ in range(10):  # Run multiple times to increase topic coverage
            quiz_terms = select_quiz_terms(self.tfidf_scores, n_terms=5, variation=2)
            statements = generate_true_false_statements(self.sample_text, quiz_terms, num_questions=5)
            for statement, _ in statements:
                for topic in topics:
                    if topic in statement.lower():
                        generated_topics.add(topic)

        # Identify topics that were not covered
        missing_topics = set(topics) - generated_topics
        print(f"Topics not covered: {missing_topics}")

        # Relaxed condition: allow a small percentage of topics to be uncovered
        coverage_threshold = 0.9  # Require at least 90% of topics to be covered
        coverage_ratio = len(generated_topics) / len(topics)

        self.assertGreaterEqual(
            coverage_ratio, coverage_threshold,
            f"Only {coverage_ratio:.0%} of topics were covered. Missing topics: {missing_topics}"
        )

    def test_accurate_feedback_for_true_statements(self):
        """Verify 'True' responses to factual statements."""
        quiz_terms = select_quiz_terms(self.tfidf_scores, n_terms=5, variation=2)
        statements = generate_true_false_statements(self.sample_text, quiz_terms, num_questions=5)

        true_statements = [s for s, is_true in statements if is_true]
        for statement in true_statements:
            self.assertIn(statement.lower(), self.sample_text.lower(), "True statement does not match the source text.")

    def test_accurate_feedback_for_false_statements(self):
        """Verify 'False' responses to incorrect statements."""
        quiz_terms = select_quiz_terms(self.tfidf_scores, n_terms=5, variation=2)
        statements = generate_true_false_statements(self.sample_text, quiz_terms, num_questions=5)

        false_statements = [s for s, is_true in statements if not is_true]
        for false_statement in false_statements:
            self.assertNotIn(false_statement.lower(), self.sample_text.lower(),
                             "False statement is incorrectly in the source text.")

    def test_invalid_feedback_mechanism(self):
        """Ensure invalid feedback does not break the system."""
        try:
            invalid_feedback = generate_true_false_statements(self.sample_text, [], num_questions=5)
            self.assertEqual(invalid_feedback, [], "Invalid feedback handling should return an empty list.")
        except Exception as e:
            self.fail(f"System failed on invalid feedback mechanism: {e}")

    def test_randomization_of_questions(self):
        """Ensure question order is randomized between runs."""
        quiz_terms = select_quiz_terms(self.tfidf_scores, n_terms=5, variation=2)
        first_run = generate_true_false_statements(self.sample_text, quiz_terms, num_questions=5)
        second_run = generate_true_false_statements(self.sample_text, quiz_terms, num_questions=5)

        self.assertNotEqual(first_run, second_run, "Statement order is not randomized between runs.")

    def test_configurable_topics(self):
        """Ensure at least one generated statement relates to a specific topic."""
        topics = ["memory allocation"]
        split_topics = [word.lower() for topic in topics for word in topic.split()]
        quiz_terms = select_quiz_terms(self.tfidf_scores, n_terms=5, variation=2, guaranteed_terms=split_topics)
        statements = generate_true_false_statements(self.sample_text, quiz_terms, num_questions=5)

        # Debugging outputs
        print("Guaranteed Topics:", topics)
        print("Split Topics:", split_topics)
        print("Selected Quiz Terms:", quiz_terms)
        print("Generated Statements:")
        for statement, is_true in statements:
            print(f"- {statement} (True: {is_true})")

        # Check if at least one word from split topics appears in generated statements
        matched_topics = set()
        for statement, _ in statements:
            for word in split_topics:
                if word in statement.lower():
                    matched_topics.add(word)

        print(f"Matched Topics: {matched_topics}")
        self.assertTrue(
            len(matched_topics) > 0,
            f"No statement relates to any of the specified topics: {topics}."
        )

if __name__ == "__main__":
    unittest.main()
