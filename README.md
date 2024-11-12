# AIVengers

This project uses TF-IDF to generate True/False quiz questions from text files.

## Instructions for Adding New Topics to the Test Bank

1. **Gather Content**: Collect relevant information or content based on the topic you want to generate quiz questions for.
2. **Create a Text File**: Save the content as a `.txt` file. The file name should reflect the topic (e.g., `Array Basics.txt`).
3. **Add to Test Bank Folder**: Move the `.txt` file into the `Test Bank` folder located in the project directory.
4. **Update Code with File Name**: In the code, add the new file’s name to the list of files processed by the quiz generator. This list is located in the `file_paths` variable.

## Instructions to Run Code Locally

1. **Install Dependencies**:
   - Ensure you are in a virtual environment (optional but recommended).
   - Run the following command to install the required libraries:
     ```bash
     pip install -r requirements.txt
     ```

2. **Run the Code**:
   - Run the main Python file to generate quiz questions from the text files in the `Test Bank` folder.

## Sample Code Execution

If the main file is `generate_quiz.py`, you can run it with:
```bash
python generate_quiz.py