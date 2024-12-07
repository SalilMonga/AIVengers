from flask import Flask, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)


@app.route('/v1/chat/completions', methods=['POST'])
def mock_chatgpt_response():
    return jsonify({
        "id": "chatcmpl-AbhmTzTz07g7d25tQVzEafGqBLe1v",
        "object": "chat.completion",
        "created": 1733549793,
        "model": "gpt-3.5-turbo",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "1. Data has specific needs for storage and manipulation. (True/False)\n"
                               "2. Linear data structures have specific characteristics and uses. (True/False)\n"
                               "3. A pointer to the right child of a binary tree is a common feature. (True/False)\n"
                               "4. A pointer to the right child in a binary tree structure is a common design element. (True/False)\n"
                               "5. Data structures serve multiple purposes beyond just organizing data. (True/False)\n"
                               "6. An ELEMENT structure serves purposes beyond just organizing ELEMENT. (True/False)\n"
                               "7. Data structures like graphs have diverse applications in various industries. (True/False)\n"
                               "8. Structured data representations like graphs are widely used across different fields. (True/False)\n"
                               "9. Arrays are a fundamental data structure used for storing elements. (True/False)\n"
                               "10. Array structures provide a systematic way to organize and access data elements. (True/False)"
                }
            }
        ],
        "usage": {
            "prompt_tokens": 200,
            "completion_tokens": 224,
            "total_tokens": 424
        }
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)