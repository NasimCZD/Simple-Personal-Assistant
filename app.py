from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import random
import json
from assistant import get_response # We import the function you already wrote!

app = Flask(__name__)
# CORS(app) # Enable Cross-Origin Resource Sharing
# Update this line to be more permissive for local testing
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/chat', methods=['POST'])
def chat():
    # Get the message from the user (JSON format)
    user_data = request.json
    user_message = user_data.get("message")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Get the response using your existing assistant logic
    bot_response = get_response(user_message)

    # Return the response as JSON
    return jsonify({"response": bot_response})

if __name__ == '__main__':
    # Run the server on localhost port 5000
    app.run(debug=True, port=5000)