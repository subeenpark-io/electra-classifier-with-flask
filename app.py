#!/usr/bin/python3
from flask import Flask, request, jsonify
from sentiment import SentimentClassifier

app = Flask(__name__)

classifier = SentimentClassifier()

@app.route('/predict', methods=["POST"])
def true_or_false():
    dataRecieve = request.get_json()
    user_input = dataRecieve["content"]

    return jsonify({
        'sentiment': f'{classifier.predict(user_input)}',
    })

@app.route('/ping')
def ping():
    return "healthy"


if __name__=="__main__":
    app.run(host="0.0.0.0", port=8000)
