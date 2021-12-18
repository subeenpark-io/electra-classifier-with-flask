#!/usr/bin/python3
from flask import Flask, request, jsonify
from sentiment import SentimentClassifier
from krwordrank.sentence import summarize_with_sentences

app = Flask(__name__)

classifier = SentimentClassifier()

@app.route('/sentiment', methods=["POST"])
def sentiment():
    dataRecieve = request.get_json()
    user_input = dataRecieve["content"]

    return jsonify({
        'sentiment': f'{classifier.predict(user_input)}',
    })

@app.route('/ping')
def ping():
    return "healthy"

@app.route('/keyword')
def keyword():
    dataRecieve = request.get_json()
    texts = [dataRecieve["text"]]
    keywords, sents = summarize_with_sentences(
        texts,
        diversity=0.5,
        num_keywords=100,
        num_keysents=10,
        verbose=False
    )

    return jsonify({
        'keywords': keywords,
    })


if __name__=="__main__":
    app.run(host="0.0.0.0", port=8000)
