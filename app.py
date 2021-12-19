#!/usr/bin/python3
from flask import Flask, request, jsonify
from sentiment import (SentimentClassifier,
                        PostKeywordExtractor)


app = Flask(__name__)

classifier = SentimentClassifier()
keyword_extractor = PostKeywordExtractor()


@app.route('/predict', methods=["POST"])
def predict_sentiment_and_keywords():
    dataRecieve = request.get_json()
    user_input = dataRecieve["content"]

    sentiment = classifier.predict(user_input)
    keywords = keyword_extractor.extract_keywords(user_input)

    return jsonify({
        'sentiment': sentiment,
        'keywords': keywords
    })

@app.route('/ping')
def ping():
    return jsonify({
        'hi': 'HI',
    })


if __name__=="__main__":
    app.run(host="0.0.0.0", port=8000)
