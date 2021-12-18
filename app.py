#!/usr/bin/python3
from flask import Flask, request, jsonify
from sentiment import SentimentClassifier
from krwordrank.word import summarize_with_keywords
from krwordrank.hangle import normalize
from konlpy.tag import Okt


app = Flask(__name__)

classifier = SentimentClassifier()
okt = Okt()

@app.route('/sentiment', methods=["POST"])
def sentiment():
    dataRecieve = request.get_json()
    user_input = dataRecieve["content"]

    return jsonify({
        'sentiment': f'{classifier.predict(user_input)}',
    })

@app.route('/ping')
def ping():
    return jsonify({
        'hi': 'HI',
    })

@app.route('/keyword', methods=["POST"])
def keyword():
    print(request)
    dataRecieve = request.get_json()
    texts = [dataRecieve["text"]]
    texts = [normalize(text, english=True, number=True) for text in texts]


    stopwords = {'너무', '정말', '로만', '원래', '구나', '건데', '돌아', '일단', '사이'}

    try:
        keywords = summarize_with_keywords(texts, min_count=2, max_length=10,
                                           beta=0.85, max_iter=10, stopwords=stopwords, verbose=True)
        keyword_nouns = []
        for word, r in sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:30]:
            for noun in okt.nouns(word):
                if len(noun) > 1:
                    keyword_nouns.append(noun)
            print('%8s:\t%.4f' % (word, r))

        if len(keyword_nouns) < 1:
            raise ValueError("Zero keywords extracted: Fallback to okt_nounts")

        return jsonify({
            'keywords': keyword_nouns,
        })
    except:
        okt_nouns = okt.nouns(texts[0])
        print(f"okt_nouns {okt_nouns}")
        keyword_nouns = [word for word in okt_nouns if len(word) > 1 and word not in stopwords]
        keyword_nouns = list(set(keyword_nouns))
        return jsonify({
            'keywords': keyword_nouns,
        })


if __name__=="__main__":
    app.run(host="0.0.0.0", port=8000)
