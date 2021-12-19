#!/usr/bin/python3
import torch
from transformers import (ElectraTokenizerFast,
                          ElectraForSequenceClassification)

from krwordrank.word import summarize_with_keywords
from krwordrank.hangle import normalize
from konlpy.tag import Okt



class SentimentClassifier:
    LABELS = ['분노', '슬픔', '불안', '당황', '상처', '기쁨', ]
    ID_LABELS = {idx: key for (idx, key) in enumerate(LABELS)}
    MAX_LEN = 512


    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = ElectraForSequenceClassification.from_pretrained("kykim/electra-kor-base",
                                                                      problem_type="multi_label_classification",
                                                                      num_labels=6).to(self.device)
        self.tokenizer = ElectraTokenizerFast.from_pretrained("kykim/electra-kor-base")
        self.model.load_state_dict(torch.load("model.pt", map_location=torch.device('cpu')))
        # print(self.dataset.describe())

    def _get_prediction_input(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=SentimentClassifier.MAX_LEN,
            pad_to_max_length=True,
            add_special_tokens=True
        )

        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]

        return input_ids, attention_mask

    def predict(self, text):
        input_ids, attention_mask = self._get_prediction_input(text)
        y_pred = self.model(input_ids.unsqueeze(0).to(self.device), attention_mask=attention_mask.unsqueeze(0).to(self.device))[0]
        _, predicted = torch.max(y_pred, 1)
        return SentimentClassifier.ID_LABELS[predicted.item()]


class PostKeywordExtractor:

    stopwords = {'너무', '정말', '로만', '원래', '구나', '건데', '돌아', '일단', '사이'}

    def __init__(self):
        self.okt = Okt()
        self.okt.nouns("시범 명사")

    def extract_keywords(self, text):
        texts = [text]
        texts = [normalize(text, english=True, number=True) for text in texts]

        keyword_nouns = []
        try:
            keywords = summarize_with_keywords(texts,
                                               min_count=2,
                                               max_length=10,
                                               beta=0.85,
                                               max_iter=10,
                                               stopwords=PostKeywordExtractor.stopwords, verbose=True)

            for word, r in sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:30]:
                for noun in self.okt.nouns(word):
                    if len(noun) > 1:
                        keyword_nouns.append(noun)
                print('%8s:\t%.4f' % (word, r))

            if len(keyword_nouns) < 1:
                raise ValueError("Zero keywords extracted: Fallback to okt_nounts")
            return keyword_nouns
        except:
            okt_nouns = self.okt.nouns(texts[0])
            print(f"okt_nouns {okt_nouns}")
            keyword_nouns = [word for word in okt_nouns if len(word) > 1 and word not in self.stopwords]
            keyword_nouns = list(set(keyword_nouns))
            return keyword_nouns