#!/usr/bin/python3
import torch
from transformers import (ElectraTokenizerFast,
                          ElectraForSequenceClassification)



class SentimentClassifier():
    LABELS = ['분노', '슬픔', '불안', '당황', '상처', '기쁨', ]
    ID_LABELS = {idx: key for (idx, key) in enumerate(LABELS)}
    MAX_LEN = 512


    def __init__(self):
        self.device = torch.device('cuda')
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