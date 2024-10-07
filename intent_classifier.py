# intent_classifier.py - Contains the BertIntentClassifier class that uses a pre-trained BERT model for intent classification.

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import logging

logging.basicConfig(level=logging.DEBUG)

class BertIntentClassifier:
    def __init__(self, model_name='bert-base-uncased', model_path=None):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        if model_path:
            self.model = BertForSequenceClassification.from_pretrained(model_path)
        else:
            self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.model.eval()

    def detect_intent(self, query):
        try:
            logging.debug(f"Tokenizing query: {query}")
            inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True)
            logging.debug(f"Tokenized inputs: {inputs}")

            outputs = self.model(**inputs)
            logging.debug(f"Model outputs: {outputs}")

            predicted_class = torch.argmax(outputs.logits).item()
            logging.debug(f"Predicted intent class for '{query}': {predicted_class}")
            return predicted_class

        except Exception as e:
            logging.error(f"Error detecting intent for query '{query}': {str(e)}")
            return None