# load_model.py
from transformers import BertForSequenceClassification, BertTokenizer
import os

def save_model_and_tokenizer(model, tokenizer, model_path):
    # Save model and tokenizer
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

def load_model(model_path):
    # Check if the model path exists
    if not os.path.exists(model_path):
        raise ValueError(f"Model path {model_path} does not exist")

    # Load model and tokenizer
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return model, tokenizer

if __name__ == "__main__":
    model_path = r"C:\Users\alayp\Desktop\Chatbot 1\combined_model"
    try:
        model, tokenizer = load_model(model_path)
        print("Combined model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    model_path = r"C:\Users\alayp\Desktop\Chatbot 1\results_wikiqa"
    try:
        model, tokenizer = load_model(model_path)
        print("Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")