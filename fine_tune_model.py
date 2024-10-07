# fine_tune_model.py
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizer
from data_loader import WikiQADataLoader
from sklearn.utils.class_weight import compute_class_weight
from transformers import EarlyStoppingCallback
import torch

def main():
    data_loader = WikiQADataLoader()
    train_dataset, val_dataset = data_loader.tokenize()

    model_save_path = r"C:\Users\alayp\Desktop\Chatbot 1\results"
    
    model = BertForSequenceClassification.from_pretrained(model_save_path)
    tokenizer = BertTokenizer.from_pretrained(model_save_path)

    # Debug: Print the structure of train_dataset
    print("Sample from train_dataset:", train_dataset[0])

    # Ensure 'labels' key exists in train_dataset
    if 'labels' not in train_dataset[0]:
        raise KeyError("The 'labels' key is missing from the train_dataset examples.")

    train_labels = [example['labels'] for example in train_dataset]
    class_weights = compute_class_weight('balanced', classes=[0, 1], y=train_labels)
    class_weights_tensor = torch.tensor(class_weights).to('cuda')  # Assuming using CUDA

if __name__ == "__main__":
    main()