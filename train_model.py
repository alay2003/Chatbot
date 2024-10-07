import os
import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForQuestionAnswering, Trainer, TrainingArguments

# Set the environment variable to use only the NVIDIA GPU (GPU 0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Check if CUDA is available and print the selected device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No CUDA device available. Using CPU.")

def preprocess_function(examples, tokenizer):
    """Preprocesses the dataset by tokenizing the input and setting start and end positions for answers."""
    inputs = tokenizer(
        examples['question'],
        examples['context'],
        truncation=True,
        padding='max_length',
        max_length=512
    )

    start_positions = []
    end_positions = []

    for i in range(len(examples['answers'])):
        if len(examples['answers'][i]['answer_start']) > 0:
            start_positions.append(int(examples['answers'][i]['answer_start'][0]))
            end_positions.append(start_positions[i] + len(examples['answers'][i]['text'][0]))
        else:
            start_positions.append(0)
            end_positions.append(0)

    inputs['start_positions'] = start_positions
    inputs['end_positions'] = end_positions
    return inputs

def load_squad_dataset(tokenizer_name='bert-base-uncased'):
    """Loads and preprocesses the SQuAD dataset."""
    squad_dataset = load_dataset('squad')
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    # Tokenize and preprocess the SQuAD dataset
    tokenized_squad = squad_dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)
    tokenized_squad.set_format('torch', columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions'])

    return tokenized_squad, tokenizer

def load_and_train_model(pretrained_model_path, dataset, tokenizer):
    """Loads the pre-trained model and trains it on the SQuAD dataset."""
    # Load the pre-trained model
    model = BertForQuestionAnswering.from_pretrained(pretrained_model_path)
    model.to(device)  # Ensure the model is on the GPU

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="steps",
        per_device_train_batch_size=4,  # Adjust batch size according to GPU memory
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=500,
        eval_steps=100,
        metric_for_best_model="f1",
        load_best_model_at_end=True,
        fp16=True,  # Enable mixed precision for faster training on GPU
        report_to="none"  # Disable reporting to any platform
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer
    )

    # Train the model
    print("Training the model on SQuAD dataset...")
    trainer.train()

    # Save the model after training
    model_save_path = 'combined_model'
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model and tokenizer saved to {model_save_path}")

def train_on_squad():
    """Main function to load and train the model on SQuAD."""
    # Specify the path to the pre-trained model (from the WikiQA dataset, for example)
    pretrained_model_path = r"C:\Users\alayp\Desktop\Chatbot 1\results"  # Path to the pre-trained model

    # Load and preprocess the SQuAD dataset
    tokenized_squad, tokenizer = load_squad_dataset(tokenizer_name='bert-base-uncased')

    # Train the model on the SQuAD dataset
    load_and_train_model(pretrained_model_path, tokenized_squad, tokenizer)

if __name__ == "__main__":
    train_on_squad()
