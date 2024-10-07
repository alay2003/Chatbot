from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import Dataset
import torch

# WikiQA Dataset Loader and Tokenization
class WikiQADataLoader:
    def __init__(self, tokenizer_name='bert-base-uncased'):
        self.dataset = load_dataset("microsoft/wiki_qa")
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    def tokenize(self, max_length=128):
        train_dataset = WikiQADataset(self.dataset['train'], self.tokenizer, max_length)
        val_dataset = WikiQADataset(self.dataset['validation'], self.tokenizer, max_length)
        return train_dataset, val_dataset

class WikiQADataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        inputs = self.tokenizer(
            item['question'],
            item['context'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        inputs['labels'] = torch.tensor(item['label'])
        return inputs

class SquadDataLoader:
    def __init__(self, tokenizer_name='bert-base-uncased'):
        self.squad_dataset = load_dataset('squad')
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    def preprocess_function(self, examples):
        inputs = self.tokenizer(
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
                start_positions.append(examples['answers'][i]['answer_start'][0])
                end_positions.append(examples['answers'][i]['answer_start'][0] + len(examples['answers'][i]['text'][0]))
            else:
                start_positions.append(0)
                end_positions.append(0)

        inputs['start_positions'] = start_positions
        inputs['end_positions'] = end_positions
        return inputs

    def load_and_preprocess(self):
        # Tokenize and preprocess the SQuAD dataset
        tokenized_squad = self.squad_dataset.map(self.preprocess_function, batched=True)
        tokenized_squad.set_format('torch', columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions'])
        return tokenized_squad, self.tokenizer

def load_squad_dataset(tokenizer_name='bert-base-uncased'):
    squad_loader = SquadDataLoader(tokenizer_name)
    return squad_loader.load_and_preprocess()