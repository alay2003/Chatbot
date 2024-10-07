from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging

class NLGManager:
    def generate_response(self, dialog_response, dialog_state):
        # Basic response generation (used if GPT-2 isn't available)
        return dialog_response

class GPT2NLGManager(NLGManager):
    def __init__(self):
        # Initialize GPT-2 model and tokenizer
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt2_model.config.pad_token_id = self.gpt2_model.config.eos_token_id
        
        # Add a padding token if it doesn't exist in GPT-2 tokenizer
        if self.gpt2_tokenizer.pad_token is None:
            self.gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # Resize model embeddings after adding special tokens
        self.gpt2_model.resize_token_embeddings(len(self.gpt2_tokenizer))

    def generate_response(self, dialog_response, dialog_state):
        # Prepare the input text for GPT-2 generation
        input_text = dialog_response
        inputs = self.gpt2_tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
        
        # Generate the response using GPT-2
        output = self.gpt2_model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=100,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
        
        # Decode and return the generated response
        return self.gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
