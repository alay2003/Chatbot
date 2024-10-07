# chatbot.py
# Main chatbot manager class that handles the conversational flow, including intent detection,
# entity extraction, and response generation.

import spacy
import logging
from dialog_manager import DialogManager
from entity_extractor import EntityExtractor
from nlg_manager import GPT2NLGManager

logging.basicConfig(level=logging.DEBUG)

class ChatbotManager:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.dialog_manager = DialogManager()
        self.entity_extractor = EntityExtractor(self.nlp)
        self.nlg_manager = GPT2NLGManager()

    def generate_response(self, user_input):
        try:
            doc = self.nlp(user_input)
            entities = self.entity_extractor.extract_entities(doc)
            logging.debug(f"Extracted entities: {entities}")
            dialog_state, response = self.dialog_manager.handle_dialog(user_input, entities)
            logging.debug(f"Dialog state: {dialog_state}, Response: {response}")
            return self.nlg_manager.generate_response(response, dialog_state)

        except Exception as e:
            logging.error(f"Error in generating response: {str(e)}")
            return "I'm sorry, I didn't understand that."