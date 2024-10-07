# dialog_manager.py
# Contains the DialogManager class, which manages the dialog state and generates responses based on user input
# and detected entities.

from intent_classifier import BertIntentClassifier
import logging

logging.basicConfig(level=logging.DEBUG)

class DialogManager:
    def __init__(self):
        self.dialog_state = {}
        self.intent_classifier = BertIntentClassifier()
        self.intent_responses = {
            0: "Hello! How can I assist you today?",
        }

    def handle_dialog(self, user_input, entities):
        try:
            intent = self.intent_classifier.detect_intent(user_input)
            self.dialog_state['intent'] = intent
            self.dialog_state['entities'] = entities
            response = self.generate_response(intent, entities)
            return self.dialog_state, response

        except Exception as e:
            logging.error(f"Error in handle_dialog: {str(e)}")
            return self.dialog_state, "I'm sorry, I didn't understand that."

    def generate_response(self, intent, entities):
        fallback_response = "I'm afraid I don't understand. Could you please rephrase your request?"
        response_template = self.intent_responses.get(intent, fallback_response)
        return response_template.format(location=entities.get('GPE', 'your location'))