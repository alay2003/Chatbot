# app.py
# Entry point of the application. It creates a Flask app and defines routes for handling requests,
# specifically for the chatbot interaction through the /chat endpoint.

from flask import Flask, request, jsonify
import logging

# Create a logger for better debugging
logging.basicConfig(level=logging.DEBUG)

def create_app():
    app = Flask(__name__)

    @app.route('/chat', methods=['POST'])
    def handle_chat():
        logging.debug("Received a POST request on /chat")
        from chatbot import ChatbotManager  # Lazy import to avoid circular dependency
        user_input = request.json.get('message')
        logging.debug(f"User input received: {user_input}")
        
        chatbot_manager = ChatbotManager()
        response = chatbot_manager.generate_response(user_input)
        logging.debug(f"Generated response: {response}")
        
        return jsonify({'response': response})

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)