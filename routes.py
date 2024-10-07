# app/routes.py
# Contains the blueprint for chatbot routes, specifically handling POST requests to the /chat endpoint
# where user input is received and processed to generate a response.

from flask import Blueprint, jsonify, request
import logging
from app.chatbot import ChatbotManager

app_bp = Blueprint('chatbot', __name__)
chatbot_manager = ChatbotManager()

@app_bp.route('/chat', methods=['POST'])
def handle_chat():
    user_input = request.json.get('message')
    logging.debug(f"User input received: {user_input}")
    response = chatbot_manager.generate_response(user_input)
    return jsonify({'response': response})
