# app/__init__.py

from flask import Flask
from .routes import app_bp  # Import the routes blueprint
from .chatbot import ChatbotManager  # Import the ChatbotManager

def create_app():
    app = Flask(__name__)

    # Initialize the ChatbotManager
    chatbot_manager = ChatbotManager()  # Store the chatbot manager in the app context

    # Register the routes blueprint
    app.register_blueprint(app_bp)

    return app
