"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Council members - list of OpenRouter model identifiers (free tier)
COUNCIL_MODELS = [
    "google/gemma-3-27b-it:free",
    "qwen/qwen-2.5-coder-32b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free"
]

# Chairman model - synthesizes final response (free tier)
CHAIRMAN_MODEL = "google/gemini-2.0-pro-exp-02-05:free"
# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Data directory for conversation storage
DATA_DIR = "data/conversations"
