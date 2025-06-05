import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = 'gemini-2.5-flash-preview-05-20'
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
TEMPERATURE = 0.2
MAX__OUT_TOKENS = 3000
