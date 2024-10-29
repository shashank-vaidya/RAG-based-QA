import os
import openai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

def load_openai_client():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    return openai

def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
