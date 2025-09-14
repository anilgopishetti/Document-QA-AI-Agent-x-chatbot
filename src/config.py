from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_store")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set — please configure .env")
