import importlib
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

print("GEMINI_API_KEY present?", bool(os.getenv("GEMINI_API_KEY")))

# test library imports
reqs = ["pymupdf", "pdfplumber", "google.generativeai", "chromadb", "pandas"]
for r in reqs:
    try:
        importlib.import_module(r)
        print(f"{r}: OK")
    except Exception as e:
        print(f"{r}: ERROR - {e}")

# test Gemini API connection
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")  # lightweight model for test
    response = model.generate_content("Hello Gemini, test connection!")
    print("Gemini response:", response.text)
except Exception as e:
    print("Gemini test failed:", e)
