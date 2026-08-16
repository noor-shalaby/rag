import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Load environment variables from .env
load_dotenv()

# Verify API key is present
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in environment variables or .env file.")

# Global configuration constants
MODEL_NAME = "gemini-3.5-flash"
EMBEDDING_MODEL = "models/text-embedding-004"

# Reusable LLM and Embedding instances
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.3)
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

if __name__ == "__main__":
    response = llm.invoke("Hello Gemini! Give me a one-sentence tip for building a RAG system.")
    print(response.content)
