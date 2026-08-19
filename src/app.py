from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from retrieve import retrieve_medical_context

# Initialize FastAPI app
app = FastAPI(
    title="Clinical RAG Assistant API",
    description="A hybrid medical RAG application powered by Gemini and Supabase",
    version="1.0.0"
)

# Enable CORS for frontend communication
origins = [
    "https://your-frontend-domain.com",
    "http://localhost:5500",             # For local development (e.g., Live Server)
    "http://localhost:3000",             # For local React/Node development if applicable
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# Initialize Gemini client
client = genai.Client()

class QueryRequest(BaseModel):
    query: str
    patient_context: str = ""


def generate_medical_answer(query: str, patient_context: str = "") -> str:
    """Retrieves 5 candidates and generates a structured HTML medical response."""

    print(f"Retrieving candidate knowledge for: '{query}'...")
    raw_results = retrieve_medical_context(query, match_threshold=0.5, match_count=5)

    context_blocks = []
    if raw_results:
        for res in raw_results:
            content = res.get("content", "")
            context_blocks.append(content)

    context_text = "\n\n---\n\n".join(context_blocks)

    prompt = f"""
    You are a professional, empathetic, and rigorous clinical AI assistant.
    A user is asking a health-related question. Provide a comprehensive, expertly structured response.

    Patient context:
    {patient_context if patient_context.strip() else "No additional patient information was provided."}

    IMPORTANT FORMATTING RULE:
    You MUST output your entire response using clean, semantic HTML tags (such as <h3>, <p>, <strong>, <ul>, <li>). Do NOT use Markdown syntax like #, **, or *.

    Reference Context from Local Database:
    {context_text if context_text else "No specific local database chunks matched, rely on standard clinical consensus."}

    Question: {query}

    Answer (in HTML tags only, no markdown):
    """

    response = client.models.generate_content(
        model="gemini-3.5-flash", contents=prompt
    )

    assert response.text is not None, "No text generated"

    return str(response.text)


@app.post("/ask")
async def ask_endpoint(payload: QueryRequest):
    """FastAPI endpoint matching the CLI testing logic with HTML formatting."""
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        answer_text = generate_medical_answer(query, payload.patient_context)
        return {
            "query": query,
            "answer": answer_text
        }
    except Exception as e:
        print(f"Error handling query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
