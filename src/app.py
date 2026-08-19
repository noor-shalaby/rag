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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini client
client = genai.Client()

class QueryRequest(BaseModel):
    query: str
    patient_context: str = ""


def generate_medical_answer(query: str, patient_context: str = "") -> tuple[str, str]:
    """Retrieves 10 candidates, reranks them to top 5, and generates a structured HTML medical response."""

    print(f"Retrieving candidate knowledge for: '{query}'...")
    raw_results = retrieve_medical_context(query, match_threshold=0.0, match_count=10)

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

    Structure your answer cleanly:
    1. <h3>Immediate Safety & Actionable Guidance</h3>
       <p>What steps the user should take immediately, and crucially, <strong>what they must avoid doing or eating</strong> (e.g., fasting, avoiding pain relievers or laxatives that mask symptoms or risk rupture).</p>
    2. <h3>Potential Causes & Clinical Overview</h3>
       <p>Standard medical consensus on why this occurs.</p>
    3. <h3>Verified Literature Insights</h3>
       <p>Integrate specific findings, clinical evaluation methods, and statistics from the provided reference context below.</p>
    4. <h3>Professional Medical Advice</h3>
       <p>Conclude clearly by emphasizing why prompt professional evaluation by a physician or emergency room is mandatory.</p>

    Reference Context from Local Database:
    {context_text if context_text else "No specific local database chunks matched, rely on standard clinical consensus."}

    Question: {query}

    Answer (in HTML tags only, no markdown):
    """

    response = client.models.generate_content(
        model="gemini-3.5-flash", contents=prompt
    )

    assert response.text is not None, "No text generated"

    # Mandatory Medical Disclaimer Header styled for HTML
    disclaimer = (
        "<div style='background-color: #fff3cd; color: #856404; border-left: 4px solid #ffeeba; "
        "padding: 12px; margin-bottom: 16px; border-radius: 4px; font-size: 0.85rem; line-height: 1.4;'>"
        "<strong>⚠️ IMPORTANT MEDICAL DISCLAIMER:</strong> This AI assistant provides educational insights synthesized "
        "from clinical literature and general medical knowledge. It is <strong>not</strong> a substitute for professional medical advice, "
        "diagnosis, or treatment. If you are experiencing a medical emergency, please contact your local emergency services "
        "or visit the nearest emergency room immediately."
        "</div>"
    )

    return disclaimer, str(response.text)


@app.post("/ask")
async def ask_endpoint(payload: QueryRequest):
    """FastAPI endpoint matching the CLI testing logic with HTML formatting."""
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        disclaimer, answer_text = generate_medical_answer(query, payload.patient_context)
        return {
            "query": query,
            "answer": disclaimer + answer_text
        }
    except Exception as e:
        print(f"Error handling query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
