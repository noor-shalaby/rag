from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from retrieve import retrieve_medical_context

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = genai.Client()

class QueryRequest(BaseModel):
    query: str
    patient_context: str = ""

def generate_medical_answer(query: str) -> str:
    # 1. Retrieval
    raw_results = retrieve_medical_context(query, match_threshold=0.0, match_count=5)
    context_text = "\n\n".join([r.get("content", "") for r in raw_results])

    # 2. Strict, Clinical-Grade Prompt
    prompt = f"""
    You are a rigorous, empathetic clinical assistant. Use the provided context to answer the user's question.

    CRITICAL SAFETY RULES:
    - If the context doesn't cover the answer, state that clearly.
    - If the user describes emergency symptoms (pain, trauma), emphasize seeking immediate care.
    - Do NOT guess or provide medical advice outside of the provided literature.

    FORMATTING:
    - Use HTML (<h3>, <p>, <ul>, <li>).
    - If the user question is simple, be concise.
    - If it's a medical condition, use headers: "Safety & Action", "Clinical Overview", "Professional Advice".
    - If the user uses Arabic, translate everything to natural, high-quality Arabic.

    Context: {context_text}
    Question: {query}
    """

    response = client.models.generate_content(model="gemini-3.5-flash", contents=prompt)

    # Disclaimer block
    disclaimer = """<div style='background-color: #fff3cd; padding: 10px; margin-bottom: 10px; border-radius: 5px; font-size: 0.8rem; border-left: 4px solid #ffeeba;'>
    <strong>⚠️ IMPORTANT:</strong> This is an AI assistant for educational purposes, not a doctor. In an emergency, seek professional care immediately.
    </div>"""

    return disclaimer + str(response.text)

@app.post("/ask")
async def ask_endpoint(payload: QueryRequest):
    try:
        return {"answer": generate_medical_answer(payload.query.strip())}
    except Exception as e:
        print(f"Error: {e}")
        return {"answer": "<p>Sorry, I'm having trouble retrieving clinical information right now. Please try again.</p>"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
