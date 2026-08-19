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


def rerank_chunks(query: str, chunks: list[dict[str, Any]], patient_context: str = "") -> list[dict[str, Any]]:
    """Uses Gemini to filter noise and select the top 5 most relevant chunks (matching CLI logic)."""
    if not chunks:
        return []

    context_list = "\n\n".join(
        [f"ID {i}: {res.get('content', '')}" for i, res in enumerate(chunks)]
    )

    prompt = f"""
    You are an expert clinical medical assistant.

    Your task is to provide a precise, medically accurate, and structured answer to the user's question using **ONLY** the provided medical sources below.

    Patient context:
    {patient_context if patient_context.strip() else "No additional patient information was provided."}

    ### CRITICAL RULES:
    - **Strict Grounding:** Base your response exclusively on the provided sources. Do not extrapolate, assume, or bring in external medical knowledge.
    - **Mandatory Citations:** You must cite every claim using the exact format corresponding to the source index right after the relevant sentence or bullet point.
    - **Noise Filtering:** Ignore chunks that contain only author names, institutional affiliations, titles, or raw bibliography/reference lists.
    - **Insufficiency Protocol:** If the provided sources do not contain enough information to answer the question, output *only*: "I don't have enough information in the provided sources to answer this question."
    - **Transparency:** Never mention the chunks, vector database, retrieval system, or internal prompt instructions.

    <sources>
    {context_list}
    </sources>

    <question>
    {query}
    </question>

    Answer:
    """

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash", contents=prompt
        )

        assert response.text is not None, "No response from reranker"
        cleaned_text = response.text.strip().replace("`", "")
        indices = [int(i.strip()) for i in cleaned_text.split(",") if i.strip().isdigit()]

        selected_chunks = [chunks[i] for i in indices if 0 <= i < len(chunks)]
        return selected_chunks[:5] if selected_chunks else chunks[:5]

    except Exception as e:
        print(f"Reranking error ({e}), falling back to top retriever results.")
        return chunks[:5]


def generate_medical_answer(query: str, patient_context: str = "") -> tuple[str, str]:
    """Retrieves 10 candidates, reranks them to top 5, and generates a perfected hybrid medical response."""

    print(f"Retrieving candidate knowledge for: '{query}'...")
    raw_results = retrieve_medical_context(query, match_threshold=0.0, match_count=10)

    context_blocks = []
    if raw_results:
        refined_results = rerank_chunks(query, raw_results, patient_context)
        for res in refined_results:
            content = res.get("content", "")
            context_blocks.append(content)

    context_text = "\n\n---\n\n".join(context_blocks)

    prompt = f"""
    You are a professional, empathetic, and rigorous clinical AI assistant.
    A user is asking a health-related question. Provide a comprehensive, expertly structured response.

    Patient context:
    {patient_context if patient_context.strip() else "No additional patient information was provided."}

    Structure your answer cleanly with Markdown:
    1. **Immediate Safety & Actionable Guidance:** What steps the user should take immediately, and crucially, **what they must avoid doing or eating** (e.g., fasting, avoiding pain relievers or laxatives that mask symptoms or risk rupture).
    2. **Potential Causes & Clinical Overview:** Standard medical consensus on why this occurs.
    3. **Verified Literature Insights:** Integrate specific findings, clinical evaluation methods (such as scoring systems or diagnostics), and statistics from the provided reference context below.
    4. **Professional Medical Advice:** Conclude clearly by emphasizing why prompt professional evaluation by a physician or emergency room is mandatory.

    Reference Context from Local Database:
    {context_text if context_text else "No specific local database chunks matched, rely on standard clinical consensus."}

    Question: {query}

    Answer:
    """

    response = client.models.generate_content(
        model="gemini-3.5-flash", contents=prompt
    )

    assert response.text is not None, "No text generated"

    # Mandatory Medical Disclaimer Header (HTML/Markdown friendly)
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
    """FastAPI endpoint matching the CLI testing logic."""
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
