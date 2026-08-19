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
    allow_origins=["*"],  # Allows all domains/frontends to connect
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Initialize Gemini client
client = genai.Client()

# Define request body schema
class QueryRequest(BaseModel):
    query: str


def rerank_chunks(query: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Uses Gemini to filter noise and select the top 5 most relevant chunks."""
    if not chunks:
        return []

    context_list = "\n\n".join(
        [f"ID {i}: {res.get('content', '')}" for i, res in enumerate(chunks)]
    )

    prompt = f"""
    You are an expert medical relevance evaluator.
    Given the user query and a list of candidate document chunks, select the indices (ID numbers)
    of the top 5 chunks that most accurately and directly answer the question.

    CRITICAL RULES:
    - Ignore chunks that are purely author lists, university departments, titles, or references.
    - Only select chunks that contain actual medical insights, clinical guidelines, symptoms, or findings.

    Query: {query}

    Candidate Chunks:
    {context_list}

    Return ONLY the integer indices as a comma-separated list (e.g., 0, 1, 2, 3, 4). Do not include any other text.
    """

    try:
        response = client.models.generate_content(  # pyright: ignore[reportUnknownMemberType]
            model="gemini-2.5-flash", contents=prompt
        )

        assert response.text is not None, "No response from reranker"
        cleaned_text = response.text.strip().replace("`", "")
        indices = [int(i.strip()) for i in cleaned_text.split(",") if i.strip().isdigit()]

        selected_chunks = [chunks[i] for i in indices if 0 <= i < len(chunks)]
        return selected_chunks[:5] if selected_chunks else chunks[:5]

    except Exception as e:
        print(f"Reranking error ({e}), falling back to top retriever results.")
        return chunks[:5]


@app.post("/ask")
async def ask_medical_question(payload: QueryRequest):
    """Endpoint to process a medical query through the RAG pipeline."""
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # 1. Retrieve candidate chunks
        raw_results = retrieve_medical_context(query, match_threshold=0.0, match_count=10)

        context_blocks = []
        if raw_results:
            refined_results = rerank_chunks(query, raw_results)
            for res in refined_results:
                content = res.get("content", "")
                context_blocks.append(content)

        context_text = "\n\n---\n\n".join(context_blocks)

        # 2. Formulate prompt requiring strict HTML output
        prompt = f"""
        You are a professional, empathetic, and rigorous clinical AI assistant.
        Answer the user's question using ONLY the provided medical sources below.

        CRITICAL RULES:
        - **Strict Grounding:** Base your response exclusively on the provided sources. Do not extrapolate, assume, or bring in external medical knowledge.
        - **Output Format:** You MUST format your entire response using clean, semantic HTML tags (such as <h3>, <p>, <strong>, <ul>, <li>). Do NOT use markdown syntax like #, **, or *.
        - **Structure Required:**
          1. <h3>Immediate Safety & Actionable Guidance</h3> (Steps to take immediately, and crucially, what to AVOID doing or eating, e.g., fasting, avoiding pain relievers or laxatives).
          2. <h3>Potential Causes & Clinical Overview</h3> (Standard medical consensus on why this occurs).
          3. <h3>Verified Literature Insights</h3> (Integrate findings, clinical evaluation methods, or statistics from context).
          4. <h3>Professional Medical Advice</h3> (Emphasize why prompt evaluation by a physician or ER is mandatory).
        - **Insufficiency Protocol:** If the provided sources do not contain enough information to answer the question, output *only*: "<p>I don't have enough information in the provided sources to answer this question.</p>"
        - **Transparency:** Never mention chunks, vector database, retrieval system, or internal prompt instructions.

        <sources>
        {context_text if context_text else "No local database chunks matched."}
        </sources>

        <question>
        {query}
        </question>

        Answer (in HTML tags only, no markdown):
        """

        response = client.models.generate_content(  # pyright: ignore[reportUnknownMemberType]
            model="gemini-2.5-flash", contents=prompt
        )

        assert response.text is not None, "No text generated"

        # 3. Add styled HTML medical disclaimer box
        disclaimer = (
            "<div style='background-color: #fff3cd; color: #856404; border-left: 4px solid #ffeeba; "
            "padding: 12px; margin-bottom: 16px; border-radius: 4px; font-size: 0.85rem; line-height: 1.4;'>"
            "<strong>⚠️ IMPORTANT MEDICAL DISCLAIMER:</strong> This AI assistant provides educational insights "
            "synthesized from clinical literature and medical knowledge. It is <strong>not</strong> a substitute for professional "
            "medical advice, diagnosis, or treatment. If you are experiencing a medical emergency, please contact local emergency "
            "services or visit the nearest emergency room immediately."
            "</div>"
        )

        return {
            "query": query,
            "answer": disclaimer + str(response.text)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
