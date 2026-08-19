from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from retrieve import retrieve_medical_context

# Initialize FastAPI app
app = FastAPI(
    title="Clinical RAG Assistant API",
    description="A hybrid medical RAG application powered by Gemini and Supabase",
    version="1.0.0"
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


@app.post("/ask")
async def ask_medical_question(payload: QueryRequest):
    """Endpoint to process a medical query through the RAG pipeline."""
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # 1. Retrieve candidates
        raw_results = retrieve_medical_context(query, match_threshold=0.0, match_count=10)

        context_blocks = []
        if raw_results:
            refined_results = rerank_chunks(query, raw_results)
            for res in refined_results:
                content = res.get("content", "")
                context_blocks.append(content)

        context_text = "\n\n---\n\n".join(context_blocks)

        # 2. Formulate prompt
        prompt = f"""
        You are a professional, empathetic, and rigorous clinical AI assistant.
        A user is asking a health-related question. Provide a comprehensive, expertly structured response.

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

        response = client.models.generate_content(  # pyright: ignore[reportUnknownMemberType]
            model="gemini-3.5-flash", contents=prompt
        )

        assert response.text is not None, "No text generated"

        # 3. Add medical disclaimer
        disclaimer = (
            "⚠️ **IMPORTANT MEDICAL DISCLAIMER:** *This AI assistant provides educational insights synthesized "
            "from clinical literature and general medical knowledge. It is **not** a substitute for professional medical advice, "
            "diagnosis, or treatment. If you are experiencing a medical emergency, please contact your local emergency services "
            "or visit the nearest emergency room immediately.*\n\n---\n\n"
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
