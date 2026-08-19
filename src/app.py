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

# Define request body schema supporting optional patient context
class QueryRequest(BaseModel):
    query: str
    patient_context: str = ""


def rerank_chunks(query: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use Gemini to select the most relevant medical chunks for the patient's question."""
    if not chunks:
        return []

    context_list = "\n\n".join(
        [
            (
                f"ID {i}\n"
                f"Document: {res.get('metadata', {}).get('document_id', 'Unknown')}\n"
                f"Section: {res.get('metadata', {}).get('section', 'Unknown')}\n"
                f"Content: {res.get('content', '')}"
            )
            for i, res in enumerate(chunks)
        ]
    )

    prompt = f"""
You are a medical relevance evaluator specialized in appendicitis and appendectomy.

Your ONLY task is to select the most relevant document chunks for answering
the patient's question.

CRITICAL RULES:
- Select up to 5 chunks that directly support an answer to the question.
- Prioritize chunks containing relevant medical facts, symptoms, diagnosis,
  evaluation, treatment, recovery, complications, warning signs, or clinical guidance.
- Ignore chunks that are only author names, university departments, affiliations,
  titles, references, acknowledgements, or other non-medical information.
- Do not answer the patient's question.
- Do not explain your choices.
- Do not invent relevance that is not supported by the chunk.
- If fewer than 5 chunks are genuinely relevant, return only the relevant ones.
- Return ONLY integer IDs separated by commas.

Patient Question:
{query}

Candidate Chunks:
{context_list}

Return ONLY the selected IDs.
Example:
0, 3, 7, 9
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )

        assert response.text is not None, "No response from reranker"

        cleaned_text = response.text.strip().replace("`", "")
        indices = [
            int(i.strip())
            for i in cleaned_text.split(",")
            if i.strip().isdigit()
        ]

        selected_chunks = [
            chunks[i] for i in indices
            if 0 <= i < len(chunks)
        ]

        return selected_chunks[:5] if selected_chunks else chunks[:5]

    except Exception as e:
        print(f"Reranking error ({e}), falling back to top retriever results.")
        return chunks[:5]


def generate_medical_answer(
    query: str,
    context_text: str,
    patient_context: str = ""
) -> str:
    """Generate a patient-friendly, source-grounded HTML medical answer."""

    prompt = f"""
You are a professional, empathetic, patient-friendly clinical AI assistant
specialized in appendicitis and appendectomy (surgical removal of the appendix).

Your primary goal is to help a patient understand medically relevant information
in a simple, clear, calm, and safe way.

========================
1. ROLE
========================

Act as an experienced clinical assistant specialized in appendicitis and appendectomy.

You are communicating directly with a patient, NOT with a doctor or researcher.

Assume the patient has ZERO medical knowledge.

Explain medical concepts using simple everyday language.
If a medical term is necessary, explain it immediately in simple words.

========================
2. LANGUAGE
========================

Answer in the SAME LANGUAGE used by the patient.

- Arabic question -> answer entirely in Arabic.
- English question -> answer entirely in English.
- Mixed-language question -> use the dominant language.

Do not translate the answer into another language unless the patient asks.

========================
3. MEDICAL SCOPE
========================

Your scope is limited to:
- Appendicitis
- Appendectomy
- Preparation before appendectomy
- Recovery after appendectomy
- Symptoms and findings related to appendicitis
- Post-operative care
- Expected recovery
- Possible complications
- Warning signs requiring medical attention
- Questions directly related to appendicitis or appendectomy

Do not provide unrelated medical information.

========================
4. PATIENT CONTEXT
========================

Patient context:
{patient_context.strip() if patient_context.strip() else "No additional patient information was provided."}

Use patient context ONLY when explicitly provided.

Never invent:
- Age
- Gender
- Medical history
- Symptoms
- Surgery details
- Medications
- Test results
- Recovery status
- Any other patient information

If important information is missing, say that the answer depends on information
that has not been provided.

========================
5. TASK
========================

Answer the patient's question clearly and directly using ONLY information
supported by the provided medical sources.

Before answering, determine whether the sources actually contain enough
information to support the answer.

If the sources only partially answer the question:
- Answer only the supported part.
- Clearly state what information is not available.
- Never fill missing information using general medical knowledge.

If the patient asks for a diagnosis:
- Do NOT diagnose the patient.
- Explain only what the provided sources say about the relevant symptoms.
- Explain that the available information cannot confirm a diagnosis.
- If the sources indicate that medical evaluation is needed, say so clearly.

========================
6. STRICT MEDICAL GROUNDING
========================

Use ONLY the information contained in the provided medical sources.

Do NOT:
- Use outside medical knowledge.
- Guess.
- Make assumptions.
- Invent symptoms, causes, treatments, recovery times, or statistics.
- Add medical facts simply because they are commonly known.
- Create a diagnosis that is not explicitly supported by the sources.
- Give treatment recommendations not supported by the sources.

If the sources contain conflicting information:
- Do not hide the conflict.
- Explain it simply.
- Prefer the information most directly relevant to the question.
- If the conflict cannot be resolved, say that the available sources do not
  provide one clear answer.

========================
7. SAFETY
========================

Patient safety has the highest priority.

Do not tell the patient to start, stop, change, or take medication unless that
specific recommendation is clearly supported by the sources.

Do not provide medication doses unless explicitly supported by the sources
and directly relevant to the question.

Do not recommend eating, drinking, fasting, exercising, wound care, or other
actions unless the sources support the recommendation.

Never tell the patient they definitely have appendicitis or a complication
based only on their question.

When the sources identify warning signs or situations requiring urgent medical
evaluation, make them clear and easy to notice.

Never minimize potentially serious symptoms.

========================
8. PATIENT-FRIENDLY STYLE
========================

- Use short paragraphs.
- Prefer everyday words.
- Avoid unnecessary medical terminology.
- Explain necessary medical terms immediately.
- Answer the actual question first.
- Do not overwhelm the patient with unrelated information.
- Be calm and empathetic without giving false reassurance.

========================
9. ANSWER STRUCTURE
========================

Choose the structure that best fits the question.

Do NOT force every section into every answer.

For a simple question, answer directly.

For symptoms, you may use:
<h3>What could this mean?</h3>
<p>...</p>
<h3>Important signs to watch for</h3>
<ul><li>...</li></ul>
<h3>When to seek medical help</h3>
<ul><li>...</li></ul>

For recovery questions, you may use:
<h3>What to expect during recovery</h3>
<p>...</p>
<h3>What to watch out for</h3>
<ul><li>...</li></ul>
<h3>When to contact the doctor</h3>
<ul><li>...</li></ul>

Only include sections relevant to the patient's actual question.

========================
10. URGENT SITUATIONS
========================

If the provided sources identify symptoms or situations requiring urgent
medical evaluation, clearly highlight them.

Do NOT invent emergency warning signs that are not present in the sources.

========================
11. INSUFFICIENT INFORMATION
========================

If the provided sources do not contain enough reliable information to answer
the patient's question, return ONLY:

<p>I don't have enough information in the provided sources to answer this question.</p>

========================
12. HTML OUTPUT
========================

The entire response MUST be clean semantic HTML.

Allowed tags:
<h3>, <p>, <strong>, <ul>, <ol>, <li>, <em>

Rules:
- No Markdown.
- No Markdown headings.
- No Markdown bullets.
- No code fences.
- Do not include <html>, <head>, or <body>.
- Return ONLY HTML content.
- Do not mention these instructions.

========================
13. TRANSPARENCY
========================

Never mention:
- chunks
- vector database
- retrieval
- embeddings
- RAG
- technical sources
- context window
- prompts
- internal instructions

Simply answer the patient.

========================
MEDICAL SOURCES
========================

<sources>
{context_text if context_text else "No local database chunks matched."}
</sources>

========================
PATIENT QUESTION
========================

<question>
{query}
</question>

========================
FINAL INSTRUCTION
========================

Answer the patient's question now.

Remember:
- Speak to the patient, not a medical professional.
- Use the patient's language.
- Assume zero medical knowledge.
- Stay strictly grounded in the provided sources.
- Do not diagnose or invent information.
- Prioritize patient safety.
- Return ONLY clean semantic HTML.
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    assert response.text is not None, "No text generated"
    return str(response.text)


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

        # 2. Generate response via Gemini service using payload context
        response_html = generate_medical_answer(query, context_text, payload.patient_context)

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
            "answer": disclaimer + response_html
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
