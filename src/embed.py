import os

from dotenv import load_dotenv
from google import genai
from supabase import Client, create_client

# Load environment variables cleanly
load_dotenv()  # pyright: ignore[reportUnusedCallResult]

# Initialize Clients
client = genai.Client()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")

assert SUPABASE_URL is not None, "SUPABASE_URL is missing in .env"

# Choose service role key if available, otherwise fall back to anon key
if SERVICE_ROLE_KEY:
    print("Using Supabase Service Role Key (Bypassing RLS)...")
    selected_key = SERVICE_ROLE_KEY
elif ANON_KEY:
    print("Service role key not found. Falling back to Supabase Anon Key...")
    selected_key = ANON_KEY
else:
    raise ValueError(
        "Neither SUPABASE_SERVICE_ROLE_KEY nor SUPABASE_ANON_KEY is set in .env!"
    )

supabase: Client = create_client(SUPABASE_URL, selected_key)


def store_chunks_to_supabase(chunks_data: list[dict[str, str | int]]) -> None:
    """Takes pre-chunked data, generates embeddings via gemini-embedding-2,

    and saves both the text chunk and vector in the same Supabase table row.
    """
    print(f"Processing {len(chunks_data)} chunks for embedding and storage...")

    for i, item in enumerate(chunks_data):
        chunk_text = str(item["content"])

        metadata: dict[str, str | int] = {
            "source": str(item.get("source", "unknown")),
            "page": int(str(item.get("page", 1))),
        }

        try:
            # Generate embedding using gemini-embedding-2
            response = client.models.embed_content(  # pyright: ignore[reportUnknownMemberType]
                model="gemini-embedding-2", contents=chunk_text
            )

            assert (
                response.embeddings is not None
            ), "No embeddings returned from Gemini API"
            first_embedding = response.embeddings[0]
            assert first_embedding is not None, "First embedding object is None"
            assert (
                first_embedding.values is not None
            ), "Embedding values are None"

            embedding_vector: list[float] = [
                float(val) for val in first_embedding.values
            ]

            row_data: dict[str, str | int | dict[str, str | int] | list[float]] = {
                "content": chunk_text,
                "metadata": metadata,
                "embedding": embedding_vector,
            }

            _ = supabase.table("medical_documents").insert(row_data).execute()
            print(f"Stored chunk {i + 1}/{len(chunks_data)} successfully.")

        except Exception as e:  # noqa: BLE001
            print(f"Error processing chunk {i + 1}: {e}")

    print("All chunks successfully embedded and stored in Supabase!")


# --- Example Execution ---
if __name__ == "__main__":
    my_preprocessed_chunks: list[dict[str, str | int]] = [
        {
            "content": "Patients should avoid heavy lifting for 2-4 weeks after laparoscopic appendectomy.",
            "source": "recovery_guide.pdf",
            "page": 2,
        },
        {
            "content": "Encourage early ambulation post-surgery to reduce the risk of thromboembolism.",
            "source": "recovery_guide.pdf",
            "page": 3,
        },
    ]

    store_chunks_to_supabase(my_preprocessed_chunks)
