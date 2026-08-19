import os
from typing import Any, cast

from dotenv import load_dotenv
from google import genai
from supabase import Client, create_client

# Load environment variables
_ = load_dotenv()

# Initialize Clients
client = genai.Client()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")

assert SUPABASE_URL is not None, "SUPABASE_URL is missing in .env"

selected_key = SERVICE_ROLE_KEY if SERVICE_ROLE_KEY else ANON_KEY
assert selected_key is not None, "No Supabase key found in .env"

supabase: Client = create_client(SUPABASE_URL, selected_key)


def retrieve_medical_context(
    query: str, match_threshold: float = 0.0, match_count: int = 10
) -> list[dict[str, Any]]:
    """Embeds a query and retrieves the top N documents from Supabase with zero filtering."""

    # 1. Embed the query using gemini-embedding-2
    response = client.models.embed_content(  # pyright: ignore[reportUnknownMemberType]
        model="gemini-embedding-2", contents=query
    )

    assert response.embeddings is not None, "No embeddings returned"
    first_embed = response.embeddings[0]
    assert first_embed is not None and first_embed.values is not None

    embedding_vector: list[float] = [float(val) for val in first_embed.values]

    # 2. Call the Supabase RPC function using match_threshold=0.0 to bypass filtering
    result = supabase.rpc(
        "match_medical_documents",
        {
            "query_embedding": embedding_vector,
            "match_threshold": match_threshold,
            "match_count": match_count,
        },
    ).execute()

    # Cast result data to satisfy type checking safely
    return cast(list[dict[str, Any]], result.data)


if __name__ == "__main__":
    user_query = input("Ask a medical question: ")

    print(f"\nFetching top matching chunks for: '{user_query}'...")
    results = retrieve_medical_context(user_query, match_threshold=0.0, match_count=5)

    if not results:
        print("No information found in database.")
    else:
        for i, res in enumerate(results):
            similarity = float(str(res.get("similarity", 0.0)))
            content = str(res.get("content", ""))
            metadata = res.get("metadata", {})

            source = "Unknown"
            if isinstance(metadata, dict):
                source = str(metadata.get("source", "Unknown"))

            print(f"\n--- Result {i+1} (Similarity: {similarity:.4f}) ---")
            print(f"Content: {content}...")
            print(f"Source: {source}")
