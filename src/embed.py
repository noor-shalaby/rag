import json
import os
import time
from typing import TypedDict, Any

from dotenv import load_dotenv
from google import genai
from supabase import Client, create_client

# Define the structure of your raw chunk data
class ChunkData(TypedDict):
    content: str
    metadata: dict[str, Any]

# Load environment variables
_ = load_dotenv()

# Initialize Clients
client = genai.Client()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")

assert SUPABASE_URL is not None, "SUPABASE_URL is missing in .env"

if SERVICE_ROLE_KEY:
    selected_key = SERVICE_ROLE_KEY
elif ANON_KEY:
    selected_key = ANON_KEY
else:
    raise ValueError("No Supabase key found in .env")

supabase: Client = create_client(SUPABASE_URL, selected_key)

def store_chunks_to_supabase(chunks_data: list[ChunkData]) -> None:
    """Embeds and stores chunks, skipping duplicates to save API quota."""
    print(f"Processing {len(chunks_data)} chunks...")

    for i, item in enumerate(chunks_data):
        chunk_text = item["content"]
        metadata = item["metadata"]
        chunk_id = metadata.get("chunk_id")

        # 1. Check if chunk already exists to save quota
        if chunk_id:
            existing = (
                supabase.table("medical_documents")
                .select("id")
                .eq("metadata->>chunk_id", str(chunk_id))
                .execute()
            )
            if existing.data:
                print(f"[{i+1}/{len(chunks_data)}] Skipping {chunk_id}: Already exists.")
                continue

        # 2. Proceed with embedding if not found
        try:
            response = client.models.embed_content(  # pyright: ignore[reportUnknownMemberType]
                model="gemini-embedding-2", contents=chunk_text
            )

            assert response.embeddings is not None, "No embeddings returned"
            values = response.embeddings[0].values
            assert values is not None

            embedding_vector: list[float] = [float(val) for val in values]

            row_data: dict[str, Any] = {
                "content": chunk_text,
                "metadata": metadata,
                "embedding": embedding_vector,
            }

            _ = supabase.table("medical_documents").insert(row_data).execute()
            print(f"[{i+1}/{len(chunks_data)}] Stored {chunk_id} successfully.")

            # 3. Pause to respect rate limits
            time.sleep(1.5)

        except Exception as e:  # noqa: BLE001
            print(f"Error processing chunk {i + 1}: {e}")

if __name__ == "__main__":
    data_folder = "chunks"

    # Process files in alphabetical order
    files = sorted(os.listdir(data_folder))

    for filename in files:
        if filename.endswith(".json"):
            file_path = os.path.join(data_folder, filename)
            print(f"\n--- Processing file: {filename} ---")

            with open(file_path, "r", encoding="utf-8") as f:
                file_data: list[ChunkData] = json.load(f)  # type: ignore[reportAny]
                store_chunks_to_supabase(file_data)

    print("\nAll tasks complete!")
