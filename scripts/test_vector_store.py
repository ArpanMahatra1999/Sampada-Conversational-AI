
import os
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "sampada-ai")
NAMESPACE = "sampada-sites"
TOP_K = 5


def main():
    # 1. Validate configuration.
    api_key = os.getenv("PINECONE_API_KEY")

    if not api_key:
        raise ValueError(
            "PINECONE_API_KEY is missing from the .env file."
        )

    # 2. Connect to Pinecone.
    pinecone = Pinecone(api_key=api_key)
    index = pinecone.Index(INDEX_NAME)

    # 3. Verify the number of stored records.
    stats = index.describe_index_stats()
    namespaces = getattr(stats, "namespaces", {}) or {}
    namespace_stats = namespaces.get(NAMESPACE, {})
    vector_count = getattr(
        namespace_stats,
        "vector_count",
        0,
    )

    print("=" * 55)
    print("SAMPADA AI VECTOR STORE VERIFICATION")
    print("=" * 55)
    print(f"Index: {INDEX_NAME}")
    print(f"Namespace: {NAMESPACE}")
    print(f"Stored records: {vector_count}")

    if not vector_count:
        raise RuntimeError(
            f"No records found in namespace '{NAMESPACE}'."
        )

    # 4. Search using Pinecone's Documents API.
    query = (
        "Ancient historic city with temples, monuments, "
        "and important cultural heritage"
    )

    print(f"\nSearch query: {query}")
    print(f"Requested results: {TOP_K}")

    results = index.documents.search(
        namespace=NAMESPACE,
        top_k=TOP_K,
        score_by=[
            {
                "type": "embed",
                "field": "chunk_text",
                "query": query,
            }
        ],
        include_fields=["title", "chunk_text"],
    )

    # 5. Display search results.
    hits = results.matches

    if not hits:
        print("\nNo search results returned.")
        return

    print(f"\nTop {len(hits)} search results:\n")

    for position, hit in enumerate(hits, start=1):
        site_id = getattr(hit, "id", "Unknown")
        score = getattr(hit, "score", None)
        title = getattr(
            hit,
            "title",
            "Unknown heritage site",
        )
        text = getattr(hit, "chunk_text", "")

        print(f"{position}. {title}")
        print(f"   Site ID: {site_id}")

        if score is not None:
            print(f"   Similarity score: {score}")

        if text:
            preview = str(text).replace("\n", " ")[:250]
            print(f"   Text preview: {preview}...")

        print()

    print("=" * 55)
    print("VECTOR STORE VERIFICATION COMPLETED")
    print("=" * 55)


if __name__ == "__main__":
    main()
