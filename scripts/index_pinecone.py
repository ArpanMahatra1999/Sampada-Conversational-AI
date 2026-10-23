
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.json"

BATCH_SIZE = 5
NAMESPACE = "sampada-sites"
MAX_RETRY_ATTEMPTS = 6
INITIAL_RETRY_DELAY = 15
MAX_RETRY_DELAY = 120
BATCH_PAUSE_SECONDS = 5


def upsert_with_retry(index, records, namespace):
    """Upload a batch, retrying rate-limit errors with backoff."""

    for attempt in range(MAX_RETRY_ATTEMPTS):
        try:
            return index.documents.upsert(
                namespace=namespace,
                documents=records,
            )

        except Exception as exc:
            error_text = str(exc)

            is_rate_limit = (
                "429" in error_text
                or "RateLimitError" in type(exc).__name__
            )

            if not is_rate_limit:
                raise

            if attempt == MAX_RETRY_ATTEMPTS - 1:
                raise

            delay = min(
                INITIAL_RETRY_DELAY * (2 ** attempt),
                MAX_RETRY_DELAY,
            )

            print(
                f"Rate limit reached. Waiting {delay} seconds "
                f"before retry {attempt + 2}/"
                f"{MAX_RETRY_ATTEMPTS}."
            )

            time.sleep(delay)


def main():
    load_dotenv(PROJECT_ROOT / ".env")

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv(
        "PINECONE_INDEX_NAME",
        "sampada-ai",
    )

    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is missing.")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}\n"
            "Run: python -m scripts.prepare_metadata"
        )

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        documents = json.load(file)

    if not isinstance(documents, list) or not documents:
        raise ValueError(
            "metadata.json must contain a non-empty list."
        )

    pinecone = Pinecone(api_key=pinecone_api_key)

    existing_indexes = pinecone.list_indexes().names()

    if index_name not in existing_indexes:
        raise ValueError(
            f"Pinecone index '{index_name}' does not exist. "
            "Create the index before running ingestion."
        )

    index = pinecone.Index(index_name)

    index_description = pinecone.indexes.describe(
        name=index_name
    )

    schema = index_description.schema

    if "chunk_text" not in schema.fields:
        raise ValueError(
            "The index schema is missing the 'chunk_text' field."
        )

    print(f"Index: {index_name}")
    print(f"Namespace: {NAMESPACE}")
    print(f"Documents to process: {len(documents)}")
    print(f"Batch size: {BATCH_SIZE}")

    uploaded = 0

    for start in range(0, len(documents), BATCH_SIZE):
        batch = documents[start:start + BATCH_SIZE]
        records = []

        for document in batch:
            if "id" not in document or "text" not in document:
                raise ValueError(
                    "Each document must contain 'id' and 'text'."
                )

            site_id = str(document["id"])
            text = document["text"]
            metadata = document.get("metadata", {})

            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Document {site_id} has empty searchable text."
                )

            if not isinstance(metadata, dict):
                raise ValueError(
                    f"Document {site_id} has invalid metadata."
                )

            # Match the existing Pinecone document schema.
            record = {
                "_id": site_id,
                "chunk_text": text,
                "title": str(
                    metadata.get(
                        "name",
                        "Unknown heritage site",
                    )
                ),
            }

            # Include additional scalar metadata fields.
            # Complex values are serialized as JSON strings.
            for key, value in metadata.items():
                if key in {"title", "chunk_text"}:
                    continue

                if value is None:
                    continue

                if isinstance(value, (dict, list)):
                    value = json.dumps(
                        value,
                        ensure_ascii=False,
                    )

                if isinstance(value, (str, int, float, bool)):
                    record[key] = value

            records.append(record)

        response = upsert_with_retry(
            index=index,
            records=records,
            namespace=NAMESPACE,
        )

        batch_count = getattr(
            response,
            "upserted_count",
            len(records),
        )

        uploaded += batch_count

        print(
            f"Processed batch: {start + 1}-"
            f"{start + len(batch)} of {len(documents)} "
            f"(reported upserted: {batch_count})"
        )

        if start + BATCH_SIZE < len(documents):
            time.sleep(BATCH_PAUSE_SECONDS)

    print("\nIngestion requests completed.")
    print(f"Index: {index_name}")
    print(f"Namespace: {NAMESPACE}")
    print(f"Reported documents uploaded: {uploaded}")


if __name__ == "__main__":
    main()
