import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "data" / "sites.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "metadata.json"

TEXT_FIELDS = [
    "name",
    "short_description",
    "description",
    "justification",
    "cultural_criteria",
    "natural_criteria",
]


def to_text(value) -> str:
    """Convert JSON values to readable text."""
    if value is None:
        return ""

    if isinstance(value, (dict, list)):
        return json.dumps(
            value,
            ensure_ascii=False,
        )

    return str(value)


def metadata_value(value):
    """Convert values to Pinecone-compatible metadata."""
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (dict, list)):
        return json.dumps(
            value,
            ensure_ascii=False,
        )

    return str(value)


def extract_year(value):
    """Extract a four-digit inscription year when available."""
    if value is None:
        return None

    match = re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", str(value))
    return int(match.group(1)) if match else None


def prepare_metadata() -> list[dict]:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {SOURCE_PATH}"
        )

    with SOURCE_PATH.open("r", encoding="utf-8") as file:
        sites = json.load(file)

    if not isinstance(sites, dict):
        raise ValueError(
            "Expected sites.json to contain a dictionary "
            "whose keys are the original site IDs."
        )

    documents = []

    for site_id, site in sites.items():
        if not isinstance(site, dict):
            print(f"Skipping invalid record: {site_id}")
            continue

        text_parts = []

        for field in TEXT_FIELDS:
            value = to_text(site.get(field)).strip()

            if value:
                text_parts.append(
                    f"{field.replace('_', ' ').title()}: {value}"
                )

        searchable_text = "\n\n".join(text_parts)

        if not searchable_text:
            print(f"Skipping empty record: {site_id}")
            continue

        metadata = {
            "site_id": str(site_id),
        }

        metadata_fields = [
            "name",
            "category",
            "region",
            "date_inscribed",
            "danger",
            "cultural_criteria",
            "natural_criteria",
            "transboundary",
            "area_hectares",
            "components_count",
        ]

        for field in metadata_fields:
            value = metadata_value(site.get(field))

            if value is not None and value != "":
                metadata[field] = value

        year = extract_year(site.get("date_inscribed"))

        if year is not None:
            metadata["inscription_year"] = year

        documents.append(
            {
                "id": str(site_id),
                "text": searchable_text,
                "metadata": metadata,
            }
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            documents,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Source records: {len(sites)}")
    print(f"Prepared documents: {len(documents)}")
    print(f"Output file: {OUTPUT_PATH}")

    return documents


if __name__ == "__main__":
    prepare_metadata()