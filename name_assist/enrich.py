"""LLM batch enrichment for name dimensions: class, heritage, origin, sentiment, sound."""

import json
import os
from pathlib import Path

import anthropic

DATA_DIR = Path(__file__).parent.parent / "data"
ENRICHMENTS_PATH = DATA_DIR / "enrichments.json"

BATCH_SIZE = 50
MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """\
You are a baby name analysis expert. Given a list of names, provide structured \
classifications for each name. Respond ONLY with a valid JSON array — no markdown, \
no explanation, no code fences. Each element must have these exact keys:

- name: the name as given
- class: one of "Upper", "Upper-middle", "Middle", "Working", "Lower"
- heritage: primary cultural/linguistic origin (e.g. "English", "Hebrew", "Greek", \
"German", "Irish", "Spanish", "Arabic", "Japanese", "African", "French", "Latin", \
"Scandinavian", "Slavic", "Italian", "Scottish", "Welsh", "Persian", "Sanskrit", "Korean", etc.)
- origin_type: one of "Biblical", "Mythological", "Literary", "Presidential", \
"Nature", "Virtue", "Invented", "Traditional", "Celebrity", "Place", "Occupational"
- sentiment_tags: list of 2-4 tags from: "Whimsical", "Mythical", "Stoic", "Strong", \
"Gentle", "Feminine", "Masculine", "Rural", "Urban", "Elegant", "Playful", "Serious", \
"Warm", "Cool", "Classic", "Modern", "Earthy", "Ethereal", "Bold", "Soft"
- sound: one of "Soft", "Sharp", "Melodic", "Punchy", "Flowing", "Crisp"
"""


def _call_llm(names: list[str]) -> list[dict]:
    """Send a batch of names to Claude for classification."""
    client = anthropic.Anthropic()
    names_text = "\n".join(f"- {n}" for n in names)

    message = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Classify these baby names:\n{names_text}",
            }
        ],
    )

    raw = message.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]
    return json.loads(raw)


def load_enrichments() -> dict[str, dict]:
    """Load enrichment data keyed by lowercase name."""
    if not ENRICHMENTS_PATH.exists():
        return {}
    with open(ENRICHMENTS_PATH) as f:
        data = json.load(f)
    return {entry["name"].lower(): entry for entry in data}


def run_batch_enrichment(names: list[str], progress_callback=None) -> None:
    """
    Enrich a list of names via Claude API and save to disk.

    If enrichments already exist, only classify missing names.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    existing = load_enrichments()
    to_classify = [n for n in names if n.lower() not in existing]

    if not to_classify:
        print(f"All {len(names)} names already enriched.")
        return

    print(f"Enriching {len(to_classify)} names in batches of {BATCH_SIZE}...")
    all_results = list(existing.values())
    errors = 0

    for i in range(0, len(to_classify), BATCH_SIZE):
        batch = to_classify[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(to_classify) + BATCH_SIZE - 1) // BATCH_SIZE

        if progress_callback:
            progress_callback(batch_num, total_batches)
        else:
            print(f"  Batch {batch_num}/{total_batches} ({len(batch)} names)...")

        try:
            results = _call_llm(batch)
            all_results.extend(results)
        except Exception as e:
            errors += 1
            print(f"  Error in batch {batch_num}: {e}")
            # Save progress so far and continue
            continue

    # Save all results
    with open(ENRICHMENTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Enrichment complete. {len(all_results)} names saved. {errors} errors.")


# ---------------------------------------------------------------------------
# CLI entry point: python -m name_assist.enrich
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd
    from name_assist.data import load_df

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: Set ANTHROPIC_API_KEY environment variable.")
        raise SystemExit(1)

    print("Loading SSA data...")
    df = load_df()

    # Get top 5000 names by total count
    top = (
        df.groupby("name")["count"]
        .sum()
        .sort_values(ascending=False)
        .head(5000)
        .index.tolist()
    )
    print(f"Top {len(top)} names selected for enrichment.")

    run_batch_enrichment(top)
