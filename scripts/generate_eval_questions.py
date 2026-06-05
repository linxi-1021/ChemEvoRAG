#!/usr/bin/env python
"""Generate evaluation QA pairs from chemistry PDFs via multimodal LLM.

For each PDF in data/pdfs/:
1. Renders every page as a high-resolution PNG image
2. Sends all page images directly to the LLM (multimodal — the LLM reads
   the paper visually, including figures, tables, chemical structures)
3. LLM generates 5 test questions covering different retrieval intents
4. Outputs per-paper QA JSON + combined manifest to data/eval/
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

_dotenv_path = PROJECT_ROOT / ".env"
if _dotenv_path.exists():
    from dotenv import load_dotenv

    load_dotenv(_dotenv_path)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

EVAL_SYSTEM_PROMPT = """\
You are evaluating ChemEvoRAG, a chemistry literature retrieval-augmented
generation (RAG) system. It parses chemistry papers, extracts structured
evidence — molecules (names, SMILES, InChIKey), chemical reactions (reactants,
products, solvents, catalysts, temperatures, yields), and factual statements —
then retrieves relevant evidence through these channels:

  entity_retrieval       — match molecules by name, alias, SMILES, or InChIKey
  reaction_retrieval     — match reactions by condition (solvent, catalyst, T)
  lexical_retrieval      — keyword/BM25 search over facts and text blocks
  provenance_backtracking— trace evidence back to paper, page, section, figure
  dense_retrieval        — semantic similarity search via embeddings

You are looking at PAGE IMAGES of a chemistry paper. Read through ALL pages
carefully — pay close attention to figures, schemes, tables, experimental
procedures, compound numbering, yields, and spectral data.

Generate 5 test questions to evaluate ChemEvoRAG end-to-end. Each question:

1. Must be answerable ONLY from this paper's content (not common knowledge)
2. Must cover a different intent type (use all 5 types below exactly once):
   - alias_resolution:        "What does [abbreviation/name] refer to?"
   - entity_lookup:           "What is the structure/SMILES/IUPAC name of [X]?"
   - reaction_condition_query:"What [solvent/catalyst/temperature/time] was used for [reaction]?"
   - reaction_comparison:     "Which gave the higher yield, [reaction A] or [reaction B]?"
   - property_query:          "What is the [yield/mp/bp/Rf/value] of [compound X]?"
3. Must require evidence from MULTIPLE retrieval channels to answer correctly
4. Must include a ground-truth answer verifiable from the paper

Return ONLY a JSON object (no markdown fences, no extra text):

{
  "paper_title": "<exact title from paper>",
  "questions": [
    {
      "id": "q1",
      "intent": "alias_resolution",
      "question": "<natural language question>",
      "ground_truth_answer": "<concise answer with specific values/names from paper>",
      "key_entities": ["<compound name, number, or identifier>"],
      "expected_evidence_type": "<molecule | reaction | fact>"
    }
  ]
}"""


def pdf_to_page_images(pdf_path: Path, dpi: int) -> list[bytes]:
    from pdf2image import convert_from_path

    images = convert_from_path(pdf_path, dpi=dpi)
    result: list[bytes] = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result.append(buf.getvalue())
    return result


def call_llm(system_prompt: str, page_images: list[bytes]) -> str | None:
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                "Here are the pages of a chemistry paper as images. "
                "Read through all pages carefully — including figures, tables, "
                "schemes, experimental sections, and supporting information. "
                "Generate 5 evaluation questions following the system prompt."
            ),
        }
    ]

    for img_bytes in page_images:
        b64 = base64.b64encode(img_bytes).decode("ascii")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    return _send_multimodal_request(system_prompt, content)


def parse_llm_output(raw: str) -> dict | None:
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


MAX_IMAGES_PER_REQUEST = 45  # API limit is 50, leave margin

BATCH_SYSTEM_PROMPT = """\
You are evaluating ChemEvoRAG, a chemistry literature RAG system.
You are looking at a PORTION (pages {start_page}–{end_page} of {total_pages})
of a chemistry paper or supporting information document.

Scan these pages carefully for compounds, reactions, yields, properties,
abbreviations, and experimental conditions that appear here.

Generate 2-3 test questions that are answerable from THESE PAGES specifically.
Each question must cover a different intent type from this list:
- alias_resolution: "What does [abbreviation/name] refer to?"
- entity_lookup: "What is the structure/SMILES of [compound]?"
- reaction_condition_query: "What solvent/catalyst/temperature was used?"
- reaction_comparison: "Which gave the higher yield?"
- property_query: "What is the [yield/mp/bp/value] of [compound]?"

Return ONLY a JSON object:
{{
  "questions": [
    {{
      "id": "q_batch{batch_num}_{{i}}",
      "intent": "<one of the five types above>",
      "question": "<natural language question>",
      "ground_truth_answer": "<concise answer with specific values>",
      "key_entities": ["<compound identifier>"],
      "expected_evidence_type": "<molecule | reaction | fact>"
    }}
  ]
}}"""


def call_llm_batched(
    all_pages: list[bytes],
    total_pages: int,
    dpi: int,
) -> list[dict]:
    """Process a long PDF in batches, generating questions from each chunk."""
    batch_size = MAX_IMAGES_PER_REQUEST
    num_batches = (total_pages + batch_size - 1) // batch_size
    all_questions: list[dict] = []

    for batch_num in range(1, num_batches + 1):
        start = (batch_num - 1) * batch_size
        end = min(batch_num * batch_size, total_pages)
        batch_pages = all_pages[start:end]

        batch_mb = sum(len(p) for p in batch_pages) / (1024 * 1024)
        print(f"  Batch {batch_num}/{num_batches}: pages {start+1}–{end} ({batch_mb:.1f} MB)")

        system_prompt = BATCH_SYSTEM_PROMPT.format(
            start_page=start + 1,
            end_page=end,
            total_pages=total_pages,
            batch_num=batch_num,
        )

        content: list[dict] = [
            {
                "type": "text",
                "text": (
                    f"Here are pages {start+1}–{end} (of {total_pages}) "
                    f"from a chemistry document. Read them and generate "
                    f"2-3 test questions based on the content in these pages."
                ),
            }
        ]

        for img_bytes in batch_pages:
            b64 = base64.b64encode(img_bytes).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })

        raw = _send_multimodal_request(system_prompt, content)
        if not raw:
            print(f"  Batch {batch_num} failed — skipping", file=sys.stderr)
            continue

        parsed = parse_llm_output(raw)
        if parsed:
            batch_qs = parsed.get("questions", [])
            all_questions.extend(batch_qs)
            print(f"  Batch {batch_num}: {len(batch_qs)} questions")
        else:
            print(f"  Batch {batch_num}: JSON parse failed", file=sys.stderr)

    return all_questions


def _send_multimodal_request(
    system_prompt: str, content: list[dict]
) -> str | None:
    """Low-level multimodal API call. Returns raw response text or None."""
    key = os.environ.get("API_KEY")
    url = os.environ.get("BASE_URL")
    model = os.environ.get("LLM_MODEL", "gpt-5-mini")

    if not key:
        return None

    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=url or None)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            temperature=0.2,
            max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "16384")),
            extra_body={},
        )
        return response.choices[0].message.content
    except Exception as exc:
        print(f"  LLM call failed: {exc}", file=sys.stderr)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf-dir", default=str(PROJECT_ROOT / "data" / "pdfs"),
        help="Directory containing PDF files.",
    )
    parser.add_argument(
        "--output-dir", default=str(PROJECT_ROOT / "data" / "eval"),
        help="Directory for evaluation JSON files.",
    )
    parser.add_argument(
        "--dpi", type=int, default=250,
        help="DPI for PDF page rendering (default: 250).",
    )
    parser.add_argument(
        "--pdf", default=None,
        help="Process a single PDF instead of the whole directory.",
    )
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.pdf:
        pdf_paths = [Path(args.pdf)]
    else:
        pdf_paths = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_paths:
        print(f"No PDFs found in {pdf_dir}", file=sys.stderr)
        return 1

    results: dict[str, dict] = {}

    for pdf_path in pdf_paths:
        print(f"\n{'='*60}")
        print(f"Processing: {pdf_path.name}")

        # Render pages
        try:
            pages = pdf_to_page_images(pdf_path, dpi=args.dpi)
        except Exception as exc:
            print(f"  Failed to render: {exc}", file=sys.stderr)
            continue

        total_mb = sum(len(p) for p in pages) / (1024 * 1024)
        print(f"  {len(pages)} pages @ {args.dpi} DPI ({total_mb:.1f} MB PNG)")

        # Route: batched for long PDFs, single-call for short ones
        model_name = os.environ.get("LLM_MODEL", "?")
        if len(pages) > MAX_IMAGES_PER_REQUEST:
            print(f"  PDF too long for single call, batching ({len(pages)} pages)")
            questions = call_llm_batched(pages, len(pages), args.dpi)
            if not questions:
                print(f"  All batches failed — skipping", file=sys.stderr)
                continue
            parsed = {"paper_title": pdf_path.stem, "questions": questions[:8]}
        else:
            print(f"  Calling LLM ({model_name}) ...")
            raw = call_llm(EVAL_SYSTEM_PROMPT, pages)

            if not raw:
                print(f"  No output — skipping", file=sys.stderr)
                continue

            parsed = parse_llm_output(raw)
            if parsed is None:
                raw_path = output_dir / f"{pdf_path.stem}_raw.txt"
                raw_path.write_text(raw, encoding="utf-8")
                print(f"  JSON parse failed — raw output → {raw_path}", file=sys.stderr)
                continue

        out_path = output_dir / f"{pdf_path.stem}_eval.json"
        out_path.write_text(
            json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        n_q = len(parsed.get("questions", []))
        print(f"  {n_q} questions → {out_path}")
        results[pdf_path.name] = parsed

    # Combined manifest — merge with existing when processing single PDFs
    if results:
        manifest_path = output_dir / "all_questions.json"
        existing: dict[str, dict] = {}
        if manifest_path.exists():
            try:
                for q in json.loads(manifest_path.read_text("utf-8")):
                    existing[q["id"]] = q
            except (json.JSONDecodeError, KeyError):
                pass

        for pdf_name, parsed in results.items():
            for q in parsed.get("questions", []):
                q["source_paper"] = pdf_name
                existing[q["id"]] = q

        all_q = sorted(existing.values(), key=lambda x: x.get("source_paper", ""))
        manifest_path.write_text(
            json.dumps(all_q, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n{'='*60}")
        print(f"Total: {len(all_q)} questions from {len(set(q['source_paper'] for q in all_q))} papers → {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
