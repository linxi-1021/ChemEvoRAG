#!/usr/bin/env python
"""Parse PDFs via MinerU batch API and build DocumentBlock evidence.

Flow:
  1. POST /api/v4/file-urls/batch  →  get batch_id + upload URLs
  2. PUT each PDF to its signed upload URL
  3. Poll GET /api/v4/extract-results/batch/{batch_id}  until all done/failed
  4. Download full_zip_url per file → extract full.md + _content_list.json
  5. Map _content_list.json entries → DocumentBlock → LocalStore.save_blocks()
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

_dotenv_path = PROJECT_ROOT / ".env"
if _dotenv_path.exists():
    from dotenv import load_dotenv

    load_dotenv(_dotenv_path)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evidence import DocumentBlock   # noqa: E402
from storage import LocalStore       # noqa: E402

MINERU_BASE = "https://mineru.net"
BATCH_URL = f"{MINERU_BASE}/api/v4/file-urls/batch"
POLL_INTERVAL = 30        # seconds between batch status checks
MAX_POLL_MINUTES = 120    # total polling timeout

TYPE_MAP = {
    "text": "paragraph",
    "paragraph": "paragraph",
    "title": "title",
    "heading": "title",
    "section": "section",
    "abstract": "abstract",
    "table": "table",
    "figure": "figure",
    "image": "figure",
    "caption": "caption",
    "equation_isolated": "paragraph",
    "equation_interline": "paragraph",
}


def _token() -> str:
    t = os.environ.get("MINERU_TOKEN", "")
    if not t or t == "your_token_here":
        raise RuntimeError("MINERU_TOKEN not set in .env")
    return t


def _headers() -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_token()}",
    }


# ── step 1: batch upload ──────────────────────────────────────


def request_upload_urls(file_names: list[str]) -> dict:
    """POST /api/v4/file-urls/batch → {batch_id, file_urls: [{name, upload_url}]}"""
    payload = {
        "files": [{"name": n, "data_id": Path(n).stem} for n in file_names],
        "model_version": "vlm",
        "enable_table": True,
        "enable_formula": True,
        "language": "en",
    }
    r = requests.post(BATCH_URL, headers=_headers(), json=payload, timeout=30)
    r.raise_for_status()
    body = r.json()
    if body.get("code") != 0:
        raise RuntimeError(f"batch create failed: {body.get('msg')} — {body}")
    return body["data"]


# ── step 2: upload files ──────────────────────────────────────


def upload_files(file_paths: list[Path], file_urls: list[str]) -> list[str]:
    """PUT each PDF to its signed URL (matched by index).  Returns list of successfully uploaded names."""
    uploaded: list[str] = []
    for fp, upload_url in tqdm(
        list(zip(file_paths, file_urls)), desc="Uploading", unit="pdf"
    ):
        if not upload_url:
            tqdm.write(f"  SKIP {fp.name}: no upload URL", file=sys.stderr)
            continue
        try:
            r = requests.put(upload_url, data=fp.read_bytes(), timeout=300)
            if r.status_code == 200:
                uploaded.append(fp.name)
            else:
                tqdm.write(f"  FAIL {fp.name}: HTTP {r.status_code}", file=sys.stderr)
        except Exception as exc:
            tqdm.write(f"  FAIL {fp.name}: {exc}", file=sys.stderr)
    return uploaded


# ── step 3: poll batch results ────────────────────────────────


def poll_batch(batch_id: str) -> list[dict]:
    """Poll GET /api/v4/extract-results/batch/{batch_id} until all tasks terminal."""
    url = f"{MINERU_BASE}/api/v4/extract-results/batch/{batch_id}"
    deadline = time.time() + MAX_POLL_MINUTES * 60

    with tqdm(desc="Parsing", unit=" files", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]") as pbar:
        while time.time() < deadline:
            r = requests.get(url, headers=_headers(), timeout=30)
            r.raise_for_status()
            body = r.json()
            if body.get("code") != 0:
                tqdm.write(f"  poll error: {body.get('msg')}", file=sys.stderr)
                time.sleep(POLL_INTERVAL)
                continue

            results: list[dict] = body["data"].get("extract_result", [])
            done = sum(1 for x in results if x["state"] in ("done", "failed"))
            total = len(results)
            pbar.total = total
            pbar.n = done
            pbar.refresh()

            pending = sum(1 for x in results if x["state"] not in ("done", "failed"))
            if pending == 0:
                pbar.close()
                return results

            time.sleep(POLL_INTERVAL)

        tqdm.write("  Timeout waiting for batch completion", file=sys.stderr)

    return []


# ── step 4: download and extract ──────────────────────────────


def download_zip(zip_url: str) -> dict[str, str]:
    """Download zip from *zip_url*, return {filename: content} for .md and .json files."""
    import tempfile

    r = requests.get(zip_url, timeout=120)
    r.raise_for_status()

    files: dict[str, str] = {}
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        for name in zf.namelist():
            if name.endswith((".md", ".json")):
                files[name] = zf.read(name).decode("utf-8", errors="replace")
    return files


# ── step 5: convert to DocumentBlock ──────────────────────────


def content_list_to_blocks(
    doc_id: str,
    content_list: list[dict],
) -> list[DocumentBlock]:
    """Convert MinerU _content_list.json entries to DocumentBlock objects."""
    blocks: list[DocumentBlock] = []
    for i, item in enumerate(content_list):
        raw_type = item.get("type", "paragraph")
        block_type = TYPE_MAP.get(raw_type, "paragraph")

        text = item.get("text", "") or ""
        page = item.get("page_idx")
        # Convert 0-indexed page to 1-indexed
        if page is not None:
            page = int(page) + 1

        bbox = item.get("bbox")
        if bbox and isinstance(bbox, list) and len(bbox) == 4:
            bbox = [float(v) for v in bbox]

        block = DocumentBlock(
            block_id=f"block_{doc_id}_{i:04d}",
            doc_id=doc_id,
            block_type=block_type,   # type: ignore[arg-type]
            text=text if text else None,
            page=page,
            bbox=bbox,
            confidence=0.85,
            raw_payload={"mineru_type": raw_type, "mineru_index": i},
        )
        blocks.append(block)

    # Link blocks
    for j in range(len(blocks)):
        if j > 0:
            blocks[j].prev_block_id = blocks[j - 1].block_id
        if j < len(blocks) - 1:
            blocks[j].next_block_id = blocks[j + 1].block_id

    return blocks


def parse_one(doc_id: str, zip_url: str) -> list[DocumentBlock] | None:
    """Download zip and convert to DocumentBlock list."""
    try:
        files = download_zip(zip_url)
    except Exception as exc:
        tqdm.write(f"  FAIL {doc_id}: download/unzip error: {exc}", file=sys.stderr)
        return None

    # Find content list JSON (MinerU naming: *_content_list.json or layout.json)
    content_json = None
    for name, content in files.items():
        if name.endswith("_content_list.json") or name == "layout.json":
            content_json = content
            break

    if not content_json:
        tqdm.write(f"  WARN {doc_id}: no content_list.json found in zip", file=sys.stderr)
        return None

    try:
        content_data = json.loads(content_json)
    except json.JSONDecodeError:
        tqdm.write(f"  WARN {doc_id}: bad content_list JSON", file=sys.stderr)
        return None

    # content_data may be a list or {"content_list": [...]}
    if isinstance(content_data, dict):
        items = content_data.get("content_list") or content_data.get("data") or []
    elif isinstance(content_data, list):
        items = content_data
    else:
        tqdm.write(f"  WARN {doc_id}: unexpected content_list shape", file=sys.stderr)
        return None

    return content_list_to_blocks(doc_id, items)


# ── main ──────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf-dir", default=str(PROJECT_ROOT / "data" / "pdfs"),
        help="Directory containing PDF files.",
    )
    parser.add_argument(
        "--pdf", default=None,
        help="Process a single PDF instead of the whole directory.",
    )
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    if args.pdf:
        pdf_paths = [Path(args.pdf)]
    else:
        pdf_paths = sorted(pdf_dir.glob("*.pdf"))

    # Skip chunk files
    pdf_paths = [p for p in pdf_paths if "_part" not in p.stem]
    if not pdf_paths:
        print("No PDFs found.", file=sys.stderr)
        return 1

    # Filter out PDFs > 200MB or > 200 pages (MinerU limits)
    valid: list[Path] = []
    for f in pdf_paths:
        size_mb = f.stat().st_size / (1024 * 1024)
        if size_mb > 200:
            print(f"  SKIP {f.name}: {size_mb:.0f}MB exceeds 200MB limit", file=sys.stderr)
            continue
        valid.append(f)

    print(f"=== MinerU Batch Parse: {len(valid)} PDFs ===\n")

    try:
        # Step 1: request upload URLs
        print(f"Step 1: Requesting upload URLs for {len(valid)} files ...")
        data = request_upload_urls([f.name for f in valid])
        batch_id = data["batch_id"]
        file_urls = data["file_urls"]
        print(f"  batch_id: {batch_id}")
        print(f"  upload URLs: {len(file_urls)}")

        # Step 2: upload
        print(f"\nStep 2: Uploading PDFs ...")
        uploaded = upload_files(valid, file_urls)
        print(f"  Uploaded: {len(uploaded)}/{len(valid)}")

        if not uploaded:
            print("No files uploaded — aborting.", file=sys.stderr)
            return 1

        # Map file name → doc_id
        name_to_doc_id = {f.name: f.stem for f in valid}

        # Step 3: wait for parsing
        print(f"\nStep 3: Waiting for parsing (polling every {POLL_INTERVAL}s) ...")
        results = poll_batch(batch_id)
        print(f"  Done: {sum(1 for r in results if r['state'] == 'done')}")
        print(f"  Failed: {sum(1 for r in results if r['state'] == 'failed')}")

        # Step 4-5: download + convert + save
        store = LocalStore(base_dir=PROJECT_ROOT)
        ok = 0
        for r in results:
            if r["state"] != "done":
                err = r.get("err_msg", "")
                tqdm.write(f"  SKIP {r['file_name']}: {r['state']} {err}", file=sys.stderr)
                continue

            file_name = r["file_name"]
            doc_id = name_to_doc_id.get(file_name, Path(file_name).stem)
            zip_url = r.get("full_zip_url", "")
            if not zip_url:
                tqdm.write(f"  SKIP {doc_id}: no full_zip_url", file=sys.stderr)
                continue

            print(f"  Processing {doc_id} ...")
            blocks = parse_one(doc_id, zip_url)
            if blocks is None:
                continue

            store.save_blocks(doc_id, blocks)
            print(f"    {len(blocks)} blocks → data/evidence/{doc_id}.blocks.json")
            ok += 1

        print(f"\nDone: {ok} documents with blocks")

    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
