"""PDF ingestion orchestration."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol

from evidence import RawExtraction
from parsing import OpenChemIEAdapter
from storage import LocalStore


class ParserAdapter(Protocol):
    def parse_pdf(self, pdf_path: str | Path, doc_id: str) -> RawExtraction:
        ...


class PDFIngestor:
    """Validate and register a PDF, then persist parser raw extraction output."""

    def __init__(
        self,
        store: LocalStore | None = None,
        parser: ParserAdapter | None = None,
        base_dir: str | Path = ".",
        copy_to_raw_dir: bool = False,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.store = store or LocalStore(base_dir=self.base_dir)
        self.parser = parser or OpenChemIEAdapter()
        self.copy_to_raw_dir = copy_to_raw_dir
        self.raw_pdf_dir = self.base_dir / "data" / "raw_pdfs"

    def ingest(self, pdf_path: str | Path, doc_id: str | None = None) -> RawExtraction:
        source_pdf = self._validate_pdf(pdf_path)
        resolved_doc_id = doc_id or source_pdf.stem
        registered_pdf = self._register_pdf(source_pdf, resolved_doc_id)

        extraction = self.parser.parse_pdf(registered_pdf, resolved_doc_id)
        self.store.save_parsed(
            resolved_doc_id,
            extraction.model_dump(mode="json"),
        )
        return extraction

    def _validate_pdf(self, pdf_path: str | Path) -> Path:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        if not path.is_file():
            raise ValueError(f"PDF path is not a file: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {path}")
        return path

    def _register_pdf(self, pdf_path: Path, doc_id: str) -> Path:
        if not self.copy_to_raw_dir:
            return pdf_path

        self.raw_pdf_dir.mkdir(parents=True, exist_ok=True)
        target = self.raw_pdf_dir / f"{doc_id}.pdf"
        if pdf_path.resolve() != target.resolve():
            shutil.copy2(pdf_path, target)
        return target

