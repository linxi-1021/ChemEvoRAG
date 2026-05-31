import pytest

from evidence import RawExtraction
from ingestion import PDFIngestor
from storage import LocalStore


class FakeParser:
    def __init__(self):
        self.calls = []

    def parse_pdf(self, pdf_path, doc_id):
        self.calls.append((pdf_path, doc_id))
        return RawExtraction(
            doc_id=doc_id,
            source_file=str(pdf_path),
            parser_name="fake",
            status="success",
            raw_payload={"ok": True},
        )


def test_ingest_pdf_saves_raw_extraction(tmp_path):
    pdf = tmp_path / "input.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    parser = FakeParser()
    store = LocalStore(base_dir=tmp_path)
    ingestor = PDFIngestor(store=store, parser=parser, base_dir=tmp_path)

    extraction = ingestor.ingest(pdf)
    loaded = store.load_parsed("input")

    assert extraction.doc_id == "input"
    assert parser.calls == [(pdf, "input")]
    assert loaded["doc_id"] == "input"
    assert loaded["parser_name"] == "fake"
    assert loaded["status"] == "success"


def test_ingest_pdf_can_use_explicit_doc_id_and_copy_to_raw_dir(tmp_path):
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    parser = FakeParser()
    ingestor = PDFIngestor(
        store=LocalStore(base_dir=tmp_path),
        parser=parser,
        base_dir=tmp_path,
        copy_to_raw_dir=True,
    )

    extraction = ingestor.ingest(pdf, doc_id="paper_001")

    copied_pdf = tmp_path / "data" / "raw_pdfs" / "paper_001.pdf"
    assert copied_pdf.exists()
    assert extraction.doc_id == "paper_001"
    assert parser.calls == [(copied_pdf, "paper_001")]


def test_ingest_pdf_rejects_missing_pdf(tmp_path):
    ingestor = PDFIngestor(
        store=LocalStore(base_dir=tmp_path),
        parser=FakeParser(),
        base_dir=tmp_path,
    )

    with pytest.raises(FileNotFoundError):
        ingestor.ingest(tmp_path / "missing.pdf")


def test_ingest_pdf_rejects_non_pdf(tmp_path):
    text = tmp_path / "input.txt"
    text.write_text("not a pdf", encoding="utf-8")
    ingestor = PDFIngestor(
        store=LocalStore(base_dir=tmp_path),
        parser=FakeParser(),
        base_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="Expected a .pdf file"):
        ingestor.ingest(text)

