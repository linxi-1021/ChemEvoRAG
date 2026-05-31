from pathlib import Path

from parsing import OpenChemIEAdapter


class FakeOpenChemIEModel:
    def extract_molecules_from_figures_in_pdf(self, pdf_path):
        return [
            {
                "page": 1,
                "molecules": [
                    {
                        "bbox": (1, 2, 3, 4),
                        "score": 0.9,
                        "smiles": "CCO",
                    }
                ],
            }
        ]

    def extract_molecules_from_text_in_pdf(self, pdf_path):
        return [{"page": 2, "molecules": [{"text": "ethanol"}]}]

    def extract_reactions_from_figures_in_pdf(self, pdf_path):
        return [{"page": 3, "reactions": [{"reactants": [], "products": []}]}]

    def extract_reactions_from_text_in_pdf(self, pdf_path):
        return [{"page": 4, "reactions": [{"tokens": ["A", "to", "B"]}]}]

    def extract_tables_from_pdf(self, pdf_path):
        return [{"page": 5, "tables": []}]


class FakeOpenChemIEAdapter(OpenChemIEAdapter):
    def _build_model(self):
        return FakeOpenChemIEModel()


def test_openchemie_adapter_returns_failed_extraction_when_init_fails(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    adapter = OpenChemIEAdapter(external_path=tmp_path / "does_not_exist")

    extraction = adapter.parse_pdf(pdf, doc_id="paper")

    assert extraction.status == "failed"
    assert extraction.parser_name == "openchemie"
    assert "OpenChemIE initialization failed" in extraction.errors[0]


def test_openchemie_adapter_maps_successful_outputs_to_raw_extraction(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    adapter = FakeOpenChemIEAdapter()

    extraction = adapter.parse_pdf(pdf, doc_id="paper")

    assert extraction.status == "success"
    assert len(extraction.molecules) == 2
    assert len(extraction.reactions) == 2
    assert len(extraction.tables) == 1
    assert extraction.molecules[0].source.page == 1
    assert extraction.reactions[1].source.page == 4
    assert extraction.raw_payload["extract_molecules_from_figures_in_pdf"][0]["page"] == 1


def test_openchemie_adapter_reports_partial_extraction(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    adapter = FakeOpenChemIEAdapter(enabled_methods=["missing_method"])

    extraction = adapter.parse_pdf(pdf, doc_id="paper")

    assert extraction.status == "failed"
    assert extraction.errors == ["OpenChemIE method not found: missing_method"]


def test_openchemie_default_path_points_to_external_openchemie():
    adapter = OpenChemIEAdapter()

    assert adapter.external_path == Path(__file__).resolve().parents[1] / "external" / "OpenChemIE"

