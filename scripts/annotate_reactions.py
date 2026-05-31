#!/usr/bin/env python
"""Draw bounding boxes with IDs on extracted figure images for verification.

Reads data/runs/test_paper_reactions.json and overlays molecule /
condition bounding boxes onto the PNG files saved under data/runs/figures/.

Usage:
    conda activate chemevorag
    cd ChemEvoRAG_Phase1-main
    python scripts/annotate_reactions.py
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REACTIONS_JSON = PROJECT_ROOT / "data" / "runs" / "test_paper_reactions.json"
FIGURES_DIR = PROJECT_ROOT / "data" / "runs" / "figures"
OUT_DIR = PROJECT_ROOT / "data" / "runs" / "annotated"


def main() -> int:
    if not REACTIONS_JSON.exists():
        print(f"Not found: {REACTIONS_JSON}")
        return 1

    data = json.loads(REACTIONS_JSON.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from PIL import Image, ImageDraw, ImageFont

    # Colours per category
    COLOURS = {
        "[Mol]":     (0, 200, 0),     # green — molecule
        "[MolR]":    (200, 100, 0),   # orange — reactant label
        "[MolP]":    (0, 100, 200),   # blue — product label
        "[Cond]":    (200, 0, 0),     # red — condition
        "[Arrow]":   (100, 100, 100), # grey — arrow
        "[MolI]":    (150, 0, 200),   # purple — identifier
        "[Sup]":     (0, 150, 150),   # teal — superscript
        "[Txt]":     (100, 50, 50),   # brown — text
    }

    drawn = 0

    for fig_idx, fig in enumerate(data):
        rxns = fig.get("reactions", [])
        if not rxns:
            continue

        page = fig.get("page", "?")
        # Find matching saved figure PNG — try a few naming patterns
        candidates = sorted(
            FIGURES_DIR.glob(f"fig_{fig_idx+1:02d}_p{page}*.png"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            # fallback: any saved fig
            candidates = sorted(FIGURES_DIR.glob("fig_*.png"))
        if not candidates:
            print(f"Fig {fig_idx+1}: no saved PNG found — skipping")
            continue

        img_path = candidates[0]
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size

        # Try to load a font (monospace if available)
        try:
            font = ImageFont.truetype("consola.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

        box_id = 0

        for rxn_idx, rxn in enumerate(rxns):
            for role, key in [("R", "reactants"), ("P", "products"), ("C", "conditions")]:
                group = rxn.get(key, [])
                for p_idx, p in enumerate(group):
                    bbox = p.get("bbox")
                    if not bbox or len(bbox) != 4:
                        continue
                    x1 = int(bbox[0] * w)
                    y1 = int(bbox[1] * h)
                    x2 = int(bbox[2] * w)
                    y2 = int(bbox[3] * h)

                    cat = p.get("category", "?")
                    colour = COLOURS.get(cat, (255, 255, 0))

                    # Short ID: Fig-Rxn-Role-Index → F2-R0-R3 = Fig2, Reaction0, Reactant3
                    # Look up in JSON: data[fig_idx-1]["reactions"][rxn_idx][key][p_idx]
                    short_id = f"F{fig_idx+1}-R{rxn_idx}-{role}{p_idx}"
                    smi = p.get("smiles", "") or ""
                    txt = p.get("text", "")
                    if isinstance(txt, list):
                        txt = " ".join(str(t) for t in txt)

                    # Thin rectangle
                    draw.rectangle([x1, y1, x2, y2], outline=colour, width=1)

                    # Short label above the box (no SMILES to avoid overlap)
                    label_text = f" {short_id} "
                    tb = draw.textbbox((x1, y1 - 12), label_text, font=font)
                    draw.rectangle(tb, fill=colour)
                    draw.text((x1, y1 - 12), label_text, fill=(255, 255, 255), font=font)

                    box_id += 1

        out_path = OUT_DIR / f"fig_{fig_idx+1:02d}_p{page}_{box_id}boxes.png"
        img.save(str(out_path))
        drawn += 1
        print(f"Fig {fig_idx+1:02d} (page {page}): {box_id} boxes → {out_path.name}")

    print(f"\nAnnotated {drawn} figure(s) saved to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
