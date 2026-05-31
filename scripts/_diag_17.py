"""Diagnose which image in doc 17 hangs OpenChemIE."""
import json, sys, time
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
MODEL_DIR = PROJECT_ROOT / "data" / "models"

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

d17 = PROJECT_ROOT / "data" / "mineru_output" / "17" / "extracted"
cl = list(d17.glob("*_content_list_v2.json"))[0]
pages = json.loads(cl.read_text("utf-8"))
img_dir = d17 / "images"

imgs = []
for page_items in pages:
    if not isinstance(page_items, list):
        continue
    for item in page_items:
        if isinstance(item, dict) and item.get("type") == "image":
            p = item.get("content", {}).get("image_source", {}).get("path", "")
            if p:
                f = img_dir / Path(p).name
                if f.exists():
                    imgs.append((f.name, f))

print(f"Testing {len(imgs)} images...")

from openchemie import OpenChemIE

model = OpenChemIE(device="cpu")

# ── load local models ──
import torch
weights = [
    ("molscribe", "swin_base_char_aux_1m680k.pth"),
    ("pdfparser", "publaynet-tf_efficientdet_d1.pth.tar"),
    ("rxnscribe", "pix2seq_reaction_full.ckpt"),
    ("moldet",    "best_hf.ckpt"),
    ("coref",     "coref_best_hf.ckpt"),
    ("chemner",   "best.ckpt"),
]
for suffix, filename in weights:
    candidate = MODEL_DIR / filename
    if not candidate.exists():
        continue
    init = getattr(model, f"init_{suffix}", None)
    if init is None:
        continue
    try:
        torch.cuda.empty_cache()
        init(str(candidate))
    except Exception as e:
        print(f"  init_{suffix}: {e}")

ms = MODEL_DIR / "swin_base_char_aux_1m680k.pth"
ms1 = MODEL_DIR / "swin_base_char_aux_1m.pth"
if ms.exists() or ms1.exists():
    try:
        from molscribe import MolScribe as MS
        ms_path = str(ms if ms.exists() else ms1)
        import rxnscribe.interface as ri
        def _local(self):
            return MS(ms_path, device=self.device)
        ri.RxnScribe.get_molscribe = _local
        ri.MolDetect.get_molscribe = _local
    except:
        pass

cre = MODEL_DIR / "chemrxnextractor-training-modules"
if cre.exists():
    try:
        model.init_chemrxnextractor(str(cre))
    except:
        pass

print("Model loaded.\n")

for idx, (name, path) in enumerate(imgs):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    print(f"[{idx+1}/{len(imgs)}] {name} ({w}x{h}) ... ", end="", flush=True)
    start = time.time()
    try:
        result = model.extract_molecules_from_figures([img], batch_size=1)
        elapsed = time.time() - start
        mols = result[0].get("molecules", []) if result else []
        print(f"OK ({elapsed:.1f}s, {len(mols)} mols)")
    except Exception as e:
        elapsed = time.time() - start
        print(f"FAIL ({elapsed:.1f}s): {str(e)[:120]}")
    torch.cuda.empty_cache()
