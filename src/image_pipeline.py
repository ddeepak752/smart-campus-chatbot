"""
Image Pipeline - OCR + CLIP hybrid.

Priority:
1. EasyOCR  → read text/signs → text pipeline → KB record
2. CLIP+FAISS → visual scene match → KB record
3. Combine both signals → best confident result

This covers:
- Sign photos: OCR reads "IT Helpdesk" → finds it_helpdesk in KB
- Scene photos: CLIP recognises library/cafeteria/gym visually
- Combined: OCR + CLIP together for ambiguous images
"""
import os, io, json, warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"]       = "1"
warnings.filterwarnings("ignore")

import numpy as np
import torch
from pathlib import Path
from PIL import Image

ROOT       = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"

CLIP_CONFIDENCE_THRESHOLD = 0.25
OCR_CONFIDENCE_THRESHOLD  = 0.5   # EasyOCR confidence score

# ── Lazy-loaded models ────────────────────────────────────────────────
_clip_model     = None
_clip_processor = None
_faiss_index    = None
_kb_ids         = None
_kb             = None
_ocr_reader     = None

MAIN_CATS = {
    "library","cafeteria","classroom","department",
    "engineering_lab","science_lab","lab","auditorium",
    "seminar_hall","administration","sports","study_area",
    "medical","student_union","workshop","innovation","meeting_room",
}


def _load_clip():
    global _clip_model, _clip_processor, _faiss_index, _kb_ids, _kb
    if _clip_model is not None:
        return
    import faiss
    from transformers import CLIPProcessor, CLIPModel
    from src.kb_utils import load_kb

    _clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    _clip_model.eval()
    _faiss_index    = faiss.read_index(str(MODELS_DIR / "clip_faiss.index"))
    with open(MODELS_DIR / "clip_kb_ids.json") as f:
        _kb_ids = json.load(f)
    _kb = {loc["id"]: loc for loc in load_kb()}


def _load_ocr():
    global _ocr_reader
    if _ocr_reader is not None:
        return
    import easyocr
    _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)


def _clip_search(img: Image.Image, top_k: int = 3) -> list:
    """CLIP image → FAISS → top_k KB records with category deduplication."""
    _load_clip()
    inputs = _clip_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        emb = _clip_model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    emb_np = emb.cpu().numpy().astype("float32")

    scores, idxs = _faiss_index.search(emb_np, k=min(40, _faiss_index.ntotal))
    seen_cats = set(); results = []
    for i in range(len(idxs[0])):
        kb_id  = _kb_ids[idxs[0][i]]
        record = _kb.get(kb_id)
        if not record: continue
        cat = record.get("category","")
        if cat in MAIN_CATS and cat not in seen_cats:
            seen_cats.add(cat)
            results.append({"record": record, "score": float(scores[0][i]),
                            "source": "clip"})
        if len(results) >= top_k: break

    if not results:
        for i in range(min(top_k, len(idxs[0]))):
            kb_id  = _kb_ids[idxs[0][i]]
            record = _kb.get(kb_id)
            if record:
                results.append({"record": record, "score": float(scores[0][i]),
                                "source": "clip"})
    return results


def _ocr_search(img: Image.Image) -> list:
    """
    EasyOCR → extract text → search KB by keyword matching.
    Returns list of {record, score, source, ocr_text}.
    """
    _load_clip()   # loads _kb
    _load_ocr()
    import numpy as np_img

    img_np = np.array(img)
    try:
        ocr_results = _ocr_reader.readtext(img_np, detail=1)
    except Exception:
        return []

    # Filter by confidence, join text, remove watermarks/noise
    NOISE_WORDS = {
        "unsplash", "unsplash+", "unsplasht", "getty", "shutterstock",
        "istock", "alamy", "dreamstime", "adobe", "depositphotos",
        "123rf", "bigstock", "freepik", "pexels", "pixabay",
        "watermark", "preview", "sample", "demo", "copyright",
    }
    confident_texts = [
        text.strip().lower()
        for _, text, conf in ocr_results
        if conf >= OCR_CONFIDENCE_THRESHOLD
        and len(text.strip()) >= 3
        and text.strip().lower().rstrip("+t") not in NOISE_WORDS
        and not any(noise in text.strip().lower()
                    for noise in ["unsplash","unspla","splash","unsp",
                                  "getty","shutterstock","istock",
                                  "watermark","©","copyright"])
    ]
    if not confident_texts:
        return []

    full_text = " ".join(confident_texts)
    print(f"  [OCR] Detected: '{full_text[:80]}'")

    # Search KB by keyword matching
    matches = []
    seen = set()
    for kb_id, record in _kb.items():
        name     = record.get("name","").lower()
        kws      = [k.lower() for k in record.get("keywords",[])]
        cat      = record.get("category","").replace("_"," ").lower()
        desc     = record.get("description","").lower()[:100]

        # Score: how many OCR words match this record
        score = 0.0
        for word in confident_texts:
            if len(word) < 3: continue
            if word in name:        score += 0.5
            if any(word in kw for kw in kws): score += 0.4
            if word in cat:         score += 0.3
            if word in desc:        score += 0.2
            # Exact name match is a strong signal
            if word == name:        score += 1.0
            if name in full_text:   score += 0.8

        if score > 0.3 and kb_id not in seen:
            seen.add(kb_id)
            matches.append({
                "record":   record,
                "score":    min(score, 0.98),
                "source":   "ocr",
                "ocr_text": full_text,
            })

    matches.sort(key=lambda x: -x["score"])
    return matches[:3]


def run_image_pipeline(image_input, top_k: int = 3) -> dict:
    """
    Hybrid OCR + CLIP image pipeline.

    Returns:
      top_match:        dict    best KB record
      top_k:            list    top records with scores
      scores:           list    float scores
      predicted_kb_id:  str
      confidence:       float
      low_confidence:   bool
      ocr_text:         str     text detected by OCR
      source:           str     'ocr' | 'clip' | 'combined'
      error:            str
    """
    result = {
        "top_match":       None,
        "top_k":           [],
        "scores":          [],
        "predicted_kb_id": "",
        "confidence":      0.0,
        "low_confidence":  True,
        "ocr_text":        "",
        "source":          "",
        "error":           "",
    }

    try:
        # Load image
        if isinstance(image_input, (str, Path)):
            img = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, bytes):
            img = Image.open(io.BytesIO(image_input)).convert("RGB")
        else:
            img = image_input.convert("RGB")

        # ── Step 1: OCR ───────────────────────────────────────────────
        ocr_results = _ocr_search(img)
        best_ocr    = ocr_results[0] if ocr_results else None

        # ── Step 2: CLIP ──────────────────────────────────────────────
        clip_results = _clip_search(img, top_k=top_k)
        best_clip    = clip_results[0] if clip_results else None

        # ── Step 3: Decision logic ────────────────────────────────────
        # OCR wins if it found something with decent confidence
        if best_ocr and best_ocr["score"] >= 0.5:
            # OCR found clear text → use OCR result, add CLIP as context
            top_records = ocr_results[:top_k]
            # Add CLIP results not already in OCR results
            seen_ids = {r["record"]["id"] for r in top_records}
            for cr in clip_results:
                if cr["record"]["id"] not in seen_ids:
                    top_records.append(cr)
                    seen_ids.add(cr["record"]["id"])
                if len(top_records) >= top_k: break
            source = "ocr"
            result["ocr_text"] = best_ocr.get("ocr_text","")

        elif best_clip and best_clip["score"] >= CLIP_CONFIDENCE_THRESHOLD:
            # CLIP found a clear visual match
            top_records = clip_results[:top_k]
            source = "clip"

        elif best_ocr and best_ocr["score"] > 0:
            # Weak OCR signal — combine with CLIP
            combined = {}
            for r in ocr_results:
                kid = r["record"]["id"]
                combined[kid] = {"record":r["record"],
                                 "score": r["score"]*0.6,
                                 "source":"combined"}
            for r in clip_results:
                kid = r["record"]["id"]
                if kid in combined:
                    combined[kid]["score"] += r["score"]*0.4
                    combined[kid]["source"] = "combined"
                else:
                    combined[kid] = {"record":r["record"],
                                     "score": r["score"]*0.4,
                                     "source":"combined"}
            top_records = sorted(combined.values(), key=lambda x:-x["score"])[:top_k]
            source = "combined"
            result["ocr_text"] = best_ocr.get("ocr_text","")

        else:
            # Fallback to CLIP only
            top_records = clip_results[:top_k]
            source = "clip"

        if top_records:
            best = top_records[0]
            result.update({
                "top_match":       best["record"],
                "top_k":           top_records,
                "scores":          [r["score"] for r in top_records],
                "predicted_kb_id": best["record"]["id"],
                "confidence":      best["score"],
                "low_confidence":  best["score"] < CLIP_CONFIDENCE_THRESHOLD,
                "source":          source,
            })

    except Exception as e:
        result["error"] = str(e)
        import traceback; traceback.print_exc()

    return result
