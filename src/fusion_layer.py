"""
Multimodal Fusion Layer v2
===========================
Supports all modality combinations:
  text only          → CLIP text embedding (512) + zero (512)
  image only         → zero (512) + CLIP image embedding (512)
  voice only         → transcribe → CLIP text embedding + zero (512)
  text + image       → CLIP text (512) + CLIP image (512)
  voice + image      → transcribe → CLIP text (512) + CLIP image (512)
  text + voice       → combine texts → CLIP text (512) + zero (512)
  text + voice + image → combined text (512) + CLIP image (512)

Both text and image use CLIP encoder → same 512-dim space → better fusion.
JOINT_DIM = 1024
"""
import os, json, warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"]       = "1"
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

TEXT_DIM  = 512
IMAGE_DIM = 512
JOINT_DIM = TEXT_DIM + IMAGE_DIM  # 1024

ROOT         = Path(__file__).parent.parent
MODELS_DIR   = ROOT / "models"
FUSION_PATH  = MODELS_DIR / "fusion_mlp.pt"

_clip_model     = None
_clip_processor = None
_faiss_index    = None
_kb_ids         = None
_kb             = None
_fusion_model   = None
_label2id       = None
_id2label_fusion = None


class FusionMLP(nn.Module):
    """Legacy v1 — kept for compatibility."""
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    def forward(self, x): return self.net(x)


class FusionMLPv2(nn.Module):
    """v2 — GELU, 3 layers, label smoothing during training."""
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2),
            nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    def forward(self, x): return self.net(x)


class FusionMLPv3(nn.Module):
    """v3 — 2 hidden layers, used for 8 visual class classification."""
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    def forward(self, x): return self.net(x)


def _load_models():
    global _clip_model, _clip_processor, _faiss_index, _kb_ids
    global _kb, _fusion_model, _label2id, _id2label_fusion
    if _clip_model is not None:
        return

    import faiss
    from transformers import CLIPProcessor, CLIPModel
    from src.kb_utils import load_kb

    _clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    _clip_model.eval()

    _faiss_index = faiss.read_index(str(MODELS_DIR / "clip_faiss.index"))
    with open(MODELS_DIR / "clip_kb_ids.json") as f:
        _kb_ids = json.load(f)

    kb_list = load_kb()
    _kb     = {loc["id"]: loc for loc in kb_list}

    # Load fusion model checkpoint first to get visual_kb_ids
    if FUSION_PATH.exists():
        ckpt = torch.load(str(FUSION_PATH), map_location="cpu")
        if isinstance(ckpt, dict) and "config" in ckpt:
            cfg            = ckpt["config"]
            input_dim      = cfg.get("input_dim", JOINT_DIM)
            hidden_dim     = cfg.get("hidden_dim", 512)
            n_classes      = cfg.get("num_classes", 8)
            visual_kb_ids  = cfg.get("visual_kb_ids", [loc["id"] for loc in kb_list])
            version        = cfg.get("version", "v3")

            # Build label maps from checkpoint's visual_kb_ids (NOT all 142 records)
            _label2id       = {kb_id: i for i, kb_id in enumerate(visual_kb_ids)}
            _id2label_fusion = {i: kb_id for i, kb_id in enumerate(visual_kb_ids)}

            # Select correct model class based on version
            if version == "v3":
                _fusion_model = FusionMLPv3(input_dim, hidden_dim, n_classes)
            else:
                _fusion_model = FusionMLPv2(input_dim, hidden_dim, n_classes)
            _fusion_model.load_state_dict(ckpt["state_dict"])
        else:
            # Legacy checkpoint — fallback to all KB records
            n_classes = len(kb_list)
            _label2id       = {loc["id"]: i for i, loc in enumerate(kb_list)}
            _id2label_fusion = {i: loc["id"] for i, loc in enumerate(kb_list)}
            _fusion_model   = FusionMLP(896, 512, n_classes)
            try:
                _fusion_model.load_state_dict(ckpt)
            except Exception:
                _fusion_model = FusionMLPv2(JOINT_DIM, 512, n_classes)
    else:
        # No checkpoint — use all KB records
        n_classes = len(kb_list)
        _label2id       = {loc["id"]: i for i, loc in enumerate(kb_list)}
        _id2label_fusion = {i: loc["id"] for i, loc in enumerate(kb_list)}
        _fusion_model   = FusionMLPv2(JOINT_DIM, 512, n_classes)

    _fusion_model.eval()


def encode_text(query: str) -> np.ndarray:
    """Encode text with CLIP text encoder → 512-dim."""
    _load_models()
    inputs = _clip_processor(text=[query], return_tensors="pt",
                              padding=True, truncation=True, max_length=77)
    inputs = {k: v for k, v in inputs.items()}
    with torch.no_grad():
        emb = _clip_model.get_text_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0].astype("float32")


def encode_image(image_input) -> np.ndarray:
    """Encode image with CLIP image encoder → 512-dim."""
    _load_models()
    from PIL import Image
    import io as _io
    if isinstance(image_input, (str, Path)):
        img = Image.open(str(image_input)).convert("RGB")
    elif isinstance(image_input, bytes):
        img = Image.open(_io.BytesIO(image_input)).convert("RGB")
    else:
        img = image_input.convert("RGB")
    inputs = _clip_processor(images=img, return_tensors="pt")
    inputs = {k: v for k, v in inputs.items()}
    with torch.no_grad():
        emb = _clip_model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0].astype("float32")


def encode_voice(audio_input, suffix=".wav") -> tuple:
    """Transcribe audio with Whisper → text → CLIP text embedding.
    Returns (transcript: str, embedding: np.ndarray)
    """
    import tempfile, os as _os
    import whisper as _whisper
    if not hasattr(encode_voice, "_whisper"):
        encode_voice._whisper = _whisper.load_model("base")
    model = encode_voice._whisper

    if isinstance(audio_input, bytes):
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_input); tmp = f.name
        try:
            result = model.transcribe(tmp, fp16=False)
        finally:
            _os.unlink(tmp)
    else:
        result = model.transcribe(str(audio_input), fp16=False)

    transcript = result.get("text","").strip()
    emb = encode_text(transcript) if transcript else np.zeros(TEXT_DIM, dtype=np.float32)
    return transcript, emb


def search_image_faiss(image_emb: np.ndarray, top_k: int = 3) -> list:
    """Direct FAISS search for image-only queries (most accurate)."""
    import faiss as _faiss
    index = _faiss.read_index(str(MODELS_DIR / "clip_faiss.index"))
    with open(MODELS_DIR / "clip_kb_ids.json") as f:
        kb_ids = json.load(f)

    img_np = image_emb.reshape(1,-1).astype("float32")
    scores, idxs = index.search(img_np, k=min(30, index.ntotal))

    MAIN_CATS = {
        "library","cafeteria","classroom","department","engineering_lab",
        "science_lab","lab","auditorium","seminar_hall","administration",
        "sports","study_area","medical","student_union","workshop","innovation",
    }
    seen_cats = set(); results = []
    for i in range(len(idxs[0])):
        kb_id  = kb_ids[idxs[0][i]]
        record = _kb.get(kb_id)
        if not record: continue
        cat = record.get("category","")
        if cat in MAIN_CATS and cat not in seen_cats:
            seen_cats.add(cat)
            results.append({"record": record, "score": float(scores[0][i])})
        if len(results) >= top_k: break

    if not results:
        for i in range(min(top_k, len(idxs[0]))):
            kb_id  = kb_ids[idxs[0][i]]
            record = _kb.get(kb_id)
            if record: results.append({"record": record, "score": float(scores[0][i])})
    return results


def run_fusion_pipeline(
    text_query:   str   = None,
    image_input         = None,
    voice_input:  bytes = None,
    voice_suffix: str   = ".wav",
    top_k:        int   = 3,
) -> dict:
    """
    Full multimodal fusion — accepts any combination of inputs.

    Modality routing:
      image only              → FAISS direct (most accurate)
      text only               → MLP with text CLIP emb + zero image
      voice only              → transcribe → same as text only
      text + image            → MLP with both CLIP embeddings
      voice + image           → transcribe → MLP with both
      text + voice            → combine queries → MLP text + zero image
      text + voice + image    → combine text+voice → MLP with image
    """
    _load_models()

    result = {
        "top_match":        None,
        "top_k_records":    [],
        "modalities_used":  [],
        "transcript":       "",
        "joint_embedding":  [],
        "error":            "",
    }

    try:
        text_emb  = None
        image_emb = None
        modalities_used = []
        transcript = ""

        # ── Encode each available modality ────────────────────────────
        if voice_input is not None:
            transcript, voice_emb = encode_voice(voice_input, suffix=voice_suffix)
            result["transcript"] = transcript
            modalities_used.append("voice")
            text_emb = voice_emb  # voice → text embedding

        if text_query and text_query.strip():
            t_emb = encode_text(text_query.strip())
            if text_emb is not None:
                # Combine text + voice by averaging
                text_emb = (text_emb + t_emb) / 2.0
                text_emb = text_emb / (np.linalg.norm(text_emb) + 1e-8)
            else:
                text_emb = t_emb
            if "voice" not in modalities_used:
                modalities_used.append("text")
            else:
                modalities_used.append("text")

        if image_input is not None:
            try:
                image_emb = encode_image(image_input)
                modalities_used.append("image")
            except Exception as e:
                result["error"] = f"Image encoding failed: {e}"

        if not modalities_used:
            result["error"] = "No valid input provided."
            return result

        result["modalities_used"] = modalities_used

        # ── Routing strategy ──────────────────────────────────────────
        # Image only → FAISS (best visual accuracy)
        if image_emb is not None and text_emb is None:
            top_records = search_image_faiss(image_emb, top_k=top_k)

        # Text/voice only or text+image/voice+image → Fusion MLP
        else:
            t = text_emb if text_emb is not None else np.zeros(TEXT_DIM, dtype=np.float32)
            i = image_emb if image_emb is not None else np.zeros(IMAGE_DIM, dtype=np.float32)
            joint = np.concatenate([t, i]).astype(np.float32)
            result["joint_embedding"] = joint.tolist()

            # Check if model expects 1024-dim or 896-dim (legacy)
            expected_dim = JOINT_DIM  # 1024
            try:
                first_layer_dim = list(_fusion_model.parameters())[0].shape[1]
            except:
                first_layer_dim = expected_dim

            if first_layer_dim == 896 and len(joint) == 1024:
                # Legacy model: use SentenceTransformer text embedding
                from sentence_transformers import SentenceTransformer
                if not hasattr(run_fusion_pipeline, "_st_model"):
                    run_fusion_pipeline._st_model = SentenceTransformer("all-MiniLM-L6-v2")
                query_text = text_query or transcript or ""
                st_emb = run_fusion_pipeline._st_model.encode([query_text])[0].astype("float32") if query_text else np.zeros(384, dtype=np.float32)
                img_part = image_emb if image_emb is not None else np.zeros(512, dtype=np.float32)
                joint = np.concatenate([st_emb, img_part])

            with torch.no_grad():
                x      = torch.tensor(joint, dtype=torch.float32).unsqueeze(0)
                logits = _fusion_model(x)[0]
                probs  = torch.softmax(logits, dim=0).numpy()

            top_indices = np.argsort(probs)[::-1][:top_k]
            top_records = []
            for idx in top_indices:
                kb_id  = _id2label_fusion[int(idx)]
                record = _kb.get(kb_id)
                if record:
                    top_records.append({
                        "record": record,
                        "score":  float(probs[idx]),
                        "rank":   len(top_records) + 1,
                    })

        if top_records:
            result["top_match"]     = top_records[0]["record"]
            result["top_k_records"] = top_records

    except Exception as e:
        result["error"] = str(e)
        import traceback; traceback.print_exc()

    return result
