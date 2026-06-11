import json
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import faiss
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
METRICS_DIR = ROOT / "outputs" / "metrics"

INDEX_PATH = MODELS_DIR / "clip_faiss.index"
IDS_PATH = MODELS_DIR / "clip_kb_ids.json"
SPLIT_PATH = MODELS_DIR / "clip_image_split.json"


def load_clip():
    print("Loading CLIP ViT-B/32 on CPU...")
    device = torch.device("cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor, device


def unique_topk(kb_ids: list[str], scores, idxs, k: int = 3) -> list[dict]:
    out, seen = [], set()
    for score, idx in zip(scores[0], idxs[0]):
        kb_id = kb_ids[int(idx)]
        if kb_id in seen:
            continue
        seen.add(kb_id)
        out.append({"kb_id": kb_id, "score": float(score)})
        if len(out) >= k:
            break
    return out


def main() -> None:
    if not INDEX_PATH.exists() or not IDS_PATH.exists() or not SPLIT_PATH.exists():
        raise SystemExit("Missing CLIP index files. Run scripts/build_clip_image_index.py first.")

    index = faiss.read_index(str(INDEX_PATH))
    kb_ids = json.loads(IDS_PATH.read_text(encoding="utf-8"))
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    test_rows = split.get("test", [])
    if not test_rows:
        raise SystemExit("No held-out test rows found in models/clip_image_split.json.")

    print(f"Index vectors: {index.ntotal}")
    print(f"Held-out test images: {len(test_rows)}")

    model, processor, device = load_clip()
    predictions = []
    top1 = top3 = 0

    for row in test_rows:
        image_path = ROOT / row["image_path"]
        true_kb = row["kb_id"]
        try:
            image = Image.open(image_path).convert("RGB")
            with torch.no_grad():
                inputs = processor(images=image, return_tensors="pt")
                inputs = {key: value.to(device) for key, value in inputs.items()}
                embedding = model.get_image_features(**inputs)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)

            search_k = min(max(20, 3), index.ntotal)
            scores, idxs = index.search(embedding.cpu().numpy().astype("float32"), k=search_k)
            top = unique_topk(kb_ids, scores, idxs, k=3)
            pred = top[0]["kb_id"] if top else ""
            top_ids = [item["kb_id"] for item in top]
            is_top1 = pred == true_kb
            is_top3 = true_kb in top_ids
            top1 += int(is_top1)
            top3 += int(is_top3)
            mark = "OK" if is_top1 else ("TOP3" if is_top3 else "MISS")
            score_text = ", ".join(f"{item['kb_id']}:{item['score']:.3f}" for item in top)
            print(f"{mark:4s} {row['generic_label']:12s} true={true_kb:22s} top=[{score_text}]")
            predictions.append({
                "image_path": row["image_path"],
                "generic_label": row["generic_label"],
                "true_kb_id": true_kb,
                "pred_top1": pred,
                "pred_top3": top_ids,
                "top_scores": top,
                "top1_correct": is_top1,
                "top3_correct": is_top3,
            })
        except Exception as exc:
            print(f"ERR  {row['image_path']}: {exc}")

    n = len(predictions)
    metrics = {
        "top1_accuracy": round(top1 / n, 4) if n else 0.0,
        "top3_accuracy": round(top3 / n, 4) if n else 0.0,
        "n_tested": n,
        "note": "Held-out images are excluded from the FAISS index.",
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "image_retrieval_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    (METRICS_DIR / "image_retrieval_predictions.json").write_text(
        json.dumps(predictions, indent=2),
        encoding="utf-8",
    )
    print(f"\nTop-1: {metrics['top1_accuracy']} ({top1}/{n})")
    print(f"Top-3: {metrics['top3_accuracy']} ({top3}/{n})")
    print("Saved outputs/metrics/image_retrieval_metrics.json")


if __name__ == "__main__":
    main()
