import csv
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import faiss
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "image_manifest.csv"
MODELS_DIR = ROOT / "models"
SPLIT_PATH = MODELS_DIR / "clip_image_split.json"
INDEX_PATH = MODELS_DIR / "clip_faiss.index"
IDS_PATH = MODELS_DIR / "clip_kb_ids.json"
META_PATH = MODELS_DIR / "clip_image_manifest.json"

SEED = 42
TEST_PER_CLASS = 3
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def read_manifest() -> list[dict]:
    if not MANIFEST.exists():
        raise SystemExit("Missing data/image_manifest.csv. Run scripts/build_image_manifest.py first.")
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    rows = [row for row in rows if (ROOT / row["image_path"]).suffix.lower() in IMAGE_EXTS]
    if not rows:
        raise SystemExit("Manifest contains no supported image files.")
    return rows


def split_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    rng = random.Random(SEED)
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["generic_label"]].append(row)

    train_rows, test_rows = [], []
    for label, label_rows in sorted(by_label.items()):
        label_rows = sorted(label_rows, key=lambda row: row["image_path"])
        rng.shuffle(label_rows)
        if len(label_rows) <= 4:
            test_count = 1
        else:
            test_count = min(TEST_PER_CLASS, max(1, len(label_rows) // 4))
        test_rows.extend(label_rows[:test_count])
        train_rows.extend(label_rows[test_count:])

    if not train_rows:
        raise SystemExit("No training images after split. Add more images.")
    return train_rows, test_rows


def load_clip():
    print("Loading CLIP ViT-B/32 on CPU...")
    device = torch.device("cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor, device


def encode_images(rows: list[dict], model, processor, device) -> tuple[np.ndarray, list[dict]]:
    embeddings, valid_rows = [], []
    for idx, row in enumerate(rows, start=1):
        image_path = ROOT / row["image_path"]
        try:
            image = Image.open(image_path).convert("RGB")
            with torch.no_grad():
                inputs = processor(images=image, return_tensors="pt")
                inputs = {key: value.to(device) for key, value in inputs.items()}
                embedding = model.get_image_features(**inputs)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            embeddings.append(embedding.cpu().numpy()[0].astype("float32"))
            valid_rows.append(row)
        except Exception as exc:
            print(f"[SKIP] {row['image_path']}: {exc}")

        if idx % 20 == 0 or idx == len(rows):
            print(f"Encoded {idx}/{len(rows)}")

    if not embeddings:
        raise SystemExit("No images could be encoded.")
    return np.asarray(embeddings, dtype="float32"), valid_rows


def main() -> None:
    rows = read_manifest()
    train_rows, test_rows = split_rows(rows)
    print(f"Manifest rows: {len(rows)}")
    print(f"Train rows:    {len(train_rows)}")
    print(f"Test rows:     {len(test_rows)}")
    print("Train counts:", dict(sorted(Counter(row["generic_label"] for row in train_rows).items())))
    print("Test counts: ", dict(sorted(Counter(row["generic_label"] for row in test_rows).items())))

    model, processor, device = load_clip()
    embeddings, valid_train_rows = encode_images(train_rows, model, processor, device)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    IDS_PATH.write_text(
        json.dumps([row["kb_id"] for row in valid_train_rows], indent=2),
        encoding="utf-8",
    )
    META_PATH.write_text(json.dumps(valid_train_rows, indent=2), encoding="utf-8")
    SPLIT_PATH.write_text(
        json.dumps({"train": valid_train_rows, "test": test_rows}, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved {INDEX_PATH.relative_to(ROOT)}")
    print(f"Indexed vectors: {index.ntotal}, dim={embeddings.shape[1]}")
    print(f"Saved {IDS_PATH.relative_to(ROOT)}")
    print(f"Saved {META_PATH.relative_to(ROOT)}")
    print(f"Saved {SPLIT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
