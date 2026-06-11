import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "data" / "images"
MANIFEST = ROOT / "data" / "image_manifest.csv"

LABEL_MAP = {
    "admin": "admin_office",
    "auditorium": "auditorium_main",
    "cafeteria": "main_cafeteria",
    "classroom": "cse_classroom_101",
    "department": "cse_department",
    "gym": "gym",
    "lab": "computer_lab",
    "library": "central_library",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    rows = []
    for label, kb_id in sorted(LABEL_MAP.items()):
        folder = IMG_DIR / label
        if not folder.exists():
            print(f"[WARN] Missing folder: {folder}")
            continue

        images = sorted(
            path for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        )
        if not images:
            print(f"[WARN] No images found in: {folder}")
            continue

        for image in images:
            rows.append({
                "image_path": image.relative_to(ROOT).as_posix(),
                "generic_label": label,
                "kb_id": kb_id,
            })
        print(f"{label:12s} {len(images):3d} images -> {kb_id}")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "generic_label", "kb_id"],
        )
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["generic_label"] for row in rows)
    print(f"\nSaved {MANIFEST.relative_to(ROOT)} with {len(rows)} rows")
    for label, count in sorted(counts.items()):
        if count < 10:
            print(f"[WARN] {label} has only {count} images; add more if possible.")

    if not rows:
        raise SystemExit("No images found. Add images under data/images/<label>/ first.")


if __name__ == "__main__":
    main()
