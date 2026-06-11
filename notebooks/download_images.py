"""
Download campus-relevant images using LoremFlickr (keyword-based, free).
LoremFlickr searches Flickr's CC-licensed photos by keyword.
"""
import requests, time
from pathlib import Path
from PIL import Image
from io import BytesIO

IMG_DIR = Path("data/images")
IMG_DIR.mkdir(exist_ok=True)

CATEGORIES = {
    "library":    ["university+library+interior", "library+books+reading", "academic+library"],
    "cafeteria":  ["university+cafeteria", "school+canteen+food", "campus+dining+hall"],
    "classroom":  ["university+classroom+lecture", "lecture+hall+university", "college+classroom"],
    "gym":        ["indoor+gym+fitness", "university+sports+gym", "fitness+center+weights"],
    "lab":        ["computer+lab+university", "science+laboratory+university", "engineering+lab"],
    "auditorium": ["university+auditorium", "lecture+theatre+seats", "conference+hall+auditorium"],
    "admin":      ["university+office+reception", "campus+administration+desk", "university+reception"],
    "department": ["university+corridor+building", "campus+hallway+academic", "university+department"],
}

headers = {"User-Agent": "Mozilla/5.0"}
total = 0

for cat, keywords in CATEGORIES.items():
    cat_dir = IMG_DIR / cat
    cat_dir.mkdir(exist_ok=True)
    for f in cat_dir.glob("*.jpg"): f.unlink()  # clear old

    count = 0
    for i, kw in enumerate(keywords):
        for attempt in range(3):  # 3 images per keyword
            try:
                url = f"https://loremflickr.com/640/480/{kw}?lock={i*10+attempt}"
                r = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
                if r.status_code == 200 and len(r.content) > 8000:
                    img = Image.open(BytesIO(r.content)).convert("RGB")
                    out = cat_dir / f"{cat}_{count+1:02d}.jpg"
                    img.save(str(out), "JPEG")
                    print(f"  [OK] {cat}/{out.name} {img.size}")
                    count += 1
                    time.sleep(0.5)
                else:
                    print(f"  [SKIP] {kw} attempt {attempt}: {r.status_code}")
            except Exception as e:
                print(f"  [ERR] {kw}: {e}")
        if count >= 9:
            break

    total += count
    print(f"  → {cat}: {count} images\n")

print(f"\nTotal: {total} images downloaded")
