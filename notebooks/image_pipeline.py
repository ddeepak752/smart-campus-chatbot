import os
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["OPENBLAS_NUM_THREADS"]   = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"]    = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"]  = "TRUE"
"""
Image Pipeline - CLIP + FAISS for campus location retrieval.
"""

import os, sys, json, pickle, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import requests, faiss, torch
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from io import BytesIO
from torchvision import transforms

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"]  = "TRUE"
sys.path.insert(0, ".")

from src.kb_utils import load_kb

BASE_DIR    = Path(".")
IMG_DIR     = BASE_DIR / "data/images"
PLOTS_DIR   = BASE_DIR / "outputs/plots"
METRICS_DIR = BASE_DIR / "outputs/metrics"
MODELS_DIR  = BASE_DIR / "models"

for d in [IMG_DIR, PLOTS_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── STEP 1: Verify downloaded images ─────────────────────────────────────────
print("Step 1: Checking downloaded images...")
CATEGORIES = ["library","cafeteria","classroom","gym","department","lab","auditorium","admin"]

downloaded = {}
for cat in CATEGORIES:
    cat_dir = IMG_DIR / cat
    if cat_dir.exists():
        files = sorted([f for f in cat_dir.iterdir() if f.suffix in (".jpg",".png")])
        downloaded[cat] = files
    else:
        downloaded[cat] = []

total_images = sum(len(v) for v in downloaded.values())
print(f"  Found {total_images} images across {len(CATEGORIES)} categories")
for cat, files in downloaded.items():
    print(f"    {cat:15s}: {len(files)} images")

# ── STEP 2: Class Distribution Plot ──────────────────────────────────────────
print("\nStep 2: Generating class distribution plot...")

counts = {cat: len(files) for cat, files in downloaded.items()}
colors = sns.color_palette("husl", len(counts))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cats = list(counts.keys())
vals = list(counts.values())

axes[0].bar(cats, vals, color=colors, alpha=0.85)
axes[0].set_title("Image Count per Campus Category")
axes[0].set_xlabel("Category")
axes[0].set_ylabel("Number of Images")
axes[0].tick_params(axis="x", rotation=30)
for i, v in enumerate(vals):
    axes[0].text(i, v + 0.05, str(v), ha="center", fontsize=10)

axes[1].pie(vals, labels=cats, colors=colors, autopct="%1.0f%%",
            startangle=90, pctdistance=0.85)
axes[1].set_title("Category Distribution")

plt.suptitle("Campus Image Dataset — Class Distribution", fontsize=13)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "image_class_distribution.png", dpi=150)
plt.close()
print("  Saved: outputs/plots/image_class_distribution.png")

# ── STEP 3: Sample Image Grid ─────────────────────────────────────────────────
print("\nStep 3: Generating annotated sample image grid...")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx, cat in enumerate(CATEGORIES):
    files = downloaded.get(cat, [])
    ax    = axes[idx]
    if not files:
        ax.set_title(f"{cat}\n(no images)")
        ax.axis("off")
        continue
    try:
        # Load image using numpy — avoids PIL/matplotlib conflict
        img_pil = Image.open(str(files[0])).convert("RGB").resize((224, 224))
        img_np  = np.array(img_pil)
        ax.imshow(img_np)
        ax.set_title(f"{cat.upper()}\n({len(files)} images)", fontsize=9, fontweight="bold")
        ax.axis("off")
    except Exception as e:
        ax.set_title(f"{cat}\n(error)")
        ax.axis("off")

plt.suptitle("Campus Image Dataset — One Sample per Category", fontsize=13)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "image_sample_grid.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: outputs/plots/image_sample_grid.png")

# ── STEP 4: Preprocessing Pipeline Visualisation ─────────────────────────────
print("\nStep 4: Visualising preprocessing pipeline...")

# Find a sample image
sample_path = None
for cat in ["library","cafeteria","gym"]:
    if downloaded.get(cat):
        sample_path = downloaded[cat][0]
        break

# Preprocessing pipeline diagram (text-based — avoids M3 PIL/matplotlib segfault)
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
steps = [
    ("1. Load & Resize", "PIL.Image.open()\nresize(224, 224)\nconvert('RGB')", "#4C72B0"),
    ("2. Normalise", "transforms.ToTensor()\nNormalize(\n  mean=[0.481,0.458,0.408]\n  std=[0.269,0.261,0.276])", "#DD8452"),
    ("3. Augment (train)", "RandomCrop(224)\nRandomHorizontalFlip()\nColorJitter()\nRandomRotation(10°)", "#55A868"),
    ("4. CLIP Input", "torch.Tensor\nshape: [3, 224, 224]\nReady for\nCLIP encoder", "#C44E52"),
]
for ax, (title, desc, color) in zip(axes, steps):
    ax.set_facecolor(color + "22")
    ax.text(0.5, 0.6, title, ha="center", va="center",
            fontsize=11, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.25, desc, ha="center", va="center",
            fontsize=8, family="monospace", transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2)
        spine.set_visible(True)

# Add arrows between steps
for i in range(3):
    axes[i].annotate("", xy=(1.05, 0.5), xytext=(1.0, 0.5),
                     xycoords="axes fraction", textcoords="axes fraction",
                     arrowprops=dict(arrowstyle="->", color="gray", lw=2))

plt.suptitle("Image Preprocessing Pipeline — CLIP Input Preparation", fontsize=12)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "image_preprocessing_pipeline.png", dpi=150)
plt.close()
print("  Saved: outputs/plots/image_preprocessing_pipeline.png")

# ── STEP 5: Build CLIP + FAISS Index ─────────────────────────────────────────
print("\nStep 5: Loading CLIP and building FAISS index...")

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from transformers import CLIPProcessor, CLIPModel

device = torch.device("cpu")   # force CPU — CLIP frozen, no GPU needed
clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
print("  CLIP ViT-B/32 loaded (CPU, transformers)")

kb = load_kb()

def build_clip_text(loc: dict) -> str:
    CAT_VISUAL = {
        "library":        "university library bookshelves reading tables study area",
        "cafeteria":      "university cafeteria canteen food counter dining tables",
        "classroom":      "university classroom lecture hall desks chairs blackboard",
        "department":     "university department office corridor academic building",
        "engineering_lab":"engineering lab workbenches electronic equipment circuits",
        "science_lab":    "science laboratory test tubes equipment benches chemicals",
        "lab":            "computer laboratory rows of computers screens university",
        "project_lab":    "project laboratory students working computers equipment",
        "auditorium":     "university auditorium rows of seats stage large hall",
        "seminar_hall":   "seminar hall conference room chairs presentation screen",
        "administration": "university administration office reception front desk staff",
        "reception":      "university reception front desk information counter lobby",
        "sports":         "university gym fitness equipment weights exercise machines",
        "study_area":     "quiet study room reading area individual desks lamps",
        "hostel":         "student hostel dormitory accommodation rooms beds",
        "parking":        "campus outdoor parking lot cars vehicles",
        "medical":        "medical room first aid station bed medicine cabinet",
        "welfare":        "counselling room welfare office chairs sofa private",
        "student_union":  "student union common room social space lounge tables",
        "store":          "campus shop stationery store shelves supplies",
        "workshop":       "engineering workshop tools machinery workbench industrial",
        "innovation":     "innovation lab startup space modern open plan whiteboards",
        "faculty_room":   "faculty staff room teachers office desks computers",
        "hod_office":     "department head office formal desk nameplate",
        "meeting_room":   "meeting room conference table chairs whiteboard",
        "washroom":       "clean washroom restroom toilet sink tiles",
        "facility":       "university campus building interior facility",
        "entrance":       "university main gate entrance security barrier",
        "common_room":    "student common room sofas tables relaxing space",
        "it_support":     "IT helpdesk technical support computers staff",
        "security":       "security cabin guard post barrier gate",
        "transport":      "bus stop transport area vehicles road",
        "research":       "research lab academic equipment computers data",
        "careers":        "careers office job boards posters professional",
        "placement":      "placement cell recruitment office career services",
    }
    cat  = loc.get("category", "")
    name = loc.get("name", "")
    base = CAT_VISUAL.get(cat, f"university {cat.replace('_',' ')} room")
    return f"{base}" s
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["OPENBLAS_NUM_THREADS"]   = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"]    = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"]  = "TRUE"
"""
Image Pipeline - CLIP + FAISS for campus location retrieval.
"""

import os, sys, json, pickle, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import requests, faiss, torch
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from io import BytesIO
from torchvision import transforms

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"]  = "TRUE"
sys.path.insert(0, ".")

from src.kb_utils import load_kb

BASE_DIR    = Path(".")
IMG_DIR     = BASE_DIR / "data/images"
PLOTS_DIR   = BASE_DIR / "outputs/plots"
METRICS_DIR = BASE_DIR / "outputs/metrics"
MODELS_DIR  = BASE_DIR / "models"

for d in [IMG_DIR, PLOTS_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── STEP 1: Verify downloaded images ─────────────────────────────────────────
print("Step 1: Checking downloaded images...")
CATEGORIES = ["library","cafeteria","classroom","gym","department","lab","auditorium","admin"]

downloaded = {}
for cat in CATEGORIES:
    cat_dir = IMG_DIR / cat
    if cat_dir.exists():
        files = sorted([f for f in cat_dir.iterdir() if f.suffix in (".jpg",".png")])
        downloaded[cat] = files
    else:
        downloaded[cat] = []

total_images = sum(len(v) for v in downloaded.values())
print(f"  Found {total_images} images across {len(CATEGORIES)} categories")
for cat, files in downloaded.items():
    print(f"    {cat:15s}: {len(files)} images")

# ── STEP 2: Class Distribution Plot ──────────────────────────────────────────
print("\nStep 2: Generating class distribution plot...")

counts = {cat: len(files) for cat, files in downloaded.items()}
colors = sns.color_palette("husl", len(counts))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cats = list(counts.keys())
vals = list(counts.values())

axes[0].bar(cats, vals, color=colors, alpha=0.85)
axes[0].set_title("Image Count per Campus Category")
axes[0].set_xlabel("Category")
axes[0].set_ylabel("Number of Images")
axes[0].tick_params(axis="x", rotation=30)
for i, v in enumerate(vals):
    axes[0].text(i, v + 0.05, str(v), ha="center", fontsize=10)

axes[1].pie(vals, labels=cats, colors=colors, autopct="%1.0f%%",
            startangle=90, pctdistance=0.85)
axes[1].set_title("Category Distribution")

plt.suptitle("Campus Image Dataset — Class Distribution", fontsize=13)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "image_class_distribution.png", dpi=150)
plt.close()
print("  Saved: outputs/plots/image_class_distribution.png")

# ── STEP 3: Sample Image Grid ─────────────────────────────────────────────────
print("\nStep 3: Generating annotated sample image grid...")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx, cat in enumerate(CATEGORIES):
    files = downloaded.get(cat, [])
    ax    = axes[idx]
    if not files:
        ax.set_title(f"{cat}\n(no images)")
        ax.axis("off")
        continue
    try:
        # Load image using numpy — avoids PIL/matplotlib conflict
        img_pil = Image.open(str(files[0])).convert("RGB").resize((224, 224))
        img_np  = np.array(img_pil)
        ax.imshow(img_np)
        ax.set_title(f"{cat.upper()}\n({len(files)} images)", fontsize=9, fontweight="bold")
        ax.axis("off")
    except Exception as e:
        ax.set_title(f"{cat}\n(error)")
        ax.axis("off")

plt.suptitle("Campus Image Dataset — One Sample per Category", fontsize=13)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "image_sample_grid.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: outputs/plots/image_sample_grid.png")

# ── STEP 4: Preprocessing Pipeline Visualisation ─────────────────────────────
print("\nStep 4: Visualising preprocessing pipeline...")

# Find a sample image
sample_path = None
for cat in ["library","cafeteria","gym"]:
    if downloaded.get(cat):
        sample_path = downloaded[cat][0]
        break

# Preprocessing pipeline diagram (text-based — avoids M3 PIL/matplotlib segfault)
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
steps = [
    ("1. Load & Resize", "PIL.Image.open()\nresize(224, 224)\nconvert('RGB')", "#4C72B0"),
    ("2. Normalise", "transforms.ToTensor()\nNormalize(\n  mean=[0.481,0.458,0.408]\n  std=[0.269,0.261,0.276])", "#DD8452"),
    ("3. Augment (train)", "RandomCrop(224)\nRandomHorizontalFlip()\nColorJitter()\nRandomRotation(10°)", "#55A868"),
    ("4. CLIP Input", "torch.Tensor\nshape: [3, 224, 224]\nReady for\nCLIP encoder", "#C44E52"),
]
for ax, (title, desc, color) in zip(axes, steps):
    ax.set_facecolor(color + "22")
    ax.text(0.5, 0.6, title, ha="center", va="center",
            fontsize=11, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.25, desc, ha="center", va="center",
            fontsize=8, family="monospace", transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2)
        spine.set_visible(True)

# Add arrows between steps
for i in range(3):
    axes[i].annotate("", xy=(1.05, 0.5), xytext=(1.0, 0.5),
                     xycoords="axes fraction", textcoords="axes fraction",
                     arrowprops=dict(arrowstyle="->", color="gray", lw=2))

plt.suptitle("Image Preprocessing Pipeline — CLIP Input Preparation", fontsize=12)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "image_preprocessing_pipeline.png", dpi=150)
plt.close()
print("  Saved: outputs/plots/image_preprocessing_pipeline.png")

# ── STEP 5: Build CLIP + FAISS Index ─────────────────────────────────────────
print("\nStep 5: Loading CLIP and building FAISS index...")

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from transformers import CLIPProcessor, CLIPModel

device = torch.device("cpu")   # force CPU — CLIP frozen, no GPU needed
clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
print("  CLIP ViT-B/32 loaded (CPU, transformers)")

kb = load_kb()

def build_clip_text(loc: dict) -> str:
    VISUAL = {
        "library":        "a photo of a university library with bookshelves and reading tables",
        "cafeteria":      "a photo of a university cafeteria or canteen with food and tables",
        "classroom":      "a photo of a university classroom with desks chairs and blackboard",
        "department":     "a photo of a university department building corridor or office",
        "engineering_lab":"a photo of an engineering lab with workbenches and equipment",
        "science_lab":    "a photo of a science laboratory with equipment and benches",
        "lab":            "a photo of a computer lab or science laboratory",
        "project_lab":    "a photo of a university project laboratory",
        "auditorium":     "a photo of a university auditorium or lecture theatre with rows of seats",
        "seminar_hall":   "a photo of a seminar hall or conference room with chairs",
        "administration": "a photo of a university administration office or reception desk",
        "reception":      "a photo of a university reception or front desk",
        "sports":         "a photo of a university gym or indoor sports facility",
        "study_area":     "a photo of a quiet study area or reading room with desks",
        "hostel":         "a photo of a university student hostel building",
        "parking":        "a photo of a campus parking area",
        "medical":        "a photo of a medical room or first aid station",
        "welfare":        "a photo of a counselling or welfare office",
        "student_union":  "a photo of a student union building or common room",
        "store":          "a photo of a campus stationery or supply store",
        "workshop":       "a photo of an engineering workshop with tools",
        "innovation":     "a photo of an innovation lab or startup space",
        "careers":        "a photo of a careers or placement office",
        "placement":      "a photo of a placement or careers office",
        "faculty_room":   "a photo of a university staff room or faculty office",
        "hod_office":     "a photo of a department head office",
        "meeting_room":   "a photo of a university conference or meeting room",
        "washroom":       "a photo of a clean university washroom or restroom",
        "facility":       "a photo of a university campus facility or building interior",
        "entrance":       "a photo of a university main gate or campus entrance",
        "common_room":    "a photo of a student common room with sofas and tables",
        "it_support":     "a photo of an IT helpdesk or technical support desk",
        "security":       "a photo of a security cabin or guard post",
        "transport":      "a photo of a university bus stop or transport area",
        "research":       "a photo of a university research centre or lab",
    }
    cat = loc.get("category", "")
    if cat in VISUAL:
        return VISUAL[cat]
    return f"a photo of a university {cat.replace('_', ' ')} building or room" 

kb_texts = [build_clip_text(loc) for loc in kb]
kb_ids   = [loc["id"] for loc in kb]

print(f"  Encoding {len(kb_texts)} KB records...")
text_embeddings = []
BATCH = 32

with torch.no_grad():
    for i in range(0, len(kb_texts), BATCH):
        batch  = kb_texts[i:i+BATCH]
        inputs = clip_processor(text=batch, return_tensors="pt",
                                padding=True, truncation=True, max_length=77)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        emb = clip_model.get_text_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        text_embeddings.append(emb.cpu().numpy())

text_embeddings = np.vstack(text_embeddings).astype("float32")
dim = text_embeddings.shape[1]
print(f"  Text embeddings: {text_embeddings.shape}")

index = faiss.IndexFlatIP(dim)
index.add(text_embeddings)
faiss.write_index(index, str(MODELS_DIR / "clip_faiss.index"))
with open(MODELS_DIR / "clip_kb_ids.json", "w") as f:
    json.dump(kb_ids, f)
print(f"  FAISS index saved: {index.ntotal} vectors, dim={dim}")

# ── STEP 6: Evaluate Retrieval ────────────────────────────────────────────────
print("\nStep 6: Evaluating CLIP retrieval on downloaded images...")

EVAL_MAP = {
    "library":    ["central_library","digital_library","reading_room"],
    "cafeteria":  ["main_cafeteria","coffee_kiosk","canteen_terrace"],
    "classroom":  ["cse_classroom_101","it_classroom_101","cse_classroom_102"],
    "gym":        ["gym","sports_ground","indoor_games_room"],
    "lab":        ["computer_lab","networking_lab","ai_lab","biotech_lab","physics_lab"],
    "auditorium": ["auditorium_main","seminar_hall_main"],
    "admin":      ["reception","admin_office","accounts_office"],
    "department": ["cse_department","it_department","ai_ds_department"],
}

top1_correct = top3_correct = n_eval = 0
retrieval_results = []

for cat, exp_ids in EVAL_MAP.items():
    files = downloaded.get(cat, [])[:2]
    for img_path in files:
        try:
            img = Image.open(str(img_path)).convert("RGB")
            with torch.no_grad():
                inputs  = clip_processor(images=img, return_tensors="pt")
                inputs  = {k: v.to(device) for k, v in inputs.items()}
                img_emb = clip_model.get_image_features(**inputs)
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                img_np  = img_emb.cpu().numpy().astype("float32")

            scores, idxs = index.search(img_np, k=3)
            top1_id  = kb_ids[idxs[0][0]]
            top3_ids = [kb_ids[idxs[0][i]] for i in range(min(3,len(idxs[0])))]

            t1 = top1_id in exp_ids
            t3 = any(x in exp_ids for x in top3_ids)
            top1_correct += int(t1)
            top3_correct += int(t3)
            n_eval += 1

            status = "✅" if t1 else ("⚠️" if t3 else "❌")
            print(f"  {status} [{cat:12s}] top1={top1_id:35s} score={scores[0][0]:.3f}")
            retrieval_results.append({
                "category":cat,"image":img_path.name,
                "top1":top1_id,"top3":top3_ids,
                "top1_correct":t1,"top3_correct":t3,
                "score":float(scores[0][0])
            })
        except Exception as e:
            print(f"  [WARN] {img_path.name}: {e}")

top1_acc = top1_correct / n_eval if n_eval else 0
top3_acc = top3_correct / n_eval if n_eval else 0
print(f"\n  Top-1: {top1_acc:.2f} | Top-3: {top3_acc:.2f}")

# ── STEP 7: Retrieval Plot ────────────────────────────────────────────────────
print("\nStep 7: Saving retrieval accuracy plot...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(["Top-1","Top-3"], [top1_acc, top3_acc],
            color=["#4C72B0","#55A868"], alpha=0.85, width=0.4)
axes[0].set_ylim(0, 1.1)
axes[0].set_title("CLIP + FAISS Retrieval Accuracy\n(zero-shot, no training)")
axes[0].set_ylabel("Accuracy")
for i, v in enumerate([top1_acc, top3_acc]):
    axes[0].text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=13)

if retrieval_results:
    sc  = [r["score"] for r in retrieval_results]
    axes[1].hist(sc, bins=8, color="#DD8452", alpha=0.85, edgecolor="white")
    axes[1].axvline(np.mean(sc), color="red", linestyle="--",
                    label=f"Mean: {np.mean(sc):.3f}")
    axes[1].set_title("CLIP Cosine Similarity Distribution")
    axes[1].set_xlabel("Cosine Similarity")
    axes[1].set_ylabel("Count")
    axes[1].legend()

plt.tight_layout()
plt.savefig(PLOTS_DIR / "clip_retrieval_accuracy.png", dpi=150)
plt.close()
print("  Saved: outputs/plots/clip_retrieval_accuracy.png")

# ── STEP 8: Save metrics ──────────────────────────────────────────────────────
image_metrics = {
    "clip_model":           "openai/clip-vit-base-patch32",
    "embedding_dim":        int(dim),
    "faiss_index_type":     "IndexFlatIP (cosine similarity)",
    "kb_records_indexed":   index.ntotal,
    "top1_accuracy":        round(float(top1_acc), 4),
    "top3_accuracy":        round(float(top3_acc), 4),
    "eval_images":          n_eval,
    "dataset_images":       total_images,
    "categories":           CATEGORIES,
    "images_per_category":  counts,
    "note": "CLIP frozen. Zero labelled training data. KB text descriptions encoded as retrieval targets."
}
with open(METRICS_DIR / "image_metrics.json", "w") as f:
    json.dump(image_metrics, f, indent=2)
with open(METRICS_DIR / "clip_retrieval_results.json", "w") as f:
    json.dump(retrieval_results, f, indent=2)

print(f"\n{'='*60}")
print(f"  IMAGE PIPELINE COMPLETE")
print(f"  Images          : {total_images} ({len(CATEGORIES)} categories)")
print(f"  CLIP index      : {index.ntotal} KB records, dim={dim}")
print(f"  Top-1 accuracy  : {top1_acc:.2f}")
print(f"  Top-3 accuracy  : {top3_acc:.2f}")
print(f"  Plots           : outputs/plots/ (4 plots)")
print(f"  Index           : models/clip_faiss.index")
print(f"{'='*60}")
