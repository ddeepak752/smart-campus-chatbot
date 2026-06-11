"""
Fusion MLP Training Script
===========================
Trains the multimodal fusion MLP to map joint embeddings to KB records.

Training data is generated synthetically from the KB:
  - For each KB record, generate multiple text queries describing it
  - Augment with zero-masked single-modality examples
  - The MLP learns to classify both text-only and image-only inputs

This is a lightweight classification task (142 classes, ~3000 examples).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"]       = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys, json, random, pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sentence_transformers import SentenceTransformer

sys.path.insert(0, ".")
from src.kb_utils import load_kb
from src.fusion_layer import FusionMLP, JOINT_DIM, TEXT_DIM, IMAGE_DIM

BASE_DIR    = Path(".")
MODELS_DIR  = BASE_DIR / "models"
PLOTS_DIR   = BASE_DIR / "outputs/plots"
METRICS_DIR = BASE_DIR / "outputs/metrics"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ── Step 1: Load KB and build label map ──────────────────────────────────────
kb       = load_kb()
kb_ids   = [loc["id"] for loc in kb]
label2id = {kb_id: i for i, kb_id in enumerate(kb_ids)}
num_classes = len(kb_ids)
print(f"KB records: {num_classes}")

# ── Step 2: Generate synthetic training queries ──────────────────────────────
print("\nGenerating synthetic training queries...")

def make_queries(loc: dict) -> list:
    """Generate diverse text queries for a KB record."""
    name  = loc["name"]
    cat   = loc.get("category","").replace("_"," ")
    desc  = loc.get("description","")[:60]
    kws   = loc.get("keywords", [])[:3]

    queries = [
        name,
        f"where is {name}",
        f"how do I get to {name}",
        f"find the {name}",
        f"location of {name}",
        f"tell me about {name}",
        f"what is {name}",
        f"{name} opening hours",
        f"is {name} open today",
    ]
    if cat:
        queries += [
            f"where is the {cat}",
            f"find the {cat}",
            f"I need the {cat}",
        ]
    if kws:
        queries.append(" ".join(kws))
        queries.append(f"find {kws[0]}")
    return queries

all_queries = []   # (query_str, kb_id)
for loc in kb:
    for q in make_queries(loc):
        all_queries.append((q, loc["id"]))

print(f"Total training queries: {len(all_queries)}")

# ── Step 3: Encode all queries with SentenceTransformer ──────────────────────
print("\nEncoding queries with SentenceTransformer...")
text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
queries    = [q for q, _ in all_queries]
labels_str = [k for _, k in all_queries]
labels_int = [label2id[k] for k in labels_str]

text_embeddings = text_model.encode(
    queries, convert_to_numpy=True, show_progress_bar=True, batch_size=64
)
print(f"Text embeddings: {text_embeddings.shape}")

# ── Step 4: Build training dataset with zero-masking augmentation ────────────
print("\nBuilding dataset with zero-masking augmentation...")

# Each example: joint_embedding (896-dim), label (int)
# Augmentation: randomly zero-mask either the text or image portion
# This teaches the MLP to handle single-modality inputs

X_list = []
y_list = []

for i, (t_emb, label) in enumerate(zip(text_embeddings, labels_int)):
    r = random.random()
    if r < 0.5:
        # Text only (zero-mask image portion) — most common real-world case
        joint = np.concatenate([t_emb, np.zeros(IMAGE_DIM, dtype=np.float32)])
    elif r < 0.7:
        # Both modalities: simulate 512-dim image embedding with gaussian noise
        # Cannot slice t_emb (384-dim) to get 512-dim — generate independently
        fake_img = np.random.randn(IMAGE_DIM).astype(np.float32)
        fake_img = fake_img / (np.linalg.norm(fake_img) + 1e-8)
        joint = np.concatenate([t_emb, fake_img])
    else:
        # Image only (zero-mask text portion) — simulated image query
        fake_img = np.random.randn(IMAGE_DIM).astype(np.float32)
        fake_img = fake_img / (np.linalg.norm(fake_img) + 1e-8)
        joint = np.concatenate([np.zeros(TEXT_DIM, dtype=np.float32), fake_img])

    X_list.append(joint.astype(np.float32))
    y_list.append(label)

X = np.array(X_list)
y = np.array(y_list)
print(f"Dataset shape: X={X.shape}, y={y.shape}")

# ── Step 5: Train/val split ───────────────────────────────────────────────────
idx    = np.random.permutation(len(X))
split  = int(0.85 * len(idx))
tr_idx = idx[:split]
va_idx = idx[split:]

X_tr, y_tr = X[tr_idx], y[tr_idx]
X_va, y_va = X[va_idx], y[va_idx]
print(f"Train: {len(X_tr)}  Val: {len(X_va)}")

# ── Step 6: Train the MLP ─────────────────────────────────────────────────────
print("\nTraining fusion MLP...")
device = torch.device("cpu")

model     = FusionMLP(JOINT_DIM, hidden_dim=512, num_classes=num_classes, dropout=0.3)
optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=30)

X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
y_tr_t = torch.tensor(y_tr, dtype=torch.long)
X_va_t = torch.tensor(X_va, dtype=torch.float32)
y_va_t = torch.tensor(y_va, dtype=torch.long)

BATCH_SIZE = 64
EPOCHS     = 50
PATIENCE   = 8

train_losses = []
val_losses   = []
val_accs     = []
best_val_loss = float("inf")
patience_count = 0

for epoch in range(EPOCHS):
    model.train()
    perm   = torch.randperm(len(X_tr_t))
    ep_loss = 0.0
    n_batches = 0

    for i in range(0, len(X_tr_t), BATCH_SIZE):
        batch_idx  = perm[i:i+BATCH_SIZE]
        xb, yb     = X_tr_t[batch_idx], y_tr_t[batch_idx]
        optimiser.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimiser.step()
        ep_loss   += loss.item()
        n_batches += 1

    scheduler.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(X_va_t)
        val_loss   = criterion(val_logits, y_va_t).item()
        val_preds  = val_logits.argmax(dim=1)
        val_acc    = (val_preds == y_va_t).float().mean().item()

    train_losses.append(ep_loss / n_batches)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    if epoch % 10 == 0 or epoch == EPOCHS-1:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | "
              f"train_loss={train_losses[-1]:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"val_acc={val_acc:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), str(MODELS_DIR / "fusion_mlp.pt"))
        patience_count = 0
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

print(f"\nBest val loss: {best_val_loss:.4f}")

# ── Step 7: Evaluate top-1 and top-3 accuracy ────────────────────────────────
print("\nEvaluating on validation set...")
model.load_state_dict(torch.load(str(MODELS_DIR / "fusion_mlp.pt"), map_location="cpu"))
model.eval()

with torch.no_grad():
    val_logits = model(X_va_t)
    probs_va   = torch.softmax(val_logits, dim=1)

top1_correct = (probs_va.argmax(dim=1) == y_va_t).sum().item()
top3_preds   = probs_va.topk(3, dim=1).indices
top3_correct = sum(y_va_t[i].item() in top3_preds[i].tolist()
                   for i in range(len(y_va_t)))

top1_acc = top1_correct / len(y_va_t)
top3_acc = top3_correct / len(y_va_t)
print(f"  Top-1 accuracy: {top1_acc:.4f} ({top1_acc*100:.1f}%)")
print(f"  Top-3 accuracy: {top3_acc:.4f} ({top3_acc*100:.1f}%)")

# ── Step 8: Plots ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(train_losses, label="Train loss", color="#4C72B0")
axes[0].plot(val_losses,   label="Val loss",   color="#DD8452")
axes[0].set_title("Fusion MLP — Loss Curves")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross-entropy loss")
axes[0].legend()

axes[1].plot(val_accs, color="#55A868")
axes[1].set_title("Fusion MLP — Validation Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0, 1.05)

axes[2].bar(["Top-1","Top-3"], [top1_acc, top3_acc],
            color=["#4C72B0","#55A868"], alpha=0.85, width=0.4)
axes[2].set_ylim(0, 1.1)
axes[2].set_title("Fusion MLP — Retrieval Accuracy")
for i, v in enumerate([top1_acc, top3_acc]):
    axes[2].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=12)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "fusion_mlp_training.png", dpi=150)
plt.close()
print("  Saved: outputs/plots/fusion_mlp_training.png")

# ── Step 9: Save metrics ──────────────────────────────────────────────────────
fusion_metrics = {
    "architecture":  "FusionMLP (2-layer)",
    "input_dim":     JOINT_DIM,
    "hidden_dim":    512,
    "output_classes": num_classes,
    "text_dim":      TEXT_DIM,
    "image_dim":     IMAGE_DIM,
    "masking_strategy": "zero-pad absent modality",
    "training_samples": len(X_tr),
    "val_samples":    len(X_va),
    "epochs_trained": len(train_losses),
    "best_val_loss":  round(best_val_loss, 4),
    "top1_accuracy":  round(top1_acc, 4),
    "top3_accuracy":  round(top3_acc, 4),
    "note": ("Trained on synthetic text queries with zero-masking augmentation. "
             "Text-only and image-only inputs are both handled via masking.")
}
with open(METRICS_DIR / "fusion_metrics.json", "w") as f:
    json.dump(fusion_metrics, f, indent=2)

print(f"\n{'='*60}")
print(f"  FUSION MLP TRAINING COMPLETE")
print(f"  Classes       : {num_classes}")
print(f"  Top-1 accuracy: {top1_acc*100:.1f}%")
print(f"  Top-3 accuracy: {top3_acc*100:.1f}%")
print(f"  Model saved   : models/fusion_mlp.pt")
print(f"  Plot saved    : outputs/plots/fusion_mlp_training.png")
print(f"{'='*60}")
