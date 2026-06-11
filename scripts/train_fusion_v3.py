
import os, sys, json, csv, random
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"]       = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from src.kb_utils import load_kb

ROOT        = Path(__file__).parent.parent
MANIFEST    = ROOT / "data/image_manifest.csv"
MODELS_DIR  = ROOT / "models"
PLOTS_DIR   = ROOT / "outputs/plots"
METRICS_DIR = ROOT / "outputs/metrics"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42); np.random.seed(42); torch.manual_seed(42)

TEXT_DIM  = 512
IMAGE_DIM = 512
JOINT_DIM = TEXT_DIM + IMAGE_DIM

VISUAL_KB_IDS = [
    "central_library", "main_cafeteria", "cse_classroom_101",
    "gym", "computer_lab", "auditorium_main", "admin_office", "cse_department",
]

print("="*60)
print("  FUSION MLP v3 — 8 visual classes")
print("="*60)

kb_full   = load_kb()
kb_visual = [loc for loc in kb_full if loc["id"] in VISUAL_KB_IDS]
label2id  = {loc["id"]: i for i, loc in enumerate(kb_visual)}
id2label  = {i: loc["id"] for i, loc in enumerate(kb_visual)}
n_classes = len(kb_visual)
print(f"Classes: {n_classes}")
for loc in kb_visual:
    print(f"  {loc['id']:25s} — {loc['name']}")

print("\nLoading CLIP...")
device         = torch.device("cpu")
clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

def enc_text(texts):
    all_embs = []
    for i in range(0, len(texts), 32):
        b = texts[i:i+32]
        inp = clip_processor(text=b, return_tensors="pt", padding=True, truncation=True, max_length=77)
        with torch.no_grad():
            e = clip_model.get_text_features(**inp)
            e = e / e.norm(dim=-1, keepdim=True)
        all_embs.append(e.cpu().numpy().astype("float32"))
    return np.vstack(all_embs)

def enc_img(p):
    img = Image.open(p).convert("RGB")
    inp = clip_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        e = clip_model.get_image_features(**inp)
        e = e / e.norm(dim=-1, keepdim=True)
    return e.cpu().numpy()[0].astype("float32")

print("\nLoading real images...")
real_embs = {}
for row in csv.DictReader(open(MANIFEST)):
    kb_id = row["kb_id"]
    if kb_id not in VISUAL_KB_IDS: continue
    try:
        e = enc_img(ROOT / row["image_path"])
        if kb_id not in real_embs: real_embs[kb_id] = []
        real_embs[kb_id].append(e)
    except: pass
for k,v in real_embs.items():
    print(f"  {k:25s}: {len(v)} images")

def make_queries(loc):
    name = loc["name"]; cat = loc.get("category","").replace("_"," ")
    kws  = loc.get("keywords",[])[:4]
    qs = [name, name.lower(), f"where is {name}", f"find {name}",
          f"how to get to {name}", f"directions to {name}",
          f"locate {name}", f"show me {name}", f"tell me about {name}",
          f"{name} location", f"{name} opening hours", f"is {name} open",
          f"what is {name}", f"I need to go to {name}", f"take me to {name}",
          f"I am looking for {name}", f"where exactly is {name}"]
    if cat:
        qs += [f"where is the {cat}", f"find the {cat}", f"I need the {cat}",
               f"{cat} location", f"{cat} on campus", f"campus {cat}",
               f"university {cat}", f"where is a {cat}", f"take me to the {cat}"]
    for kw in kws:
        qs += [kw, f"find {kw}", f"where is {kw}"]
    return [q for q in qs if q.strip()]

all_queries = []
for loc in kb_visual:
    for q in make_queries(loc):
        all_queries.append((q, loc["id"]))
print(f"\nText queries: {len(all_queries)}")

texts  = [q for q,_ in all_queries]
labels = [label2id[k] for _,k in all_queries]
print("Encoding text with CLIP...")
text_embs = enc_text(texts)

print("\nBuilding dataset...")
X_list, y_list = [], []
for t_emb, label in zip(text_embs, labels):
    kb_id = id2label[label]
    r = random.random()
    if r < 0.35:
        joint = np.concatenate([t_emb, np.zeros(IMAGE_DIM, dtype=np.float32)])
    elif r < 0.70:
        i_emb = random.choice(real_embs[kb_id])
        joint  = np.concatenate([t_emb, i_emb])
    else:
        i_emb = random.choice(real_embs[kb_id])
        joint  = np.concatenate([np.zeros(TEXT_DIM, dtype=np.float32), i_emb])
    X_list.append(joint.astype(np.float32)); y_list.append(label)

for kb_id, embs in real_embs.items():
    if kb_id not in label2id: continue
    label = label2id[kb_id]
    for i_emb in embs:
        for _ in range(5):
            joint = np.concatenate([np.zeros(TEXT_DIM, dtype=np.float32), i_emb])
            X_list.append(joint.astype(np.float32)); y_list.append(label)
            noise = np.random.randn(TEXT_DIM).astype(np.float32)*0.05
            X_list.append(np.concatenate([noise, i_emb]).astype(np.float32)); y_list.append(label)

X = np.array(X_list); y = np.array(y_list)
print(f"Dataset: {len(X)} examples")
idx = np.random.permutation(len(X)); split = int(0.85*len(idx))
X_tr,y_tr = X[idx[:split]],y[idx[:split]]
X_va,y_va = X[idx[split:]],y[idx[split:]]
print(f"Train: {len(X_tr)}  Val: {len(X_va)}")

class MLP(nn.Module):
    def __init__(self,i,h,o,d=0.2):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(i,h),nn.LayerNorm(h),nn.GELU(),nn.Dropout(d),
            nn.Linear(h,h//2),nn.LayerNorm(h//2),nn.GELU(),nn.Dropout(d),
            nn.Linear(h//2,o))
    def forward(self,x): return self.net(x)

model=MLP(JOINT_DIM,512,n_classes,0.2)
opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
crit=nn.CrossEntropyLoss(label_smoothing=0.05)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=80)

Xtt=torch.tensor(X_tr,dtype=torch.float32); ytt=torch.tensor(y_tr,dtype=torch.long)
Xvt=torch.tensor(X_va,dtype=torch.float32); yvt=torch.tensor(y_va,dtype=torch.long)

EPOCHS=80; BATCH=32; PAT=15
tl=[]; vl=[]; va=[]; bv=float("inf"); pc=0
print("\nTraining...")
for ep in range(EPOCHS):
    model.train()
    perm=torch.randperm(len(Xtt)); el=0; nb=0
    for i in range(0,len(Xtt),BATCH):
        xb=Xtt[perm[i:i+BATCH]]; yb=ytt[perm[i:i+BATCH]]
        opt.zero_grad(); loss=crit(model(xb),yb); loss.backward(); opt.step()
        el+=loss.item(); nb+=1
    sched.step()
    model.eval()
    with torch.no_grad():
        vlo=crit(model(Xvt),yvt).item()
        vac=(model(Xvt).argmax(1)==yvt).float().mean().item()
    tl.append(el/nb); vl.append(vlo); va.append(vac)
    if ep%10==0 or ep==EPOCHS-1:
        print(f"  Epoch {ep+1:3d} | train={tl[-1]:.4f} val={vlo:.4f} acc={vac:.4f}")
    if vlo<bv:
        bv=vlo
        torch.save({"state_dict":model.state_dict(),
                    "config":{"input_dim":JOINT_DIM,"hidden_dim":512,
                               "num_classes":n_classes,"version":"v3",
                               "text_dim":TEXT_DIM,"image_dim":IMAGE_DIM,
                               "visual_kb_ids":VISUAL_KB_IDS}},
                   str(MODELS_DIR/"fusion_mlp.pt")); pc=0
    else:
        pc+=1
        if pc>=PAT: print(f"  Early stop ep {ep+1}"); break

ckpt=torch.load(str(MODELS_DIR/"fusion_mlp.pt"),map_location="cpu")
model.load_state_dict(ckpt["state_dict"]); model.eval()
with torch.no_grad():
    probs=torch.softmax(model(Xvt),dim=1)
t1=(probs.argmax(1)==yvt).sum().item()
t3=sum(yvt[i].item() in probs[i].topk(min(3,n_classes)).indices.tolist() for i in range(len(yvt)))
a1=t1/len(yvt); a3=t3/len(yvt)
print(f"\nTop-1: {a1*100:.1f}%  Top-3: {a3*100:.1f}%")

fig,axes=plt.subplots(1,3,figsize=(15,4))
axes[0].plot(tl,label="Train",color="#4C72B0"); axes[0].plot(vl,label="Val",color="#DD8452")
axes[0].set_title("Fusion MLP v3 Loss"); axes[0].legend()
axes[1].plot(va,color="#55A868"); axes[1].set_title("Val Accuracy"); axes[1].set_ylim(0,1.05)
axes[2].bar(["Top-1","Top-3"],[a1,a3],color=["#4C72B0","#55A868"],alpha=0.85,width=0.4)
axes[2].set_ylim(0,1.1); axes[2].set_title("Retrieval Accuracy")
for i,v in enumerate([a1,a3]):
    axes[2].text(i,v+0.02,f"{v:.2f}",ha="center",fontsize=12)
plt.tight_layout(); plt.savefig(PLOTS_DIR/"fusion_mlp_training.png",dpi=150); plt.close()

json.dump({"architecture":"FusionMLP v3 (8 visual classes)","version":"v3",
           "input_dim":JOINT_DIM,"hidden_dim":512,"output_classes":n_classes,
           "text_dim":TEXT_DIM,"image_dim":IMAGE_DIM,"visual_kb_ids":VISUAL_KB_IDS,
           "training_samples":len(X_tr),"val_samples":len(X_va),
           "epochs_trained":len(tl),"best_val_loss":round(bv,4),
           "top1_accuracy":round(a1,4),"top3_accuracy":round(a3,4),
           "design_rationale":"Fusion MLP classifies only 8 visually-grounded KB records. Text pipeline handles all non-visual queries."},
          open(METRICS_DIR/"fusion_metrics.json","w"),indent=2)
print(f"\n{'='*60}")
print(f"  FUSION v3 COMPLETE — {n_classes} visual classes")
print(f"  Top-1: {a1*100:.1f}%  Top-3: {a3*100:.1f}%")
print(f"{'='*60}")
