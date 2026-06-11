"""
Part 2 — Voice Preprocessing, Feature Extraction & Whisper WER Evaluation

Steps:
  1. Load all synthetic audio files
  2. Extract MFCCs using Librosa (preprocessing demonstration)
  3. Visualise MFCC features and audio waveforms
  4. Normalize and pad MFCC sequences for batch processing
  5. Run Whisper transcription on all files
  6. Calculate Word Error Rate (WER) per intent and overall
  7. Save all plots to outputs/plots/ and metrics to outputs/metrics/
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from jiwer import wer

sys.path.insert(0, ".")
BASE_DIR = Path(".")

AUDIO_DIR  = BASE_DIR / "data/audio/synthetic"
PLOTS_DIR  = BASE_DIR / "outputs/plots"
METRICS_DIR = BASE_DIR / "outputs/metrics"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load ground truth ────────────────────────────────────────────────────────
with open(BASE_DIR / "data/audio/ground_truth.json") as f:
    ground_truth = json.load(f)

# Collect all audio files
audio_files = []
for intent_dir in sorted(AUDIO_DIR.iterdir()):
    if intent_dir.is_dir():
        for af in sorted(intent_dir.iterdir()):
            if af.suffix in (".mp3", ".wav"):
                audio_files.append(af)

print(f"Found {len(audio_files)} audio files across {len(list(AUDIO_DIR.iterdir()))} intents\n")

# ── STEP 1: MFCC Extraction ──────────────────────────────────────────────────
print("Step 1: Extracting MFCCs...")

N_MFCC    = 40
SR        = 22050
MAX_LEN   = 200   # pad/truncate all sequences to this length

all_mfccs  = []
file_labels = []

for af in audio_files:
    try:
        y, sr = librosa.load(str(af), sr=SR, mono=True)
        mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

        # Normalize per file
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)

        # Pad or truncate to MAX_LEN
        if mfcc.shape[1] < MAX_LEN:
            pad_w = MAX_LEN - mfcc.shape[1]
            mfcc  = np.pad(mfcc, ((0,0),(0,pad_w)), mode="constant")
        else:
            mfcc = mfcc[:, :MAX_LEN]

        all_mfccs.append(mfcc)
        file_labels.append(af.parent.name)   # intent name as label
    except Exception as e:
        print(f"  [WARN] Could not process {af.name}: {e}")

all_mfccs = np.array(all_mfccs)
print(f"  MFCC array shape: {all_mfccs.shape}  (files x n_mfcc x time_steps)")

# ── STEP 2: Visualise MFCCs ──────────────────────────────────────────────────
print("\nStep 2: Generating MFCC visualisations...")

# Plot 1: Sample MFCC heatmaps for 6 different intents
sample_intents = ["find_location", "menu_query", "emergency",
                  "ask_hours", "lost_found", "ask_event"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, intent in enumerate(sample_intents):
    # Find first file for this intent
    matches = [i for i, l in enumerate(file_labels) if l == intent]
    if not matches:
        continue
    mfcc = all_mfccs[matches[0]]
    img = librosa.display.specshow(
        mfcc, x_axis="time", ax=axes[idx], cmap="viridis"
    )
    axes[idx].set_title(f"MFCC — {intent.replace('_',' ').title()}", fontsize=10)
    axes[idx].set_xlabel("Time frames")
    axes[idx].set_ylabel("MFCC coefficients")
    fig.colorbar(img, ax=axes[idx])

plt.suptitle("MFCC Feature Maps — Sample Voice Queries per Intent", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "mfcc_sample_heatmaps.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: outputs/plots/mfcc_sample_heatmaps.png")

# Plot 2: Waveform + MFCC side by side for one sample
sample_file = None
for af in audio_files:
    if "emergency" in str(af):
        sample_file = af
        break

if sample_file:
    y, sr = librosa.load(str(sample_file), sr=SR)
    mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    librosa.display.waveshow(y, sr=sr, ax=axes[0], color="#4C72B0")
    axes[0].set_title(f'Waveform — "{sample_file.stem}"', fontsize=11)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")

    img = librosa.display.specshow(mfcc, x_axis="time", y_axis="mel",
                                   sr=sr, ax=axes[1], cmap="magma")
    axes[1].set_title("MFCC Features (40 coefficients)", fontsize=11)
    fig.colorbar(img, ax=axes[1])

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "waveform_mfcc_example.png", dpi=150)
    plt.close()
    print("  Saved: outputs/plots/waveform_mfcc_example.png")

# Plot 3: Audio duration distribution per intent
durations_by_intent = {}
for af in audio_files:
    try:
        y, sr = librosa.load(str(af), sr=SR)
        dur   = librosa.get_duration(y=y, sr=sr)
        intent = af.parent.name
        durations_by_intent.setdefault(intent, []).append(dur)
    except Exception:
        pass

intents_sorted = sorted(durations_by_intent.keys())
means  = [np.mean(durations_by_intent[i]) for i in intents_sorted]
stds   = [np.std(durations_by_intent[i])  for i in intents_sorted]

fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(len(intents_sorted))
bars = ax.bar(x, means, yerr=stds, capsize=4, color="#55A868", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([i.replace("_", "\n") for i in intents_sorted],
                   fontsize=8)
ax.set_ylabel("Duration (seconds)")
ax.set_title("Average Audio Duration per Intent (with std dev)", fontsize=12)
ax.axhline(np.mean(means), color="red", linestyle="--", alpha=0.6,
           label=f"Overall mean: {np.mean(means):.2f}s")
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "audio_duration_by_intent.png", dpi=150)
plt.close()
print("  Saved: outputs/plots/audio_duration_by_intent.png")

# Plot 4: MFCC coefficient energy distribution
mean_energy = all_mfccs.mean(axis=(0, 2))   # shape: (n_mfcc,)
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(range(N_MFCC), mean_energy, color="#4C72B0", alpha=0.8)
ax.set_xlabel("MFCC Coefficient Index")
ax.set_ylabel("Mean Energy")
ax.set_title("Mean MFCC Coefficient Energy Across All Audio Files")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "mfcc_coefficient_energy.png", dpi=150)
plt.close()
print("  Saved: outputs/plots/mfcc_coefficient_energy.png")

# ── STEP 3: Whisper Transcription & WER ─────────────────────────────────────
print("\nStep 3: Running Whisper transcription (this takes 3-5 minutes)...")

import whisper
whisper_model = whisper.load_model("base")
print("  Whisper 'base' model loaded\n")

transcriptions  = {}
wer_per_intent  = {}
all_refs        = []
all_hyps        = []

for intent_dir in sorted(AUDIO_DIR.iterdir()):
    if not intent_dir.is_dir():
        continue

    intent     = intent_dir.name
    intent_refs = []
    intent_hyps = []

    for af in sorted(intent_dir.iterdir()):
        if af.suffix not in (".mp3", ".wav"):
            continue

        # Get ground truth
        gt_key = str(af.relative_to(BASE_DIR))
        ref    = ground_truth.get(gt_key, "").strip()
        if not ref:
            continue

        # Transcribe
        try:
            result = whisper_model.transcribe(str(af), language="en", fp16=False)
            hyp    = result["text"].strip()
        except Exception as e:
            print(f"  [WARN] Whisper failed on {af.name}: {e}")
            hyp = ""

        transcriptions[gt_key] = {"reference": ref, "hypothesis": hyp}
        intent_refs.append(ref.lower())
        intent_hyps.append(hyp.lower())
        all_refs.append(ref.lower())
        all_hyps.append(hyp.lower())

        print(f"  [{intent:15s}] REF: {ref[:45]:45s} | HYP: {hyp[:45]}")

    # WER per intent
    if intent_refs:
        intent_wer = wer(intent_refs, intent_hyps)
        wer_per_intent[intent] = round(intent_wer, 4)

# Overall WER
overall_wer = wer(all_refs, all_hyps) if all_refs else 0.0
print(f"\n  Overall WER: {overall_wer:.4f} ({overall_wer*100:.1f}%)")

# ── STEP 4: WER Plots ────────────────────────────────────────────────────────
print("\nStep 4: Generating WER visualisations...")

# Plot 5: WER per intent bar chart
fig, ax = plt.subplots(figsize=(14, 5))
intents_w = list(wer_per_intent.keys())
wers_w    = list(wer_per_intent.values())
colors    = ["#C44E52" if w > 0.1 else "#55A868" for w in wers_w]

ax.bar(range(len(intents_w)), wers_w, color=colors, alpha=0.85)
ax.set_xticks(range(len(intents_w)))
ax.set_xticklabels([i.replace("_", "\n") for i in intents_w], fontsize=8)
ax.set_ylabel("Word Error Rate")
ax.set_title(f"Whisper WER per Intent (Overall WER = {overall_wer*100:.1f}%)", fontsize=12)
ax.axhline(overall_wer, color="navy", linestyle="--", alpha=0.7,
           label=f"Overall WER: {overall_wer*100:.1f}%")
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "whisper_wer_per_intent.png", dpi=150)
plt.close()
print("  Saved: outputs/plots/whisper_wer_per_intent.png")

# Plot 6: Sample transcriptions table
fig, ax = plt.subplots(figsize=(15, 6))
ax.axis("off")
samples = list(transcriptions.items())[:10]
table_data = [
    [os.path.basename(k).replace(".mp3",""),
     v["reference"][:40],
     v["hypothesis"][:40],
     "✓" if v["reference"].lower()[:20] in v["hypothesis"].lower() else "✗"]
    for k, v in samples
]
table = ax.table(
    cellText=table_data,
    colLabels=["File", "Reference", "Hypothesis", "Match"],
    loc="center", cellLoc="left"
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.2, 1.8)
plt.title("Whisper Transcription Samples", fontsize=12, pad=20)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "whisper_transcription_samples.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: outputs/plots/whisper_transcription_samples.png")

# ── STEP 5: Save metrics ─────────────────────────────────────────────────────
print("\nStep 5: Saving metrics...")

voice_metrics = {
    "overall_wer":       round(float(overall_wer), 4),
    "overall_wer_pct":   round(float(overall_wer) * 100, 2),
    "wer_per_intent":    wer_per_intent,
    "total_files":       len(audio_files),
    "whisper_model":     "base",
    "sample_rate":       SR,
    "n_mfcc":            N_MFCC,
    "mfcc_max_len":      MAX_LEN,
    "mfcc_array_shape":  list(all_mfccs.shape),
    "note": "WER is a reporting metric only. Whisper is frozen — not retrained."
}

with open(METRICS_DIR / "voice_metrics.json", "w") as f:
    json.dump(voice_metrics, f, indent=2)

with open(BASE_DIR / "data/audio/transcriptions.json", "w") as f:
    json.dump(transcriptions, f, indent=2)

# Save preprocessed MFCCs
np.save(BASE_DIR / "data/audio/mfcc_features.npy", all_mfccs)
np.save(BASE_DIR / "data/audio/mfcc_labels.npy",   np.array(file_labels))

print(f"\n{'='*60}")
print(f"  VOICE PREPROCESSING COMPLETE")
print(f"  Files processed : {len(audio_files)}")
print(f"  MFCC shape      : {all_mfccs.shape}")
print(f"  Overall WER     : {overall_wer*100:.1f}%")
print(f"  Metrics saved   : outputs/metrics/voice_metrics.json")
print(f"  Plots saved     : outputs/plots/ (6 new plots)")
print(f"{'='*60}")
print("\nNext step: create src/voice_pipeline.py")
