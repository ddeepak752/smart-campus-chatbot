```markdown
# 🏛️ BSBI Campus Navigator

A multimodal AI chatbot for smart campus orientation at Berlin School of Business and Innovation. Accepts text, voice and image inputs and returns campus locations, opening hours, events, directions and service information from a structured knowledge base.

---

## What It Does

- **Text** — Type any campus question and get an instant structured response
- **Voice** — Record or upload audio; Whisper transcribes it locally and routes it through the same pipeline
- **Image** — Upload or capture a photo of a campus building or space; CLIP + FAISS identifies the location
- **Multimodal** — Combine image + text + voice for context-aware responses

---

## Architecture

```
User Input (Text / Voice / Image)
        │
        ├── Text ──► DistilBERT Intent Classifier ──► Semantic Retrieval
        │
        ├── Voice ──► Whisper ASR (local) ──► Transcript ──► Text Pipeline
        │
        └── Image ──► CLIP ViT-B/32 ──► FAISS Cosine Search ──► KB Record
                                │
                        Fusion MLP (8 visual classes)
                                │
                    Campus Knowledge Base (142 records)
                                │
                    Response Generator (LLM rewrite or template fallback)
                                │
                        Streamlit Chat UI
```

---

## Models Used

| Component | Model | Mode |
|---|---|---|
| Intent Classification | DistilBERT | Fine-tuned |
| Speech Transcription | Whisper base | Frozen |
| Image Retrieval | CLIP ViT-B/32 | Frozen |
| Vector Search | FAISS CPU | Index |
| Multimodal Fusion | Custom MLP | Trained |
| Semantic Retrieval | SentenceTransformers all-MiniLM-L6-v2 | Frozen |

---

## Evaluation Results

| Component | Metric | Result |
|---|---|---|
| DistilBERT Intent | Accuracy / F1 | 100% / 100% |
| Semantic Retrieval | Top-1 / Top-3 | 90% / 100% |
| CLIP + FAISS | Top-1 / Top-3 | 88.89% / 100% |
| Fusion MLP | Top-1 / Top-3 | 100% / 100% |
| Whisper ASR | WER | 17.44% |

---

## Local Setup

**Requirements:** Python 3.10, FFmpeg

```bash
# Clone the repo
git clone https://github.com/ddeepak752/smart-campus-chatbot.git
cd smart-campus-chatbot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your Groq API key to .env (optional - system works without it using template fallback)
```

---

## Run Locally

```bash
streamlit run app/app.py
```

Open `http://localhost:8501` in your browser.

---

## Docker

```bash
# Build
docker build -t smart-campus-chatbot .

# Run
docker run -p 8501:8501 smart-campus-chatbot
```

Open `http://localhost:8501` in your browser.

---

## Project Structure

```
smart_campus_chatbot/
├── app/
│   └── app.py                  # Streamlit UI
├── data/
│   ├── kb/knowledge_base.json  # 142 campus records
│   ├── images/                 # 99 campus images
│   ├── audio/                  # 65 voice query files
│   └── text/                   # Intent dataset CSVs
├── models/
│   ├── distilbert_intent/      # Fine-tuned intent classifier
│   ├── clip_faiss.index        # FAISS vector index
│   ├── fusion_mlp.pt           # Trained fusion model
│   └── id2label.json           # Intent label mapping
├── outputs/
│   ├── plots/                  # Evaluation figures
│   └── metrics/                # Saved metric files
├── scripts/                    # Training and evaluation scripts
├── src/                        # Core pipeline modules
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Knowledge Base

The knowledge base is a structured JSON file containing 142 campus records. Each record includes:

- `name` — location name
- `category` — type of location
- `description` — what the location offers
- `opening_hours` — when it is open
- `map_reference` — campus zone and coordinates
- `directions` — how to get there
- `events` — upcoming events
- `keywords` — for deterministic routing

---

## Privacy and GDPR

- Voice input is transcribed locally using Whisper — no audio is sent to external APIs
- Uploaded images are processed in memory and not stored
- No personal data is collected or retained between sessions
- The system processes only the minimum data required to answer the query

---

## Live Demo

Deployed on Hugging Face Spaces:
👉 https://huggingface.co/spaces/ddeepak752/smart-campus-chatbot

---

## Academic Context

Developed as part of the MSc in Artificial Intelligence programme at BSBI / University for the Creative Arts (2026). Module: Multi-Modal Chatbots.
```