---
title: Smart Campus Chatbot
emoji: 🏛️
colorFrom: blue
colorTo: blue
sdk: docker
sdk_version: "1.35.0"
app_file: app/app.py
pinned: false
---

# BSBI Campus Navigator

A multimodal AI chatbot for smart campus orientation at Berlin School of Business and Innovation. It accepts text, voice and image inputs, then returns campus locations, opening hours, events, directions and service information from a structured knowledge base.

## What It Does

- **Text** - Type a campus question and get a structured response.
- **Voice** - Record or upload audio; Whisper transcribes it locally and routes it through the text pipeline.
- **Image** - Upload or capture a campus photo; CLIP and FAISS identify the closest campus location.
- **Multimodal** - Combine image, text and voice for context-aware responses.

## Architecture

```text
User Input (Text / Voice / Image)
        |
        |-- Text --> DistilBERT Intent Classifier --> Semantic Retrieval
        |
        |-- Voice --> Whisper ASR (local) --> Transcript --> Text Pipeline
        |
        |-- Image --> CLIP ViT-B/32 --> FAISS Cosine Search --> KB Record
                                |
                        Fusion MLP (8 visual classes)
                                |
                    Campus Knowledge Base (142 records)
                                |
                    Response Generator (LLM rewrite or template fallback)
                                |
                        Streamlit Chat UI
```

## Models Used

| Component | Model | Mode |
|---|---|---|
| Intent Classification | DistilBERT | Fine-tuned |
| Speech Transcription | Whisper base | Frozen |
| Image Retrieval | CLIP ViT-B/32 | Frozen |
| Vector Search | FAISS CPU | Index |
| Multimodal Fusion | Custom MLP | Trained |
| Semantic Retrieval | SentenceTransformers all-MiniLM-L6-v2 | Frozen |

## Evaluation Results

| Component | Metric | Result |
|---|---|---|
| DistilBERT Intent | Accuracy / F1 | 100% / 100% |
| Semantic Retrieval | Top-1 / Top-3 | 90% / 100% |
| CLIP + FAISS | Top-1 / Top-3 | 88.89% / 100% |
| Fusion MLP | Top-1 / Top-3 | 100% / 100% |
| Whisper ASR | WER | 17.44% |

## Local Setup

Requirements: Python 3.10 and FFmpeg.

```bash
git clone https://github.com/ddeepak752/smart-campus-chatbot.git
cd smart-campus-chatbot

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

Add a Groq API key to `.env` if LLM rewriting is required. The system also works without it using template fallback responses.

## Run Locally

```bash
streamlit run app/app.py
```

Open `http://localhost:8501` in your browser.

## Docker

```bash
docker build -t smart-campus-chatbot .
docker run -p 8501:8501 smart-campus-chatbot
```

## Project Structure

```text
smart_campus_chatbot/
├── app/
│   └── app.py                  # Streamlit UI
├── data/
│   ├── kb/knowledge_base.json  # 142 campus records
│   ├── images/                 # 99 campus images
│   ├── audio/                  # Voice query files
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

## Knowledge Base

The knowledge base is a structured JSON file containing 142 campus records. Each record includes:

- `name` - location name
- `category` - type of location
- `description` - what the location offers
- `opening_hours` - when it is open
- `map_reference` - campus zone and coordinates
- `directions` - how to get there
- `events` - upcoming events
- `keywords` - deterministic routing terms

## Privacy and GDPR

- Voice input is transcribed locally using Whisper.
- Uploaded images are processed for inference and are not intentionally stored by the application.
- No personal data is required to use the chatbot.
- The system processes only the minimum input needed to answer a query.

## Live Demo

Deployed on Hugging Face Spaces:
https://huggingface.co/spaces/ddeepak752/smart-campus-chatbot

## Academic Context

Developed as part of the MSc in Artificial Intelligence programme at BSBI / University for the Creative Arts (2026). Module: Multi-Modal Chatbots.
