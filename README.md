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

# Smart Campus Navigator — BSBI

Multimodal AI chatbot for campus orientation. Supports text, voice, and image inputs.

## Pipeline

- Text: DistilBERT intent classifier → semantic retrieval → LLM response
- Voice: Whisper ASR → transcript → text pipeline
- Image: EasyOCR + CLIP ViT-B/32 + FAISS → KB record → LLM response
- Fusion: MLP (8 visual classes) combining CLIP text + image embeddings

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Run

```bash
streamlit run app/app.py
```

## Docker

```bash
docker build -t campus-navigator .
docker run -p 8501:8501 campus-navigator
```

## GDPR Note

Voice and image data processed locally. No personal data stored or transmitted.
