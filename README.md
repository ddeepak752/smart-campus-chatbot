# Smart Campus Chatbot - BSBI Campus Navigator

Multimodal campus assistant for text, voice, and image queries.

## Required Pipeline

- Text input -> DistilBERT intent classifier -> semantic retrieval -> KB response
- Voice input -> local Whisper ASR -> transcript -> text pipeline
- Image input -> frozen CLIP ViT-B/32 -> FAISS cosine search -> KB record
- Fusion -> combines text/image evidence for final routing and response generation
- UI -> Streamlit
- Docker -> planned later

## Image Dataset

Images are manually selected GDPR-safe indoor/campus-style images and stored in:

```bash
data/images/admin/
data/images/auditorium/
data/images/cafeteria/
data/images/classroom/
data/images/department/
data/images/gym/
data/images/lab/
data/images/library/
```

Avoid identifiable faces, ID cards, private documents, and license plates.

## Build The Image Pipeline

Run from the project root:

```bash
./venv/bin/python scripts/build_image_manifest.py
./venv/bin/python scripts/build_clip_image_index.py
./venv/bin/python scripts/test_image_retrieval.py
```

Outputs:

```bash
data/image_manifest.csv
models/clip_faiss.index
models/clip_kb_ids.json
models/clip_image_manifest.json
models/clip_image_split.json
outputs/metrics/image_retrieval_metrics.json
outputs/metrics/image_retrieval_predictions.json
```

The test script uses held-out images that are excluded from the FAISS index.

## Run App

```bash
./venv/bin/streamlit run app/app.py
```

## Notes

- CLIP and SentenceTransformer models must be available locally or downloadable on first run.
- If offline, run the model-loading scripts once while connected to the internet so Hugging Face caches the models.
- Docker files are not included yet; Docker deployment can be added later.
