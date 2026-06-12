FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg git libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install --no-cache-dir "setuptools==69.5.1" wheel

RUN pip install --no-cache-dir --no-build-isolation openai-whisper==20231117

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
COPY data/kb/ ./data/kb/
COPY data/images/ ./data/images/
COPY data/image_manifest.csv ./data/
COPY models/ ./models/
COPY scripts/ ./scripts/
COPY chatbot.py .
COPY README.md .
COPY .streamlit/ ./.streamlit/

ENV KMP_DUPLICATE_LIB_OK=TRUE
ENV OMP_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false

EXPOSE 7860

HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/app.py", \
            "--server.port=7860", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--server.enableCORS=false", \
            "--server.enableXsrfProtection=false"]
