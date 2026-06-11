"""
Voice Pipeline - inference module.

Flow:
  audio file → Whisper ASR → transcript text
                                    ↓
                           text_pipeline.run_text_pipeline()
                                    ↓
                              LLM response

Whisper is frozen (not retrained). It runs locally with no API key.
The transcript feeds directly into the same DistilBERT + retrieval +
LLM pipeline used for typed text queries.
"""

import os
import ssl
import warnings
import tempfile
import librosa
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

# Fix SSL on macOS
ssl._create_default_https_context = ssl._create_unverified_context

from src.logger        import get_logger, log_interaction
from src.text_pipeline import run_text_pipeline

logger = get_logger()

# Whisper loaded once on first call
_whisper_model = None


def _load_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        logger.info("Loading Whisper ASR model (base)...")
        _whisper_model = whisper.load_model("base")
        logger.info("Whisper loaded.")
    return _whisper_model


def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe an audio file using Whisper.

    Returns:
      {
        transcript:  str,   # recognised text
        language:    str,   # detected language code
        duration_s:  float, # audio duration in seconds
        mfcc_shape:  list,  # shape of extracted MFCC features
        error:       str,   # empty if no error
      }
    """
    model  = _load_whisper()
    result = {"transcript": "", "language": "en",
              "duration_s": 0.0, "mfcc_shape": [], "error": ""}

    try:
        # Extract basic audio stats using librosa
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        result["duration_s"]  = round(float(duration), 2)
        result["mfcc_shape"]  = list(mfcc.shape)

        # Whisper transcription
        whisper_result       = model.transcribe(audio_path, language="en", fp16=False)
        result["transcript"] = whisper_result["text"].strip()
        result["language"]   = whisper_result.get("language", "en")

        logger.info(f"Whisper transcript: '{result['transcript']}'")

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Whisper transcription error: {e}")

    return result


def run_voice_pipeline(audio_path: str) -> dict:
    """
    Full voice pipeline: audio → Whisper → text_pipeline → response.

    Returns the same dict as run_text_pipeline() plus:
      transcript:   str   — what Whisper heard
      duration_s:   float — audio length
      mfcc_shape:   list  — MFCC feature dimensions
      asr_error:    str   — empty if no error
    """
    audio_path = str(audio_path)

    if not os.path.exists(audio_path):
        return {
            "response":          "Audio file not found.",
            "transcript":        "",
            "intent":            "error",
            "intent_confidence": 0.0,
            "matched_location":  "",
            "retrieval_score":   0.0,
            "llm_used":          False,
            "kb_result":         "",
            "duration_s":        0.0,
            "mfcc_shape":        [],
            "asr_error":         "File not found",
        }

    # Step 1: Transcribe
    asr = transcribe_audio(audio_path)

    if asr["error"]:
        return {
            "response":          f"Could not process audio: {asr['error']}",
            "transcript":        "",
            "intent":            "error",
            "intent_confidence": 0.0,
            "matched_location":  "",
            "retrieval_score":   0.0,
            "llm_used":          False,
            "kb_result":         "",
            "duration_s":        asr["duration_s"],
            "mfcc_shape":        asr["mfcc_shape"],
            "asr_error":         asr["error"],
        }

    transcript = asr["transcript"]

    if not transcript.strip():
        return {
            "response":          "I couldn't hear anything clearly. Please try again.",
            "transcript":        "",
            "intent":            "fallback",
            "intent_confidence": 0.0,
            "matched_location":  "",
            "retrieval_score":   0.0,
            "llm_used":          False,
            "kb_result":         "",
            "duration_s":        asr["duration_s"],
            "mfcc_shape":        asr["mfcc_shape"],
            "asr_error":         "",
        }

    # Step 2: Run through the same text pipeline as typed queries
    pipeline_result = run_text_pipeline(transcript, modality="voice")

    # Merge ASR info into result
    pipeline_result["transcript"] = transcript
    pipeline_result["duration_s"] = asr["duration_s"]
    pipeline_result["mfcc_shape"] = asr["mfcc_shape"]
    pipeline_result["asr_error"]  = ""

    return pipeline_result


def run_voice_pipeline_from_bytes(audio_bytes: bytes,
                                   suffix: str = ".mp3") -> dict:
    """
    Convenience wrapper for Streamlit — accepts raw bytes from
    st.audio_input() or st.file_uploader() and writes to a temp file.
    """
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        result = run_voice_pipeline(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    return result
