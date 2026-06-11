"""
Persistent logger - survives VS Code restarts.
Every interaction is written to outputs/logs/interactions.jsonl
That file is never wiped - you can read past sessions anytime.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("outputs/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_logger(name: str = "campus_bot") -> logging.Logger:
    """Logger that writes to terminal (INFO+) and a daily file (DEBUG+)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    log_file = LOG_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def log_interaction(
    query: str,
    modality: str,
    intent: str,
    intent_confidence: float,
    matched_location: str,
    retrieval_score: float,
    llm_used: bool,
    response: str,
    error: str = ""
):
    """
    Appends one JSON line per chatbot call to interactions.jsonl.

    Fields:
      modality          - 'text', 'voice', or 'image'
      intent            - predicted intent label
      intent_confidence - model confidence 0.0-1.0
      matched_location  - KB record name that was returned
      retrieval_score   - semantic similarity score
      llm_used          - True if Groq formatted the answer
      response          - first 300 chars of the reply shown to user
      error             - exception message if something failed
    """
    record = {
        "timestamp":         datetime.now().isoformat(),
        "modality":          modality,
        "query":             query,
        "intent":            intent,
        "intent_confidence": round(float(intent_confidence), 4),
        "matched_location":  matched_location,
        "retrieval_score":   round(float(retrieval_score), 4),
        "llm_used":          llm_used,
        "response":          response[:300],
        "error":             error,
    }
    with open(LOG_DIR / "interactions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_logs(last_n: int = 20) -> list:
    log_file = LOG_DIR / "interactions.jsonl"
    if not log_file.exists():
        return []
    with open(log_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    return [json.loads(l) for l in lines[-last_n:]]


def print_log_summary():
    logs = read_logs(20)
    if not logs:
        print("No interactions logged yet.")
        return
    print(f"\n{'='*72}")
    print(f"  LAST {len(logs)} INTERACTIONS")
    print(f"{'='*72}")
    for r in logs:
        status = "OK" if not r["error"] else "ERR"
        llm    = "LLM" if r["llm_used"] else "TPL"
        print(
            f"[{status}] {r['timestamp'][11:19]} | "
            f"{r['modality']:5s} | {r['intent']:20s} | "
            f"conf={r['intent_confidence']:.2f} | "
            f"score={r['retrieval_score']:.2f} | "
            f"{llm} | -> {r['matched_location']}"
        )
    print(f"{'='*72}\n")
