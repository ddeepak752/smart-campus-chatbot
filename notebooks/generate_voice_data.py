"""
Part 1 — Synthetic Voice Data Generation
Generates .wav audio files from text queries using gTTS (Google Text-to-Speech).
These are used to:
  1. Demonstrate the voice pipeline with real audio
  2. Evaluate Whisper WER against known ground truth transcripts
  3. Show data acquisition for the assignment

Output:
  data/audio/synthetic/<intent>/<filename>.wav
  data/audio/ground_truth.json   — {filename: transcript} pairs for WER
"""

import os
import json
import time
from pathlib import Path
from gtts import gTTS

BASE_DIR  = Path(".")
AUDIO_DIR = BASE_DIR / "data/audio/synthetic"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# ── Voice queries per intent ────────────────────────────────────────────────
# 5 queries per intent = 65 total audio files
# Covers all 13 intents, diverse phrasings including accessibility requests
VOICE_QUERIES = {
    "find_location": [
        "Where is the library",
        "How do I get to the cafeteria",
        "Take me to the computer lab",
        "Which floor is the AI department on",
        "How do I go from the civil department to the mechanical department",
    ],
    "ask_hours": [
        "Is the library open on Saturday",
        "What time does the cafeteria close",
        "Is the gym open now",
        "What are the opening hours of the reading room",
        "Can I visit the library this evening",
    ],
    "ask_event": [
        "Any events happening today",
        "Are there any seminars this week",
        "What workshops are available on campus",
        "Show me upcoming events at the student union",
        "Any guest lectures today",
    ],
    "menu_query": [
        "What is the price of tea",
        "How much does coffee cost",
        "What is on the menu today",
        "Price of samosa",
        "What snacks are available right now",
    ],
    "faculty_query": [
        "Where is the civil department staff room",
        "Where can I meet my project supervisor",
        "Where do lecturers sit in the mechanical department",
        "Where is the HOD office of the AI department",
        "Where can I find my professor",
    ],
    "service_query": [
        "Where can I charge my laptop",
        "Where can I print my assignment",
        "My wifi is not working who do I contact",
        "Where can I get a bonafide certificate",
        "Where is the IT helpdesk",
    ],
    "recommend_place": [
        "Where can I study quietly",
        "Suggest a peaceful place to study",
        "Where can I sit with my laptop",
        "I need a quiet corner with good wifi",
        "Where is the best place to prepare for exams",
    ],
    "lost_found": [
        "I lost my wallet where should I go",
        "I found a phone near the library what do I do",
        "Where is the lost and found office",
        "I lost my student ID card",
        "I think someone took my bag",
    ],
    "emergency": [
        "I feel sick where can I get help",
        "My friend fainted in the robotics lab what do I do",
        "I cut my finger where is first aid",
        "I need medical assistance urgently",
        "I hurt my leg and need help",
    ],
    "facility_info": [
        "Tell me about the central library",
        "What facilities are available in the student union",
        "What does the innovation centre offer",
        "Give me details about the sports complex",
        "What is on the third floor",
    ],
    "ask_contact": [
        "Who should I contact for hostel issues",
        "Where can visitors ask for help",
        "Who handles document queries on campus",
        "Who is the contact for international students",
        "Where can I ask general questions",
    ],
    "ask_department": [
        "Tell me about the computer science department",
        "Where is the mechanical engineering department",
        "What labs are in the AI department",
        "Which floor is the civil engineering department on",
        "What does the robotics department offer",
    ],
    "fallback": [
        "What is the weather today",
        "Tell me a joke",
        "Who won the cricket match yesterday",
        "What is the price of bitcoin",
        "Play some music for me",
    ],
}

# ── Generate audio files ────────────────────────────────────────────────────
ground_truth = {}
total = 0
errors = 0

print("Generating synthetic voice queries...")
print(f"Output directory: {AUDIO_DIR}\n")

for intent, queries in VOICE_QUERIES.items():
    intent_dir = AUDIO_DIR / intent
    intent_dir.mkdir(exist_ok=True)

    for i, query in enumerate(queries):
        filename = f"{intent}_{i+1:02d}.wav"
        filepath = intent_dir / filename

        if filepath.exists():
            print(f"  [SKIP] {filename} already exists")
            ground_truth[str(filepath.relative_to(BASE_DIR))] = query
            total += 1
            continue

        try:
            tts = gTTS(text=query, lang="en", slow=False)
            # gTTS saves as mp3, we save with .wav extension
            # Whisper handles both formats via ffmpeg
            mp3_path = str(filepath).replace(".wav", ".mp3")
            tts.save(mp3_path)

            # Rename to consistent naming
            os.rename(mp3_path, str(filepath).replace(".wav", ".mp3"))
            final_path = str(filepath).replace(".wav", ".mp3")

            ground_truth[final_path.replace(str(BASE_DIR) + "/", "")] = query
            total += 1
            print(f"  [OK] {intent}/{filename.replace('.wav','.mp3')} — \"{query}\"")

            # Small delay to avoid rate limiting
            time.sleep(0.3)

        except Exception as e:
            print(f"  [ERROR] {filename}: {e}")
            errors += 1

# Save ground truth transcripts for WER evaluation
gt_path = BASE_DIR / "data/audio/ground_truth.json"
with open(gt_path, "w") as f:
    json.dump(ground_truth, f, indent=2)

print(f"\n{'='*60}")
print(f"  Generated : {total} audio files")
print(f"  Errors    : {errors}")
print(f"  Intents   : {len(VOICE_QUERIES)}")
print(f"  GT saved  : {gt_path}")
print(f"{'='*60}")
print("\nNext step: run notebooks/voice_preprocessing.py")
