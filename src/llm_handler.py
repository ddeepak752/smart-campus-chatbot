"""
LLM Handler - Groq (free) with template fallback.
Get a free key at https://console.groq.com (30 seconds to sign up).
Add to .env:  GROQ_API_KEY=gsk_xxxx
"""

import os
from dotenv import load_dotenv

GROQ_MODEL = "llama-3.1-8b-instant"

def _get_api_key() -> str:
    load_dotenv(override=True)
    return os.getenv("GROQ_API_KEY", "")


def _clean_template_response(kb_result: str) -> str:
    """Readable local fallback when Groq is unavailable."""
    if not kb_result:
        return "I couldn't find a specific campus match. Please visit Main Reception for help."

    lines = [line.strip() for line in kb_result.splitlines()]
    data = {"events": [], "related": [], "courses": []}
    section = ""

    for line in lines:
        if not line:
            continue
        low = line.lower()
        if low.startswith("events:"):
            section = "events"; continue
        if low.startswith("additional related locations:"):
            section = "related"; continue
        if low.startswith("courses offered:"):
            section = "courses"; continue

        if section in ("events", "related", "courses") and line.startswith("- "):
            data[section].append(line[2:].strip())
            continue

        if ":" in line:
            key, value = line.split(":", 1)
            data[key.strip().lower()] = value.strip()

    if "price for" in kb_result.lower() or "today's menu" in kb_result.lower():
        return kb_result

    title = data.get("location") or data.get("urgent") or "Campus location"
    about = data.get("about", "")
    hours = data.get("opening hours") or data.get("available", "")
    map_ref = data.get("map reference", "")

    parts = []
    if "urgent" in data:
        parts.append(f"Please go to {title}.")
    else:
        parts.append(f"{title} is the best place for this.")
    if about:
        parts.append(about)
    if map_ref:
        parts.append(f"Location: {map_ref}.")
    if hours:
        parts.append(f"Hours: {hours}")
    if data["events"] and "No upcoming events" not in " ".join(data["events"]):
        parts.append("Events: " + "; ".join(data["events"][:2]))
    if data["courses"]:
        parts.append("Courses include: " + "; ".join(data["courses"][:8]))

    return "\n\n".join(parts)

SYSTEM_PROMPT = """You are a helpful campus assistant for BSBI (Berlin School of Business and Innovation).
You help students find locations, check opening hours, get directions, find menu prices, and discover events.

Core rules:
1. Answer using ONLY the KB_RESULT provided. Never invent facts.
2. Sound warm and natural — like a helpful senior student.
3. For prices, give only the specific item price requested.
4. For directions, use clear numbered steps.
5. Keep replies under 120 words unless directions need more.
6. Treat the first "Location:" or "URGENT:" record as the primary answer. Do not make another related location sound like the main match.
7. Avoid uncertain filler like "I think", "you might mean", or "usually" unless the KB_RESULT explicitly says uncertainty.

Handling specific situations:
- If asked about faculty/HOD/staff timing: say office hours vary by staff member, suggest contacting the department office or Main Reception.
- If asked about dean/principal/director: refer to the Principal Office record and its hours.
- If asked about courses/programmes/what to study: list the courses from KB_RESULT if available.
- Always mention specific course names when the KB_RESULT contains a courses_offered field.
- If someone asks how to make friends or about social life: mention the Student Union, clubs, and campus events as great places to meet people.
- If someone mentions depression, anxiety, stress, or mental health: prioritise the Student Counselling Room if present; for immediate danger, also suggest Security Cabin/Main Reception.
- If the query is completely off-topic (weather, sports scores, general knowledge): politely say you only cover campus information.
- If KB_RESULT has no specific match: be honest, suggest visiting Main Reception or the relevant department.
- Never suggest campus rooms for romantic or personal activities."""


def ask_llm(user_query: str, kb_result: str) -> tuple:
    """
    Returns (response_text, llm_was_used).
    Falls back to returning kb_result as-is if Groq is not configured.
    """
    GROQ_API_KEY = _get_api_key()
    if not GROQ_API_KEY or GROQ_API_KEY.startswith("your_"):
        return _clean_template_response(kb_result), False

    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)

        user_msg = (
            f"User question: {user_query}\n\n"
            f"KB_RESULT:\n{kb_result}\n\n"
            "Give a natural, helpful answer based strictly on the KB_RESULT above. "
            "Make the first listed location the main answer."
        )

        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=350,
        )
        return resp.choices[0].message.content.strip(), True

    except Exception as e:
        print(f"[LLM] Groq unavailable ({e}), using template response.")
        return _clean_template_response(kb_result), False
