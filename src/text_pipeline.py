"""
Text Pipeline - inference only, loads trained models from disk.
Never retrain unless you change your data.

Flow:
  query -> predict_intent() -> semantic_retrieve() -> build_kb_result() -> ask_llm()
"""

import json
import pickle
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.kb_utils        import load_kb, search, get_by_id
from src.answer_generator import build_kb_result, build_kb_result_multi
from src.llm_handler     import ask_llm
from src.logger          import get_logger, log_interaction

logger = get_logger()

# ── Paths ─────────────────────────────────────────────────────────────
MODEL_DIR   = Path("models/distilbert_intent")
EMBED_PATH  = Path("models/retrieval_embeddings.pkl")
ID2LABEL    = Path("models/id2label.json")

# ── Globals (loaded once on first call) ───────────────────────────────
_intent_tokenizer  = None
_intent_model      = None
_id2label          = None
_retrieval_model   = None
_embeddings        = None
_kb_ids            = None


def _load_models():
    global _intent_tokenizer, _intent_model, _id2label
    global _retrieval_model, _embeddings, _kb_ids

    if _intent_model is not None:
        return  # already loaded

    logger.info("Loading intent classifier...")
    _intent_tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    _intent_model     = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    _intent_model.eval()

    with open(ID2LABEL) as f:
        _id2label = {int(k): v for k, v in json.load(f).items()}

    logger.info("Loading retrieval model...")
    _retrieval_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    with open(EMBED_PATH, "rb") as f:
        data = pickle.load(f)
    _embeddings = data["embeddings"]
    _kb_ids     = data["kb_ids"]

    logger.info("All models loaded.")


def predict_intent(query: str) -> dict:
    """
    Returns {intent: str, confidence: float, all_scores: dict}
    """
    _load_models()
    inputs = _intent_tokenizer(
        query, return_tensors="pt",
        truncation=True, padding=True, max_length=64
    )
    with torch.no_grad():
        logits = _intent_model(**inputs).logits
    probs   = torch.softmax(logits, dim=1)[0]
    pred_id = torch.argmax(probs).item()
    return {
        "intent":     _id2label[pred_id],
        "confidence": float(probs[pred_id]),
        "all_scores": {_id2label[i]: float(probs[i]) for i in range(len(probs))}
    }


def semantic_retrieve(query: str, top_k: int = 3) -> list:
    """
    Encodes the query and finds the most similar KB records.
    Returns list of {record, score} dicts sorted best first.
    """
    _load_models()
    kb      = load_kb()
    q_emb   = _retrieval_model.encode([query])
    scores  = cosine_similarity(q_emb, _embeddings)[0]
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [{"record": kb[i], "score": float(scores[i])} for i in top_idx]


# Sub-location types that should rank lower than main departments
_PENALISE_CATEGORIES = {
    "washroom", "water_dispenser", "emergency_exit",
    "classroom", "lift", "corridor",
    "faculty_room", "hod_office"
}
_PENALISE_NAMES = {
    "washroom", "water dispenser", "classroom",
    "lift area", "emergency exit", "dispenser",
    "faculty room", "hod office"
}


def _keyword_route(query: str) -> dict:
    """
    Small deterministic routing layer for high-value campus service queries.
    It corrects common intent/retrieval misses without replacing the ML model.
    """
    q = query.lower()
    q_short = q.strip(" ?!.")

    if q_short == "water":
        return {"intent": "find_location", "kb_id": "water_dispenser_ground", "reason": "water dispenser"}

    routes = [
        (["new student", "new on campus", "first day", "go first", "where should i go first", "just joined"], "recommend_place", "reception", "new student help"),
        (["principal", "director"], "ask_contact", "principal_office", "principal office"),  # Note: direction queries bypass this
        (["library", "central library"], "find_location", "central_library", "library"),
        (["computer lab", "computing lab"], "find_location", "computer_lab", "computer lab"),
        (["quiet place", "quiet study", "study quietly", "silent study", "exam preparation", "exam prep"], "recommend_place", "reading_room", "quiet study"),
        (["group study", "study with classmates", "study with friends", "team study", "group work", "sit with classmates", "study together", "study room"], "recommend_place", "group_study_room", "group study"),
        (["academic journal", "academic journals", "journals", "e-journal", "e-journals", "research portal", "research portals"], "service_query", "digital_library", "academic journals"),
        (["print", "printing", "photocopy", "photocopying", "scan ", "scanning"], "service_query", "printing_room", "printing"),
        (["hungry", "food", "eat", "lunch", "dinner", "snack", "meal"], "menu_query", "main_cafeteria", "food query"),
        (["cafeteria", "canteen", "main cafeteria", "main canteen", "dining hall"], "find_location", "main_cafeteria", "cafeteria"),
        (["water dispenser", "drinking water", "drink water", "fill bottle", "refill bottle", "get water", "need water", "want water", "where is water", "water point", "get me water", "i need water", "where can i get water", "i want water", "get water please"], "find_location", "water_dispenser_ground", "water dispenser"),
        (["hostel", "dormitory", "room allotment", "warden"], "ask_hostel", "hostel_office", "hostel"),
        (["civil faculty", "ce faculty", "civil staff room", "civil lecturer", "civil teacher", "civil professor", "civil engineering faculty", "civil engineering professor", "civil engineering staff", "ce department faculty"], "faculty_query", "civil_faculty_room", "civil faculty room"),
        (["mechanical faculty", "me faculty", "mechanical staff", "mechanical lecturer", "mechanical teacher", "mechanical professor"], "faculty_query", "mechanical_faculty_room", "mechanical faculty room"),
        (["cse faculty", "computer science faculty", "cs faculty", "cs staff", "cse professor", "computer science professor"], "faculty_query", "cse_faculty_room", "cse faculty room"),
        (["it faculty", "information technology faculty", "it staff", "it professor", "information technology professor"], "faculty_query", "it_faculty_room", "it faculty room"),
        (["ai faculty", "data science faculty", "ai ds faculty", "ai&ds faculty", "ai professor", "data science professor"], "faculty_query", "ai_ds_faculty_room", "ai faculty room"),
        (["department faculty", "my faculty", "my professor", "meet professor", "speak to professor", "talk to professor"], "faculty_query", "reception", "generic faculty guidance"),
        (["fees", "fee ", "tuition", "payment", "pay fee", "account"], "service_query", "accounts_office", "fees/accounts"),
        (["visa", "immigration", "brp", "residence permit"], "service_query", "international_office", "visa support"),
        (["student email", "email", "wifi", "wi-fi", "password", "laptop", "technical support"], "service_query", "it_helpdesk", "IT support"),
        (["exam", "exams", "assessment", "result", "results"], "ask_exam", "exam_cell", "exam support"),
        (["general help", "just entered", "new here", "information desk", "reception"], "find_location", "reception", "general help"),
        (["new id card", "get id card", "student id card", "replace id", "id card lost", "new student id", "get my student id", "where can i get my student id"], "service_query", "admin_office", "student ID card"),
        (["make friend", "make friends", "meet people", "lonely", "no friends"], "social_life", "student_union", "student life"),
        (["lost", "misplaced", "id card", "student card", "wallet"], "lost_found", "lost_and_found", "lost item"),
        (["unsafe", "security", "danger", "threat"], "emergency", "security_cabin", "security"),
        (["exam", "examination", "timetable", "hall ticket", "result", "re-evaluation", "marksheet", "transcript"], "ask_exam", "exam_cell", "exam"),
        (["fest", "festival", "annual day", "hackathon", "cultural", "techfest", "sports day", "farewell", "freshers"], "ask_fest", "student_union", "fest"),
        (["join club", "student club", "coding club", "music club", "drama club", "photography club", "debate club", "dance club"], "ask_club", "student_union", "club"),
        (["sports team", "sports club", "join sports", "football team", "cricket team", "athletics team", "basketball team"], "ask_club", "sports_ground", "sports team"),
        (["placement", "placement cell", "recruitment", "campus drive", "internship", "cv help", "career"], "ask_placement", "placement_cell", "placement"),
        (["scholarship", "bursary", "financial aid", "fee waiver", "fee concession"], "ask_scholarship", "accounts_office", "scholarship"),
        (["admission", "apply", "enroll", "enrolment", "courses offered", "course list", "programme", "fee structure"], "ask_admission", "admin_office", "admission"),
        (["depression", "depressed", "anxiety", "mental health", "counselling", "counseling", "stress"], "emergency", "counselling_room", "wellbeing"),
        (["hurt", "injured", "stomach ache", "headache", "sick", "first aid", "medical"], "emergency", "medical_room", "medical"),
    ]

    for needles, intent, kb_id, reason in routes:
        if any(needle in q for needle in needles):
            return {"intent": intent, "kb_id": kb_id, "reason": reason}
    return {}

def hybrid_retrieve(query: str, intent: str, top_k: int = 3) -> list:
    """
    Combines semantic retrieval with keyword search.
    Penalises sub-locations (washrooms, classrooms, dispensers)
    so main departments always rank higher when both match.
    """
    semantic_results = semantic_retrieve(query, top_k=top_k + 3)
    keyword_results  = search(query, top_k=top_k)

    seen   = {r["record"]["id"] for r in semantic_results}
    merged = list(semantic_results)
    for rec in keyword_results:
        if rec["id"] not in seen:
            merged.append({"record": rec, "score": 0.28})
            seen.add(rec["id"])

    # Apply penalty to sub-locations
    washroom_query = any(w in query.lower() for w in
                         ["washroom", "toilet", "bathroom", "restroom"])

    def penalise(item):
        rec  = item["record"]
        cat  = rec.get("category", "")
        name = rec.get("name", "").lower()
        # Don't penalise washrooms if user is asking about washrooms
        if cat == "washroom" and washroom_query:
            return item["score"]
        if cat in _PENALISE_CATEGORIES:
            return item["score"] * 0.4
        if any(kw in name for kw in _PENALISE_NAMES):
            return item["score"] * 0.4
        return item["score"]

    for item in merged:
        item["score"] = penalise(item)

    merged.sort(key=lambda x: -x["score"])
    return merged[:top_k]


# ── Main pipeline function ────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.35   # below this, treat as fallback
RETRIEVAL_THRESHOLD  = 0.32   # below this, no confident match

def run_text_pipeline(query: str, modality: str = "text") -> dict:
    """
    Full pipeline: query -> intent -> retrieve -> format -> LLM -> response.

    Returns:
      {
        response:          str,   # final answer shown to user
        intent:            str,
        intent_confidence: float,
        matched_location:  str,
        retrieval_score:   float,
        llm_used:          bool,
        kb_result:         str,   # structured result before LLM
      }
    """
    query = query.strip()
    if not query:
        return {"response": "Please type a question.", "intent": "fallback",
                "intent_confidence": 0.0, "matched_location": "",
                "retrieval_score": 0.0, "llm_used": False, "kb_result": ""}

    error_msg = ""
    kb_result = ""

    # Greeting shortcut — handle before ML pipeline
    GREETINGS = {
        "hi", "hello", "hey", "hiya", "howdy",
        "good morning", "good afternoon", "good evening",
        "bye", "goodbye", "see you", "take care", "cya",
        "thank you", "thanks", "thank you so much", "many thanks",
        "ok", "okay", "alright", "sure", "great", "nice",
    }
    q_clean = query.lower().strip().rstrip("!.,")
    if q_clean in GREETINGS:
        greeting_responses = {
            "bye": "Goodbye! Have a great day on campus! 👋",
            "goodbye": "Goodbye! Take care! 👋",
            "see you": "See you around campus! 👋",
            "take care": "You too! Take care! 😊",
            "thank you": "You're welcome! Let me know if you need anything else. 😊",
            "thanks": "Happy to help! Let me know if you need anything else. 😊",
            "thank you so much": "You're very welcome! 😊",
            "many thanks": "My pleasure! 😊",
        }
        if any(q_clean.startswith(g) for g in ["bye", "goodbye", "see you", "take care"]):
            response = greeting_responses.get(q_clean, "Goodbye! Take care! 👋")
        elif any(q_clean.startswith(g) for g in ["thank", "thanks"]):
            response = greeting_responses.get(q_clean, "You're welcome! 😊")
        else:
            response = (
                "Hello! 👋 Welcome to the Smart Campus Assistant. "
                "I can help you with locations, directions, opening hours, "
                "menu prices, and events. What would you like to know?"
            )
        log_interaction(query, modality, "greeting", 1.0,
                        "greeting", 1.0, False, response)
        return {"response": response, "intent": "greeting",
                "intent_confidence": 1.0, "matched_location": "greeting",
                "retrieval_score": 1.0, "llm_used": False, "kb_result": ""}


    try:
        # Step 1: Intent
        intent_result     = predict_intent(query)
        intent            = intent_result["intent"]
        intent_confidence = intent_result["confidence"]
        override          = _keyword_route(query)

        if override:
            override_record = get_by_id(override["kb_id"])
            if override_record:
                # Don't override with ask_contact if user is asking for directions
                direction_words = ["direction", "directions", "how do i get", "how to get",
                                   "how to go", "how do i go", "route", "from ", "get to "]
                is_direction_q = any(w in query.lower() for w in direction_words)
                if is_direction_q and override.get("intent") == "ask_contact":
                    override = {}  # let direction intent pass through
                else:
                    logger.debug(
                        f"Keyword route='{override['reason']}' -> {override_record['name']}"
                    )
                    intent = override["intent"]
                    intent_confidence = max(intent_confidence, 0.88)

        logger.debug(f"Query='{query}' Intent={intent} Conf={intent_confidence:.2f}")

        # Step 2: Fallback shortcut
        if not override and (intent == "fallback" or intent_confidence < CONFIDENCE_THRESHOLD):
            response = (
                "I can help with campus locations, directions, opening hours, "
                "menu prices, and events. Try asking something like:\n"
                "• \"Where is the library?\"\n"
                "• \"Is the cafeteria open on Saturday?\"\n"
                "• \"Price of coffee\"\n"
                "• \"How do I get to the gym?\""
            )
            log_interaction(query, modality, intent, intent_confidence,
                            "fallback", 0.0, False, response)
            return {"response": response, "intent": intent,
                    "intent_confidence": intent_confidence,
                    "matched_location": "fallback",
                    "retrieval_score": 0.0, "llm_used": False,
                    "kb_result": ""}

        # Step 3: Retrieve KB record
        # For navigation, extract the TO destination from the query
        # and retrieve that specifically — prevents matching the FROM location
        retrieve_query = query
        if intent == "find_location":
            import re as _re
            # Match "from X to Y" pattern — extract Y as destination
            from_to = _re.search(
                r'from\s+.+?\s+to\s+(?:the\s+)?(.+?)(?:\s*$)', query.lower()
            )
            # Match "to Y from X" pattern
            to_from = _re.search(
                r'\bto\s+(?:the\s+)?([a-z][a-z\s]+?)(?:\s+from\b)', query.lower()
            )
            if from_to:
                retrieve_query = from_to.group(1).strip(" ?!.,")
            elif to_from:
                retrieve_query = to_from.group(1).strip(" ?!.,")

        if override and override_record:
            related = hybrid_retrieve(retrieve_query, intent, top_k=3)
            results = [{"record": override_record, "score": 0.95}]
            seen = {override_record["id"]}
            for item in related:
                if item["record"]["id"] not in seen:
                    results.append(item)
                    seen.add(item["record"]["id"])
                if len(results) >= 3:
                    break
        else:
            results = hybrid_retrieve(retrieve_query, intent, top_k=3)

        if not results:
            response = (
                "I couldn't find a specific campus match. "
                "Please visit Main Reception on the Ground Floor for help."
            )
            log_interaction(query, modality, intent, intent_confidence,
                            "no_match", 0.0, False, response)
            return {"response": response, "intent": intent,
                    "intent_confidence": intent_confidence,
                    "matched_location": "no_match",
                    "retrieval_score": 0.0,
                    "llm_used": False, "kb_result": ""}

        best            = results[0]
        record          = best["record"]
        retrieval_score = best["score"]

        logger.debug(f"Matched='{record['name']}' Score={retrieval_score:.2f}")

        # Step 4: Low confidence retrieval
        # Use lower threshold for location/direction queries
        # as these often have complex multi-entity phrasing
        effective_threshold = RETRIEVAL_THRESHOLD
        if intent == "find_location":
            effective_threshold = 0.20
        if retrieval_score < effective_threshold:
            intent_defaults = {
                "menu_query":      "main_cafeteria",
                "emergency":       "medical_room",
                "lost_found":      "lost_and_found",
                "service_query":   "international_office",
                "ask_event":       "student_union",
                "ask_hours":       "reception",
                "ask_contact":     "reception",
                "recommend_place": "reading_room",
                "facility_info":   "reception",
                "ask_admission":   "reception",
                "ask_placement":   "placement_cell",
                "ask_fest":        "student_union",
                "ask_club":        "student_union",
                "ask_exam":        "exam_cell",
                "ask_hostel":      "hostel_office",
                "ask_scholarship": "accounts_office",
                "social_life":     "student_union",
                "faculty_query":   "reception",
            }
            if intent in intent_defaults:
                fallback_record = get_by_id(intent_defaults[intent])
                if fallback_record:
                    record = fallback_record
                    retrieval_score = 0.31
                    results = [{"record": record, "score": 0.31}]
            else:
                response = (
                    f"I understood you want help with '{intent.replace('_',' ')}', "
                    "but I couldn't find a specific match. "
                    "Please visit Main Reception on the Ground Floor for help."
                )
                log_interaction(query, modality, intent, intent_confidence,
                                "no_match", retrieval_score, False, response)
                return {"response": response, "intent": intent,
                        "intent_confidence": intent_confidence,
                        "matched_location": "no_match",
                        "retrieval_score": retrieval_score,
                        "llm_used": False, "kb_result": ""}

        # Step 5: Format KB result
        # Pass all retrieved records so LLM can synthesise multi-record answers
        all_records = [r["record"] for r in results]
        kb_result   = build_kb_result_multi(query, intent, all_records)

        # Step 6: LLM natural language response
        response, llm_used = ask_llm(query, kb_result)

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Pipeline error: {e}")
        response        = "Sorry, something went wrong. Please try again."
        intent          = "error"
        intent_confidence = 0.0
        record          = {"name": "error"}
        retrieval_score = 0.0
        llm_used        = False
        kb_result       = ""

    # Step 7: Log
    log_interaction(
        query, modality, intent, intent_confidence,
        record.get("name", ""), retrieval_score,
        llm_used, response, error_msg
    )

    return {
        "response":          response,
        "intent":            intent,
        "intent_confidence": intent_confidence,
        "matched_location":  record.get("name", ""),
        "retrieval_score":   retrieval_score,
        "llm_used":          llm_used,
        "kb_result":         kb_result,
    }
