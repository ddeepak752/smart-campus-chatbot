"""
Knowledge Base utilities - all KB access goes through here.
"""

import json
from datetime import datetime
from pathlib import Path

KB_PATH = Path("data/kb/knowledge_base.json")
_KB = None


def load_kb() -> list:
    global _KB
    if _KB is None:
        with open(KB_PATH, "r", encoding="utf-8") as f:
            _KB = json.load(f)
    return _KB


def get_by_id(loc_id: str):
    for r in load_kb():
        if r.get("id") == loc_id:
            return r
    return None


def search(query: str, top_k: int = 3) -> list:
    """
    Score every KB record against the query.
    Scoring uses: exact match, partial match, alias match, keyword overlap.
    Returns top_k records sorted best first.
    """
    kb       = load_kb()
    q        = query.lower().strip()
    q_tokens = set(q.split())
    scored   = []

    for loc in kb:
        score = 0.0
        name  = loc.get("name", "").lower()

        if q == name:              score += 12.0
        elif q in name:            score += 6.0
        elif name in q:            score += 4.0

        for alias in loc.get("aliases", []):
            a = alias.lower()
            if q == a:
                score += 10.0
                break
            elif q in a or a in q:
                score += 4.0
                break

        kw_tokens = set(" ".join(loc.get("keywords", [])).lower().split())
        score    += len(q_tokens & kw_tokens) * 1.5
        score    *= loc.get("search_priority", 1.0)

        if score > 0:
            scored.append((score, loc))

    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored[:top_k]]


def find_menu_item(item_query: str) -> list:
    """
    Search every cafeteria menu for a specific item.
    Deduplicates so each unique (item, price) appears once.
    Returns list of dicts: {location, day, meal_type, item, price}
    """
    results = []
    seen    = set()
    q       = item_query.lower()

    for loc in load_kb():
        weekly = loc.get("menu", {}).get("weekly_menu", {})
        for day, meals in weekly.items():
            for meal_type, items in meals.items():
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    if q in item.get("item", "").lower():
                        key = (item["item"].lower(), item.get("price", ""))
                        if key not in seen:
                            seen.add(key)
                            results.append({
                                "location":  loc["name"],
                                "day":       day.capitalize(),
                                "meal_type": meal_type.replace("_", " ").title(),
                                "item":      item["item"],
                                "price":     item["price"],
                            })
    return results


def get_todays_menu(loc_id: str = "main_cafeteria") -> dict:
    loc = get_by_id(loc_id)
    if not loc:
        return {}
    today = datetime.now().strftime("%A").lower()
    return loc.get("menu", {}).get("weekly_menu", {}).get(today, {})


def get_all_events() -> list:
    events = []
    for loc in load_kb():
        for ev in loc.get("events", []):
            if ev:
                events.append({"location": loc["name"], "event": ev})
    return events

# Locations to exclude from floor listings — too granular
_FLOOR_EXCLUDE_CATEGORIES = {
    "washroom", "water_dispenser", "emergency_exit",
    "classroom", "lift", "corridor"
}

_FLOOR_EXCLUDE_KEYWORDS = {
    "washroom", "water dispenser", "emergency exit", "lift area"
}

def search_by_floor(floor_number: int) -> list:
    """
    Returns all meaningful locations on a given floor.
    Excludes washrooms, dispensers, classrooms and other sub-locations
    so the result is a useful floor summary.
    """
    kb = load_kb()
    import re
    results = []

    for loc in kb:
        coord = loc.get("coordinates", "")

        # Parse floor from coordinate
        if "K-1" in coord.upper():
            loc_floor = -1
        else:
            m = re.search(r'(\d+)', coord)
            loc_floor = int(m.group(1)) if m else 0

        if loc_floor != floor_number:
            continue

        # Skip excluded categories
        if loc.get("category", "") in _FLOOR_EXCLUDE_CATEGORIES:
            continue

        # Skip excluded keywords in name
        name_lower = loc.get("name", "").lower()
        if any(kw in name_lower for kw in _FLOOR_EXCLUDE_KEYWORDS):
            continue

        # Skip classrooms (name contains "Classroom")
        if "classroom" in name_lower:
            continue

        results.append(loc)

    return results

def build_doc(loc: dict) -> str:
    """
    Builds rich text for embedding. Only main locations get floor text
    so floor queries match departments, not washrooms or classrooms.
    """
    import re as _re

    coord     = loc.get("coordinates", "")
    floor_num = None
    if "K-1" in coord.upper():
        floor_num = -1
    else:
        m = _re.search(r'(\d+)', coord)
        if m:
            floor_num = int(m.group(1))

    # Only add floor text to main locations — not washrooms, classrooms etc.
    MAIN_CATEGORIES = {
        "department", "library", "cafeteria", "admin", "office",
        "medical", "sports", "student_services", "facility",
        "lab", "study_area", "hostel", "auditorium"
    }

    floor_text = ""
    if loc.get("category", "") in MAIN_CATEGORIES:
        floor_text = {
            -1: "basement floor gym basement floor",
             0: "ground floor entrance ground floor reception cafeteria admin ground floor",
             1: "first floor 1st floor library cse it computer science first floor",
             2: "second floor 2nd floor artificial intelligence electronics ece second floor",
             3: "third floor 3rd floor three third floor mechanical engineering electrical engineering me department ee department third floor",
             4: "fourth floor 4th floor four fourth floor civil engineering chemical engineering ce department che department fourth floor",
             5: "fifth floor 5th floor robotics biotechnology rob bio fifth floor",
             6: "sixth floor 6th floor mba humanities management hum sixth floor",
        }.get(floor_num, "")

    # Repeat keywords for locations with many synonyms to boost semantic signal
    keywords_text = " ".join(loc.get("keywords", []))
    keywords_boosted = keywords_text + " " + keywords_text  # repeat for signal strength

    return " ".join(filter(None, [
        floor_text,
        loc.get("name", ""),
        loc.get("category", "").replace("_", " "),
        loc.get("description", ""),
        " ".join(loc.get("keywords", [])),
        " ".join(loc.get("aliases", [])),
    ]))