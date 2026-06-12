"""
Answer Generator
Routes each intent to the correct formatting function,
then returns a structured string for the LLM to rewrite naturally.
"""

from datetime import datetime
from src.kb_utils         import load_kb, find_menu_item, get_todays_menu
from src.direction_engine import get_directions_from_query


def _fmt_events(events: list) -> str:
    return "\n".join(f"  - {e}" for e in events) if events else "  No upcoming events."


def _fmt_hours(record: dict) -> str:
    return record.get("opening_hours", "Opening hours not available.")


def _menu_handler(query: str, record: dict) -> str:
    ITEMS = sorted([
        "tea","coffee","samosa","sandwich","thali","biryani","juice",
        "cold coffee","muffin","brownie","lassi","idli","dosa","paratha",
        "noodles","pasta","veg fried rice","chicken fried rice","fried rice","rice","fries","cookies","smoothie","lemonade",
        "iced tea","omelette","pancakes","pakora","upma","poha",
    ], key=len, reverse=True)
    DAYS = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]

    q = query.lower()
    import re as _re
    found = next(
        (i for i in ITEMS if _re.search(r'\b' + _re.escape(i) + r'\b', q)),
        None
    )

    if found:
        hits = find_menu_item(found)
        if hits:
            seen, lines = set(), []
            for h in hits:
                key = (h["item"].lower(), h["price"])
                if key not in seen:
                    seen.add(key)
                    lines.append(f"  - {h['item']}: {h['price']}")
            pay = ", ".join(record.get("menu", {}).get("payment_options", []))
            return (
                f"Location: {record['name']}\n"
                f"Price for '{found}':\n" + "\n".join(lines[:6]) +
                (f"\nPayment accepted: {pay}" if pay else "")
            )
        return f"'{found}' was not found on the menu at {record['name']}."

    # Check if user asked for a specific day
    asked_day = next((d for d in DAYS if d in q), None)
    if asked_day:
        weekly = record.get("menu", {}).get("weekly_menu", {})
        menu = weekly.get(asked_day, {})
        if not menu:
            return f"Menu not available for {asked_day.capitalize()} at {record['name']}."
        lines = [f"{asked_day.capitalize()} Menu at {record['name']}:"]
        for meal_type, items in menu.items():
            lines.append(f"\n  {meal_type.title()}:")
            for item in items:
                if isinstance(item, dict):
                    lines.append(f"    - {item.get('item')}: {item.get('price')}")
        pay = ", ".join(record.get("menu", {}).get("payment_options", []))
        if pay:
            lines.append(f"\nPayment: {pay}")
        return "\n".join(lines)

    # General menu query - today only
    day  = datetime.now().strftime("%A")
    menu = get_todays_menu(record.get("id", "main_cafeteria"))
    if not menu:
        return f"Menu not available for today ({day}) at {record['name']}."

    lines = [f"Today's Menu ({day}) at {record['name']}:"]
    for meal_type, items in menu.items():
        lines.append(f"\n  {meal_type.title()}:")
        for item in items:
            if isinstance(item, dict):
                lines.append(f"    - {item.get('item')}: {item.get('price')}")
    pay = ", ".join(record.get("menu", {}).get("payment_options", []))
    if pay:
        lines.append(f"\nPayment: {pay}")
    return "\n".join(lines)


def _location_handler(query: str, record: dict) -> str:
    kb         = load_kb()
    directions = get_directions_from_query(query, record, kb)
    return (
        f"Location: {record['name']}\n"
        f"Category: {record.get('category','')}\n"
        f"About: {record.get('description','')}\n\n"
        f"Map Reference: {record.get('map_reference','')}\n\n"
        f"Directions:\n{directions}\n\n"
        f"Opening Hours: {_fmt_hours(record)}\n\n"
        f"Events:\n{_fmt_events(record.get('events',[]))}"
    )


def _hours_handler(record: dict) -> str:
    return (
        f"Location: {record['name']}\n"
        f"Opening Hours: {_fmt_hours(record)}\n"
        f"About: {record.get('description','')}"
    )


def _events_handler(record: dict) -> str:
    return (
        f"Location: {record['name']}\n"
        f"Events:\n{_fmt_events(record.get('events',[]))}\n"
        f"Opening Hours: {_fmt_hours(record)}"
    )


def _admission_handler(record: dict) -> str:
    courses = record.get("courses_offered", [])
    lines = [
        f"Location: {record['name']}",
        f"About: {record.get('description','')}",
        f"Opening Hours: {_fmt_hours(record)}",
        f"Map Reference: {record.get('map_reference','')}",
    ]
    if courses:
        lines.append("Courses Offered:")
        lines.extend(f"  - {course}" for course in courses)
    return "\n".join(lines)


def _emergency_handler(query: str, record: dict) -> str:
    kb         = load_kb()
    directions = get_directions_from_query(query, record, kb)
    return (
        f"URGENT: {record['name']}\n"
        f"About: {record.get('description','')}\n\n"
        f"How to get there:\n{directions}\n\n"
        f"Available: {_fmt_hours(record)}"
    )


def build_kb_result(query: str, intent: str, record=None) -> str:
    """
    Main entry point. Takes query + intent + matched KB record.
    Returns a structured string that gets passed to the LLM.
    """
    if not record:
        return (
            "No matching campus location found for this query. "
            "Please visit Main Reception on the Ground Floor for assistance."
        )

    if intent == "menu_query":
        return _menu_handler(query, record)
    if intent == "ask_hours":
        return _hours_handler(record)
    if intent == "ask_event":
        return _events_handler(record)
    if intent == "ask_admission":
        return _admission_handler(record)
    if intent in ("emergency", "lost_found"):
        return _emergency_handler(query, record)

    # Contact/hours — include description and hours, no step-by-step directions
    if intent == "ask_contact":
        return (
            f"Location: {record['name']}\n"
            f"About: {record.get('description','')}\n"
            f"Opening Hours: {_fmt_hours(record)}\n"
            f"Map Reference: {record.get('map_reference','')}\n"
            f"Events:\n{_fmt_events(record.get('events',[]))}"
        )

    # find_location, faculty_query, ask_department, service_query,
    # recommend_place, facility_info, fallback
    return _location_handler(query, record)

def build_kb_result_multi(query: str, intent: str, records: list) -> str:
    """
    Multi-record version of build_kb_result.
    Passes context from top-3 KB records to the LLM so it can
    synthesise answers for floor listings, washroom queries,
    multi-location questions etc. — without any rule-based special cases.

    Single-record intents (menu, hours, emergency) still use only the
    best record to avoid confusing the LLM with irrelevant context.
    """
    if not records:
        return (
            "No matching campus location found. "
            "Please visit Main Reception on the Ground Floor."
        )

    # These intents need focused single-record answers
    single_record_intents = {
        "menu_query", "ask_hours", "emergency", "lost_found"
    }

    if intent in single_record_intents:
        return build_kb_result(query, intent, records[0])

    # For all other intents pass all records as context
    best   = records[0]
    others = records[1:]

    # Build primary answer from best match
    primary = build_kb_result(query, intent, best)

    if not others:
        return primary

    # Append additional context from records 2 and 3
    extra_parts = []
    for rec in others:
        name    = rec.get("name", "")
        cat     = rec.get("category", "")
        desc    = rec.get("description", "")
        map_ref = rec.get("map_reference", "")
        hours   = rec.get("opening_hours", "")
        extra_parts.append(
            f"- {name} ({cat}): {desc} | Location: {map_ref} | Hours: {hours}"
        )

    extra = "\n".join(extra_parts)
    return (
        f"{primary}\n\n"
        f"Additional related locations:\n{extra}"
    )
