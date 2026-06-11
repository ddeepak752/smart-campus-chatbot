"""
Direction Engine
Generates step-by-step navigation between any two KB locations.

Coordinate system used in the KB:
  CSE1       -> Floor 1, CSE Block
  CE4        -> Floor 4, Civil Block
  AI2        -> Floor 2, AI Block
  CAF0       -> Ground Floor, Cafeteria
  K-1        -> Basement
  PARK-STAFF -> Outdoor parking
"""

import re

FLOOR_NAMES = {
    -1: "Basement",
     0: "Ground Floor",
     1: "First Floor",
     2: "Second Floor",
     3: "Third Floor",
     4: "Fourth Floor",
     5: "Fifth Floor",
     6: "Sixth Floor",
}

ZONE_DISPLAY = {
    "CSE":   "CSE Block",
    "IT":    "IT Block",
    "AI":    "AI & Data Science Block",
    "ECE":   "ECE Block",
    "EE":    "Electrical Engineering Block",
    "ME":    "Mechanical Block",
    "CE":    "Civil Block",
    "CHE":   "Chemical Block",
    "BIO":   "Biotech Block",
    "ROB":   "Robotics Block",
    "MBA":   "MBA / Management Block",
    "HUM":   "Humanities Block",
    "ADM":   "Admin Block",
    "CAF":   "Cafeteria Block",
    "LIB":   "Library Block",
    "A":     "Main Building",
    "SCI":   "Science Block",
    "WS":    "Workshop Block",
    "R":     "Innovation Block",
    "C":     "Student Services Block",
    "K":     "Basement Block",
    "SEM":   "Seminar Hall Area",
    "AUD":   "Auditorium Block",
    "CAR":   "Careers Block",
    "PLC":   "Placement Block",
    "W":      "Washroom Area",
    "EXIT":   "Emergency Exit",
    "LIFT":   "Main Lift Area",
    "U":      "Hostel Office Block",
    "HOSTEL": "Hostel Block",
    "TR":     "Transport Area",
    "STU":    "Student Common Area",
    "STUDY":  "Open Study Area",
    "ROBO":   "Robotics Block",
}

OUTDOOR_PREFIXES = {"SP", "PARK", "GH", "BH"}
OUTDOOR_EXACT    = {"A-GATE", "A-SEC", "A-ATM"}


def parse_coord(coord: str) -> dict:
    """
    Parse coordinate string into floor and zone.

    Returns dict with keys: floor (int), zone (str), is_outdoor (bool)
    """
    if not coord:
        return {"floor": 0, "zone": "A", "is_outdoor": False}

    c = coord.upper().strip()

    if c.startswith("K-1"):
        return {"floor": -1, "zone": "K", "is_outdoor": False}

    if c in OUTDOOR_EXACT:
        zone = c.split("-")[0]
        return {"floor": 0, "zone": zone, "is_outdoor": True}

    # Standard: letters then digit(s), e.g. CSE1, AI2, CAF0, MBA6
    m = re.match(r'^([A-Z]+?)(\d+)', c)
    if m:
        zone  = m.group(1)
        floor = int(m.group(2))
        return {"floor": floor, "zone": zone, "is_outdoor": zone in OUTDOOR_PREFIXES}

    # Hyphenated codes: PARK-STAFF, HOSTEL-MESS, A-MED, A-LOST
    m2 = re.match(r'^([A-Z]+)-', c)
    if m2:
        zone = m2.group(1)
        return {"floor": 0, "zone": zone, "is_outdoor": zone in OUTDOOR_PREFIXES}

    return {"floor": 0, "zone": "A", "is_outdoor": False}


def generate_directions(from_loc: dict, to_loc: dict) -> str:
    """
    Core navigation function.
    Takes two KB records, returns numbered step-by-step directions.
    """
    fc = parse_coord(from_loc.get("coordinates", ""))
    tc = parse_coord(to_loc.get("coordinates", ""))

    from_name  = from_loc.get("name", "your location")
    to_name    = to_loc.get("name",   "destination")
    to_zone    = ZONE_DISPLAY.get(tc["zone"], f"{tc['zone']} Block")
    from_floor = FLOOR_NAMES.get(fc["floor"], f"Floor {fc['floor']}")
    to_floor   = FLOOR_NAMES.get(tc["floor"], f"Floor {tc['floor']}")
    to_map_ref = to_loc.get("map_reference", "")

    steps    = []
    step_num = [1]

    def add(text):
        steps.append(f"  {step_num[0]}. {text}")
        step_num[0] += 1

    add(f"Start at {from_name} ({from_floor}).")

    # Outdoor destination
    if tc["is_outdoor"]:
        if fc["floor"] != 0:
            add("Go down to the Ground Floor using the lift or stairs.")
        add("Exit through the main building entrance.")
        add(f"Head to {to_name} — {to_map_ref or to_zone}.")
        return "\n".join(steps)

    # Same floor, same zone — very close together
    if fc["floor"] == tc["floor"] and fc["zone"] == tc["zone"]:
        add(f"You are already in the right area ({to_zone}, {to_floor}).")
        add(f"Look for {to_name} — {to_map_ref or 'nearby'}.")
        return "\n".join(steps)

    # Same floor, different zone
    if fc["floor"] == tc["floor"]:
        add(f"Stay on the {to_floor}.")
        add(f"Walk towards the {to_zone}.")
        add(f"Find {to_name} — {to_map_ref or 'in that section'}.")
        return "\n".join(steps)

    # Different floors
    floor_context = {
        -1: "Basement (Gym)",
         0: "Ground Floor (Reception, Cafeteria, Admin)",
         1: "First Floor (Library, CSE, IT)",
         2: "Second Floor (AI, ECE)",
         3: "Third Floor (Mechanical, Electrical)",
         4: "Fourth Floor (Civil, Chemical)",
         5: "Fifth Floor (Robotics, Biotechnology)",
         6: "Sixth Floor (MBA, Humanities)",
    }
    dest_label = floor_context.get(tc["floor"], to_floor)
    diff = abs(tc["floor"] - fc["floor"])
    if tc["floor"] == -1:
        add("Take the stairs down to the Basement (Gym).")
    elif tc["floor"] > fc["floor"]:
        add(f"Take the lift or stairs UP {diff} floor(s) to the {dest_label}.")
    else:
        add(f"Take the lift or stairs DOWN {diff} floor(s) to the {dest_label}.")

    if fc["zone"] != tc["zone"]:
        add(f"On the {to_floor}, walk towards the {to_zone}.")

    add(f"Find {to_name} — {to_map_ref or 'in this block'}.")
    return "\n".join(steps)


def get_directions_from_query(query: str, to_record: dict, all_kb: list) -> str:
    """
    Checks if the user said 'from X' in their query.
    If yes, finds X in KB and generates directions from X to destination.
    If no 'from' found, defaults to directions from Main Reception.
    """
    q = query.lower()
    from_loc = None

    # Detect "from <place>" pattern
    m = re.search(r'\bfrom\s+(?:the\s+)?(.+?)(?:\s+to\b|\s+how|\s+where|\s*$)', q)
    if m:
        hint = m.group(1).strip()
        # Strip common suffixes so "civil department" matches "Civil Engineering Department"
        for suffix in [" department", " block", " room", " office", " lab", " floor"]:
            hint = hint.replace(suffix, "").strip()
        for loc in all_kb:
            if hint in loc.get("name", "").lower():
                from_loc = loc
                break
            if any(hint in a.lower() for a in loc.get("aliases", [])):
                from_loc = loc
                break
            if any(hint in k.lower() for k in loc.get("keywords", [])):
                from_loc = loc
                break
    # Default: Main Reception
    if not from_loc:
        for loc in all_kb:
            if loc.get("id") == "reception":
                from_loc = loc
                break

    if not from_loc:
        return to_record.get("directions", "Directions not available.")

    return generate_directions(from_loc, to_record)
