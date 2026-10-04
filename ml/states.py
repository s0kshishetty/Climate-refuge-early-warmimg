"""Reference table for Indian states/UTs + helpers to clean state names.

Columns: name -> (lat, lon, coastal, himalayan)  (approximate centroids)
Ladakh is merged into 'Jammu and Kashmir' and Telangana is its own state
(events before 2014 that name Telangana districts will only match if the
word 'Telangana' appears in the text).
"""
import re

STATES = {
    "Andhra Pradesh":    (15.9, 79.7, 1, 0),
    "Arunachal Pradesh": (28.2, 94.7, 0, 1),
    "Assam":             (26.2, 92.9, 0, 0),
    "Bihar":             (25.1, 85.3, 0, 0),
    "Chhattisgarh":      (21.3, 81.9, 0, 0),
    "Delhi":             (28.6, 77.2, 0, 0),
    "Goa":               (15.3, 74.1, 1, 0),
    "Gujarat":           (22.3, 71.2, 1, 0),
    "Haryana":           (29.1, 76.1, 0, 0),
    "Himachal Pradesh":  (31.9, 77.2, 0, 1),
    "Jammu and Kashmir": (33.8, 76.6, 0, 1),
    "Jharkhand":         (23.6, 85.3, 0, 0),
    "Karnataka":         (15.3, 75.7, 1, 0),
    "Kerala":            (10.5, 76.3, 1, 0),
    "Madhya Pradesh":    (23.5, 78.0, 0, 0),
    "Maharashtra":       (19.7, 75.7, 1, 0),
    "Manipur":           (24.7, 93.9, 0, 0),
    "Meghalaya":         (25.5, 91.4, 0, 0),
    "Mizoram":           (23.2, 92.8, 0, 0),
    "Nagaland":          (26.1, 94.5, 0, 0),
    "Odisha":            (20.5, 84.4, 1, 0),
    "Punjab":            (31.0, 75.4, 0, 0),
    "Rajasthan":         (26.6, 73.8, 0, 0),
    "Sikkim":            (27.6, 88.5, 0, 1),
    "Tamil Nadu":        (11.1, 78.4, 1, 0),
    "Telangana":         (17.9, 79.0, 0, 0),
    "Tripura":           (23.8, 91.7, 0, 0),
    "Uttar Pradesh":     (26.8, 80.9, 0, 0),
    "Uttarakhand":       (30.1, 79.2, 0, 1),
    "West Bengal":       (23.0, 87.9, 1, 0),
}

ALIASES = {
    "orissa": "Odisha",
    "uttaranchal": "Uttarakhand",
    "jammu & kashmir": "Jammu and Kashmir",
    "jammu and kashmir": "Jammu and Kashmir",
    "j&k": "Jammu and Kashmir",
    "ladakh": "Jammu and Kashmir",
    "nct of delhi": "Delhi",
    "new delhi": "Delhi",
    "andhra": "Andhra Pradesh",
    "bengal": "West Bengal",
    "tamilnadu": "Tamil Nadu",
}

# Small city -> state hint list (EM-DAT 'Location' often has only city/district names)
CITY_HINTS = {
    "mumbai": "Maharashtra", "pune": "Maharashtra", "kolhapur": "Maharashtra",
    "chennai": "Tamil Nadu", "kolkata": "West Bengal", "calcutta": "West Bengal",
    "hyderabad": "Telangana", "bangalore": "Karnataka", "bengaluru": "Karnataka",
    "patna": "Bihar", "lucknow": "Uttar Pradesh", "ahmedabad": "Gujarat",
    "surat": "Gujarat", "kochi": "Kerala", "cochin": "Kerala", "guwahati": "Assam",
    "bhubaneswar": "Odisha", "puri": "Odisha", "jaipur": "Rajasthan",
    "srinagar": "Jammu and Kashmir", "shimla": "Himachal Pradesh",
    "dehradun": "Uttarakhand", "kedarnath": "Uttarakhand", "chamoli": "Uttarakhand",
    "visakhapatnam": "Andhra Pradesh", "vijayawada": "Andhra Pradesh",
    "bhopal": "Madhya Pradesh", "ranchi": "Jharkhand", "raipur": "Chhattisgarh",
}

_lookup = {s.lower(): s for s in STATES}
_lookup.update({k: v for k, v in ALIASES.items()})


def normalize_state(name):
    """Return canonical state name or None."""
    if not isinstance(name, str):
        return None
    n = name.strip().lower().replace("&", "and")
    n = re.sub(r"\s+", " ", n)
    if n in _lookup:
        return _lookup[n]
    n2 = n.replace("and ", "& ") if "and " in n else n
    return _lookup.get(n2)


def _pattern(term):
    if term == "bengal":
        return re.compile(r"(?<!bay of )(?<!west )\bbengal\b")
    return re.compile(r"\b" + re.escape(term) + r"\b")


_PATTERNS = [(_pattern(t), s) for t, s in _lookup.items()]
_PATTERNS += [(_pattern(t), s) for t, s in CITY_HINTS.items()]


def extract_states(text):
    """Find every state mentioned in free text (e.g. EM-DAT 'Location')."""
    if not isinstance(text, str):
        return set()
    t = text.lower().replace("&", "and")
    found = set()
    for pat, state in _PATTERNS:
        if pat.search(t):
            found.add(state)
    # 'jammu and kashmir' written with '&' was already normalised above
    return found
