"""Maharashtra districts used by the Maharashtra section.

lat/lon are district headquarters (geographic reference points, used only to
ask NASA POWER for the grid cell that covers the district). They are NOT
climate data. Census 2001/2011 had 35 districts: Palghar (2014) was part of
Thane, so Palghar is linked to the Thane census row (`census_id`).

`aliases` = spellings used in Census tables (Aurangabad, Osmanabad, Buldana,
Gondiya, Raigarh, Ahmadnagar ...), matched after lower-casing and removing
everything except letters.
"""
import re

# id, name, region, lat, lon, coastal, census_id, aliases
_ROWS = [
    ("ahilyanagar", "Ahilyanagar (Ahmednagar)", "North Maharashtra", 19.095, 74.749, False, "ahilyanagar", ["ahmadnagar", "ahmednagar", "ahilyanagar"]),
    ("akola", "Akola", "Vidarbha", 20.709, 77.002, False, "akola", []),
    ("amravati", "Amravati", "Vidarbha", 20.933, 77.752, False, "amravati", []),
    ("chh-sambhajinagar", "Chh. Sambhajinagar (Aurangabad)", "Marathwada", 19.876, 75.343, False, "chh-sambhajinagar", ["aurangabad", "chhatrapatisambhajinagar", "sambhajinagar"]),
    ("beed", "Beed", "Marathwada", 18.989, 75.756, False, "beed", ["bid"]),
    ("bhandara", "Bhandara", "Vidarbha", 21.167, 79.650, False, "bhandara", []),
    ("buldhana", "Buldhana", "Vidarbha", 20.531, 76.184, False, "buldhana", ["buldana"]),
    ("chandrapur", "Chandrapur", "Vidarbha", 19.961, 79.296, False, "chandrapur", []),
    ("dhule", "Dhule", "North Maharashtra", 20.902, 74.774, False, "dhule", []),
    ("gadchiroli", "Gadchiroli", "Vidarbha", 20.181, 80.000, False, "gadchiroli", []),
    ("gondia", "Gondia", "Vidarbha", 21.460, 80.192, False, "gondia", ["gondiya"]),
    ("hingoli", "Hingoli", "Marathwada", 19.718, 77.150, False, "hingoli", []),
    ("jalgaon", "Jalgaon", "North Maharashtra", 21.007, 75.563, False, "jalgaon", []),
    ("jalna", "Jalna", "Marathwada", 19.841, 75.880, False, "jalna", []),
    ("kolhapur", "Kolhapur", "Western Maharashtra", 16.705, 74.243, False, "kolhapur", []),
    ("latur", "Latur", "Marathwada", 18.408, 76.560, False, "latur", []),
    ("mumbai-city", "Mumbai City", "Konkan", 18.950, 72.835, True, "mumbai-city", ["mumbai", "mumbaicity", "greatermumbai"]),
    ("mumbai-suburban", "Mumbai Suburban", "Konkan", 19.076, 72.878, True, "mumbai-suburban", ["mumbaisuburban"]),
    ("nagpur", "Nagpur", "Vidarbha", 21.146, 79.088, False, "nagpur", []),
    ("nanded", "Nanded", "Marathwada", 19.138, 77.321, False, "nanded", []),
    ("nandurbar", "Nandurbar", "North Maharashtra", 21.370, 74.240, False, "nandurbar", []),
    ("nashik", "Nashik", "North Maharashtra", 19.998, 73.790, False, "nashik", ["nasik"]),
    ("dharashiv", "Dharashiv (Osmanabad)", "Marathwada", 18.186, 76.042, False, "dharashiv", ["osmanabad", "dharashiv"]),
    ("palghar", "Palghar", "Konkan", 19.697, 72.766, True, "thane", []),
    ("parbhani", "Parbhani", "Marathwada", 19.268, 76.771, False, "parbhani", []),
    ("pune", "Pune", "Western Maharashtra", 18.520, 73.857, False, "pune", []),
    ("raigad", "Raigad", "Konkan", 18.641, 72.878, True, "raigad", ["raigarh"]),
    ("ratnagiri", "Ratnagiri", "Konkan", 16.994, 73.300, True, "ratnagiri", []),
    ("sangli", "Sangli", "Western Maharashtra", 16.855, 74.565, False, "sangli", []),
    ("satara", "Satara", "Western Maharashtra", 17.680, 74.018, False, "satara", []),
    ("sindhudurg", "Sindhudurg", "Konkan", 16.349, 73.559, True, "sindhudurg", []),
    ("solapur", "Solapur", "Western Maharashtra", 17.659, 75.906, False, "solapur", ["sholapur"]),
    ("thane", "Thane", "Konkan", 19.218, 72.978, True, "thane", []),
    ("wardha", "Wardha", "Vidarbha", 20.745, 78.602, False, "wardha", []),
    ("washim", "Washim", "Vidarbha", 20.111, 77.133, False, "washim", []),
    ("yavatmal", "Yavatmal", "Vidarbha", 20.389, 78.120, False, "yavatmal", []),
]

DISTRICTS = [
    dict(id=i, name=n, region=r, lat=la, lon=lo, coastal=c, census_id=ci, aliases=al)
    for (i, n, r, la, lo, c, ci, al) in _ROWS
]


def norm(s) -> str:
    return re.sub(r"[^a-z]", "", str(s).lower())


def census_lookup() -> dict:
    """normalised census spelling -> census district id (Palghar is NOT a census district)."""
    m = {}
    for d in DISTRICTS:
        if d["id"] == "palghar":
            continue
        for k in [d["id"], d["name"], *d["aliases"]]:
            m[norm(k)] = d["id"]
        # "Ahilyanagar (Ahmednagar)" style names: also register the text before/inside brackets
        for part in re.split(r"[()]", d["name"]):
            if part.strip():
                m[norm(part)] = d["id"]
    return m
