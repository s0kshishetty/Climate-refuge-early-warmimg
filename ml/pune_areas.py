"""Pune local areas used by the Pune early-warning section.

lat/lon are approximate area centres (verify on a map if you need precision).

`exposure` (0-1) is a flood-exposure ESTIMATE: riverside / low-lying / known
waterlogging areas get higher values. It is a hand-made assumption, NOT a
measured value. Replace it later with real data (PMC flood maps, OSM distance
to river, drainage data).
"""

AREAS = [
    {"id": "shivajinagar",     "name": "Shivajinagar",     "lat": 18.5308, "lon": 73.8475, "exposure": 0.6},
    {"id": "deccan",           "name": "Deccan Gymkhana",  "lat": 18.5167, "lon": 73.8412, "exposure": 0.6},
    {"id": "kothrud",          "name": "Kothrud",          "lat": 18.5074, "lon": 73.8077, "exposure": 0.5},
    {"id": "warje",            "name": "Warje",            "lat": 18.4880, "lon": 73.8000, "exposure": 0.6},
    {"id": "sinhagad-road",    "name": "Sinhagad Road",    "lat": 18.4795, "lon": 73.8240, "exposure": 0.9},
    {"id": "aundh",            "name": "Aundh",            "lat": 18.5580, "lon": 73.8075, "exposure": 0.5},
    {"id": "baner",            "name": "Baner",            "lat": 18.5590, "lon": 73.7868, "exposure": 0.4},
    {"id": "pashan",           "name": "Pashan",           "lat": 18.5380, "lon": 73.7920, "exposure": 0.3},
    {"id": "hinjewadi",        "name": "Hinjewadi",        "lat": 18.5912, "lon": 73.7389, "exposure": 0.7},
    {"id": "wakad",            "name": "Wakad",            "lat": 18.5980, "lon": 73.7600, "exposure": 0.5},
    {"id": "pimpri-chinchwad", "name": "Pimpri-Chinchwad", "lat": 18.6298, "lon": 73.7997, "exposure": 0.7},
    {"id": "nigdi",            "name": "Nigdi",            "lat": 18.6590, "lon": 73.7710, "exposure": 0.5},
    {"id": "bhosari",          "name": "Bhosari",          "lat": 18.6460, "lon": 73.8480, "exposure": 0.4},
    {"id": "yerwada",          "name": "Yerwada",          "lat": 18.5530, "lon": 73.8890, "exposure": 0.7},
    {"id": "sangamwadi",       "name": "Sangamwadi",       "lat": 18.5340, "lon": 73.8700, "exposure": 0.8},
    {"id": "viman-nagar",      "name": "Viman Nagar",      "lat": 18.5679, "lon": 73.9143, "exposure": 0.4},
    {"id": "kharadi",          "name": "Kharadi",          "lat": 18.5515, "lon": 73.9475, "exposure": 0.5},
    {"id": "wagholi",          "name": "Wagholi",          "lat": 18.5800, "lon": 73.9830, "exposure": 0.4},
    {"id": "hadapsar",         "name": "Hadapsar",         "lat": 18.5089, "lon": 73.9260, "exposure": 0.5},
    {"id": "kondhwa",          "name": "Kondhwa",          "lat": 18.4637, "lon": 73.8940, "exposure": 0.4},
    {"id": "katraj",           "name": "Katraj",           "lat": 18.4575, "lon": 73.8677, "exposure": 0.8},
    {"id": "bibwewadi",        "name": "Bibwewadi",        "lat": 18.4730, "lon": 73.8590, "exposure": 0.3},
    {"id": "swargate",         "name": "Swargate",         "lat": 18.5018, "lon": 73.8636, "exposure": 0.4},
    {"id": "camp",             "name": "Pune Camp",        "lat": 18.5196, "lon": 73.8790, "exposure": 0.4},
]
