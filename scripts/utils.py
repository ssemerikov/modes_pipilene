"""
Shared utilities for the computational analysis pipeline.
"""
import re
import json
from pathlib import Path
from collections import Counter

# ── Paths ──
BASE_DIR = Path("/home/cc/claude_code/hamaniuk3")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TEXTS_DIR = OUTPUT_DIR / "texts"
PAPER_DIR = BASE_DIR / "paper"
SCRIPTS_DIR = BASE_DIR / "scripts"

PDF_FILES = {
    "applebaum": DATA_DIR / "Applebaum_Between_East_and_West_1994_2015.pdf",
    "nicolay": DATA_DIR / "Nicolay_The_Humorless_Ladies_2016.pdf",
    "brumme": DATA_DIR / "Brumme_Im_Schatten_des_Krieges.pdf",
    "orth": DATA_DIR / "Stephan _Orth_Couchsurfing_Ukraine.pdf",
}

BOOK_META = {
    "applebaum": {
        "author": "Anne Applebaum",
        "title": "Between East and West: Across the Borderlands of Europe",
        "lang": "en",
        "year": 1994,
        "model": "historical_documentary",
    },
    "nicolay": {
        "author": "Franz Nicolay",
        "title": "The Humorless Ladies of Border Control",
        "lang": "en",
        "year": 2016,
        "model": "experiential_documentary",
    },
    "brumme": {
        "author": "Christoph Brumme",
        "title": "Im Schatten des Krieges",
        "lang": "de",
        "year": 2022,
        "model": "wartime_diary",
    },
    "orth": {
        "author": "Stephan Orth",
        "title": "Couchsurfing in der Ukraine",
        "lang": "de",
        "year": 2023,
        "model": "travel_documentary",
    },
}

# ── Ukraine keyword sets ──
UKRAINE_KEYWORDS_EN = {
    "ukraine", "ukrainian", "ukrainians", "ukraine's", "kyiv", "kiev",
    "lviv", "lvov", "odessa", "odesa", "kharkiv", "kharkov",
    "dnipro", "dnipropetrovsk", "zaporizhzhia", "donetsk", "donbas",
    "crimea", "chernobyl", "chernivtsi", "poltava", "kherson",
    "mykolaiv", "sumy", "zhytomyr", "vinnytsia", "ternopil",
    "ivano-frankivsk", "uzhhorod", "cherkasy", "chernihiv", "lutsk",
    "rivne", "kropyvnytskyi", "maidan", "euromaidan",
}

UKRAINE_KEYWORDS_DE = {
    "ukraine", "ukrainisch", "ukrainische", "ukrainischen", "ukrainischer",
    "ukrainer", "ukrainerin", "kyjiw", "kiew", "lwiw", "lemberg",
    "odesa", "odessa", "charkiw", "dnipro", "saporischschja",
    "donezk", "donbas", "krim", "tschernobyl", "tscherniwzi",
    "poltawa", "cherson", "mykolajiw", "sumy", "schytomyr",
    "winnyzja", "ternopil", "iwano-frankiwsk", "uschhorod",
    "tscherkasy", "tschernihiw", "luzk", "riwne", "maidan",
}

UKRAINE_KEYWORDS_ALL = UKRAINE_KEYWORDS_EN | UKRAINE_KEYWORDS_DE

# ── Toponym normalization ──
TOPONYM_NORMALIZE = {
    # Kyiv
    "kiev": "Kyiv", "kyiv": "Kyiv", "kyjiw": "Kyiv", "kiew": "Kyiv",
    "kijew": "Kyiv", "kiow": "Kyiv",
    # Lviv
    "lviv": "Lviv", "lvov": "Lviv", "lwów": "Lviv", "lwiw": "Lviv",
    "lwow": "Lviv", "lemberg": "Lviv", "lwóv": "Lviv", "l'viv": "Lviv",
    # Odesa
    "odessa": "Odesa", "odesa": "Odesa",
    # Kharkiv
    "kharkiv": "Kharkiv", "kharkov": "Kharkiv", "charkiw": "Kharkiv",
    "charkov": "Kharkiv", "charkow": "Kharkiv",
    # Dnipro
    "dnipro": "Dnipro", "dnipropetrovsk": "Dnipro", "dnipropetrowsk": "Dnipro",
    "dnjepropetrowsk": "Dnipro", "dnepr": "Dnipro",
    # Donetsk/Donbas
    "donetsk": "Donetsk", "donezk": "Donetsk",
    "donbas": "Donbas", "donbass": "Donbas",
    # Zaporizhzhia
    "zaporizhzhia": "Zaporizhzhia", "saporischschja": "Zaporizhzhia",
    "saporischja": "Zaporizhzhia", "zaporozhye": "Zaporizhzhia",
    # Crimea
    "crimea": "Crimea", "krim": "Crimea",
    # Chernobyl
    "chernobyl": "Chornobyl", "tschernobyl": "Chornobyl", "chornobyl": "Chornobyl",
    # Others
    "chernivtsi": "Chernivtsi", "czernowitz": "Chernivtsi", "tscherniwzi": "Chernivtsi",
    "cernauti": "Chernivtsi", "cernăuți": "Chernivtsi",
    "poltava": "Poltava", "poltawa": "Poltava",
    "kherson": "Kherson", "cherson": "Kherson",
    "mykolaiv": "Mykolaiv", "mykolajiw": "Mykolaiv", "nikolaev": "Mykolaiv",
    "sumy": "Sumy",
    "zhytomyr": "Zhytomyr", "schytomyr": "Zhytomyr",
    "vinnytsia": "Vinnytsia", "winnyzja": "Vinnytsia",
    "ternopil": "Ternopil",
    "ivano-frankivsk": "Ivano-Frankivsk", "iwano-frankiwsk": "Ivano-Frankivsk",
    "uzhhorod": "Uzhhorod", "uschhorod": "Uzhhorod",
    "cherkasy": "Cherkasy", "tscherkasy": "Cherkasy",
    "chernihiv": "Chernihiv", "tschernihiw": "Chernihiv",
    "lutsk": "Lutsk", "luzk": "Lutsk",
    "rivne": "Rivne", "riwne": "Rivne",
    "mariupol": "Mariupol",
    "bucha": "Bucha", "butscha": "Bucha",
    "irpin": "Irpin",
    "minsk": "Minsk",
    "brest": "Brest",
    "kaliningrad": "Kaliningrad", "königsberg": "Kaliningrad", "konigsberg": "Kaliningrad",
    "vilnius": "Vilnius", "wilno": "Vilnius",
    "moscow": "Moscow", "moskau": "Moscow",
    "berlin": "Berlin",
    "warsaw": "Warsaw", "warschau": "Warsaw",
    "krakow": "Krakow", "kraków": "Krakow", "krakau": "Krakow",
    "prague": "Prague", "prag": "Prague",
    "budapest": "Budapest",
    "vienna": "Vienna", "wien": "Vienna",
    "saint petersburg": "Saint Petersburg", "st. petersburg": "Saint Petersburg",
    "petersburg": "Saint Petersburg",
    "kishinev": "Chișinău", "chișinău": "Chișinău", "chisinau": "Chișinău",
    "kamenets podolsky": "Kamianets-Podilskyi",
    "drohobych": "Drohobych",
    "kobrin": "Kobrin",
    "prypjat": "Prypiat", "pripyat": "Prypiat",
    "saltiwka": "Saltivka",
    "bohorodytschne": "Bohorodychne",
}

# Region classification for Ukrainian cities
UKRAINE_REGIONS = {
    "Western": ["Lviv", "Ternopil", "Ivano-Frankivsk", "Uzhhorod", "Lutsk", "Rivne", "Drohobych"],
    "Central": ["Kyiv", "Vinnytsia", "Cherkasy", "Zhytomyr", "Poltava", "Chernihiv", "Irpin", "Bucha", "Prypiat"],
    "Eastern": ["Kharkiv", "Donetsk", "Donbas", "Dnipro", "Sumy", "Mariupol", "Saltivka", "Bohorodychne"],
    "Southern": ["Odesa", "Kherson", "Mykolaiv", "Zaporizhzhia", "Crimea"],
    "Non-Ukraine": ["Minsk", "Brest", "Kobrin", "Kaliningrad", "Vilnius", "Moscow",
                     "Berlin", "Warsaw", "Krakow", "Prague", "Budapest", "Vienna",
                     "Saint Petersburg", "Chișinău", "Chernivtsi"],
}


def normalize_toponym(name):
    """Normalize a toponym to its canonical form."""
    key = name.lower().strip()
    return TOPONYM_NORMALIZE.get(key, name)


def get_region(normalized_name):
    """Get the region for a normalized toponym."""
    for region, cities in UKRAINE_REGIONS.items():
        if normalized_name in cities:
            return region
    return "Other"


# ── War vocabulary ──
WAR_VOCAB_EN = {
    "conflict": {"war", "conflict", "battle", "fight", "combat", "siege", "assault",
                 "attack", "offensive", "invasion", "occupation", "resistance"},
    "weapons": {"weapon", "gun", "rifle", "tank", "missile", "rocket", "bomb",
                "artillery", "ammunition", "drone", "howitzer", "mortar"},
    "suffering": {"death", "dead", "kill", "killed", "wound", "wounded", "injury",
                  "refugee", "displaced", "flee", "destroy", "destruction", "ruin",
                  "devastation", "casualty", "victim", "trauma", "grief", "mourn"},
    "military": {"army", "soldier", "military", "troops", "commander", "general",
                 "battalion", "regiment", "front", "frontline", "defense", "defence",
                 "retreat", "advance", "mobilization"},
}

WAR_VOCAB_DE = {
    "conflict": {"krieg", "konflikt", "kampf", "schlacht", "angriff", "offensive",
                 "invasion", "besetzung", "widerstand", "gefecht"},
    "weapons": {"waffe", "waffen", "gewehr", "panzer", "rakete", "bombe",
                "artillerie", "munition", "drohne", "haubitze", "mörser"},
    "suffering": {"tod", "tot", "tote", "töten", "getötet", "verwundet", "verletzt",
                  "flüchtling", "vertrieben", "flucht", "zerstörung", "zerstört",
                  "ruine", "opfer", "trauma", "trauer"},
    "military": {"armee", "soldat", "soldaten", "militär", "truppen", "kommandant",
                 "general", "bataillon", "regiment", "front", "frontlinie",
                 "verteidigung", "rückzug", "mobilisierung", "mobilmachung"},
}


# ── Text cleaning ──
def dehyphenate(text):
    """Remove end-of-line hyphenation."""
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
    return text


def clean_ocr_text(text):
    """Clean common OCR artifacts."""
    # Fix common substitutions
    text = re.sub(r'(?<=[a-zäöü])\s{2,}(?=[a-zäöü])', ' ', text)
    # Remove isolated single characters that are OCR noise
    text = re.sub(r'\n[|l1I]\s*\n', '\n', text)
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def remove_headers_footers(text, page_num=None):
    """Remove page numbers and common headers/footers."""
    lines = text.split('\n')
    cleaned = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Skip standalone page numbers
        if re.match(r'^\d{1,3}$', stripped):
            continue
        # Skip very short lines at start/end that look like headers
        if (i < 2 or i > len(lines) - 3) and len(stripped) < 5 and not stripped.isalpha():
            continue
        cleaned.append(line)
    return '\n'.join(cleaned)


# ── KWIC ──
def kwic(text, pattern, window=5, max_results=100):
    """
    Keyword-in-context concordance.
    Returns list of (left_context, match, right_context) tuples.
    """
    tokens = text.split()
    results = []
    pat = re.compile(pattern, re.IGNORECASE)
    for i, tok in enumerate(tokens):
        if pat.search(tok):
            left = ' '.join(tokens[max(0, i - window):i])
            right = ' '.join(tokens[i + 1:i + 1 + window])
            results.append((left, tok, right))
            if len(results) >= max_results:
                break
    return results


def contains_ukraine_keyword(text, lang="en"):
    """Check if text mentions Ukraine-related keywords."""
    text_lower = text.lower()
    keywords = UKRAINE_KEYWORDS_EN if lang == "en" else UKRAINE_KEYWORDS_DE
    return any(kw in text_lower for kw in keywords)


def save_json(data, path):
    """Save data as JSON with proper encoding."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"  Saved: {path}")


def load_json(path):
    """Load JSON data."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
