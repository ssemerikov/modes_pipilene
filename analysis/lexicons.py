#!/usr/bin/env python3
"""Every word list the analysis uses, in one place.

The lists descend from the first submission's pipeline. Three things changed,
and all three are reported in the supplementary appendix:

1. Entries are lemmas and are matched against the lemma *and* the surface form
   of each token, so inflected forms count in both languages (the first version
   matched surface forms only, which favoured German entries listed in several
   inflections over English ones listed in one).
2. The English and German lists were brought into line concept by concept.
   Items with no counterpart in the other language, or too ambiguous to keep
   ("gerade", "eben", "advance"), were dropped.
3. German closed compounds count when a war term is their first or last element
   ("Raketenangriff", "Kriegsgebiet"); English writes these as two words, both
   of which already count.

Cross-language comparisons of lexicon densities remain approximate, which is why
the article rests its period claims on contrasts inside one language and inside
one author.
"""
from __future__ import annotations

# ── Narrator presence ─────────────────────────────────────────────────────────
FIRST_PERSON = {
    "en": {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"},
    "de": {"ich", "mir", "mich", "wir", "uns"},
}
# German possessives inflect (mein, meine, meinem ...; unser, unsere ...).
FIRST_PERSON_PREFIX_DE = ("mein", "unser", "unsr")

# ── Genre markers ─────────────────────────────────────────────────────────────
# Diary: deixis anchored in the day of writing, and calendar dates in the text.
DIARY = {
    "en": {"today", "yesterday", "tomorrow", "tonight", "now", "later", "earlier"},
    "de": {"heute", "gestern", "morgen", "jetzt", "später", "vorhin", "soeben", "derzeit"},
}
DIARY_ADV_ONLY = {"de": {"morgen"}, "en": set()}       # "Morgen" the noun is a morning
DIARY_PHRASES = {
    "en": [r"\bthis (?:morning|afternoon|evening)\b"],
    "de": [],            # "heute" and adverbial "morgen" are already counted as tokens
}
DATE_PATTERN = {
    "en": (r"\b(?:\d{1,2}(?:st|nd|rd|th)? (?:January|February|March|April|May|June|July|"
           r"August|September|October|November|December)|"
           r"(?:January|February|March|April|May|June|July|August|September|October|"
           r"November|December) \d{1,2})\b"),
    "de": (r"\b\d{1,2}\. ?(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|"
           r"Oktober|November|Dezember)\b"),
}

# Travel: movement and sensory observation.
TRAVEL = {
    "en": {"arrive", "depart", "travel", "journey", "drive", "walk", "cross", "reach",
           "leave", "pass", "border",
           "see", "hear", "smell", "taste", "feel", "look", "sound", "touch",
           "beautiful", "cold", "warm", "dark", "bright"},
    "de": {"ankommen", "abfahren", "reisen", "reise", "fahren", "gehen", "überqueren",
           "erreichen", "verlassen", "passieren", "grenze",
           "sehen", "hören", "riechen", "schmecken", "fühlen", "aussehen", "klingen",
           "berühren", "schön", "kalt", "warm", "dunkel", "hell"},
}

# Historical: temporal distance and inherited past.
HISTORICAL = {
    "en": {"century", "decade", "year", "ancient", "medieval", "historical", "formerly",
           "once", "tradition", "heritage", "memory", "remember"},
    "de": {"jahrhundert", "jahrzehnt", "jahr", "antik", "mittelalterlich", "historisch",
           "ehemals", "einst", "tradition", "erbe", "erinnerung", "erinnern"},
}

# ── Evaluative and modal stance markers ("subjectivity" in the first version) ─
STANCE = {
    "en": {"beautiful", "ugly", "wonderful", "terrible", "amazing", "horrible",
           "excellent", "awful", "great", "bad", "strange", "odd", "remarkable",
           "impressive", "disturbing", "sad", "happy", "grim", "bleak", "vibrant",
           "charming",
           "might", "could", "would", "should", "may", "must",
           "perhaps", "maybe", "somewhat", "rather", "quite", "fairly", "apparently",
           "seemingly", "allegedly",
           "very", "extremely", "incredibly", "absolutely", "utterly", "really",
           "deeply", "profoundly", "tremendously"},
    "de": {"schön", "hässlich", "wunderbar", "schrecklich", "toll", "furchtbar",
           "großartig", "schlimm", "seltsam", "merkwürdig", "beeindruckend",
           "beunruhigend", "traurig", "fröhlich", "düster", "trostlos", "lebendig",
           "bezaubernd", "schlecht", "ausgezeichnet",
           "könnte", "würde", "sollte", "müsste", "dürfte", "mag",
           "vielleicht", "möglicherweise", "etwas", "ziemlich", "anscheinend",
           "offenbar", "wohl", "vermutlich", "angeblich",
           "sehr", "extrem", "unglaublich", "absolut", "völlig", "wirklich",
           "zutiefst", "enorm", "ungeheuer"},
}

# ── War vocabulary, four categories ───────────────────────────────────────────
WAR = {
    "en": {
        "conflict": {"war", "conflict", "battle", "fight", "fighting", "combat", "siege",
                     "assault", "attack", "offensive", "invasion", "occupation",
                     "resistance"},
        "weapons": {"weapon", "gun", "rifle", "tank", "missile", "rocket", "bomb",
                    "artillery", "ammunition", "drone", "howitzer", "mortar", "shell",
                    "shelling", "grenade", "mine"},
        "suffering": {"death", "dead", "kill", "wound", "wounded", "injury", "injured",
                      "refugee", "displaced", "flee", "destroy", "destruction", "ruin",
                      "devastation", "casualty", "victim", "trauma", "grief", "mourn"},
        "military": {"army", "soldier", "military", "troop", "commander", "general",
                     "battalion", "regiment", "frontline", "defense", "defence",
                     "retreat", "mobilization", "mobilisation"},
    },
    "de": {
        "conflict": {"krieg", "konflikt", "kampf", "schlacht", "kämpfen", "belagerung",
                     "angriff", "offensive", "invasion", "besatzung", "besetzung",
                     "widerstand", "gefecht"},
        "weapons": {"waffe", "gewehr", "panzer", "rakete", "bombe", "artillerie",
                    "munition", "drohne", "haubitze", "mörser", "granate", "beschuss",
                    "mine", "geschoss"},
        "suffering": {"tod", "tot", "tote", "töten", "verwunden", "verwundet",
                      "verletzen", "verletzt", "verletzung", "flüchtling", "vertrieben",
                      "flucht", "fliehen", "flüchten", "zerstören", "zerstört",
                      "zerstörung", "ruine", "verwüstung", "opfer", "trauma", "trauer",
                      "trauern"},
        "military": {"armee", "soldat", "militär", "militärisch", "truppe",
                     "kommandant", "kommandeur", "general", "bataillon", "regiment",
                     "front", "frontlinie", "verteidigung", "rückzug", "mobilisierung",
                     "mobilmachung"},
    },
}
WAR_CATEGORIES = ["conflict", "weapons", "suffering", "military"]

# Entries that are war terms only as nouns ("the general", "a mine").
WAR_NOUN_ONLY = {"en": {"general", "mine", "shell", "tank", "front"},
                 "de": {"general", "mine"}}
# English "front" counts only as "the front" not followed by "of".

# German stems that mark a closed compound as a war term when they open or close
# it. Short stems ("tod", "tot", "mine") are left out: too many false compounds.
WAR_COMPOUND_STEMS_DE = {
    "conflict": ["krieg", "kampf", "schlacht", "angriff", "invasion", "besatzung",
                 "gefecht", "belagerung"],
    "weapons": ["waffe", "gewehr", "panzer", "rakete", "bombe", "artillerie",
                "munition", "drohne", "granate", "beschuss"],
    "suffering": ["flüchtling", "zerstörung", "trümmer", "opfer"],
    "military": ["armee", "soldat", "militär", "truppe", "front", "bataillon",
                 "verteidigung"],
}


def all_lexicons() -> dict:
    """Everything above as plain data, for the supplementary material."""
    srt = lambda s: sorted(s)
    return {
        "first_person": {k: srt(v) for k, v in FIRST_PERSON.items()},
        "first_person_prefix_de": list(FIRST_PERSON_PREFIX_DE),
        "diary": {k: srt(v) for k, v in DIARY.items()},
        "diary_phrases": DIARY_PHRASES,
        "date_pattern": DATE_PATTERN,
        "travel": {k: srt(v) for k, v in TRAVEL.items()},
        "historical": {k: srt(v) for k, v in HISTORICAL.items()},
        "stance": {k: srt(v) for k, v in STANCE.items()},
        "war": {l: {c: srt(w) for c, w in cats.items()} for l, cats in WAR.items()},
        "war_noun_only": {k: srt(v) for k, v in WAR_NOUN_ONLY.items()},
        "war_compound_stems_de": WAR_COMPOUND_STEMS_DE,
    }
