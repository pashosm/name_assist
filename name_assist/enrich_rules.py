"""Rule-based name enrichment — no API key required.

Classifies names by heritage, class, origin_type, sentiment_tags, and sound
using curated lookup tables and phonetic heuristics.

Usage:
    python -m name_assist.enrich_rules
"""

import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
ENRICHMENTS_PATH = DATA_DIR / "enrichments.json"

# ---------------------------------------------------------------------------
# Heritage lookup — maps names (lowercase) to primary cultural origin
# ---------------------------------------------------------------------------

_HEBREW_NAMES = {
    "aaron", "abigail", "abraham", "adam", "adah", "amos", "anna", "ariel",
    "asher", "benjamin", "bethany", "caleb", "daniel", "david", "deborah",
    "delilah", "dinah", "eden", "eli", "eliana", "elias", "elijah",
    "elizabeth", "emanuel", "ephraim", "esther", "ethan", "eve", "ezekiel",
    "ezra", "gabriel", "gideon", "hannah", "isaac", "isaiah", "israel",
    "jacob", "jael", "james", "jedidiah", "jeremiah", "jesse", "joel",
    "john", "jonah", "jonathan", "jordan", "joseph", "joshua", "josiah",
    "judith", "leah", "levi", "malachi", "mary", "matthew", "micah",
    "michael", "miriam", "mordecai", "moses", "naomi", "nathan", "nehemiah",
    "noah", "obadiah", "rachel", "rebecca", "ruth", "samson", "samuel",
    "sarah", "seth", "simeon", "simon", "solomon", "susanna", "tabitha",
    "thomas", "tobias", "uriah", "zachariah", "zachary", "zion",
}

_GREEK_NAMES = {
    "achilles", "adonis", "agatha", "agnes", "aiden", "alexander", "alexandra",
    "alexis", "anastasia", "andrew", "angel", "angelica", "angelina",
    "anthony", "ariadne", "athena", "basil", "calista", "cassandra",
    "catherine", "chloe", "christian", "christopher", "constance", "cora",
    "cyrus", "damian", "daphne", "demetrius", "diana", "dorothy", "echo",
    "eleanor", "elena", "eugenia", "eugene", "evangeline", "george",
    "gregory", "helen", "hector", "irene", "iris", "jason", "katherine",
    "lydia", "margaret", "melissa", "nicholas", "penelope", "peter",
    "philip", "phoebe", "rhea", "selena", "sophia", "sophie", "stephen",
    "theodore", "theresa", "timothy", "vanessa", "veronica", "zoe",
}

_LATIN_NAMES = {
    "adrian", "amara", "amanda", "amber", "antonia", "august", "augustine",
    "aurora", "beatrice", "benedict", "camilla", "carmen", "cecil", "celeste",
    "clara", "clarence", "claudia", "clement", "cordelia", "cornelius",
    "dominic", "emily", "emilia", "felix", "flora", "florence", "francis",
    "gloria", "grace", "ignatius", "julia", "julian", "june", "justin",
    "laurel", "laura", "lilian", "lillian", "lucia", "lucille", "luna",
    "lyric", "magnus", "marcus", "marina", "mark", "martin", "max",
    "maximilian", "miles", "miranda", "natalie", "natasha", "nova", "olivia",
    "patricia", "patrick", "paul", "regina", "rex", "roman", "rosalie",
    "ruby", "sabrina", "serena", "silvia", "stella", "sylvia", "valentine",
    "valeria", "victor", "victoria", "vincent", "violet", "virgil", "vivian",
}

_GERMAN_NAMES = {
    "albert", "alice", "arnold", "bertha", "carl", "charles", "charlotte",
    "conrad", "dietrich", "emma", "ernest", "frederick", "fritz", "gertrude",
    "greta", "hans", "harold", "heidi", "henry", "herman", "hilda", "karl",
    "kurt", "leonard", "lewis", "louis", "louise", "ludwig", "matilda",
    "otto", "ralph", "raymond", "richard", "robert", "roger", "roland",
    "rudolph", "sigrid", "walter", "warren", "william", "wilma",
}

_IRISH_NAMES = {
    "aidan", "brady", "brendan", "brian", "bridget", "casey", "ciara",
    "colleen", "connor", "declan", "deirdre", "donovan", "erin", "finley",
    "finn", "grady", "kelly", "kennedy", "kieran", "liam", "maeve",
    "megan", "murphy", "neil", "niamh", "nolan", "owen", "patrick",
    "quinn", "riley", "rory", "ryan", "sean", "shannon", "sienna",
    "sullivan", "tara",
}

_SCOTTISH_NAMES = {
    "ainsley", "blair", "bruce", "cameron", "campbell", "craig", "douglas",
    "duncan", "gordon", "graham", "hamish", "isla", "kyle", "logan",
    "malcolm", "ross", "scott", "stuart", "wallace",
}

_WELSH_NAMES = {
    "brynn", "carys", "dylan", "gareth", "gavin", "gwendolyn", "lloyd",
    "meredith", "morgan", "reese", "rhys", "seren", "trevor", "wynn",
}

_FRENCH_NAMES = {
    "adele", "aimee", "andre", "annette", "blaise", "blanche", "brice",
    "brigitte", "bruce", "chantal", "colette", "denise", "desiree",
    "dominique", "eloise", "faye", "genevieve", "jacques", "jean",
    "jolie", "leon", "lucien", "madeleine", "marguerite", "monique",
    "nicole", "noel", "odette", "pierre", "renee", "rochelle", "simone",
    "vivienne", "yvette", "yvonne",
}

_SPANISH_NAMES = {
    "alejandro", "alma", "alvaro", "ana", "angelina", "carlos", "catalina",
    "cruz", "diego", "dolores", "elena", "esperanza", "esteban", "fernando",
    "francisco", "guadalupe", "isabella", "jaime", "jose", "juan", "juana",
    "linda", "lola", "lucia", "luis", "luz", "manuel", "maria", "mario",
    "mercedes", "miguel", "paloma", "pedro", "pilar", "rafael", "ramon",
    "roberto", "rosa", "salvador", "santiago", "sofia", "teresa",
}

_ITALIAN_NAMES = {
    "alessandra", "angelica", "angelo", "antonio", "bianca", "bruno",
    "carlo", "dante", "elisa", "emilio", "enrico", "francesca", "franco",
    "gia", "gianna", "gino", "giovanni", "giuliana", "giuseppe", "luca",
    "lucia", "marco", "matteo", "nico", "paolo", "rocco", "romeo",
    "salvatore", "valentina", "vincenzo",
}

_SCANDINAVIAN_NAMES = {
    "anders", "astrid", "axel", "bjorn", "dagny", "erik", "freya",
    "gunnar", "gustav", "helga", "ingrid", "ivar", "karina", "lars",
    "leif", "magnus", "nils", "olaf", "oscar", "sigrid", "soren",
    "sven", "thor", "tove",
}

_SLAVIC_NAMES = {
    "boris", "dmitri", "ivan", "katarina", "mila", "milena", "misha",
    "nadia", "natalya", "nikita", "nina", "olga", "sasha", "tatiana",
    "vera", "vladimir", "yuri",
}

_ARABIC_NAMES = {
    "aaliyah", "ahmad", "aisha", "ali", "amir", "amira", "fatima",
    "hana", "hassan", "hussein", "jamal", "jasmine", "karim", "khalid",
    "layla", "leila", "malik", "mariam", "mohammed", "nadia", "omar",
    "rashid", "salim", "samira", "tariq", "yasmin", "zahra", "zara",
}

_PERSIAN_NAMES = {
    "aria", "cyrus", "darius", "jasper", "roxana", "shirin",
}

_SANSKRIT_NAMES = {
    "ananda", "arjun", "devi", "indira", "kali", "kamala", "karma",
    "krishna", "lakshmi", "maya", "priya", "raj", "ram", "sita", "tara",
    "uma", "veda",
}

_JAPANESE_NAMES = {
    "akira", "emiko", "hana", "haruki", "hiro", "kai", "kenji", "kenzo",
    "maki", "mio", "naomi", "ren", "riku", "sakura", "yuki", "yumi",
}

_AFRICAN_NAMES = {
    "amara", "ashanti", "ayanna", "ebony", "imani", "jabari", "jada",
    "kamari", "kaya", "keisha", "kenya", "kwame", "makena", "malik",
    "nia", "nyala", "sanaa", "shani", "simba", "zuri",
}

_CHINESE_NAMES = {
    "chen", "jin", "li", "mei", "ming", "wei", "xin",
}

_KOREAN_NAMES = {
    "eunji", "hana", "jimin", "minho", "soo", "yuna",
}

_NATIVE_AMERICAN_NAMES = {
    "dakota", "cheyenne", "sequoia", "winona",
}

def _heritage_lookup(name: str) -> str:
    n = name.lower()
    if n in _HEBREW_NAMES: return "Hebrew"
    if n in _GREEK_NAMES: return "Greek"
    if n in _LATIN_NAMES: return "Latin"
    if n in _GERMAN_NAMES: return "German"
    if n in _IRISH_NAMES: return "Irish"
    if n in _SCOTTISH_NAMES: return "Scottish"
    if n in _WELSH_NAMES: return "Welsh"
    if n in _FRENCH_NAMES: return "French"
    if n in _SPANISH_NAMES: return "Spanish"
    if n in _ITALIAN_NAMES: return "Italian"
    if n in _SCANDINAVIAN_NAMES: return "Scandinavian"
    if n in _SLAVIC_NAMES: return "Slavic"
    if n in _ARABIC_NAMES: return "Arabic"
    if n in _PERSIAN_NAMES: return "Persian"
    if n in _SANSKRIT_NAMES: return "Sanskrit"
    if n in _JAPANESE_NAMES: return "Japanese"
    if n in _AFRICAN_NAMES: return "African"
    if n in _CHINESE_NAMES: return "Chinese"
    if n in _KOREAN_NAMES: return "Korean"
    if n in _NATIVE_AMERICAN_NAMES: return "Native American"
    return _heritage_from_pattern(n)


def _heritage_from_pattern(name: str) -> str:
    """Guess heritage from common name endings and patterns."""
    n = name.lower()
    if n.endswith(("owski", "inski", "ska", "ski")): return "Slavic"
    if n.endswith(("sson", "sen", "dottir")): return "Scandinavian"
    if n.endswith(("ello", "ella", "etta", "etto", "ini", "ino")): return "Italian"
    if n.endswith(("ique", "ette", "elle", "aine")): return "French"
    if n.endswith(("ito", "ita", "cion", "ez")): return "Spanish"
    if n.endswith(("iko", "uki", "iko", "uro", "aru")): return "Japanese"
    if n.endswith(("ullah", "din", "eem")): return "Arabic"
    if n.endswith(("esh", "ish", "inder", "deep", "preet")): return "Sanskrit"
    if n.endswith(("agh", "ough", "een", "leen")): return "Irish"
    # Default to English for Western names
    return "English"


# ---------------------------------------------------------------------------
# Origin type classification
# ---------------------------------------------------------------------------

_BIBLICAL = {
    "aaron", "abel", "abigail", "abraham", "adam", "amos", "anna",
    "bartholomew", "benjamin", "caleb", "daniel", "david", "deborah",
    "delilah", "eli", "elijah", "elizabeth", "esther", "eve", "ezekiel",
    "ezra", "gabriel", "gideon", "hannah", "isaac", "isaiah", "israel",
    "jacob", "james", "jeremiah", "jesse", "joel", "john", "jonah",
    "jonathan", "joseph", "joshua", "josiah", "judith", "leah", "levi",
    "luke", "malachi", "mark", "martha", "mary", "matthew", "micah",
    "michael", "miriam", "mordecai", "moses", "naomi", "nathan", "noah",
    "paul", "peter", "rachel", "rebecca", "ruth", "samson", "samuel",
    "sarah", "seth", "simeon", "simon", "solomon", "stephen", "susanna",
    "tabitha", "thomas", "timothy", "tobias", "uriah", "zachariah", "zachary",
}

_MYTHOLOGICAL = {
    "achilles", "adonis", "aphrodite", "apollo", "ares", "ariadne", "artemis",
    "athena", "atlas", "aurora", "cassandra", "clio", "daphne", "diana",
    "echo", "freya", "hector", "helen", "hercules", "hermes", "iris",
    "isis", "jason", "juno", "mars", "minerva", "neptune", "odin",
    "orion", "penelope", "persephone", "phoenix", "rhea", "selena",
    "thor", "titan", "troy", "venus", "zeus",
}

_LITERARY = {
    "ariel", "cordelia", "desdemona", "hamlet", "hermione", "juliet",
    "miranda", "ophelia", "romeo", "rosalind", "sebastian", "viola",
    "atticus", "darcy", "heathcliff", "scarlett", "holden",
}

_PRESIDENTIAL = {
    "abraham", "andrew", "barack", "benjamin", "calvin", "carter",
    "chester", "clinton", "donald", "dwight", "franklin", "george",
    "gerald", "grant", "grover", "harrison", "hayes", "herbert",
    "jackson", "jefferson", "jimmy", "kennedy", "lincoln", "madison",
    "monroe", "nixon", "obama", "pierce", "reagan", "roosevelt",
    "theodore", "truman", "tyler", "washington", "wilson", "woodrow",
}

_NATURE = {
    "amber", "ash", "aspen", "aurora", "bay", "birch", "brook", "brooke",
    "cedar", "clay", "cliff", "clover", "coral", "crystal", "daisy",
    "dawn", "delta", "fern", "flint", "flora", "forest", "gale", "glen",
    "hazel", "heath", "holly", "iris", "ivy", "jade", "jasmine", "juniper",
    "lake", "laurel", "lily", "magnolia", "maple", "meadow", "ocean",
    "olive", "pearl", "petal", "poppy", "rain", "river", "robin", "rose",
    "rowan", "ruby", "sage", "savanna", "sierra", "sky", "skylar", "star",
    "stella", "stone", "storm", "summer", "sunny", "terra", "violet",
    "willow", "winter", "wren",
}

_VIRTUE = {
    "charity", "constance", "faith", "felicity", "grace", "harmony",
    "honor", "hope", "joy", "justice", "mercy", "patience", "prudence",
    "serenity", "temperance", "trinity", "truth", "valor", "verity",
}

_PLACE = {
    "asia", "austin", "berlin", "bethany", "bethlehem", "boston", "brooklyn",
    "cairo", "carolina", "charlotte", "chelsea", "cheyenne", "china",
    "dakota", "dallas", "denver", "dublin", "florence", "georgia",
    "guadalupe", "havana", "helena", "houston", "india", "israel",
    "jordan", "kenya", "london", "madison", "memphis", "milan", "montana",
    "paris", "phoenix", "regina", "rio", "rome", "santiago", "savannah",
    "sedona", "sienna", "sydney", "troy", "valencia", "venice", "verona",
    "vienna", "virginia",
}

_OCCUPATIONAL = {
    "archer", "bailey", "carter", "chase", "clark", "cole", "colton",
    "cooper", "deacon", "dean", "fletcher", "forrest", "foster", "hunter",
    "judge", "knight", "marshall", "mason", "page", "palmer", "parker",
    "porter", "ranger", "sawyer", "scout", "shepherd", "spencer", "tanner",
    "taylor", "tucker", "tyler", "walker", "ward", "weaver",
}

_CELEBRITY = {
    "beyonce", "cher", "elvis", "harlow", "hendrix", "jagger", "lennon",
    "madonna", "marley", "monroe", "oprah", "presley", "rihanna",
}


def _origin_type(name: str) -> str:
    n = name.lower()
    if n in _BIBLICAL: return "Biblical"
    if n in _MYTHOLOGICAL: return "Mythological"
    if n in _LITERARY: return "Literary"
    if n in _PRESIDENTIAL: return "Presidential"
    if n in _NATURE: return "Nature"
    if n in _VIRTUE: return "Virtue"
    if n in _PLACE: return "Place"
    if n in _OCCUPATIONAL: return "Occupational"
    if n in _CELEBRITY: return "Celebrity"
    return "Traditional"


# ---------------------------------------------------------------------------
# Sound classification — based on phonetic patterns
# ---------------------------------------------------------------------------

_SOFT_ENDINGS = ("a", "ah", "ia", "iah", "ie", "na", "la", "ra", "ya", "ee")
_SHARP_ENDINGS = ("k", "ck", "x", "t", "d", "p", "g")
_MELODIC_ENDINGS = ("ella", "iana", "aria", "elia", "ina", "ana", "ola")
_PUNCHY_PATTERNS = re.compile(r"^(b|d|g|j|k|p|t)[aeiou]", re.I)
_FLOWING_ENDINGS = ("lyn", "line", "lene", "leen", "lynn", "ine", "een")

def _classify_sound(name: str) -> str:
    n = name.lower()
    if any(n.endswith(e) for e in _MELODIC_ENDINGS): return "Melodic"
    if any(n.endswith(e) for e in _FLOWING_ENDINGS): return "Flowing"
    if any(n.endswith(e) for e in _SOFT_ENDINGS) and len(n) > 3: return "Soft"
    if any(n.endswith(e) for e in _SHARP_ENDINGS): return "Sharp"
    if _PUNCHY_PATTERNS.match(n) and len(n) <= 5: return "Punchy"
    # Count vowel-consonant flow
    vowels = sum(1 for c in n if c in "aeiou")
    ratio = vowels / max(len(n), 1)
    if ratio > 0.5: return "Melodic"
    if ratio < 0.3: return "Crisp"
    return "Crisp"


# ---------------------------------------------------------------------------
# Class (socioeconomic perception)
# ---------------------------------------------------------------------------

_UPPER_NAMES = {
    "adelaide", "alexandra", "arabella", "archibald", "augustine", "beatrice",
    "benedict", "bianca", "caroline", "cecilia", "charlotte", "constance",
    "cordelia", "cornelius", "daphne", "edmund", "eleanor", "elizabeth",
    "emilia", "evangeline", "florence", "frederick", "genevieve", "harriet",
    "henrietta", "isadora", "josephine", "katherine", "leopold", "louisa",
    "madeleine", "margaret", "montgomery", "octavia", "penelope", "persephone",
    "philippa", "reginald", "rosalind", "sebastian", "theodore", "valentine",
    "victoria", "vivienne", "william", "winifred",
}

_UPPER_MIDDLE_NAMES = {
    "abigail", "alexander", "alice", "amelia", "andrew", "benjamin",
    "campbell", "catherine", "charles", "claire", "daniel", "edward",
    "eloise", "emma", "ethan", "grant", "henry", "isabelle", "james",
    "jane", "juliet", "lucas", "maxwell", "natalie", "nathaniel",
    "nicholas", "oliver", "olivia", "owen", "peter", "quinn", "rachel",
    "samuel", "sarah", "simon", "sophia", "thomas", "violet", "walker",
}

_WORKING_NAMES = {
    "billy", "bobby", "brandy", "bubba", "bud", "buddy", "butch",
    "cletus", "cody", "crystal", "darryl", "darlene", "dwayne", "earl",
    "floyd", "garth", "grady", "hank", "harley", "jethro", "jimmy",
    "johnny", "jolene", "junior", "larry", "leroy", "lonnie", "loretta",
    "merle", "myrtle", "norma", "otis", "patsy", "peggy", "reba",
    "ricky", "ronnie", "rusty", "tammy", "toby", "travis", "troy",
    "vernon", "virgil", "wanda", "wayne", "willie",
}


def _classify_class(name: str) -> str:
    n = name.lower()
    if n in _UPPER_NAMES: return "Upper"
    if n in _UPPER_MIDDLE_NAMES: return "Upper-middle"
    if n in _WORKING_NAMES: return "Working"
    # Heuristics for remaining names
    if len(n) >= 9: return "Upper-middle"
    if n.endswith(("lyn", "lynn", "leigh")): return "Middle"
    if n.endswith(("ington", "sworth")): return "Upper"
    return "Middle"


# ---------------------------------------------------------------------------
# Sentiment tags — based on name characteristics
# ---------------------------------------------------------------------------

def _sentiment_tags(name: str) -> list[str]:
    n = name.lower()
    tags = []

    # Gender associations
    soft_feminine = n.endswith(("a", "ia", "ie", "ina", "elle", "ette", "lyn"))
    hard_masculine = n.endswith(("on", "er", "or", "us", "ard", "ck"))

    if soft_feminine:
        tags.append("Feminine")
    if hard_masculine:
        tags.append("Masculine")

    # Classic vs Modern
    if n in _BIBLICAL or n in _GREEK_NAMES or n in _LATIN_NAMES:
        tags.append("Classic")
    elif len(n) <= 3 or n.endswith(("den", "lyn", "leigh", "lee")):
        tags.append("Modern")

    # Strong vs Gentle/Soft
    if n.endswith(("ax", "ex", "ox", "ck", "nk", "rk")) or n in {"rex", "max", "knox", "jax"}:
        tags.append("Strong")
        tags.append("Bold")
    elif soft_feminine or n.endswith(("ose", "isa", "ila", "ela")):
        tags.append("Gentle")
        tags.append("Soft")

    # Elegant vs Playful
    if len(n) >= 8 and n.endswith(("ine", "ina", "ella", "ette", "iana")):
        tags.append("Elegant")
    elif len(n) <= 4:
        tags.append("Playful")

    # Nature names → Earthy
    if n in _NATURE:
        tags.append("Earthy")

    # Mythological → Mythical
    if n in _MYTHOLOGICAL:
        tags.append("Mythical")

    # Warm vs Cool
    if n.startswith(("w", "m")) and n.endswith(("a", "ah", "y")):
        tags.append("Warm")
    elif n.startswith(("c", "k", "z")) and n.endswith(("e", "o", "x")):
        tags.append("Cool")

    # Whimsical — short + unusual patterns
    if n in _NATURE and n.endswith(("er", "ow", "y")):
        tags.append("Whimsical")

    # Serious — long, traditional
    if len(n) >= 8 and n in (_BIBLICAL | _PRESIDENTIAL):
        tags.append("Serious")

    # Urban
    if n in _CELEBRITY or n.endswith(("ique", "ell", "elle")):
        tags.append("Urban")

    # Ethereal
    if n.endswith(("iel", "ael", "ael", "iel")) or n in {"aurora", "celeste", "luna", "nova", "seraphina"}:
        tags.append("Ethereal")

    # Deduplicate and limit to 2-4
    seen = set()
    unique = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    if len(unique) < 2:
        # Ensure minimum 2 tags
        defaults = ["Classic", "Warm", "Gentle", "Serious"]
        for d in defaults:
            if d not in seen:
                unique.append(d)
                seen.add(d)
            if len(unique) >= 2:
                break
    return unique[:4]


# ---------------------------------------------------------------------------
# Main enrichment function
# ---------------------------------------------------------------------------

def enrich_name(name: str) -> dict:
    """Classify a single name across all enrichment dimensions."""
    return {
        "name": name,
        "class": _classify_class(name),
        "heritage": _heritage_lookup(name),
        "origin_type": _origin_type(name),
        "sentiment_tags": _sentiment_tags(name),
        "sound": _classify_sound(name),
    }


def run_rule_enrichment(names: list[str]) -> None:
    """Enrich all names using rules and save to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results = [enrich_name(n) for n in names]
    with open(ENRICHMENTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Rule-based enrichment complete. {len(results)} names saved to {ENRICHMENTS_PATH}")


# ---------------------------------------------------------------------------
# CLI entry point: python -m name_assist.enrich_rules
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from name_assist.data import load_df

    print("Loading SSA data...")
    df = load_df()

    top = (
        df.groupby("name")["count"]
        .sum()
        .sort_values(ascending=False)
        .head(5000)
        .index.tolist()
    )
    print(f"Top {len(top)} names selected for enrichment.")
    run_rule_enrichment(top)
