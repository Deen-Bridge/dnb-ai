"""Soroban authorization entry safety explainer for the Deen Bridge AI service.

Soroban smart contracts on the Stellar network rely on explicit, fine-grained
authorization entries (SorobanAuthorizationEntry). When a user interacts with
a smart contract, their non-custodial wallet signs an authorization entry that
specifies the authorizing identity, replay protection (nonce), signature
expiration ledger, and the exact invocation tree (contract function, args, and
nested sub-invocations).

This module provides an explicit, network-aware, and non-custodial safety
explainer for Soroban authorization entries:
1. Deeply unpacks credentials (source account, address credentials, nonces,
   and expiration ledgers).
2. Recursively inspects the invocation tree, uncovering nested sub-invocations
   that could perform secondary token transfers or allowance grants.
3. Classifies operations by risk level (safe, caution, warning, critical).
4. Strictly detects, redacts, and refuses any secret keys (S...) or seed phrases,
   ensuring zero exposure of sensitive private data.
5. Provides plain-language explanations with actionable safety recommendations
   and official stellar.expert explorer links.
"""

import logging
import os
import re
from enum import Enum
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from stellar_sdk import Address, StrKey, scval, xdr

logger = logging.getLogger(__name__)

router = APIRouter(tags=["stellar-soroban"])

# Network configuration
STELLAR_NETWORK = os.getenv("STELLAR_NETWORK", "testnet")

# Regular expressions for Stellar and Soroban identifiers
PUBLIC_KEY_PATTERN = re.compile(r"\bG[A-Z2-7]{55}\b")
CONTRACT_KEY_PATTERN = re.compile(r"\bC[A-Z2-7]{55}\b")
SECRET_KEY_PATTERN = re.compile(r"\bS[A-Z2-7]{55}\b")

# Common mnemonic / seed phrase indicators (12 or 24 words, or explicit labeling)
SEED_PHRASE_INDICATORS = (
    "seed phrase",
    "recovery phrase",
    "mnemonic phrase",
    "private key",
    "secret seed",
)
_BIP39_SAMPLE_WORDS = {
    "abandon",
    "ability",
    "able",
    "about",
    "above",
    "absent",
    "absorb",
    "abstract",
    "absurd",
    "abuse",
    "access",
    "accident",
    "account",
    "accuse",
    "achieve",
    "acid",
    "acoustic",
    "acquire",
    "across",
    "act",
    "action",
    "actor",
    "actress",
    "actual",
    "adapt",
    "add",
    "addict",
    "address",
    "adjust",
    "admit",
    "adult",
    "advance",
    "advice",
    "aerobic",
    "affair",
    "afford",
    "afraid",
    "again",
    "age",
    "agent",
    "agree",
    "ahead",
    "aim",
    "air",
    "airport",
    "aisle",
    "alarm",
    "album",
    "alcohol",
    "alert",
    "alien",
    "all",
    "alley",
    "allow",
    "almost",
    "alone",
    "alpha",
    "already",
    "also",
    "alter",
    "always",
    "amateur",
    "amazing",
    "among",
    "amount",
    "amused",
    "analyst",
    "anchor",
    "ancient",
    "anger",
    "angle",
    "angry",
    "animal",
    "ankle",
    "announce",
    "annual",
    "another",
    "answer",
    "antenna",
    "antique",
    "anxiety",
    "any",
    "apart",
    "apology",
    "appear",
    "apple",
    "approve",
    "april",
    "arch",
    "arctic",
    "area",
    "arena",
    "argue",
    "arm",
    "armed",
    "armor",
    "army",
    "around",
    "arrange",
    "arrest",
    "arrive",
    "arrow",
    "art",
    "artefact",
    "artist",
    "artwork",
    "ask",
    "aspect",
    "assault",
    "asset",
    "assist",
    "assume",
    "asthma",
    "athlete",
    "atom",
    "attack",
    "attend",
    "attitude",
    "attract",
    "auction",
    "audit",
    "august",
    "aunt",
    "author",
    "auto",
    "autumn",
    "average",
    "avocado",
    "avoid",
    "awake",
    "aware",
    "away",
    "awesome",
    "awful",
    "awkward",
    "axis",
    "baby",
    "bachelor",
    "bacon",
    "badge",
    "bag",
    "balance",
    "balcony",
    "ball",
    "bamboo",
    "banana",
    "banner",
    "bar",
    "barely",
    "bargain",
    "barrel",
    "base",
    "basic",
    "basket",
    "battle",
    "beach",
    "bean",
    "beauty",
    "because",
    "become",
    "beef",
    "before",
    "begin",
    "behave",
    "behind",
    "believe",
    "below",
    "belt",
    "bench",
    "benefit",
    "best",
    "betray",
    "better",
    "between",
    "beyond",
    "bicycle",
    "bid",
    "bike",
    "bind",
    "biology",
    "bird",
    "birth",
    "bitter",
    "black",
    "blade",
    "blame",
    "blanket",
    "blast",
    "bleak",
    "bless",
    "blind",
    "blood",
    "blossom",
    "blouse",
    "blue",
    "blur",
    "blush",
    "board",
    "boat",
    "body",
    "boil",
    "bomb",
    "bone",
    "bonus",
    "book",
    "boost",
    "border",
    "boring",
    "borrow",
    "boss",
    "bottom",
    "bounce",
    "box",
    "boy",
    "bracket",
    "brain",
    "brand",
    "brass",
    "brave",
    "bread",
    "breeze",
    "brick",
    "bridge",
    "brief",
    "bright",
    "bring",
    "brisk",
    "broccoli",
    "broken",
    "bronze",
    "broom",
    "brother",
    "brown",
    "brush",
    "bubble",
    "buddy",
    "budget",
    "buffalo",
    "build",
    "bulb",
    "bulk",
    "bullet",
    "bundle",
    "bunker",
    "burden",
    "burger",
    "burst",
    "bus",
    "business",
    "busy",
    "butter",
    "buyer",
    "buzz",
    "cabbage",
    "cabin",
    "cable",
    "cactus",
    "cage",
    "cake",
    "call",
    "calm",
    "camera",
    "camp",
    "can",
    "canal",
    "cancel",
    "candy",
    "cannon",
    "canoe",
    "canvas",
    "canyon",
    "capable",
    "capital",
    "captain",
    "car",
    "carbon",
    "card",
    "cargo",
    "carpet",
    "carry",
    "cart",
    "case",
    "cash",
    "casino",
    "castle",
    "casual",
    "cat",
    "catalog",
    "catch",
    "category",
    "cattle",
    "caught",
    "cause",
    "caution",
    "cave",
    "ceiling",
    "celery",
    "cement",
    "census",
    "century",
    "cereal",
    "certain",
    "chair",
    "chalk",
    "champion",
    "change",
    "chaos",
    "chapter",
    "charge",
    "chase",
    "chat",
    "cheap",
    "check",
    "cheese",
    "chef",
    "cherry",
    "chest",
    "chicken",
    "chief",
    "child",
    "chimney",
    "choice",
    "choose",
    "chronic",
    "chuckle",
    "chunk",
    "churn",
    "cigar",
    "cinnamon",
    "circle",
    "citizen",
    "city",
    "civil",
    "claim",
    "clap",
    "clarify",
    "claw",
    "clay",
    "clean",
    "clerk",
    "clever",
    "click",
    "client",
    "cliff",
    "climb",
    "clinic",
    "clip",
    "clock",
    "clog",
    "close",
    "cloth",
    "cloud",
    "clown",
    "club",
    "clump",
    "cluster",
    "clutch",
    "coach",
    "coast",
    "coconut",
    "coin",
    "collect",
    "color",
    "column",
    "combine",
    "come",
    "comfort",
    "comic",
    "common",
    "company",
    "concert",
    "conduct",
    "confirm",
    "congress",
    "connect",
    "consider",
    "control",
    "convince",
    "cook",
    "cool",
    "copper",
    "copy",
    "coral",
    "core",
    "corn",
    "correct",
    "cost",
    "cotton",
    "couch",
    "country",
    "couple",
    "course",
    "cousin",
    "cover",
    "coyote",
    "crack",
    "cradle",
    "craft",
    "cram",
    "crane",
    "crash",
    "crater",
    "crawl",
    "crazy",
    "cream",
    "credit",
    "creek",
    "crew",
    "cricket",
    "crime",
    "crisp",
    "critic",
    "crop",
    "cross",
    "crouch",
    "crowd",
    "crucial",
    "cruel",
    "cruise",
    "crumble",
    "crunch",
    "crush",
    "cry",
    "crystal",
    "cube",
    "culture",
    "cup",
    "cupboard",
    "curious",
    "current",
    "curtain",
    "curve",
    "cushion",
    "custom",
    "cute",
    "cycle",
    "dad",
    "damage",
    "damp",
    "dance",
    "danger",
    "daring",
    "dash",
    "daughter",
    "dawn",
    "day",
    "deal",
    "debate",
    "debris",
    "decade",
    "december",
    "decide",
    "decline",
    "decorate",
    "decrease",
    "deer",
    "defense",
    "define",
    "defy",
    "degree",
    "delay",
    "deliver",
    "demand",
    "demise",
    "denial",
    "dentist",
    "deny",
    "depart",
    "depend",
    "deposit",
    "depth",
    "deputy",
    "derive",
    "describe",
    "desert",
    "design",
    "desk",
    "despair",
    "destroy",
    "detail",
    "detect",
    "develop",
    "device",
    "devote",
    "diagram",
    "dial",
    "diamond",
    "diary",
    "dice",
    "diesel",
    "diet",
    "differ",
    "digital",
    "dignity",
    "dilemma",
    "dinner",
    "dinosaur",
    "direct",
    "dirt",
    "disagree",
    "discover",
    "disease",
    "dish",
    "dismiss",
    "disorder",
    "display",
    "distance",
    "divert",
    "divide",
    "divorce",
    "dizzy",
    "doctor",
    "document",
    "dog",
    "doll",
    "dolphin",
    "domain",
    "donate",
    "donkey",
    "donor",
    "door",
    "dose",
    "double",
    "dove",
    "draft",
    "dragon",
    "drama",
    "drastic",
    "draw",
    "dream",
    "dress",
    "drift",
    "drill",
    "drink",
    "drip",
    "drive",
    "drop",
    "drum",
    "dry",
    "duck",
    "dumb",
    "dune",
    "during",
    "dust",
    "dutch",
    "duty",
    "dwarf",
    "dynamic",
    "eager",
    "eagle",
    "early",
    "earn",
    "earth",
    "easily",
    "east",
    "easy",
    "echo",
    "ecology",
    "economy",
    "edge",
    "edit",
    "educate",
    "effort",
    "egg",
    "eight",
    "either",
    "elbow",
    "elder",
    "electric",
    "elegant",
    "element",
    "elephant",
    "elevator",
    "elite",
    "else",
    "embark",
    "embody",
    "embrace",
    "emerge",
    "emotion",
    "employ",
    "empower",
    "empty",
    "enable",
    "enact",
    "end",
    "endless",
    "endorse",
    "enemy",
    "energy",
    "enforce",
    "engage",
    "engine",
    "enhance",
    "enjoy",
    "enlist",
    "enough",
    "enrich",
    "enroll",
    "ensure",
    "enter",
    "entire",
    "entry",
    "envelope",
    "episode",
    "equal",
    "equip",
    "era",
    "erase",
    "erode",
    "erosion",
    "error",
    "erupt",
    "escape",
    "essay",
    "essence",
    "estate",
    "eternal",
    "ethics",
    "evidence",
    "evil",
    "evoke",
    "evolve",
    "exact",
    "example",
    "excess",
    "exchange",
    "excite",
    "exclude",
    "excuse",
    "execute",
    "exercise",
    "exhaust",
    "exhibit",
    "exile",
    "exist",
    "exit",
    "exotic",
    "expand",
    "expect",
    "expire",
    "explain",
    "expose",
    "express",
    "extend",
    "extra",
    "eye",
    "eyebrow",
    "fabric",
    "face",
    "faculty",
    "fade",
    "faint",
    "faith",
    "fall",
    "false",
    "fame",
    "family",
    "famous",
    "fan",
    "fancy",
    "fantasy",
    "farm",
    "fashion",
    "fat",
    "fatal",
    "father",
    "fatigue",
    "fault",
    "favorite",
    "feature",
    "february",
    "federal",
    "fee",
    "feed",
    "feel",
    "female",
    "fence",
    "festival",
    "fetch",
    "fever",
    "few",
    "fiber",
    "fiction",
    "field",
    "figure",
    "file",
    "film",
    "filter",
    "final",
    "find",
    "fine",
    "finger",
    "finish",
    "fire",
    "firm",
    "first",
    "fiscal",
    "fish",
    "fit",
    "fitness",
    "fix",
    "flag",
    "flame",
    "flash",
    "flat",
    "flavor",
    "flee",
    "flight",
    "flip",
    "float",
    "flock",
    "floor",
    "flower",
    "fluid",
    "flush",
    "fly",
    "foam",
    "focus",
    "fog",
    "foil",
    "fold",
    "follow",
    "food",
    "foot",
    "force",
    "forest",
    "forget",
    "fork",
    "fortune",
    "forum",
    "forward",
    "fossil",
    "foster",
    "found",
    "fox",
    "fragile",
    "frame",
    "frequent",
    "fresh",
    "friend",
    "fringe",
    "frog",
    "front",
    "frost",
    "frown",
    "frozen",
    "fruit",
    "fuel",
    "fun",
    "funny",
    "furnace",
    "fury",
    "future",
    "gadget",
    "gain",
    "galaxy",
    "gallery",
    "game",
    "gap",
    "garage",
    "garbage",
    "garden",
    "garlic",
    "garment",
    "gas",
    "gasp",
    "gate",
    "gather",
    "gauge",
    "gaze",
    "general",
    "genius",
    "genre",
    "gentle",
    "genuine",
    "gesture",
    "ghost",
    "giant",
    "gift",
    "giggle",
    "ginger",
    "giraffe",
    "girl",
    "give",
    "glad",
    "glance",
    "glare",
    "glass",
    "glide",
    "glimpse",
    "globe",
    "gloom",
    "glory",
    "glove",
    "glow",
    "glue",
    "goat",
    "goddess",
    "gold",
    "good",
    "goose",
    "gorilla",
    "gospel",
    "gossip",
    "govern",
    "gown",
    "grab",
    "grace",
    "grain",
    "grant",
    "grape",
    "grass",
    "gravity",
    "great",
    "green",
    "grid",
    "grief",
    "grit",
    "grocery",
    "group",
    "grow",
    "grunt",
    "guard",
    "guess",
    "guide",
    "guilt",
    "guitar",
    "gun",
    "gym",
    "habit",
    "hair",
    "half",
    "hammer",
    "hamster",
    "hand",
    "happy",
    "harbor",
    "hard",
    "harsh",
    "harvest",
    "hat",
    "have",
    "hawk",
    "hazard",
    "head",
    "health",
    "heart",
    "heavy",
    "hedgehog",
    "height",
    "hello",
    "helmet",
    "help",
    "hen",
    "hero",
    "hidden",
    "high",
    "hill",
    "hint",
    "hip",
    "hire",
    "history",
    "hobby",
    "hockey",
    "hold",
    "hole",
    "holiday",
    "hollow",
    "home",
    "honey",
    "hood",
    "hope",
    "horn",
    "horror",
    "horse",
    "hospital",
    "host",
    "hotel",
    "hour",
    "hover",
    "hub",
    "huge",
    "human",
    "humble",
    "humor",
    "hundred",
    "hungry",
    "hunt",
    "hurdle",
    "hurry",
    "hurt",
    "husband",
    "hybrid",
    "ice",
    "icon",
    "idea",
    "identify",
    "idle",
    "ignore",
    "ill",
    "illegal",
    "illness",
    "image",
    "imitate",
    "immense",
    "immune",
    "impact",
    "impose",
    "improve",
    "impulse",
    "inch",
    "include",
    "income",
    "increase",
    "index",
    "indicate",
    "indoor",
    "industry",
    "infant",
    "inflict",
    "inform",
    "initial",
    "inject",
    "injury",
    "inmate",
    "inner",
    "innocent",
    "input",
    "inquiry",
    "insane",
    "insect",
    "inside",
    "inspire",
    "install",
    "intact",
    "interest",
    "into",
    "invest",
    "invite",
    "involve",
    "iron",
    "island",
    "isolate",
    "issue",
    "item",
    "ivory",
    "jacket",
    "jaguar",
    "jar",
    "jazz",
    "jealous",
    "jeans",
    "jelly",
    "jewel",
    "job",
    "join",
    "joke",
    "journey",
    "joy",
    "judge",
    "juice",
    "jump",
    "jungle",
    "junior",
    "junk",
    "just",
    "kangaroo",
    "keen",
    "keep",
    "ketchup",
    "key",
    "kick",
    "kid",
    "kidney",
    "kind",
    "kingdom",
    "kiss",
    "kit",
    "kitchen",
    "kite",
    "kitten",
    "kiwi",
    "knee",
    "knife",
    "knock",
    "know",
    "lab",
    "label",
    "labor",
    "ladder",
    "lady",
    "lake",
    "lamp",
    "language",
    "laptop",
    "large",
    "later",
    "latin",
    "laugh",
    "laundry",
    "lava",
    "law",
    "lawn",
    "lawsuit",
    "layer",
    "lazy",
    "leader",
    "leaf",
    "learn",
    "leave",
    "lecture",
    "left",
    "leg",
    "legal",
    "legend",
    "leisure",
    "lemon",
    "lend",
    "length",
    "lens",
    "leopard",
    "lesson",
    "letter",
    "level",
    "liar",
    "liberty",
    "library",
    "license",
    "life",
    "lift",
    "light",
    "like",
    "limb",
    "limit",
    "link",
    "lion",
    "liquid",
    "list",
    "little",
    "live",
    "lizard",
    "load",
    "loan",
    "lobster",
    "local",
    "lock",
    "logic",
    "lonely",
    "long",
    "loop",
    "lottery",
    "loud",
    "lounge",
    "love",
    "loyal",
    "lucky",
    "luggage",
    "lumber",
    "lunar",
    "lunch",
    "luxury",
    "lyrics",
    "machine",
    "mad",
    "magic",
    "magnet",
    "maid",
    "mail",
    "main",
    "major",
    "make",
    "mammal",
    "man",
    "manage",
    "mandate",
    "mango",
    "mansion",
    "manual",
    "maple",
    "marble",
    "march",
    "margin",
    "marine",
    "market",
    "marriage",
    "mask",
    "mass",
    "master",
    "match",
    "material",
    "math",
    "matrix",
    "matter",
    "maximum",
    "maze",
    "meadow",
    "mean",
    "measure",
    "meat",
    "mechanic",
    "medal",
    "media",
    "melody",
    "melt",
    "member",
    "memory",
    "mention",
    "menu",
    "mercy",
    "merge",
    "merit",
    "merry",
    "mesh",
    "message",
    "metal",
    "method",
    "middle",
    "midnight",
    "milk",
    "million",
    "mimic",
    "mind",
    "minimum",
    "minor",
    "minute",
    "miracle",
    "mirror",
    "misery",
    "miss",
    "mistake",
    "mix",
    "mixed",
    "mixture",
    "mobile",
    "model",
    "modify",
    "mom",
    "moment",
    "monitor",
    "monkey",
    "monster",
    "month",
    "moon",
    "moral",
    "more",
    "morning",
    "mosquito",
    "mother",
    "motion",
    "motor",
    "mountain",
    "mouse",
    "move",
    "movie",
    "much",
    "muffin",
    "mule",
    "multiply",
    "muscle",
    "museum",
    "mushroom",
    "music",
    "must",
    "mutual",
    "myself",
    "mystery",
    "myth",
    "naive",
    "name",
    "napkin",
    "narrow",
    "nasty",
    "nation",
    "nature",
    "near",
    "neck",
    "need",
    "negative",
    "neglect",
    "neither",
    "nephew",
    "nerve",
    "nest",
    "net",
    "network",
    "neutral",
    "never",
    "news",
    "next",
    "nice",
    "night",
    "noble",
    "noise",
    "nominee",
    "noodle",
    "normal",
    "north",
    "nose",
    "notable",
    "note",
    "nothing",
    "notice",
    "novel",
    "now",
    "nuclear",
    "number",
    "nurse",
    "nut",
    "oak",
    "obey",
    "object",
    "oblige",
    "obscure",
    "observe",
    "obtain",
    "obvious",
    "occur",
    "ocean",
    "october",
    "odor",
    "off",
    "offer",
    "office",
    "often",
    "oil",
    "okay",
    "old",
    "olive",
    "olympic",
    "omit",
    "once",
    "one",
    "onion",
    "online",
    "only",
    "open",
    "opera",
    "opinion",
    "oppose",
    "option",
    "orange",
    "orbit",
    "orchard",
    "order",
    "ordinary",
    "organ",
    "orient",
    "original",
    "orphan",
    "ostrich",
    "other",
    "outdoor",
    "outer",
    "output",
    "outside",
    "oval",
    "oven",
    "over",
    "own",
    "owner",
    "oxygen",
    "oyster",
    "ozone",
    "pact",
    "paddle",
    "page",
    "pair",
    "palace",
    "palm",
    "panda",
    "panel",
    "panic",
    "panther",
    "paper",
    "parade",
    "parent",
    "park",
    "parrot",
    "party",
    "pass",
    "patch",
    "path",
    "patient",
    "patrol",
    "pattern",
    "pause",
    "pave",
    "payment",
    "peace",
    "peanut",
    "pear",
    "peasant",
    "pelican",
    "pen",
    "penalty",
    "pencil",
    "people",
    "pepper",
    "perfect",
    "permit",
    "person",
    "pet",
    "phone",
    "photo",
    "phrase",
    "physical",
    "piano",
    "picnic",
    "picture",
    "piece",
    "pig",
    "pigeon",
    "pill",
    "pilot",
    "pink",
    "pioneer",
    "pipe",
    "pistol",
    "pitch",
    "pizza",
    "place",
    "planet",
    "plastic",
    "plate",
    "play",
    "please",
    "pledge",
    "pluck",
    "plug",
    "plunge",
    "poem",
    "poet",
    "point",
    "polar",
    "pole",
    "police",
    "pond",
    "pony",
    "pool",
    "popular",
    "portion",
    "position",
    "possible",
    "post",
    "potato",
    "pottery",
    "poverty",
    "powder",
    "power",
    "practice",
    "praise",
    "predict",
    "prefer",
    "prepare",
    "present",
    "pretty",
    "prevent",
    "price",
    "pride",
    "primary",
    "print",
    "priority",
    "prison",
    "private",
    "prize",
    "problem",
    "process",
    "produce",
    "profit",
    "program",
    "project",
    "promote",
    "proof",
    "property",
    "prosper",
    "protect",
    "proud",
    "provide",
    "public",
    "pudding",
    "pull",
    "pulp",
    "pulse",
    "pumpkin",
    "punch",
    "pupil",
    "puppy",
    "purchase",
    "purity",
    "purpose",
    "purse",
    "push",
    "put",
    "puzzle",
    "pyramid",
    "quality",
    "quantum",
    "quarter",
    "question",
    "quick",
    "quit",
    "quiz",
    "quote",
    "rabbit",
    "raccoon",
    "race",
    "rack",
    "radar",
    "radio",
    "rail",
    "rain",
    "raise",
    "rally",
    "ramp",
    "ranch",
    "random",
    "range",
    "rapid",
    "rare",
    "rate",
    "rather",
    "raven",
    "raw",
    "razor",
    "ready",
    "real",
    "reason",
    "rebel",
    "rebuild",
    "recall",
    "receive",
    "recipe",
    "record",
    "recycle",
    "reduce",
    "reflect",
    "reform",
    "refuse",
    "region",
    "regret",
    "regular",
    "reject",
    "relax",
    "release",
    "relief",
    "rely",
    "remain",
    "remember",
    "remind",
    "remove",
    "render",
    "renew",
    "rent",
    "reopen",
    "repair",
    "repeat",
    "replace",
    "report",
    "require",
    "rescue",
    "resemble",
    "resist",
    "resource",
    "response",
    "result",
    "retire",
    "retreat",
    "return",
    "reunion",
    "reveal",
    "review",
    "reward",
    "rhythm",
    "rib",
    "ribbon",
    "rice",
    "rich",
    "ride",
    "ridge",
    "rifle",
    "right",
    "rigid",
    "ring",
    "riot",
    "ripple",
    "risk",
    "ritual",
    "rival",
    "river",
    "road",
    "roast",
    "robot",
    "robust",
    "rocket",
    "romance",
    "roof",
    "rookie",
    "room",
    "rose",
    "rotate",
    "rough",
    "round",
    "route",
    "royal",
    "rubber",
    "rude",
    "rug",
    "rule",
    "run",
    "runway",
    "rural",
    "sad",
    "saddle",
    "sadness",
    "safe",
    "sail",
    "salad",
    "salmon",
    "salon",
    "salt",
    "salute",
    "same",
    "sample",
    "sand",
    "satisfy",
    "satoshi",
    "sauce",
    "sausage",
    "save",
    "say",
    "scale",
    "scan",
    "scare",
    "scatter",
    "scene",
    "scheme",
    "school",
    "science",
    "scissors",
    "scorpion",
    "scout",
    "scrap",
    "screen",
    "script",
    "scrub",
    "sea",
    "search",
    "season",
    "seat",
    "second",
    "secret",
    "section",
    "security",
    "seduce",
    "see",
    "seed",
    "seek",
    "segment",
    "select",
    "sell",
    "seminar",
    "senior",
    "sense",
    "sentence",
    "series",
    "service",
    "session",
    "settle",
    "setup",
    "seven",
    "shadow",
    "shaft",
    "shallow",
    "share",
    "shed",
    "shell",
    "sheriff",
    "shield",
    "shift",
    "shine",
    "ship",
    "shiver",
    "shock",
    "shoe",
    "shoot",
    "shop",
    "short",
    "shoulder",
    "shove",
    "shrimp",
    "shrug",
    "shuffle",
    "shy",
    "sibling",
    "sick",
    "side",
    "siege",
    "sight",
    "sign",
    "silent",
    "silk",
    "silly",
    "silver",
    "similar",
    "simple",
    "since",
    "sing",
    "siren",
    "sister",
    "situate",
    "six",
    "size",
    "skate",
    "sketch",
    "ski",
    "skill",
    "skin",
    "skirt",
    "skull",
    "slab",
    "slam",
    "sleep",
    "slender",
    "slice",
    "slide",
    "slight",
    "slim",
    "slogan",
    "slot",
    "slow",
    "slush",
    "small",
    "smart",
    "smile",
    "smoke",
    "smooth",
    "snack",
    "snake",
    "snap",
    "sniff",
    "snow",
    "soap",
    "soccer",
    "social",
    "sock",
    "soda",
    "soft",
    "solar",
    "soldier",
    "solid",
    "solution",
    "solve",
    "someone",
    "song",
    "soon",
    "sorry",
    "sort",
    "soul",
    "sound",
    "soup",
    "source",
    "south",
    "space",
    "spare",
    "spatial",
    "spawn",
    "speak",
    "special",
    "speed",
    "spell",
    "spend",
    "sphere",
    "spice",
    "spider",
    "spike",
    "spin",
    "spirit",
    "split",
    "spoil",
    "sponsor",
    "spoon",
    "sport",
    "spot",
    "spray",
    "spread",
    "spring",
    "spy",
    "square",
    "squeeze",
    "squirrel",
    "stable",
    "stadium",
    "staff",
    "stage",
    "stairs",
    "stamp",
    "stand",
    "start",
    "state",
    "stay",
    "steak",
    "steel",
    "stem",
    "step",
    "stereo",
    "stick",
    "still",
    "sting",
    "stock",
    "stomach",
    "stone",
    "stool",
    "story",
    "stove",
    "strategy",
    "street",
    "strike",
    "strong",
    "struggle",
    "student",
    "stuff",
    "stumble",
    "style",
    "subject",
    "submit",
    "subway",
    "success",
    "such",
    "sudden",
    "suffer",
    "sugar",
    "suggest",
    "suit",
    "summer",
    "sun",
    "sunny",
    "sunset",
    "super",
    "supply",
    "supreme",
    "sure",
    "surface",
    "surge",
    "surprise",
    "surround",
    "survey",
    "suspect",
    "sustain",
    "swallow",
    "swamp",
    "swap",
    "swarm",
    "swear",
    "sweet",
    "swift",
    "swim",
    "swing",
    "switch",
    "sword",
    "symbol",
    "symptom",
    "syrup",
    "system",
    "table",
    "tackle",
    "tag",
    "tail",
    "talent",
    "talk",
    "tank",
    "tape",
    "target",
    "task",
    "taste",
    "tattoo",
    "taxi",
    "teach",
    "team",
    "tell",
    "ten",
    "tenant",
    "tennis",
    "term",
    "test",
    "text",
    "thank",
    "that",
    "theme",
    "then",
    "theory",
    "there",
    "they",
    "thing",
    "this",
    "thought",
    "three",
    "thrive",
    "throw",
    "thumb",
    "thunder",
    "ticket",
    "tide",
    "tiger",
    "tilt",
    "timber",
    "time",
    "tiny",
    "tip",
    "tired",
    "tissue",
    "title",
    "toast",
    "tobacco",
    "today",
    "toddler",
    "toe",
    "together",
    "toilet",
    "token",
    "tomato",
    "tomorrow",
    "tone",
    "tongue",
    "tonight",
    "tool",
    "tooth",
    "top",
    "topic",
    "topple",
    "torch",
    "tornado",
    "tortoise",
    "toss",
    "total",
    "tourist",
    "toward",
    "tower",
    "town",
    "toy",
    "track",
    "trade",
    "traffic",
    "tragic",
    "train",
    "transfer",
    "trap",
    "trash",
    "travel",
    "tray",
    "treat",
    "tree",
    "trend",
    "trial",
    "tribe",
    "trick",
    "trigger",
    "trim",
    "trip",
    "trophy",
    "trouble",
    "truck",
    "true",
    "truly",
    "trumpet",
    "trust",
    "truth",
    "try",
    "tube",
    "tuition",
    "tumble",
    "tuna",
    "tunnel",
    "turkey",
    "turn",
    "turtle",
    "twelve",
    "twenty",
    "twice",
    "twin",
    "twist",
    "two",
    "type",
    "typical",
    "ugly",
    "umbrella",
    "unable",
    "unaware",
    "uncle",
    "uncover",
    "under",
    "undo",
    "unfair",
    "unfold",
    "unhappy",
    "uniform",
    "unique",
    "unit",
    "universe",
    "unknown",
    "unlock",
    "until",
    "unusual",
    "unveil",
    "update",
    "upgrade",
    "uphold",
    "upon",
    "upper",
    "upset",
    "urban",
    "urge",
    "usage",
    "use",
    "used",
    "useful",
    "useless",
    "usual",
    "utility",
    "vacant",
    "vacuum",
    "vague",
    "valid",
    "valley",
    "valve",
    "van",
    "vanish",
    "vapor",
    "various",
    "vast",
    "vault",
    "vehicle",
    "velvet",
    "vendor",
    "venture",
    "venue",
    "verb",
    "verify",
    "version",
    "very",
    "vessel",
    "veteran",
    "viable",
    "vibrant",
    "vicious",
    "victory",
    "video",
    "view",
    "village",
    "vintage",
    "violin",
    "virtual",
    "virus",
    "visa",
    "visit",
    "visual",
    "vital",
    "vivid",
    "vocal",
    "voice",
    "void",
    "volcano",
    "volume",
    "vote",
    "voyage",
    "wage",
    "wagon",
    "wait",
    "walk",
    "wall",
    "walnut",
    "want",
    "warfare",
    "warm",
    "warrior",
    "wash",
    "wasp",
    "waste",
    "water",
    "wave",
    "way",
    "wealth",
    "weapon",
    "wear",
    "weasel",
    "weather",
    "web",
    "wedding",
    "week",
    "weird",
    "welcome",
    "west",
    "wet",
    "whale",
    "what",
    "wheat",
    "wheel",
    "when",
    "where",
    "whip",
    "whisper",
    "wide",
    "width",
    "wife",
    "wild",
    "will",
    "win",
    "window",
    "wine",
    "wing",
    "wink",
    "winner",
    "winter",
    "wire",
    "wisdom",
    "wise",
    "wish",
    "witness",
    "wolf",
    "woman",
    "wonder",
    "wood",
    "wool",
    "word",
    "work",
    "world",
    "worry",
    "worth",
    "wrap",
    "wreck",
    "wrestle",
    "wrist",
    "write",
    "wrong",
    "yard",
    "year",
    "yellow",
    "you",
    "young",
    "youth",
    "zebra",
    "zero",
    "zone",
    "zoo",
}

SECRET_KEY_PLACEHOLDER = "[REDACTED STELLAR SECRET KEY]"
SEED_PHRASE_PLACEHOLDER = "[REDACTED SEED PHRASE]"

SECRET_KEY_WARNING = (
    "CRITICAL SECURITY ALERT: Your input appears to contain a Stellar SECRET key (starting with S). "
    "Do not paste, enter, or share your secret key with any AI assistant or website! "
    "Soroban authorization entries are signed locally within your non-custodial wallet "
    "(such as Freighter or Lobstr) without ever exposing your secret key. "
    "Because this key was shared, treat it as compromised and transfer your assets to a fresh wallet immediately."
)

SEED_PHRASE_WARNING = (
    "CRITICAL SECURITY ALERT: Your input appears to contain a wallet seed phrase or recovery mnemonic. "
    "Never paste your recovery phrase into any prompt or application! "
    "Deen Bridge and the Stellar network never require your recovery phrase. "
    "Treat this wallet as compromised and migrate your funds to a new wallet immediately."
)

SOROBAN_AUTH_DISCLAIMER = (
    "This is an automated safety explainer based strictly on Soroban authorization metadata. "
    "It is non-custodial and read-only. It explains what contract invocations and permissions "
    "are authorized by this entry, but does not execute bytecode or guarantee smart contract "
    "correctness. Always verify contract IDs and sign only via trusted non-custodial wallets."
)


class RiskLevel(str, Enum):
    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"


def explorer_contract_url(contract_id: str, network: str | None = None) -> str:
    """Return public stellar.expert URL for a contract."""
    net = (network or STELLAR_NETWORK).lower()
    net_name = "public" if net in ("public", "mainnet") else "testnet"
    return f"https://stellar.expert/explorer/{net_name}/contract/{contract_id}"


def explorer_account_url(account_id: str, network: str | None = None) -> str:
    """Return public stellar.expert URL for an account."""
    net = (network or STELLAR_NETWORK).lower()
    net_name = "public" if net in ("public", "mainnet") else "testnet"
    return f"https://stellar.expert/explorer/{net_name}/account/{account_id}"


def contains_secret_key(*texts: str | None) -> bool:
    """True if any text contains a Stellar secret key (S...)."""
    return any(bool(SECRET_KEY_PATTERN.search(text or "")) for text in texts)


def contains_seed_phrase(*texts: str | None) -> bool:
    """True if any text appears to contain a 12-24 word BIP-39 mnemonic phrase."""
    for text in texts:
        if not text:
            continue
        lowered = text.casefold()
        if any(ind in lowered for ind in SEED_PHRASE_INDICATORS):
            return True
        # Check for sequences of BIP-39 words (e.g. 12 or 24 words)
        words = re.findall(r"\b[a-z]{3,12}\b", lowered)
        if len(words) >= 12:
            match_count = sum(1 for w in words if w in _BIP39_SAMPLE_WORDS)
            if match_count >= 12 and (match_count / len(words)) >= 0.75:
                return True
    return False


def contains_sensitive_key_material(*texts: str | None) -> bool:
    """True if any text contains a secret key or seed phrase."""
    return contains_secret_key(*texts) or contains_seed_phrase(*texts)


def redact_sensitive_material(text: str | None) -> str | None:
    """Scrub secret keys and apparent seed phrases from user-supplied text."""
    if not text:
        return text
    scrubbed = SECRET_KEY_PATTERN.sub(SECRET_KEY_PLACEHOLDER, text)
    if contains_seed_phrase(scrubbed):
        # Redact labeled phrases
        for ind in SEED_PHRASE_INDICATORS:
            scrubbed = re.sub(
                rf"(?i){re.escape(ind)}\s*[:=]?\s*([a-zA-Z\s]{{20,200}})", SEED_PHRASE_PLACEHOLDER, scrubbed
            )
    return scrubbed


def sc_address_to_str(sc_addr: xdr.SCAddress) -> str:
    """Convert an SCAddress XDR structure into a public G... or C... address."""
    try:
        val = scval.to_native(xdr.SCVal(xdr.SCValType.SCV_ADDRESS, address=sc_addr))
        if isinstance(val, Address):
            return str(val.address)
        addr_attr = getattr(val, "address", None)
        if addr_attr is not None:
            return str(addr_attr)
        return str(val)
    except Exception:
        if sc_addr.type == xdr.SCAddressType.SC_ADDRESS_TYPE_ACCOUNT:
            account_id = sc_addr.account_id
            if account_id and account_id.account_id and account_id.account_id.ed25519:
                return StrKey.encode_ed25519_public_key(account_id.account_id.ed25519.uint256)
        elif sc_addr.type == xdr.SCAddressType.SC_ADDRESS_TYPE_CONTRACT:
            contract_id = sc_addr.contract_id
            if contract_id and contract_id.contract_id:
                return StrKey.encode_contract(contract_id.contract_id.hash)
        return str(sc_addr)


def scval_to_native_safe(val: xdr.SCVal) -> Any:
    """Safely convert SCVal into human-readable native Python representation."""
    try:
        native = scval.to_native(val)
        if isinstance(native, Address):
            return native.address
        if isinstance(native, bytes):
            try:
                return native.decode("utf-8")
            except UnicodeDecodeError:
                return f"0x{native.hex()}"
        if isinstance(native, list):
            return [scval_to_native_safe(item) if isinstance(item, xdr.SCVal) else item for item in native]
        if isinstance(native, dict):
            return {str(k): scval_to_native_safe(v) if isinstance(v, xdr.SCVal) else v for k, v in native.items()}
        return native
    except Exception:
        return str(val)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class InvocationDetail(BaseModel):
    """Detailed view of an authorized function call in the invocation tree."""

    contract_id: str | None = Field(None, description="Target smart contract address (C...)")
    contract_explorer: str | None = Field(None, description="Stellar.expert link for contract")
    function_name: str = Field(..., description="Function invoked on the contract")
    args: list[Any] = Field(default_factory=list, description="Decoded function arguments")
    sub_invocations: list["InvocationDetail"] = Field(
        default_factory=list, description="Nested child invocations authorized under this call"
    )
    risk_factors: list[str] = Field(default_factory=list, description="Risk flags specific to this invocation")


class TransferDetail(BaseModel):
    """Summary of an asset or token transfer identified in the entry."""

    token_contract_id: str | None = None
    token_explorer: str | None = None
    from_address: str | None = None
    to_address: str | None = None
    amount: str | None = None
    is_sub_invocation: bool = False


class AllowanceDetail(BaseModel):
    """Summary of a token allowance approval identified in the entry."""

    token_contract_id: str | None = None
    token_explorer: str | None = None
    from_address: str | None = None
    spender_address: str | None = None
    amount: str | None = None
    live_until_ledger: int | None = None
    is_unlimited: bool = False
    is_sub_invocation: bool = False


class SorobanAuthExplanation(BaseModel):
    """Complete safety explanation of a Soroban authorization entry."""

    network: str = Field(..., description="Stellar network (testnet, public)")
    risk_level: RiskLevel = Field(..., description="Overall risk classification")
    summary: str = Field(..., description="Plain-language description of authorized actions")
    credentials_type: str = Field(..., description="source_account, address, address_v2, etc.")
    authorized_address: str | None = Field(default=None, description="Address granting authorization (G... or C...)")
    authorized_address_explorer: str | None = Field(default=None, description="Stellar.expert account link")
    nonce: int | None = Field(default=None, description="Replay protection nonce counter")
    signature_expiration_ledger: int | None = Field(
        default=None, description="Ledger sequence after which auth expires"
    )
    root_invocation: InvocationDetail | None = None
    total_invocations_count: int = 0
    sub_invocations_count: int = 0
    transfers: list[TransferDetail] = Field(default_factory=list)
    allowances: list[AllowanceDetail] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    secret_key_detected: bool = False
    is_non_custodial: bool = True
    disclaimer: str = SOROBAN_AUTH_DISCLAIMER


class SorobanAuthExplainRequest(BaseModel):
    """Input payload to explain a Soroban authorization entry."""

    entry_xdr: str | None = Field(default=None, description="Base64-encoded SorobanAuthorizationEntry XDR")
    entry_json: dict[str, Any] | None = Field(default=None, description="Parsed JSON Soroban authorization entry")
    contract_id: str | None = Field(default=None, description="Direct contract ID override (C...)")
    function_name: str | None = Field(default=None, description="Direct function name override")
    args: list[Any] | None = Field(default=None, description="Direct function arguments")
    auth_address: str | None = Field(default=None, description="Authorizing address (G... or C...)")
    nonce: int | None = Field(default=None, description="Nonce counter")
    signature_expiration_ledger: int | None = Field(default=None, description="Expiration ledger sequence")
    network: str | None = Field(default=None, description="Network override ('testnet' or 'public')")


class SorobanAuthInfo(BaseModel):
    """Grounding info provided to chat responses."""

    explained: bool = Field(..., description="True if an authorization entry was parsed and explained")
    risk_level: str | None = None
    contract_id: str | None = None
    function_name: str | None = None
    authorized_address: str | None = None
    sub_invocations_count: int = 0
    secret_key_detected: bool = False


class SorobanAuthContext(BaseModel):
    """Chat grounding context for a turn involving Soroban smart contract authorization."""

    info: SorobanAuthInfo
    prompt_block: str
    explanation: SorobanAuthExplanation | None = None


# ---------------------------------------------------------------------------
# Core Analysis Logic
# ---------------------------------------------------------------------------


def _analyze_invocation(
    inv: xdr.SorobanAuthorizedInvocation,
    network: str,
    depth: int = 0,
) -> tuple[InvocationDetail, list[TransferDetail], list[AllowanceDetail], list[str]]:
    """Recursively analyze an authorized invocation and its sub-invocations."""
    func = inv.function
    contract_id: str | None = None
    contract_url: str | None = None
    function_name = "unknown"
    args: list[Any] = []
    risks: list[str] = []
    transfers: list[TransferDetail] = []
    allowances: list[AllowanceDetail] = []

    if func.type == xdr.SorobanAuthorizedFunctionType.SOROBAN_AUTHORIZED_FUNCTION_TYPE_CONTRACT_FN:
        assert func.contract_fn is not None
        contract_id = sc_address_to_str(func.contract_fn.contract_address)
        contract_url = explorer_contract_url(contract_id, network)
        function_name = func.contract_fn.function_name.sc_symbol.decode("utf-8", errors="replace")
        args = [scval_to_native_safe(arg) for arg in func.contract_fn.args]
    elif func.type == xdr.SorobanAuthorizedFunctionType.SOROBAN_AUTHORIZED_FUNCTION_TYPE_CREATE_CONTRACT_HOST_FN:
        function_name = "create_contract"
        risks.append("Deploys a new smart contract instance")
    elif func.type == xdr.SorobanAuthorizedFunctionType.SOROBAN_AUTHORIZED_FUNCTION_TYPE_CREATE_CONTRACT_V2_HOST_FN:
        function_name = "create_contract_v2"
        risks.append("Deploys a new smart contract instance with constructor arguments")

    # Inspect function semantics
    lower_fn = function_name.lower()
    is_sub = depth > 0

    if lower_fn in ("transfer", "transfer_from"):
        from_addr = str(args[0]) if len(args) > 0 else None
        to_addr = str(args[1]) if len(args) > 1 else None
        amount = str(args[2]) if len(args) > 2 else None
        if lower_fn == "transfer_from" and len(args) > 3:
            # transfer_from(spender, from, to, amount)
            from_addr = str(args[1])
            to_addr = str(args[2])
            amount = str(args[3])

        transfers.append(
            TransferDetail(
                token_contract_id=contract_id,
                token_explorer=contract_url,
                from_address=from_addr,
                to_address=to_addr,
                amount=amount,
                is_sub_invocation=is_sub,
            )
        )
        if is_sub:
            risks.append(f"Sub-invocation transfers tokens to {to_addr or 'recipient'}")
        else:
            risks.append(f"Authorizes transfer of {amount or 'tokens'} to {to_addr or 'recipient'}")

    elif lower_fn == "approve":
        # approve(from, spender, amount, live_until_ledger)
        from_addr = str(args[0]) if len(args) > 0 else None
        spender_addr = str(args[1]) if len(args) > 1 else None
        amount = str(args[2]) if len(args) > 2 else None
        live_until = int(args[3]) if len(args) > 3 and isinstance(args[3], int) else None

        is_unlimited = False
        if amount:
            try:
                amt_int = int(amount)
                # Max or near max int128 / int64 or > 1,000,000,000,000
                if amt_int >= 2**63 - 1 or amt_int > 10**12:
                    is_unlimited = True
            except ValueError:
                pass

        allowances.append(
            AllowanceDetail(
                token_contract_id=contract_id,
                token_explorer=contract_url,
                from_address=from_addr,
                spender_address=spender_addr,
                amount=amount,
                live_until_ledger=live_until,
                is_unlimited=is_unlimited,
                is_sub_invocation=is_sub,
            )
        )
        if is_unlimited:
            risks.append(f"CRITICAL: Unlimited token spending allowance granted to {spender_addr}")
        else:
            risks.append(f"Grants spending allowance of {amount} to {spender_addr}")

    elif lower_fn in ("set_admin", "set_owner", "upgrade", "update_wasm"):
        risks.append(f"Administrative privilege or contract logic update ({function_name})")

    elif lower_fn in ("burn", "clawback"):
        risks.append(f"Permanent asset reduction/removal ({function_name})")

    # Recursively analyze sub-invocations
    sub_details: list[InvocationDetail] = []
    for sub in inv.sub_invocations:
        sub_detail, sub_transfers, sub_allowances, sub_risks = _analyze_invocation(sub, network, depth + 1)
        sub_details.append(sub_detail)
        transfers.extend(sub_transfers)
        allowances.extend(sub_allowances)
        risks.extend(sub_risks)

    detail = InvocationDetail(
        contract_id=contract_id,
        contract_explorer=contract_url,
        function_name=function_name,
        args=args,
        sub_invocations=sub_details,
        risk_factors=risks,
    )
    return detail, transfers, allowances, risks


def _count_invocations(detail: InvocationDetail) -> int:
    """Count total invocations in tree (root + all sub-invocations)."""
    count = 1
    for sub in detail.sub_invocations:
        count += _count_invocations(sub)
    return count


def analyze_soroban_authorization(
    entry: xdr.SorobanAuthorizationEntry,
    network: str = STELLAR_NETWORK,
) -> SorobanAuthExplanation:
    """Analyze a parsed SorobanAuthorizationEntry and produce a safety explanation."""
    warnings: list[str] = []
    recommendations: list[str] = []

    # Credentials
    creds = entry.credentials
    creds_type = "source_account"
    auth_address: str | None = None
    nonce: int | None = None
    expiration_ledger: int | None = None

    if creds.type == xdr.SorobanCredentialsType.SOROBAN_CREDENTIALS_SOURCE_ACCOUNT:
        creds_type = "source_account"
    elif creds.type == xdr.SorobanCredentialsType.SOROBAN_CREDENTIALS_ADDRESS:
        creds_type = "address"
        assert creds.address is not None
        auth_address = sc_address_to_str(creds.address.address)
        nonce = int(creds.address.nonce.int64)
        expiration_ledger = int(creds.address.signature_expiration_ledger.uint32)
    elif creds.type == xdr.SorobanCredentialsType.SOROBAN_CREDENTIALS_ADDRESS_V2:
        creds_type = "address_v2"
        assert creds.address_v2 is not None
        auth_address = sc_address_to_str(creds.address_v2.address)
        nonce = int(creds.address_v2.nonce.int64)
        expiration_ledger = int(creds.address_v2.signature_expiration_ledger.uint32)
    elif creds.type == xdr.SorobanCredentialsType.SOROBAN_CREDENTIALS_ADDRESS_WITH_DELEGATES:
        creds_type = "address_with_delegates"
        assert creds.address_with_delegates is not None
        addr_creds = creds.address_with_delegates.address_credentials
        auth_address = sc_address_to_str(addr_creds.address)
        nonce = int(addr_creds.nonce.int64)
        expiration_ledger = int(addr_creds.signature_expiration_ledger.uint32)

    auth_addr_url = explorer_account_url(auth_address, network) if auth_address else None

    # Invocation Tree Analysis
    root_detail, transfers, allowances, risks = _analyze_invocation(entry.root_invocation, network, depth=0)
    total_invocations = _count_invocations(root_detail)
    sub_count = total_invocations - 1

    # Replay and Expiration evaluation
    if expiration_ledger is not None:
        if expiration_ledger == 0:
            warnings.append("Signature expiration ledger is 0: authorization has no expiration and could be replayed.")
        elif expiration_ledger > 50_000_000:
            warnings.append(
                f"Very high expiration ledger ({expiration_ledger}): authorization stays valid for an extended duration."
            )

    if creds_type != "source_account" and nonce == 0:
        recommendations.append("Nonce is 0: verify this is the initial authorization on this address.")

    # Sub-invocation risk flags
    if sub_count > 0:
        warnings.append(
            f"Contains {sub_count} nested sub-invocation(s). By signing, you authorize all nested actions in the tree."
        )

    for tr in transfers:
        if tr.is_sub_invocation:
            warnings.append(
                f"Sub-invocation transfer: Token {tr.token_contract_id or 'contract'} transfers {tr.amount or 'funds'} "
                f"from {tr.from_address or 'sender'} to {tr.to_address or 'recipient'}."
            )

    for al in allowances:
        if al.is_unlimited:
            warnings.append(
                f"High-risk allowance: unlimited spending permission granted to {al.spender_address or 'spender'}."
            )
        elif al.is_sub_invocation:
            warnings.append(
                f"Sub-invocation allowance: permission granted to {al.spender_address or 'spender'} inside sub-call."
            )

    # Determine Overall Risk Level
    has_unlimited_allowance = any(al.is_unlimited for al in allowances)
    has_admin_calls = any("Administrative" in r for r in risks)
    has_sub_transfers = any(tr.is_sub_invocation for tr in transfers)

    if has_unlimited_allowance:
        risk_level = RiskLevel.WARNING
    elif has_sub_transfers or has_admin_calls:
        risk_level = RiskLevel.WARNING
    elif len(transfers) > 0 or len(allowances) > 0 or sub_count > 0:
        risk_level = RiskLevel.CAUTION
    else:
        risk_level = RiskLevel.SAFE

    # Recommendations
    if root_detail.contract_id:
        recommendations.append(f"Verify contract address matches expected dapp: {root_detail.contract_id}")
    if transfers:
        recommendations.append("Double check recipient address and amount before signing in your wallet.")
    recommendations.append("Always sign using a non-custodial wallet; never provide secret keys or seed phrases.")

    # Build human summary
    summary_parts = []
    fn = root_detail.function_name
    c_id = root_detail.contract_id or "smart contract"
    signer = auth_address or "transaction source account"
    summary_parts.append(f"Authorizes {signer} to execute '{fn}' on contract {c_id}")
    if transfers:
        summary_parts.append(f"including {len(transfers)} asset transfer(s)")
    if allowances:
        summary_parts.append(f"and {len(allowances)} allowance approval(s)")
    if sub_count > 0:
        summary_parts.append(f"with {sub_count} nested sub-call(s)")
    summary = ", ".join(summary_parts) + "."

    return SorobanAuthExplanation(
        network=network,
        risk_level=risk_level,
        summary=summary,
        credentials_type=creds_type,
        authorized_address=auth_address,
        authorized_address_explorer=auth_addr_url,
        nonce=nonce,
        signature_expiration_ledger=expiration_ledger,
        root_invocation=root_detail,
        total_invocations_count=total_invocations,
        sub_invocations_count=sub_count,
        transfers=transfers,
        allowances=allowances,
        warnings=warnings,
        recommendations=recommendations,
        secret_key_detected=False,
        is_non_custodial=True,
        disclaimer=SOROBAN_AUTH_DISCLAIMER,
    )


def explain_soroban_authorization(request: SorobanAuthExplainRequest) -> SorobanAuthExplanation:
    """Entry point for explaining a Soroban authorization entry."""
    net = (request.network or STELLAR_NETWORK).lower()
    net_name = "public" if net in ("public", "mainnet") else "testnet"

    # Check for secret keys or seed phrases in request
    if contains_sensitive_key_material(
        request.entry_xdr,
        request.contract_id,
        request.function_name,
        request.auth_address,
        str(request.entry_json or ""),
        *[str(a) for a in (request.args or [])],
    ):
        logger.warning("Secret key or seed phrase detected in Soroban auth request; refusing")
        return SorobanAuthExplanation(
            network=net_name,
            risk_level=RiskLevel.CRITICAL,
            summary="Refused: input contains sensitive secret key or seed phrase.",
            credentials_type="unknown",
            warnings=[SECRET_KEY_WARNING],
            recommendations=["Never share private keys or seed phrases with anyone or any AI service."],
            secret_key_detected=True,
            is_non_custodial=True,
        )

    # 1. Parse from XDR string if provided
    if request.entry_xdr:
        clean_xdr = request.entry_xdr.strip()
        try:
            entry = xdr.SorobanAuthorizationEntry.from_xdr(clean_xdr)
            return analyze_soroban_authorization(entry, network=net_name)
        except Exception as exc:
            logger.info("Failed to parse SorobanAuthorizationEntry from XDR: %s", exc)
            raise HTTPException(
                status_code=400,
                detail=f"Invalid Soroban authorization entry XDR: {exc}",
            ) from None

    # 2. Parse from JSON dict if provided
    if request.entry_json:
        try:
            entry = xdr.SorobanAuthorizationEntry.from_json_dict(request.entry_json)
            return analyze_soroban_authorization(entry, network=net_name)
        except Exception:
            # Fallback: construct manual entry if simplified dictionary was passed
            c_id = request.entry_json.get("contract_id") or request.contract_id
            fn = request.entry_json.get("function_name") or request.function_name or "call"
            args = request.entry_json.get("args") or request.args or []
            auth_addr = request.entry_json.get("auth_address") or request.auth_address
            nonce = request.entry_json.get("nonce", request.nonce)
            exp = request.entry_json.get("signature_expiration_ledger", request.signature_expiration_ledger)

            return _manual_explanation(c_id, fn, args, auth_addr, nonce, exp, net_name)

    # 3. Direct field parameters
    if request.contract_id or request.function_name:
        return _manual_explanation(
            request.contract_id,
            request.function_name or "call",
            request.args or [],
            request.auth_address,
            request.nonce,
            request.signature_expiration_ledger,
            net_name,
        )

    raise HTTPException(
        status_code=400,
        detail="Missing Soroban authorization entry. Please provide 'entry_xdr', 'entry_json', or contract details.",
    )


def _manual_explanation(
    contract_id: str | None,
    function_name: str,
    args: list[Any],
    auth_address: str | None,
    nonce: int | None,
    expiration_ledger: int | None,
    network: str,
) -> SorobanAuthExplanation:
    """Build explanation from high-level fields when raw XDR is not supplied."""
    contract_url = explorer_contract_url(contract_id, network) if contract_id else None
    auth_url = explorer_account_url(auth_address, network) if auth_address else None

    risks: list[str] = []
    transfers: list[TransferDetail] = []
    allowances: list[AllowanceDetail] = []
    warnings: list[str] = []
    recommendations: list[str] = []

    lower_fn = function_name.lower()
    if lower_fn in ("transfer", "transfer_from"):
        from_a = str(args[0]) if len(args) > 0 else auth_address
        to_a = str(args[1]) if len(args) > 1 else None
        amt = str(args[2]) if len(args) > 2 else None
        transfers.append(
            TransferDetail(
                token_contract_id=contract_id,
                token_explorer=contract_url,
                from_address=from_a,
                to_address=to_a,
                amount=amt,
            )
        )
        risks.append(f"Authorizes transfer of {amt or 'assets'} to {to_a or 'recipient'}")
    elif lower_fn == "approve":
        spender_a = str(args[1]) if len(args) > 1 else None
        amt = str(args[2]) if len(args) > 2 else None
        is_unlimited = False
        if amt:
            try:
                if int(amt) > 10**12:
                    is_unlimited = True
            except ValueError:
                pass
        allowances.append(
            AllowanceDetail(
                token_contract_id=contract_id,
                token_explorer=contract_url,
                from_address=auth_address,
                spender_address=spender_a,
                amount=amt,
                live_until_ledger=expiration_ledger,
                is_unlimited=is_unlimited,
            )
        )
        if is_unlimited:
            risks.append(f"CRITICAL: Unlimited spending allowance to {spender_a}")
            warnings.append(f"Unlimited token allowance granted to {spender_a}.")
        else:
            risks.append(f"Grants spending allowance of {amt} to {spender_a}")

    risk_level = (
        RiskLevel.WARNING
        if any(al.is_unlimited for al in allowances)
        else (RiskLevel.CAUTION if (transfers or allowances) else RiskLevel.SAFE)
    )

    detail = InvocationDetail(
        contract_id=contract_id,
        contract_explorer=contract_url,
        function_name=function_name,
        args=args,
        risk_factors=risks,
    )

    if contract_id:
        recommendations.append(f"Verify contract address matches official registry: {contract_id}")
    recommendations.append("Sign transactions only via non-custodial wallet extensions or hardware keys.")

    summary = (
        f"Authorizes {auth_address or 'source account'} to invoke '{function_name}' "
        f"on contract {contract_id or 'unknown'}."
    )

    return SorobanAuthExplanation(
        network=network,
        risk_level=risk_level,
        summary=summary,
        credentials_type="address" if auth_address else "source_account",
        authorized_address=auth_address,
        authorized_address_explorer=auth_url,
        nonce=nonce,
        signature_expiration_ledger=expiration_ledger,
        root_invocation=detail,
        total_invocations_count=1,
        sub_invocations_count=0,
        transfers=transfers,
        allowances=allowances,
        warnings=warnings,
        recommendations=recommendations,
        secret_key_detected=False,
        is_non_custodial=True,
        disclaimer=SOROBAN_AUTH_DISCLAIMER,
    )


# ---------------------------------------------------------------------------
# Chat Integration
# ---------------------------------------------------------------------------

SOROBAN_KEYWORDS = (
    "soroban",
    "auth entry",
    "authorization entry",
    "soroban authorization",
    "contract authorization",
    "auth simulation",
    "smart contract auth",
    "simulateTransaction",
    "InvokeHostFunction",
)


def is_soroban_auth_question(text: str) -> bool:
    """True if user is asking about Soroban authorization or safety."""
    lowered = (text or "").casefold()
    if any(kw.casefold() in lowered for kw in SOROBAN_KEYWORDS):
        return True
    if CONTRACT_KEY_PATTERN.search(text or "") and any(term in lowered for term in ("auth", "sign", "safe", "permit")):
        return True
    return False


def extract_entry_xdr(*texts: str | None) -> str | None:
    """Attempt to find a Base64-encoded SorobanAuthorizationEntry XDR string."""
    for text in texts:
        if not text:
            continue
        # Base64 strings with padding or alphanumeric sequences > 40 chars
        candidates = re.findall(r"\b[A-Za-z0-9+/=]{40,}\b", text)
        for cand in candidates:
            try:
                # Try decoding to see if it is a valid SorobanAuthorizationEntry
                xdr.SorobanAuthorizationEntry.from_xdr(cand)
                return cand
            except Exception:
                continue
    return None


def build_soroban_auth_prompt_block(explanation: SorobanAuthExplanation) -> str:
    """Format explanation as factual prompt grounding for LLM chat."""
    lines = [
        "",
        "SOROBAN AUTHORIZATION SAFETY ANALYSIS (real on-chain metadata for user):",
        f"- Network: {explanation.network}",
        f"- Risk level: {explanation.risk_level.value.upper()}",
        f"- Summary: {explanation.summary}",
        f"- Credentials type: {explanation.credentials_type}",
        f"- Authorizing address: {explanation.authorized_address or 'Source account'}",
    ]
    if explanation.nonce is not None:
        lines.append(f"- Nonce: {explanation.nonce}")
    if explanation.signature_expiration_ledger is not None:
        lines.append(f"- Expiration ledger: {explanation.signature_expiration_ledger}")
    if explanation.root_invocation:
        lines.append(f"- Contract: {explanation.root_invocation.contract_id}")
        lines.append(f"- Function: {explanation.root_invocation.function_name}")
        if explanation.root_invocation.contract_explorer:
            lines.append(f"- Explorer link: {explanation.root_invocation.contract_explorer}")

    lines.append(f"- Total invocations in tree: {explanation.total_invocations_count}")
    lines.append(f"- Sub-invocations count: {explanation.sub_invocations_count}")

    if explanation.transfers:
        lines.append("Transfers authorized:")
        for t in explanation.transfers:
            lines.append(
                f"  * From: {t.from_address} -> To: {t.to_address}, Amount: {t.amount} (Token: {t.token_contract_id})"
            )

    if explanation.allowances:
        lines.append("Allowances authorized:")
        for a in explanation.allowances:
            unlim = " [UNLIMITED/HIGH RISK]" if a.is_unlimited else ""
            lines.append(f"  * Spender: {a.spender_address}, Amount: {a.amount}{unlim}")

    if explanation.warnings:
        lines.append("Safety Warnings:")
        for w in explanation.warnings:
            lines.append(f"  ! {w}")

    if explanation.recommendations:
        lines.append("Recommendations:")
        for r in explanation.recommendations:
            lines.append(f"  - {r}")

    lines.extend(
        [
            "",
            "RULES FOR THE ASSISTANT:",
            "- Explain what this Soroban authorization does clearly and neutrally.",
            "- Explicitly highlight if any token transfer or allowance is authorized.",
            "- If sub-invocations exist, warn the user that signing authorizes the entire sub-tree.",
            "- Remind them to check the contract address on stellar.expert.",
            "- State plainly that secret keys and recovery seed phrases must never be shared.",
            f'Close with the disclaimer: "{SOROBAN_AUTH_DISCLAIMER}"',
        ]
    )
    return "\n".join(lines)


NO_AUTH_ENTRY_EDUCATIONAL_NOTE = (
    "\n\nSOROBAN AUTHORIZATION SAFETY QUESTION: The user asked about Soroban smart contract "
    "authorization or auth entries, but no entry XDR was provided. Explain how Soroban "
    "authorization entries work (fine-grained signing of contract invocations, sub-invocation trees, "
    "nonces for replay protection, and expiration ledgers). Advise them on what to check before signing: "
    "1. Contract ID matches official address. 2. Function and arguments (especially transfers or allowances). "
    "3. Sub-invocations that could move assets behind the scenes. 4. Never enter or expose secret keys or "
    "seed phrases; always sign inside non-custodial wallets like Freighter or Lobstr. Invite them to provide "
    "an auth entry XDR if they want a specific breakdown."
)


async def build_chat_soroban_auth_context(
    prompt: str,
    entry_xdr: str | None = None,
    context: str | None = None,
) -> SorobanAuthContext | None:
    """Analyze Soroban authorization context for a chat turn, or None if not relevant."""
    if not is_soroban_auth_question(prompt) and not entry_xdr:
        return None

    if contains_sensitive_key_material(prompt, entry_xdr, context):
        logger.warning("Secret key or seed phrase in Soroban chat turn; refusing")
        return SorobanAuthContext(
            info=SorobanAuthInfo(explained=False, secret_key_detected=True),
            prompt_block=f"\n\n{SECRET_KEY_WARNING}",
            explanation=None,
        )

    resolved_xdr = entry_xdr or extract_entry_xdr(prompt, context)
    if not resolved_xdr:
        # User asked a conceptual question about Soroban auth safety
        return SorobanAuthContext(
            info=SorobanAuthInfo(explained=False),
            prompt_block=NO_AUTH_ENTRY_EDUCATIONAL_NOTE,
            explanation=None,
        )

    try:
        req = SorobanAuthExplainRequest(entry_xdr=resolved_xdr)
        explanation = explain_soroban_authorization(req)
        return SorobanAuthContext(
            info=SorobanAuthInfo(
                explained=True,
                risk_level=explanation.risk_level.value,
                contract_id=explanation.root_invocation.contract_id if explanation.root_invocation else None,
                function_name=explanation.root_invocation.function_name if explanation.root_invocation else None,
                authorized_address=explanation.authorized_address,
                sub_invocations_count=explanation.sub_invocations_count,
            ),
            prompt_block=build_soroban_auth_prompt_block(explanation),
            explanation=explanation,
        )
    except Exception as exc:
        logger.info("Soroban auth explanation for chat could not parse XDR: %s", exc)
        return SorobanAuthContext(
            info=SorobanAuthInfo(explained=False),
            prompt_block=f"\n\nCOULD NOT PARSE SOROBAN AUTH XDR: {exc}. {NO_AUTH_ENTRY_EDUCATIONAL_NOTE}",
            explanation=None,
        )


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@router.post("/stellar/soroban/auth/explain", response_model=SorobanAuthExplanation)
async def explain_auth_endpoint(request: SorobanAuthExplainRequest):
    """Explain the permissions, actions, and safety risks of a Soroban authorization entry."""
    return explain_soroban_authorization(request)


@router.post("/stellar/soroban-auth-explain", response_model=SorobanAuthExplanation)
async def explain_auth_alias(request: SorobanAuthExplainRequest):
    """Alias for /stellar/soroban/auth/explain."""
    return explain_soroban_authorization(request)
