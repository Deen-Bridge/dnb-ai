"""Arabic Stellar terminology evaluator checking non-custodial and network-explicit translations (#667)."""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
from dataclasses import asdict, dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

PUBLIC_PASSPHRASE = "Public Global Stellar Network ; September 2015"
TESTNET_PASSPHRASE = "Test SDF Network ; September 2015"
STROOPS_PER_UNIT = Decimal("10000000")
MAX_INT64_STROOPS = (1 << 63) - 1
_SECRET_KEY_RE = re.compile(r"(?<![A-Z2-7])S[A-Z2-7]{55}(?![A-Z2-7])")
_PUBLIC_KEY_RE = re.compile(r"^G[A-Z2-7]{55}$")
_ASSET_CODE_RE = re.compile(r"^[A-Za-z0-9]{1,12}$")
_TX_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SEED_PHRASE_RE = re.compile(
    r"\b(mnemonic|seed\s+phrase|recovery\s+phrase|recovery\s+words|12\s+words|24\s+words|private\s+key|secret\s+seed)\b",
    re.IGNORECASE,
)
# BIP-39 English wordlist (2048 words), embedded so mnemonic detection does not
# depend on a network fetch or an unpinned dependency. Source:
# https://github.com/bitcoin/bips/blob/master/bip-0039/english.txt
_BIP39_WORDLIST = frozenset(
    (
        "abandon ability able about above absent absorb abstract absurd abuse access accident account "
        "accuse achieve acid acoustic acquire across act action actor actress actual adapt add addict "
        "address adjust admit adult advance advice aerobic affair afford afraid again age agent agree "
        "ahead aim air airport aisle alarm album alcohol alert alien all alley allow almost alone alpha "
        "already also alter always amateur amazing among amount amused analyst anchor ancient anger "
        "angle angry animal ankle announce annual another answer antenna antique anxiety any apart "
        "apology appear apple approve april arch arctic area arena argue arm armed armor army around "
        "arrange arrest arrive arrow art artefact artist artwork ask aspect assault asset assist assume "
        "asthma athlete atom attack attend attitude attract auction audit august aunt author auto autumn "
        "average avocado avoid awake aware away awesome awful awkward axis baby bachelor bacon badge bag "
        "balance balcony ball bamboo banana banner bar barely bargain barrel base basic basket battle "
        "beach bean beauty because become beef before begin behave behind believe below belt bench "
        "benefit best betray better between beyond bicycle bid bike bind biology bird birth bitter black "
        "blade blame blanket blast bleak bless blind blood blossom blouse blue blur blush board boat "
        "body boil bomb bone bonus book boost border boring borrow boss bottom bounce box boy bracket "
        "brain brand brass brave bread breeze brick bridge brief bright bring brisk broccoli broken "
        "bronze broom brother brown brush bubble buddy budget buffalo build bulb bulk bullet bundle "
        "bunker burden burger burst bus business busy butter buyer buzz cabbage cabin cable cactus cage "
        "cake call calm camera camp can canal cancel candy cannon canoe canvas canyon capable capital "
        "captain car carbon card cargo carpet carry cart case cash casino castle casual cat catalog "
        "catch category cattle caught cause caution cave ceiling celery cement census century cereal "
        "certain chair chalk champion change chaos chapter charge chase chat cheap check cheese chef "
        "cherry chest chicken chief child chimney choice choose chronic chuckle chunk churn cigar "
        "cinnamon circle citizen city civil claim clap clarify claw clay clean clerk clever click client "
        "cliff climb clinic clip clock clog close cloth cloud clown club clump cluster clutch coach "
        "coast coconut code coffee coil coin collect color column combine come comfort comic common "
        "company concert conduct confirm congress connect consider control convince cook cool copper "
        "copy coral core corn correct cost cotton couch country couple course cousin cover coyote crack "
        "cradle craft cram crane crash crater crawl crazy cream credit creek crew cricket crime crisp "
        "critic crop cross crouch crowd crucial cruel cruise crumble crunch crush cry crystal cube "
        "culture cup cupboard curious current curtain curve cushion custom cute cycle dad damage damp "
        "dance danger daring dash daughter dawn day deal debate debris decade december decide decline "
        "decorate decrease deer defense define defy degree delay deliver demand demise denial dentist "
        "deny depart depend deposit depth deputy derive describe desert design desk despair destroy "
        "detail detect develop device devote diagram dial diamond diary dice diesel diet differ digital "
        "dignity dilemma dinner dinosaur direct dirt disagree discover disease dish dismiss disorder "
        "display distance divert divide divorce dizzy doctor document dog doll dolphin domain donate "
        "donkey donor door dose double dove draft dragon drama drastic draw dream dress drift drill "
        "drink drip drive drop drum dry duck dumb dune during dust dutch duty dwarf dynamic eager eagle "
        "early earn earth easily east easy echo ecology economy edge edit educate effort egg eight "
        "either elbow elder electric elegant element elephant elevator elite else embark embody embrace "
        "emerge emotion employ empower empty enable enact end endless endorse enemy energy enforce "
        "engage engine enhance enjoy enlist enough enrich enroll ensure enter entire entry envelope "
        "episode equal equip era erase erode erosion error erupt escape essay essence estate eternal "
        "ethics evidence evil evoke evolve exact example excess exchange excite exclude excuse execute "
        "exercise exhaust exhibit exile exist exit exotic expand expect expire explain expose express "
        "extend extra eye eyebrow fabric face faculty fade faint faith fall false fame family famous fan "
        "fancy fantasy farm fashion fat fatal father fatigue fault favorite feature february federal fee "
        "feed feel female fence festival fetch fever few fiber fiction field figure file film filter "
        "final find fine finger finish fire firm first fiscal fish fit fitness fix flag flame flash flat "
        "flavor flee flight flip float flock floor flower fluid flush fly foam focus fog foil fold "
        "follow food foot force forest forget fork fortune forum forward fossil foster found fox fragile "
        "frame frequent fresh friend fringe frog front frost frown frozen fruit fuel fun funny furnace "
        "fury future gadget gain galaxy gallery game gap garage garbage garden garlic garment gas gasp "
        "gate gather gauge gaze general genius genre gentle genuine gesture ghost giant gift giggle "
        "ginger giraffe girl give glad glance glare glass glide glimpse globe gloom glory glove glow "
        "glue goat goddess gold good goose gorilla gospel gossip govern gown grab grace grain grant "
        "grape grass gravity great green grid grief grit grocery group grow grunt guard guess guide "
        "guilt guitar gun gym habit hair half hammer hamster hand happy harbor hard harsh harvest hat "
        "have hawk hazard head health heart heavy hedgehog height hello helmet help hen hero hidden high "
        "hill hint hip hire history hobby hockey hold hole holiday hollow home honey hood hope horn "
        "horror horse hospital host hotel hour hover hub huge human humble humor hundred hungry hunt "
        "hurdle hurry hurt husband hybrid ice icon idea identify idle ignore ill illegal illness image "
        "imitate immense immune impact impose improve impulse inch include income increase index "
        "indicate indoor industry infant inflict inform inhale inherit initial inject injury inmate "
        "inner innocent input inquiry insane insect inside inspire install intact interest into invest "
        "invite involve iron island isolate issue item ivory jacket jaguar jar jazz jealous jeans jelly "
        "jewel job join joke journey joy judge juice jump jungle junior junk just kangaroo keen keep "
        "ketchup key kick kid kidney kind kingdom kiss kit kitchen kite kitten kiwi knee knife knock "
        "know lab label labor ladder lady lake lamp language laptop large later latin laugh laundry lava "
        "law lawn lawsuit layer lazy leader leaf learn leave lecture left leg legal legend leisure lemon "
        "lend length lens leopard lesson letter level liar liberty library license life lift light like "
        "limb limit link lion liquid list little live lizard load loan lobster local lock logic lonely "
        "long loop lottery loud lounge love loyal lucky luggage lumber lunar lunch luxury lyrics machine "
        "mad magic magnet maid mail main major make mammal man manage mandate mango mansion manual maple "
        "marble march margin marine market marriage mask mass master match material math matrix matter "
        "maximum maze meadow mean measure meat mechanic medal media melody melt member memory mention "
        "menu mercy merge merit merry mesh message metal method middle midnight milk million mimic mind "
        "minimum minor minute miracle mirror misery miss mistake mix mixed mixture mobile model modify "
        "mom moment monitor monkey monster month moon moral more morning mosquito mother motion motor "
        "mountain mouse move movie much muffin mule multiply muscle museum mushroom music must mutual "
        "myself mystery myth naive name napkin narrow nasty nation nature near neck need negative "
        "neglect neither nephew nerve nest net network neutral never news next nice night noble noise "
        "nominee noodle normal north nose notable note nothing notice novel now nuclear number nurse nut "
        "oak obey object oblige obscure observe obtain obvious occur ocean october odor off offer office "
        "often oil okay old olive olympic omit once one onion online only open opera opinion oppose "
        "option orange orbit orchard order ordinary organ orient original orphan ostrich other outdoor "
        "outer output outside oval oven over own owner oxygen oyster ozone pact paddle page pair palace "
        "palm panda panel panic panther paper parade parent park parrot party pass patch path patient "
        "patrol pattern pause pave payment peace peanut pear peasant pelican pen penalty pencil people "
        "pepper perfect permit person pet phone photo phrase physical piano picnic picture piece pig "
        "pigeon pill pilot pink pioneer pipe pistol pitch pizza place planet plastic plate play please "
        "pledge pluck plug plunge poem poet point polar pole police pond pony pool popular portion "
        "position possible post potato pottery poverty powder power practice praise predict prefer "
        "prepare present pretty prevent price pride primary print priority prison private prize problem "
        "process produce profit program project promote proof property prosper protect proud provide "
        "public pudding pull pulp pulse pumpkin punch pupil puppy purchase purity purpose purse push put "
        "puzzle pyramid quality quantum quarter question quick quit quiz quote rabbit raccoon race rack "
        "radar radio rail rain raise rally ramp ranch random range rapid rare rate rather raven raw "
        "razor ready real reason rebel rebuild recall receive recipe record recycle reduce reflect "
        "reform refuse region regret regular reject relax release relief rely remain remember remind "
        "remove render renew rent reopen repair repeat replace report require rescue resemble resist "
        "resource response result retire retreat return reunion reveal review reward rhythm rib ribbon "
        "rice rich ride ridge rifle right rigid ring riot ripple risk ritual rival river road roast "
        "robot robust rocket romance roof rookie room rose rotate rough round route royal rubber rude "
        "rug rule run runway rural sad saddle sadness safe sail salad salmon salon salt salute same "
        "sample sand satisfy satoshi sauce sausage save say scale scan scare scatter scene scheme school "
        "science scissors scorpion scout scrap screen script scrub sea search season seat second secret "
        "section security seed seek segment select sell seminar senior sense sentence series service "
        "session settle setup seven shadow shaft shallow share shed shell sheriff shield shift shine "
        "ship shiver shock shoe shoot shop short shoulder shove shrimp shrug shuffle shy sibling sick "
        "side siege sight sign silent silk silly silver similar simple since sing siren sister situate "
        "six size skate sketch ski skill skin skirt skull slab slam sleep slender slice slide slight "
        "slim slogan slot slow slush small smart smile smoke smooth snack snake snap sniff snow soap "
        "soccer social sock soda soft solar soldier solid solution solve someone song soon sorry sort "
        "soul sound soup source south space spare spatial spawn speak special speed spell spend sphere "
        "spice spider spike spin spirit split spoil sponsor spoon sport spot spray spread spring spy "
        "square squeeze squirrel stable stadium staff stage stairs stamp stand start state stay steak "
        "steel stem step stereo stick still sting stock stomach stone stool story stove strategy street "
        "strike strong struggle student stuff stumble style subject submit subway success such sudden "
        "suffer sugar suggest suit summer sun sunny sunset super supply supreme sure surface surge "
        "surprise surround survey suspect sustain swallow swamp swap swarm swear sweet swift swim swing "
        "switch sword symbol symptom syrup system table tackle tag tail talent talk tank tape target "
        "task taste tattoo taxi teach team tell ten tenant tennis tent term test text thank that theme "
        "then theory there they thing this thought three thrive throw thumb thunder ticket tide tiger "
        "tilt timber time tiny tip tired tissue title toast tobacco today toddler toe together toilet "
        "token tomato tomorrow tone tongue tonight tool tooth top topic topple torch tornado tortoise "
        "toss total tourist toward tower town toy track trade traffic tragic train transfer trap trash "
        "travel tray treat tree trend trial tribe trick trigger trim trip trophy trouble truck true "
        "truly trumpet trust truth try tube tuition tumble tuna tunnel turkey turn turtle twelve twenty "
        "twice twin twist two type typical ugly umbrella unable unaware uncle uncover under undo unfair "
        "unfold unhappy uniform unique unit universe unknown unlock until unusual unveil update upgrade "
        "uphold upon upper upset urban urge usage use used useful useless usual utility vacant vacuum "
        "vague valid valley valve van vanish vapor various vast vault vehicle velvet vendor venture "
        "venue verb verify version very vessel veteran viable vibrant vicious victory video view village "
        "vintage violin virtual virus visa visit visual vital vivid vocal voice void volcano volume vote "
        "voyage wage wagon wait walk wall walnut want warfare warm warrior wash wasp waste water wave "
        "way wealth weapon wear weasel weather web wedding weekend weird welcome west wet whale what "
        "wheat wheel when where whip whisper wide width wife wild will win window wine wing wink winner "
        "winter wire wisdom wise wish witness wolf woman wonder wood wool word work world worry worth "
        "wrap wreck wrestle wrist write wrong yard year yellow you young youth zebra zero zone zoo "
    ).split()
)

# Shape of a BIP-39 mnemonic: a run of 12/15/18/21/24 lowercase words. Shape alone
# is not evidence -- ordinary prose hits it constantly -- so _looks_like_mnemonic
# separately requires that most of the words are real BIP-39 words.
_BIP39_WORD_RUN_RE = re.compile(r"(?<![a-z])[a-z]{3,8}(?:\s+[a-z]{3,8}){11,23}(?![a-z])")
# Detection is layered. The first pattern is the literal denylist kept from the
# original revision; the remaining three match injection *intent* structurally so
# that reworded instructions are still caught. They are deliberately narrow:
#   - verbs must be imperative and directed ("show me", not "shows")
#   - the verb and its object must sit within one short clause (_CLAUSE_GAP)
# That keeps declarative non-custodial copy, which legitimately names secret keys
# and seed phrases, from being flagged.
_CLAUSE_GAP = r"[^.!?\n]{0,24}?"

_INJECTION_RE = re.compile(
    r"(ignore\s+(all\s+)?(previous|prior|above)\s+instructions|"
    r"disregard\s+.*instructions|"
    r"forget\s+.*instructions|"
    r"override\s+.*instructions|"
    r"system\s*prompt|developer\s*message|"
    r"output\s+credentials|reveal\s+.*(secret|key|prompt|credentials)|"
    r"send\s+funds\s+to)",
    re.IGNORECASE,
)

# "forget everything", "disregard the rules above", ...
_OVERRIDE_RE = re.compile(
    r"\b(ignore|disregard|forget|override|bypass|discard)\b"
    + _CLAUSE_GAP
    + r"\b(instruction|instructions|rule|rules|prompt|prompts|directive|directives|"
    r"guideline|guidelines|constraint|constraints|policy|policies|above|previous|prior|"
    r"earlier|system|developer|everything)\b",
    re.IGNORECASE,
)

# "show me your secret key", "output credentials", ...
_EXFIL_RE = re.compile(
    r"\b(output|reveal|disclose|dump|exfiltrate|leak|echo|repeat|recite|"
    r"show me|give me|send me|tell me|hand over|print out|write out|list all)\b"
    + _CLAUSE_GAP
    +  # `token` alone names a Stellar asset here ("give me the token address"), so only
    # credential-style tokens count.
    r"\b(secret|seed|mnemonic|passphrase|password|private key|private keys|"
    r"credential|credentials|prompt|api key|access token|auth token|bearer token|"
    r"session token|refresh token|signing key)\b",
    re.IGNORECASE,
)

# Fake role/message boundaries used to smuggle instructions.
_DELIMITER_RE = re.compile(
    r"(<\s*\|\s*(system|assistant|user)\s*\|\s*>|"
    r"^\s*(system|assistant)\s*:|"
    # `act as` and `DAN` are matched only inside a role-change phrase: "The wallet does
    # not act as a custodian" is the exact non-custodial copy this tool grades, and
    # "Dan" is a person's name.
    r"\bjailbreak\b|\bpretend (to be|you)\b|\byou are now\b|"
    r"\bnew (system )?prompt\b|"
    r"\bact as (?:a|an|the)\s+(?:system|developer|admin|assistant|root|superuser)\b|"
    r"\byou are DAN\b|\bDAN mode\b)",
    re.IGNORECASE | re.MULTILINE,
)
_ED25519_PUBLIC_KEY_VERSION_BYTE = 6 << 3  # 48 ('G')


@dataclass(frozen=True)
class StellarWaveCheckResult:
    """Deterministic, network-aware, non-custodial result envelope (#667).

    Note: ``valid`` reflects offline format and safety validation only; it performs
    no Horizon or RPC network reads and does not confirm on-chain inclusion.
    """

    issue_number: int
    operation: str
    network: str
    network_passphrase: str
    valid: bool
    non_custodial: bool
    summary: str
    details: dict[str, Any]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _crc16_xmodem(data: bytes) -> int:
    crc = 0x0000
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def is_valid_stellar_account_id(value: str) -> bool:
    """Validate a Stellar SEP-0023 ``G...`` Ed25519 public key including CRC16-XMODEM checksum."""
    candidate = value.strip()
    if not _PUBLIC_KEY_RE.match(candidate):
        return False
    try:
        raw = base64.b32decode(candidate, casefold=False)
    except (binascii.Error, ValueError):
        return False
    if len(raw) != 35 or raw[0] != _ED25519_PUBLIC_KEY_VERSION_BYTE:
        return False
    canonical = base64.b32encode(raw).decode("ascii").rstrip("=")
    if canonical != candidate:
        return False
    expected_crc = _crc16_xmodem(raw[:33]).to_bytes(2, "little")
    return raw[33:] == expected_crc


# BIP-39 defines phrases of exactly these many words.
MNEMONIC_LENGTHS = (12, 15, 18, 21, 24)


def _looks_like_mnemonic(text: str) -> bool:
    """True when any 12/15/18/21/24-word window is almost entirely BIP-39 words.

    Word count alone is far too weak: ordinary sentences hit it constantly, and this
    evaluator exists to grade non-custodial copy, which is prose. Requiring that all
    but at most one word of a window come from the BIP-39 list keeps real recovery
    phrases while letting normal sentences through.

    The window must slide. Scoring a whole run in one go lets a phrase hide inside
    padding: "stellar wallet <12-word phrase>" matches as a single 14-word run whose
    two leading words are not in the list, and the phrase slips through.
    """
    for match in _BIP39_WORD_RUN_RE.finditer(text):
        run = match.group(0).lower().split()
        for length in MNEMONIC_LENGTHS:
            if length > len(run):
                continue
            for start in range(len(run) - length + 1):
                window = run[start : start + length]
                hits = sum(1 for w in window if w in _BIP39_WORDLIST)
                # allow one non-list word, so a phrase with one typo still counts
                if hits >= length - 1:
                    return True
    return False


def contains_secret_material(text: str) -> bool:
    """Return True if text contains a Stellar S-seed, recovery label, or 12-24 word mnemonic."""
    return bool(_SECRET_KEY_RE.search(text) or _SEED_PHRASE_RE.search(text) or _looks_like_mnemonic(text))


def looks_like_injection(text: str) -> bool:
    """Return True if untrusted text reads as an attempt to steer the consumer.

    Heuristic by nature: it is a guard rail, not a proof of safety. The security
    property does not rest on it, because no untrusted text is ever echoed back --
    see ``evaluate_arabic_stellar_terminology``.
    """
    return bool(
        _INJECTION_RE.search(text) or _OVERRIDE_RE.search(text) or _EXFIL_RE.search(text) or _DELIMITER_RE.search(text)
    )


def resolve_network(network: str) -> tuple[str, str]:
    """Normalize network identifier and return (network_name, canonical_passphrase)."""
    cleaned = network.strip().lower()
    if cleaned in ("public", "mainnet", "pubnet"):
        return ("public", PUBLIC_PASSPHRASE)
    if cleaned == "testnet":
        return ("testnet", TESTNET_PASSPHRASE)
    raise ValueError("Unsupported Stellar network; expected 'public' or 'testnet'")


def format_stroops_to_decimal(stroops: int) -> str:
    """Convert bounded int64 stroops to a 7-decimal Stellar amount string."""
    if isinstance(stroops, bool) or not isinstance(stroops, int) or stroops < 0 or stroops > MAX_INT64_STROOPS:
        raise ValueError("stroops must be a bounded non-negative int64 integer")
    try:
        val = (Decimal(stroops) / STROOPS_PER_UNIT).quantize(Decimal("0.0000001"), rounding=ROUND_HALF_UP)
    except InvalidOperation as exc:
        raise ValueError("stroops exceeds supported decimal precision") from exc
    return f"{val:.7f}"


def evaluate_arabic_stellar_terminology(
    *,
    network: str = "testnet",
    payload: str = "",
    amount_stroops: int = 10_000_000,
    asset_code: str = "USDC",
    issuer: str = "GBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5",
    ledger_sequence: int = 1_024,
    tx_hash: str = "a" * 64,
) -> StellarWaveCheckResult:
    """Execute offline Arabic Stellar terminology check (#667); ``valid`` checks local format only."""
    # Each field is scanned on its own. Joining them first let the field names and
    # defaults ("testnet", "USDC") contribute words to the mnemonic run, so a payload
    # of only ten ordinary words could be refused.
    for field in (network, payload, asset_code, issuer, tx_hash, amount_stroops, ledger_sequence):
        if contains_secret_material(str(field)):
            raise ValueError("Refusing secret keys or seed phrases in non-custodial Stellar tool")

    net_name, passphrase = resolve_network(network)
    warnings: list[str] = []

    printable_payload = "".join(ch for ch in str(payload) if ch.isprintable()).strip()[:240]

    # Untrusted text is never echoed. Any payload collapses to a one-way digest that
    # still supports correlation but cannot carry attacker-chosen wording, so the
    # sanitisation guarantee does not depend on the detector above being complete.
    if printable_payload:
        digest = hashlib.sha256(printable_payload.encode("utf-8")).hexdigest()[:16]
        clean_payload = f"[UNTRUSTED_INPUT_REDACTED sha256:{digest}]"
    else:
        clean_payload = ""

    # Only the payload is untrusted prose. An asset code that contains whitespace
    # already fails _ASSET_CODE_RE and gets invalid_asset_code, and scanning it here
    # both rejected legal codes (DAN) and left an unbounded input on a pattern with
    # .* quantifiers.
    if looks_like_injection(printable_payload):
        warnings.append("untrusted_metadata_injection_blocked")

    raw_code = str(asset_code).strip()
    code_ok = bool(_ASSET_CODE_RE.match(raw_code))
    norm_code = raw_code if code_ok else "INVALID"
    if not code_ok:
        warnings.append("invalid_asset_code")

    is_native = code_ok and norm_code == "XLM"
    issuer_ok = True if is_native else is_valid_stellar_account_id(str(issuer))
    if not issuer_ok:
        warnings.append("invalid_issuer_public_key")

    norm_hash = str(tx_hash).strip().lower()
    hash_ok = bool(_TX_HASH_RE.match(norm_hash))
    if not hash_ok:
        warnings.append("invalid_transaction_hash")

    ledger_ok = isinstance(ledger_sequence, int) and not isinstance(ledger_sequence, bool) and ledger_sequence > 0
    safe_ledger_sequence = ledger_sequence if ledger_ok else 0
    if not ledger_ok:
        warnings.append("invalid_ledger_sequence")

    try:
        formatted_amount = format_stroops_to_decimal(amount_stroops)
        amount_ok = True
        safe_amount_stroops = amount_stroops
    except ValueError:
        formatted_amount = "0.0000000"
        amount_ok = False
        safe_amount_stroops = 0
        warnings.append("invalid_amount_stroops")

    if net_name == "testnet":
        warnings.append("testnet_environment_not_real_funds")

    valid = (
        code_ok
        and issuer_ok
        and hash_ok
        and ledger_ok
        and amount_ok
        and ("untrusted_metadata_injection_blocked" not in warnings)
    )

    details: dict[str, Any] = {
        "asset_code": norm_code if code_ok else "",
        "issuer": None if is_native else (issuer.strip() if issuer_ok else ""),
        "formatted_amount": f"{formatted_amount} {norm_code} ({net_name})",
        "amount_stroops": safe_amount_stroops,
        "ledger_sequence": safe_ledger_sequence,
        "tx_hash": norm_hash if hash_ok else "",
        "sanitized_payload": clean_payload,
        # False on every path: untrusted text is never returned, only a digest.
        "payload_echoed": False,
        "explorer_url": (
            f"https://stellar.expert/explorer/{'public' if net_name == 'public' else 'testnet'}/tx/{norm_hash}"
            if hash_ok
            else ""
        ),
        "arabic_term": "شبكة ستيلر غير الوصائية",
        "swahili_term": "Mtandao wa Stellar usioshikilia funguo za siri",
    }

    return StellarWaveCheckResult(
        issue_number=667,
        operation="evaluate_arabic_stellar_terminology",
        network=net_name,
        network_passphrase=passphrase,
        valid=valid,
        non_custodial=True,
        summary=(
            f"[{net_name.upper()}] Arabic Stellar terminology evaluator checking non-custodial "
            f"and network-explicit translations: {formatted_amount} {norm_code}"
        ),
        details=details,
        warnings=tuple(warnings),
    )
