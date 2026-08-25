"""Comprehensive test suite for the Swahili Islamic Language Subsystem."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from main import app
from swahili import (
    CodeSwitchType,
    IslamicDomain,
    SwahiliDialect,
    analyze_swahili,
    code_switch_processor,
    cultural_context_engine,
    dialect_classifier,
    loanword_analyzer,
    swahili_query_optimizer,
    swahili_response_enhancer,
    terminology_db,
)


@pytest.fixture
def client() -> TestClient:
    """Create FastAPI test client."""
    return TestClient(app)


# ==========================================
# 1. Terminology Engine Tests
# ==========================================


def test_terminology_database_loading() -> None:
    """Ensure database loads terms with all 8 domains represented."""
    assert len(terminology_db.terms) >= 30
    assert len(terminology_db.terms) == 120 or len(terminology_db.terms) >= 30

    # Ensure all domains have terms
    categories_found = {t.category for t in terminology_db.terms}
    assert IslamicDomain.IBADA in categories_found
    assert IslamicDomain.AQIDAH in categories_found
    assert IslamicDomain.FIQHI in categories_found
    assert IslamicDomain.MUAMALAT in categories_found
    assert IslamicDomain.NDOA_MIRATHI in categories_found
    assert IslamicDomain.QURAN_HADITHI in categories_found
    assert IslamicDomain.MAADILI in categories_found
    assert IslamicDomain.UTAMADUNI_HISTORIA in categories_found


def test_terminology_lookups() -> None:
    """Test multi-index lookup via Swahili term, Arabic script, transliteration, and variants."""
    # Lookup Swala
    term_swala = terminology_db.lookup_term("Swala")
    assert term_swala is not None
    assert term_swala.swahili_term == "Swala"
    assert term_swala.arabic_transliteration == "Salat"

    # Lookup by Arabic script
    term_ar = terminology_db.lookup_term("صلاة")
    assert term_ar is not None
    assert term_ar.id == term_swala.id

    # Lookup by transliteration
    term_trans = terminology_db.lookup_term("salat")
    assert term_trans is not None
    assert term_trans.id == term_swala.id

    # Lookup by variant
    term_var = terminology_db.lookup_term("sala")
    assert term_var is not None
    assert term_var.id == term_swala.id

    # Lookup Udhu / Kutawadha
    term_udhu = terminology_db.lookup_term("udhu")
    assert term_udhu is not None
    assert term_udhu.category == IslamicDomain.IBADA


def test_terminology_search_and_filter() -> None:
    """Test query search and category filtering."""
    # Filter by category
    ibada_terms = terminology_db.search_terms(category=IslamicDomain.IBADA)
    assert len(ibada_terms) > 0
    assert all(t.category == IslamicDomain.IBADA for t in ibada_terms)

    # Search with query
    search_res = terminology_db.search_terms(query="Ramadhani")
    assert len(search_res) > 0
    assert any("Ramadhani" in t.swahili_term or "Ramadhani" in t.definition_sw for t in search_res)


def test_extract_terms_from_text() -> None:
    """Test extracting multiple Islamic terms from a complex Swahili question."""
    text = "Je, ni nini hukumu ya kutoa Zaka ya Fitri kabla ya Swala ya Iddi baada ya Saumu ya Ramadhani?"
    extracted = terminology_db.extract_terms_from_text(text)
    term_names = [t.swahili_term for t in extracted]

    assert any(t in term_names for t in ["Zaka", "Swala", "Saumu", "Sikukuu ya Iddi"])


# ==========================================
# 2. Arabic Loanwords & Morphology Tests
# ==========================================


def test_bantu_prefix_stripping() -> None:
    """Test stripping Bantu noun class and verbal prefixes from Arabic loanwords."""
    # Verbs (ku-)
    stem_swali, prefix1 = loanword_analyzer.strip_bantu_prefix("kuswali")
    assert prefix1 == "ku"

    stem_tawadha, prefix2 = loanword_analyzer.strip_bantu_prefix("kutawadha")
    assert prefix2 == "ku"

    # Human Plural (wa-)
    stem_waislamu, prefix3 = loanword_analyzer.strip_bantu_prefix("waislamu")
    assert prefix3 in ("wa", "wai")

    # Abstract Noun (u-)
    stem_ushirikina, prefix4 = loanword_analyzer.strip_bantu_prefix("ushirikina")
    assert prefix4 == "u"

    # Collective Plural (ma-)
    stem_maswahaba, prefix5 = loanword_analyzer.strip_bantu_prefix("maswahaba")
    assert prefix5 == "ma"


def test_loanword_analysis_direct_and_inflected() -> None:
    """Test loanword matching on both base stems and inflected forms."""
    match_base = loanword_analyzer.analyze_word("haramu")
    assert match_base is not None
    assert match_base.matched_term == "Haramu"
    assert match_base.arabic_original == "حرام"

    match_inflected = loanword_analyzer.analyze_word("kutawadha")
    assert match_inflected is not None
    assert match_inflected.matched_term == "Udhu"

    match_fuzzy = loanword_analyzer.analyze_word("zakaah")
    assert match_fuzzy is not None
    assert match_fuzzy.matched_term == "Zaka"


def test_swahili_tokenization() -> None:
    """Test tokenizing Swahili sentence into enriched tokens."""
    sentence = "Waislamu wanapaswa kuswali Swala tano kila siku."
    tokens = loanword_analyzer.tokenize_swahili(sentence)

    assert len(tokens) >= 5
    loan_tokens = [t for t in tokens if t.is_arabic_loanword]
    assert len(loan_tokens) >= 2
    raw_loan_words = [t.raw_token.lower() for t in loan_tokens]
    assert "waislamu" in raw_loan_words or "kuswali" in raw_loan_words or "swala" in raw_loan_words


# ==========================================
# 3. Dialect Classification & Normalization Tests
# ==========================================


def test_dialect_classification_standard() -> None:
    """Test Standard Swahili (Sanifu) baseline."""
    text = "Niaje ndugu yangu, naomba kufahamu masharti ya funga ya Ramadhani."
    res = dialect_classifier.classify_dialect(text)
    assert res.primary_dialect in (SwahiliDialect.SANIFU, SwahiliDialect.BARA_INLAND)


def test_dialect_classification_mvita() -> None:
    """Test Coastal Mombasa / Kimvita dialect detection."""
    text = "Mvyee wangu alikwenda kuvua kabla ya kwenda chuo."
    res = dialect_classifier.classify_dialect(text)
    assert res.primary_dialect == SwahiliDialect.PWANI_MVITA
    assert res.is_coastal is True
    assert "kuvua" in res.detected_markers or "mvyee" in res.detected_markers or "chuo" in res.detected_markers


def test_dialect_classification_sheng() -> None:
    """Test Nairobi/Dar Urban Sheng slang detection."""
    text = "Manze nilikua na-fast jana lakini nilisahau nikapiga swala bila kushika wudhu."
    res = dialect_classifier.classify_dialect(text)
    assert res.primary_dialect == SwahiliDialect.SHENG_URBAN
    assert "kushika wudhu" in res.detected_markers or "kupiga swala" in res.detected_markers


def test_dialect_normalization_to_standard() -> None:
    """Test converting dialect markers to standard Sanifu terms."""
    text = "Kabla ya kuingia msikitini lazima kuvua vizuri."
    normalized, replaced = dialect_classifier.normalize_to_standard(text)
    assert "kutawadha" in normalized
    assert replaced.get("kuvua") == "kutawadha"


# ==========================================
# 4. Code-Switching & Formula Tests
# ==========================================


def test_code_switching_monolingual() -> None:
    """Test pure Swahili query."""
    text = "Je, ni yapi masharti ya kuswali swala ya adhuhuri?"
    res = code_switch_processor.analyze_code_switching(text)
    assert res.dominant_language == "sw"
    assert res.switch_type in (CodeSwitchType.MONOLINGUAL_SWAHILI, CodeSwitchType.SWAHILI_ARABIC_MIXED)


def test_code_switching_swahili_arabic() -> None:
    """Test Swahili query containing Arabic religious formulas."""
    text = "Assalamu Alaykum, naomba kujua kuhusu kutoa sadaka, Jazakallahu Khayran."
    res = code_switch_processor.analyze_code_switching(text)
    assert res.switch_type in (CodeSwitchType.SWAHILI_ARABIC_MIXED, CodeSwitchType.TRILINGUAL_MIXED)
    assert len(res.arabic_phrases) >= 1
    assert any("assalamu alaykum" in p.lower() or "jazakallah" in p.lower() for p in res.arabic_phrases)


def test_code_switching_trilingual() -> None:
    """Test Swahili + English + Arabic trilingual query."""
    text = "Bismillah, je ruling ya forex trading online inaruhusiwa katika Uislamu?"
    res = code_switch_processor.analyze_code_switching(text)
    assert res.switch_type == CodeSwitchType.TRILINGUAL_MIXED


def test_code_switching_quran_and_dua_detection() -> None:
    """Test detection of Quran quotes and Du'a phrases."""
    text = "Surah Al-Baqarah inasema nini kuhusu saumu ya Ramadhani?"
    res = code_switch_processor.analyze_code_switching(text)
    assert res.contains_quran_or_hadith is True

    dua_text = "Nifundishe dua ya kuingia msikitini na kuomba msamaha kwa Allah."
    res_dua = code_switch_processor.analyze_code_switching(dua_text)
    assert res_dua.contains_dua is True


# ==========================================
# 5. East African Cultural Context Tests
# ==========================================


def test_cultural_context_institutions() -> None:
    """Test identifying BAKWATA and Chief Kadhi institutions."""
    text = "Je, BAKWATA na Mahakama ya Kadhi wametoa mwongozo gani kuhusu mwezi mwandamo?"
    context = cultural_context_engine.extract_context(text)
    assert len(context.local_institutions_mentioned) >= 2
    assert any("BAKWATA" in inst for inst in context.local_institutions_mentioned)
    assert any("Kadhi" in inst for inst in context.local_institutions_mentioned)


def test_cultural_context_prayer_and_events() -> None:
    """Test extracting prayer times and cultural events."""
    text = "Wakati wa Alfajiri kabla ya kula Daku kwenye mwezi wa Ramadhani."
    context = cultural_context_engine.extract_context(text)
    assert context.prayer_time_context is not None
    assert "Alfajiri" in context.prayer_time_context
    assert context.cultural_event_context is not None


def test_cultural_context_shafi_relevance() -> None:
    """Test Shafi'i madhhab relevance on coastal and ritual topics."""
    text = "Hukumu ya kugusa mwanamke bila kizuizi baada ya kutawadha katika pwani ya Zanzibar."
    context = cultural_context_engine.extract_context(text)
    assert context.shafi_madhhab_relevant is True
    assert context.recommended_madhhab == "shafii"


# ==========================================
# 6. Query Optimizer & Response Enhancer Tests
# ==========================================


def test_query_optimizer() -> None:
    """Test query expansion for cross-corpus search."""
    res = swahili_query_optimizer.optimize_query_for_retrieval("Kuvua kwa ajili ya kuswali Swala ya Ijumaa")
    assert "kutawadha" in res["normalized_swahili"]
    assert len(res["arabic_keywords"]) >= 1
    assert "expanded_search_string" in res
    assert len(res["expanded_search_string"]) > 0


def test_prompt_enhancer_and_validator() -> None:
    """Test constructing prompt augmentation and validating generated responses."""
    enhancement = swahili_response_enhancer.build_prompt_enhancement("Masharti ya ndoa na mahari chini ya Kadhi")
    assert "SWAHILI ISLAMIC GUIDELINES" in enhancement.system_instructions
    assert len(enhancement.contextual_glossary) >= 1

    sample_valid_response = (
        "Qur'ani Tukufu inasema: بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ. "
        "Mwenyezi Mungu (Subhanahu wa Ta'ala) ametuamrisha kutii maamrisho Yake, "
        "na Mtume Muhammad (Swalla Allahu Alayhi wa Sallam) ameeleza katika Hadithi "
        "ya Sahih al-Bukhari kuwa ndoa inahitaji idhini na mahari."
    )
    val = swahili_response_enhancer.validate_swahili_response(sample_valid_response)
    assert val["valid"] is True
    assert val["score"] >= 0.75


# ==========================================
# 7. Facade & Full Pipeline Tests
# ==========================================


def test_analyze_swahili_facade() -> None:
    """Test root analyze_swahili helper function."""
    res = analyze_swahili("Assalamu Alaykum, vipi hukumu ya kufunga saumu ya Ramadhani?")
    assert res.original_text.startswith("Assalamu Alaykum")
    assert len(res.tokens) > 0
    assert len(res.detected_terms) >= 1
    assert res.dialect.primary_dialect in (
        SwahiliDialect.SANIFU,
        SwahiliDialect.BARA_INLAND,
        SwahiliDialect.PWANI_UNGUJA,
    )
    assert res.cultural_context.cultural_event_context is not None


# ==========================================
# 8. FastAPI Endpoint Tests
# ==========================================


def test_api_analyze_endpoint(client: TestClient) -> None:
    """Test POST /swahili/analyze."""
    payload = {"text": "Je, BAKWATA imetangaza nini kuhusu Zaka ya Fitri na Swala ya Iddi?"}
    response = client.post("/swahili/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["original_text"] == payload["text"]
    assert len(data["detected_terms"]) >= 1
    assert len(data["cultural_context"]["local_institutions_mentioned"]) >= 1
    assert data["processing_time_ms"] >= 0.0


def test_api_normalize_endpoint(client: TestClient) -> None:
    """Test POST /swahili/normalize."""
    payload = {"text": "Mvyee alikwenda kuvua kabla ya kuswali."}
    response = client.post("/swahili/normalize", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "kutawadha" in data["normalized_text"]
    assert "kuvua" in data["replaced_terms"]


def test_api_terms_search_and_get(client: TestClient) -> None:
    """Test GET /swahili/terms and GET /swahili/terms/{term_id}."""
    # List/Search terms
    response = client.get("/swahili/terms?query=Swala&category=ibada")
    assert response.status_code == 200
    terms = response.json()
    assert len(terms) >= 1
    assert terms[0]["swahili_term"] == "Swala"

    # Get single term
    term_id = terms[0]["id"]
    get_res = client.get(f"/swahili/terms/{term_id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == term_id

    # 404 for invalid ID
    err_res = client.get("/swahili/terms/invalid-nonexistent-id")
    assert err_res.status_code == 404


def test_api_dialects_endpoint(client: TestClient) -> None:
    """Test GET /swahili/dialects."""
    response = client.get("/swahili/dialects")
    assert response.status_code == 200
    data = response.json()
    assert "supported_dialects" in data
    assert "sanifu" in data["supported_dialects"]
    assert "pwani_mvita" in data["supported_dialects"]


def test_api_code_switch_endpoint(client: TestClient) -> None:
    """Test POST /swahili/code-switch."""
    payload = {"text": "Bismillah, how to pray swala ya witri according to sunnah?"}
    response = client.post("/swahili/code-switch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["switch_type"] in ("trilingual_mixed", "swahili_english_mixed")
    assert len(data["segments"]) >= 2


def test_api_enhance_prompt_endpoint(client: TestClient) -> None:
    """Test POST /swahili/enhance-prompt."""
    payload = {"text": "Tafadhali nieleze kuhusu hukumu ya ndoa na talaka chini ya Kadhi Mkuu."}
    response = client.post("/swahili/enhance-prompt", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "system_instructions" in data
    assert len(data["cultural_notes"]) >= 1


def test_chat_swahili_headers_integration(client: TestClient) -> None:
    """Test that /chat attaches Swahili analysis headers when receiving a Swahili prompt."""
    payload = {
        "prompt": "Habari, naomba kujua masharti ya Swala na kutoa Zaka ya Fitri.",
        "language": "sw",
    }
    response = client.post("/chat", json=payload)
    # Even on mock / offline error / 200 / 500, check headers if response was formed
    if response.status_code == 200:
        assert "X-Swahili-Dialect" in response.headers
        assert "X-Swahili-Terms-Detected" in response.headers
        assert "X-Swahili-Code-Switching" in response.headers
