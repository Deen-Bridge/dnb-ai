"""Offline tests for Arabic–English cross-lingual retrieval (#232).

No network anywhere: the autouse fixture blanks GEMINI_API_KEY (a real key on
a dev machine must never turn these into live calls) and pins the offline
embedder, and every Gemini-dependent code path is either unconfigured or
injected with fakes.
"""

import numpy as np
import pytest

import crosslingual as xl
from crosslingual import (
    CrosslingualIndex,
    Document,
    HashingCrossScriptEmbedder,
    crosslingual_search,
    detect_script,
    gemini_configured,
    get_glossary,
    normalize_arabic_token,
    normalize_english_text,
    translate_query,
)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """Force fully offline behavior regardless of the host environment."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("CROSS_LINGUAL_EMBEDDER", "hashing")


class FakeEmbedder:
    """Two-dim toy space: [arabic-script mass, ascii-letter mass]."""

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(2, dtype=np.float32)
        vec[0] = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF")
        vec[1] = sum(1 for ch in text if ch.isascii() and ch.isalpha())
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


class TestDetectScript:
    def test_pure_arabic(self):
        det = detect_script("ما حكم الصلاة للمسافر")
        assert det.lang == "ar"
        assert all(t.lang == "ar" for t in det.token_langs)

    def test_pure_english(self):
        det = detect_script("How do I pray fajr while travelling?")
        assert det.lang == "en"
        assert all(t.lang == "en" for t in det.token_langs)

    def test_code_switched_mixed(self):
        det = detect_script("كيف أؤدي صلاة الفجر أثناء travel إلى another city")
        assert det.lang == "mixed"
        langs = {t.lang for t in det.token_langs}
        assert {"ar", "en"} <= langs

    def test_single_english_word_in_arabic_is_mixed(self):
        assert detect_script("ما حكم salah").lang == "mixed"

    def test_token_count_matches(self):
        det = detect_script("زكاة الذهب والفضة")
        assert len(det.token_langs) == 3
        assert [t.token for t in det.token_langs] == ["زكاة", "الذهب", "والفضة"]

    def test_letterless_falls_back_to_en(self):
        det = detect_script("123 ?!")
        assert det.lang == "en"
        assert all(t.lang is None for t in det.token_langs)

    def test_empty_string_is_en_default(self):
        assert detect_script("").lang == "en"

    def test_arabic_indic_digits_are_arabic_script(self):
        assert detect_script("٤٥").lang == "ar"


# ---------------------------------------------------------------------------
# Glossary loading and lookups across spelling variants
# ---------------------------------------------------------------------------

VARIANT_CASES = [
    ("salah", "صلاة"),
    ("salaat", "صلاة"),
    ("salat", "صلاة"),
    ("namaz", "صلاة"),
    ("zakat", "زكاة"),
    ("zakaah", "زكاة"),
    ("zekat", "زكاة"),
    ("tafseer", "تفسير"),
    ("hadeeth", "حديث"),
    ("hadith", "حديث"),
    ("tauheed", "توحيد"),
    ("wudhu", "وضوء"),
    ("riba", "ربا"),
]


class TestGlossary:
    @pytest.mark.parametrize(("surface", "expected_ar"), VARIANT_CASES)
    def test_variant_lookup(self, surface, expected_ar):
        term = get_glossary().lookup(surface)
        assert term is not None
        assert term.ar == expected_ar

    def test_case_and_diacritic_insensitive(self):
        term = get_glossary().lookup("Salât")
        assert term is not None
        assert term.id == "salah"

    def test_apostrophe_optional(self):
        assert get_glossary().lookup("bidah") is not None
        assert get_glossary().lookup("bid'ah").id == get_glossary().lookup("bidah").id

    def test_unknown_surface_returns_none(self):
        assert get_glossary().lookup("flibbertigibbet") is None

    def test_glossary_has_substantial_coverage(self):
        assert len(get_glossary()) >= 80

    def test_phrase_lookup_surah_al_fatiha(self):
        matches = get_glossary().find_terms("What does surah al-fatiha mean?")
        by_id = {m.term_id: m for m in matches}
        assert "al_fatihah" in by_id
        assert by_id["al_fatihah"].canonical_ar == "سورة الفاتحة"
        # The longer phrase wins over its bare-name substring.
        fatihah_match = next(m for m in matches if m.matched_text.endswith("fatiha"))
        assert "surah" in fatihah_match.matched_text

    def test_multiple_terms_found_in_one_query(self):
        matches = get_glossary().find_terms("Can I delay salaat and combine zakaah payments?")
        ids = {m.term_id for m in matches}
        assert {"salah", "zakat"} <= ids

    def test_arabic_text_matches_canonical_term(self):
        matches = get_glossary().find_terms("أهمية الصلاة في الإسلام")
        assert any(m.term_id == "salah" for m in matches)

    def test_matches_are_non_overlapping_and_ordered(self):
        matches = get_glossary().find_terms("surah al-fatiha recitation")
        spans = [(m.start, m.end) for m in matches]
        assert spans == sorted(spans)
        for (_, end1), (start2, _) in zip(spans, spans[1:]):
            assert end1 <= start2

    def test_word_boundary_prevents_substring_hits(self):
        # "nas" (An-Nas) must not fire inside an unrelated English word.
        assert get_glossary().find_terms("unstable nasality samples") == []


# ---------------------------------------------------------------------------
# Morphology normalizers (hand-written expectations)
# ---------------------------------------------------------------------------


class TestNormalizers:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("بالصلاة", "صلاه"),  # بـ + ال stripped, teh marbuta folded
            ("والقرآن", "قران"),  # و + ال stripped, alef variants folded
            ("ٱلرَّحِيمِ", "رحيم"),  # marks stripped, alef wasla folded, ال stripped
            ("في", "في"),  # stripping ف would leave one letter: kept intact
            ("بِوَجْهِ", "وجه"),  # marks stripped; single-clitic strip refused
            ("وضوء", "وضوء"),  # nothing to strip
            ("كتابُهم", "كتابهم"),  # harakat only
        ],
    )
    def test_arabic_token(self, raw, expected):
        assert normalize_arabic_token(raw) == expected

    def test_double_prefix_stripped(self):
        assert normalize_arabic_token("فبالصدقة") == "صدقه"

    def test_english_tokens_lowercase_and_strip_punctuation(self):
        assert normalize_english_text("The ZAKAAT, was paid!") == ["the", "zakaat", "was", "paid"]

    def test_english_stopword_drop_is_opt_in(self):
        assert normalize_english_text("What is the Riba?", drop_stopwords=True) == ["what", "riba"]


# ---------------------------------------------------------------------------
# Translation adapter fallback flags
# ---------------------------------------------------------------------------


class TestTranslationFallback:
    async def test_en_to_ar_uses_glossary_without_key(self):
        result = await translate_query("What exactly is riba?")
        assert result.source_lang == "en"
        assert result.target_lang == "ar"
        assert result.translation_source == "glossary"
        assert "ربا" in result.translated_text
        assert any(p.canonical_ar == "ربا" for p in result.protected_terms)

    async def test_ar_to_en_substitutes_english_gloss(self):
        result = await translate_query("ما هي زكاة المال")
        assert result.translation_source == "glossary"
        assert "alms" in result.translated_text

    async def test_unmatched_query_passes_through_flagged_none(self):
        result = await translate_query("purple elephants dancing quietly")
        assert result.translation_source == "none"
        assert result.protected_terms == []
        assert result.translated_text == "purple elephants dancing quietly"

    async def test_same_language_passthrough(self):
        result = await translate_query("ما حكم الصلاة", target_lang="ar")
        assert result.translation_source == "none"
        assert result.target_lang == "ar"

    async def test_placeholder_keys_never_enable_gemini(self, monkeypatch):
        for key in ("dummy", "test-key", "your_api_key_here", ""):
            monkeypatch.setenv("GEMINI_API_KEY", key)
            assert gemini_configured() is False
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaRealLookingKey1234567890")
        assert gemini_configured() is True

    async def test_gemini_failure_degrades_to_glossary(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaRealLookingKey1234567890")

        def _boom(text, target, protections):
            raise TimeoutError("network unreachable")

        monkeypatch.setattr(xl, "_gemini_translate_sync", _boom)
        result = await translate_query("explain riba in loans")
        assert result.translation_source == "glossary"
        assert any("fallback" in note for note in result.notes)

    async def test_code_switched_query_mirrors_toward_non_dominant_script(self):
        result = await translate_query("ما حكم salah")
        assert result.source_lang == "mixed"
        # Arabic dominates, so the mirror target is English.
        assert result.target_lang == "en"


# ---------------------------------------------------------------------------
# Offline hashing embedder: determinism, L2 norm, cross-script affinity
# ---------------------------------------------------------------------------


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


class TestHashingEmbedder:
    def test_deterministic_across_calls(self):
        emb = HashingCrossScriptEmbedder(dim=128)
        v1 = emb.embed("Zakaah on gold savings")
        v2 = emb.embed("Zakaah on gold savings")
        assert np.array_equal(v1, v2)

    def test_output_l2_normalized(self):
        emb = HashingCrossScriptEmbedder(dim=128)
        vec = emb.embed("Explain the rules of ghusl after janabah")
        assert vec.shape == (128,)
        assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5

    def test_empty_input_gives_zero_vector(self):
        vec = HashingCrossScriptEmbedder(dim=64).embed("")
        assert np.isfinite(vec).all()
        assert float(np.linalg.norm(vec)) == 0.0

    def test_cross_script_affinity_via_glossary_bridge(self):
        emb = HashingCrossScriptEmbedder(dim=256)
        romanized = emb.embed("how many rakat in maghrib")
        arabic = emb.embed("صلاة المغرب")
        unrelated = emb.embed("quarterly market forecast report")
        assert cosine(romanized, arabic) > cosine(romanized, unrelated)
        assert cosine(romanized, arabic) > 0.05

    def test_different_inputs_differ(self):
        emb = HashingCrossScriptEmbedder(dim=128)
        assert not np.array_equal(emb.embed("hajj"), emb.embed("wudu"))


# ---------------------------------------------------------------------------
# Retrieval: lang_pref filtering and response contract
# ---------------------------------------------------------------------------

DOCS = [
    Document(doc_id="d-ar-only", ar="إن الله مع الصابرين"),
    Document(doc_id="d-en-only", en="Allah is with those who are patient"),
    Document(doc_id="d-bilingual", ar="وأقيموا الصلاة", en="Establish regular prayer"),
]


class TestLangPrefFiltering:
    async def test_pref_en_only_returns_english_sides(self):
        resp = await crosslingual_search(
            "establish prayer",
            k=5,
            lang_pref="en",
            documents=DOCS,
            embedder=FakeEmbedder(),
            translate=False,
        )
        assert resp.results_in == "en"
        assert resp.query_lang == "en"
        assert resp.results
        for hit in resp.results:
            assert hit.text_lang == "en"

    async def test_pref_ar_excludes_english_sides(self):
        resp = await crosslingual_search(
            "أقم الصلاة",
            k=5,
            lang_pref="ar",
            documents=DOCS,
            embedder=FakeEmbedder(),
            translate=False,
        )
        assert resp.results
        assert {hit.text_lang for hit in resp.results} == {"ar"}

    async def test_pref_any_can_return_both_scripts(self):
        resp = await crosslingual_search(
            "prayer صلاة",
            k=10,
            lang_pref="any",
            documents=DOCS,
            embedder=FakeEmbedder(),
            translate=False,
        )
        assert resp.query_lang == "mixed"
        scripts = {hit.text_lang for hit in resp.results}
        assert scripts == {"ar", "en"}

    async def test_invalid_lang_pref_rejected(self):
        with pytest.raises(ValueError):
            await crosslingual_search(
                "prayer", k=3, lang_pref="fr", documents=DOCS, embedder=FakeEmbedder()
            )

    async def test_hit_contract_fields(self):
        resp = await crosslingual_search(
            "salaat",
            k=2,
            lang_pref="any",
            documents=DOCS,
            embedder=HashingCrossScriptEmbedder(),
            translate=True,
        )
        assert resp.translation.translation_source in {"glossary", "none"}
        for hit in resp.results:
            assert hit.doc_id
            assert hit.text
            assert hit.score <= 1.0001
            if hit.mirrored_snippet is not None:
                assert isinstance(hit.mirrored_snippet, str)


class TestEquivalenceNotes:
    async def test_glossary_bridge_note_present(self):
        resp = await crosslingual_search(
            "salaat",
            k=3,
            lang_pref="ar",
            documents=[Document(doc_id="x", ar="حكم الصلاة", en=None)],
            embedder=HashingCrossScriptEmbedder(),
            translate=False,
        )
        assert resp.results, "romanized query should retrieve the Arabic side via bridge"
        joined = " ".join(note for hit in resp.results for note in hit.equivalence_notes)
        assert "صلاة" in joined


class TestDefaultCorpusLoading:
    def test_bundled_corpus_loads_bilingual_docs(self):
        docs = xl.load_default_documents()
        assert docs, "bundled quran_uthmani.json should yield documents"
        bilingual = [d for d in docs if d.ar and d.en]
        assert bilingual
        assert all(d.doc_id.startswith("quran:") for d in docs)

    def test_index_empty_collection_is_safe(self):
        index = CrosslingualIndex([], FakeEmbedder())
        vec = FakeEmbedder().embed("anything")
        assert index.search_vectors(vec, k=3) == []


# ---------------------------------------------------------------------------
# Route wiring (guarded: skips when heavy main.py deps are unavailable)
# ---------------------------------------------------------------------------


def test_route_registered(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    main = pytest.importorskip("main")
    paths = {getattr(route, "path", None) for route in main.app.routes}
    assert "/search/crosslingual" in paths


@pytest.mark.asyncio
async def test_endpoint_smoke_offline(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    main = pytest.importorskip("main")
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/search/crosslingual", json={"query": "salaat times", "k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["query_lang"] in {"ar", "en", "mixed"}
    assert body["results_in"] == "any"
    assert isinstance(body["results"], list)
    assert body["translation"]["translation_source"] in {"gemini", "glossary", "none"}
