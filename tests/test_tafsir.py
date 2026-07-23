"""Tests for the tafsir layer — reference validation, retrieval assembly,
attribution, graceful degradation, caching, and the chat integration.

Every test runs offline against ``FakeTafsirSource`` and the bundled surah
index; no live API calls and no GEMINI_API_KEY needed.
"""

import asyncio

import pytest

from semantic_cache import get_keyed_cache
from tafsir import (
    DEFAULT_TAFSIR_KEYS,
    MAX_AYAT_PER_REQUEST,
    TAFSIR_REGISTRY,
    AyahRef,
    FakeTafsirSource,
    InvalidReference,
    TafsirRequest,
    TafsirWork,
    VerseText,
    build_chat_tafsir_context,
    build_tafsir_prompt_block,
    build_tafsir_response,
    detect_ayah_references,
    fetch_tafsirs_for_ayah,
    load_surah_index,
    normalize_tafsir_key,
    parse_reference,
    parse_tafsir_payload,
    resolve_requested_tafsirs,
    strip_html,
    summarize_tafsir_context,
    surah_by_name,
    surah_by_number,
    tafsir_system_context,
    validate_reference,
)


@pytest.fixture(autouse=True)
def clear_tafsir_cache():
    """Each test starts with an empty tafsir cache."""
    get_keyed_cache("tafsir").clear()
    yield
    get_keyed_cache("tafsir").clear()


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Fixtures: canned source payloads
#
# Tafsir text is paraphrased placeholder prose, not the real works — these
# tests check plumbing and attribution, and bundling real tafsir text would
# raise licensing questions the repo deliberately avoids (data/quran/
# PROVENANCE.md). The payload *shape* mirrors the Quran.com API exactly.
# ---------------------------------------------------------------------------

IBN_KATHIR_103 = {
    "verses": {"103:1": {"id": 6177}, "103:2": {"id": 6178}, "103:3": {"id": 6179}},
    "resource_id": 169,
    "resource_name": "Ibn Kathir (Abridged)",
    "slug": "en-tafisr-ibn-kathir",
    "translated_name": {"name": "Ibn Kathir (Abridged)", "language_name": "english"},
    "text": "<h2>Al-'Asr</h2><p>Al-'Asr is the time in which the deeds of "
            "the children of Adam occur, whether good or bad.</p>",
}

SAADI_103 = {
    "verses": {"103:2": {"id": 6178}},
    "resource_id": 91,
    "resource_name": "Arabic Saddi Tafseer",
    "slug": "ar-tafseer-al-saddi",
    "translated_name": {"name": "السعدي Al-Sa'di", "language_name": "arabic"},
    "text": "<p>Allah swears by time, that mankind is at a loss except those "
            "who possess the four described qualities.</p>",
}

TABARI_103 = {
    "verses": {"103:2": {"id": 6178}},
    "resource_id": 15,
    "resource_name": "Tafsir al-Tabari",
    "slug": "ar-tafsir-al-tabari",
    "translated_name": {"name": "Tafsir al-Tabari", "language_name": "arabic"},
    "text": "<p>The people of interpretation differed over the meaning of "
            "al-'Asr: some held it to be the age of time itself, others the "
            "hour of the afternoon prayer.</p>",
}

EMPTY_QURTUBI_103 = {
    "verses": {"103:2": {"id": 6178}},
    "resource_id": 90,
    "resource_name": "Al-Qurtubi",
    "slug": "ar-tafseer-al-qurtubi",
    "translated_name": {"name": "Al-Qurtubi", "language_name": "arabic"},
    "text": "",
}


def make_source(**overrides) -> FakeTafsirSource:
    tafsirs = {
        ("en-tafisr-ibn-kathir", "103:2"): IBN_KATHIR_103,
        ("ar-tafseer-al-saddi", "103:2"): SAADI_103,
        ("ar-tafsir-al-tabari", "103:2"): TABARI_103,
    }
    tafsirs.update(overrides.pop("tafsirs", {}))
    verses = {
        "103:2": VerseText(
            arabic="إن الإنسان لفي خسر",
            translation="Indeed, mankind is in loss,",
            translation_language="en",
        ),
    }
    verses.update(overrides.pop("verses", {}))
    return FakeTafsirSource(tafsirs=tafsirs, verses=verses)


# ---------------------------------------------------------------------------
# Surah index
# ---------------------------------------------------------------------------


class TestSurahIndex:
    def test_index_has_114_surahs(self):
        assert len(load_surah_index()) == 114

    def test_total_ayah_count_is_kufan(self):
        assert sum(s.ayah_count for s in load_surah_index()) == 6236

    @pytest.mark.parametrize("number,name,count", [
        (1, "Al-Fatihah", 7),
        (2, "Al-Baqarah", 286),
        (9, "At-Tawbah", 129),
        (103, "Al-'Asr", 3),
        (114, "An-Nas", 6),
    ])
    def test_known_surahs(self, number, name, count):
        surah = surah_by_number(number)
        assert surah is not None
        assert surah.name == name
        assert surah.ayah_count == count

    @pytest.mark.parametrize("number", [0, -1, 115, 999])
    def test_out_of_range_number_returns_none(self, number):
        assert surah_by_number(number) is None

    @pytest.mark.parametrize("name,expected", [
        ("Al-'Asr", 103),
        ("al asr", 103),
        ("asr", 103),
        ("Surah al-Asr", 103),
        ("Al-Baqarah", 2),
        ("baqara", 2),
        ("Al-Kahf", 18),
        ("Ya-Sin", 36),
        ("yaseen", 36),
        ("An-Nas", 114),
        ("الفاتحة", 1),
    ])
    def test_lookup_by_name(self, name, expected):
        surah = surah_by_name(name)
        assert surah is not None and surah.number == expected

    def test_unknown_name_returns_none(self):
        assert surah_by_name("Al-Nonexistent") is None


# ---------------------------------------------------------------------------
# Reference validation
# ---------------------------------------------------------------------------


class TestReferenceValidation:
    def test_valid_reference(self):
        ref = validate_reference(103, 2)
        assert ref.key == "103:2"

    @pytest.mark.parametrize("surah", [0, 115, 200, -3])
    def test_surah_out_of_range(self, surah):
        with pytest.raises(InvalidReference, match="1 to 114"):
            validate_reference(surah, 1)

    def test_ayah_out_of_range_names_the_bound(self):
        with pytest.raises(InvalidReference) as exc:
            validate_reference(2, 300)
        message = str(exc.value)
        assert "286" in message
        assert "Al-Baqarah" in message

    @pytest.mark.parametrize("ayah", [0, -1, 4])
    def test_ayah_out_of_range_short_surah(self, ayah):
        with pytest.raises(InvalidReference):
            validate_reference(103, ayah)

    @pytest.mark.parametrize("raw,expected", [
        ("103:1", ["103:1"]),
        (" 103 : 1 ", ["103:1"]),
        ("103.1", ["103:1"]),
        ("103:1-3", ["103:1", "103:2", "103:3"]),
        ("103:1 to 3", ["103:1", "103:2", "103:3"]),
        ("Al-Asr 1-3", ["103:1", "103:2", "103:3"]),
        ("Surah al-Baqarah 255", ["2:255"]),
    ])
    def test_parse_reference(self, raw, expected):
        assert [ref.key for ref in parse_reference(raw)] == expected

    @pytest.mark.parametrize("raw", ["", "   ", "hello", "2:", ":255", "abc:def"])
    def test_unparseable_reference(self, raw):
        with pytest.raises(InvalidReference):
            parse_reference(raw)

    def test_out_of_range_range_end_rejected(self):
        with pytest.raises(InvalidReference, match="3 ayat"):
            parse_reference("103:1-9")

    def test_reversed_range_rejected(self):
        with pytest.raises(InvalidReference, match="before the first"):
            parse_reference("2:10-5")

    def test_oversized_range_rejected(self):
        with pytest.raises(InvalidReference, match="at most"):
            parse_reference(f"2:1-{MAX_AYAT_PER_REQUEST + 5}")


# ---------------------------------------------------------------------------
# Tafsir key normalization
# ---------------------------------------------------------------------------


class TestTafsirKeys:
    @pytest.mark.parametrize("raw,expected", [
        ("ibn-kathir", "ibn-kathir"),
        ("Ibn Kathir", "ibn-kathir"),
        ("ibnkathir", "ibn-kathir"),
        ("KATHIR", "ibn-kathir"),
        ("al-tabari", "tabari"),
        ("Sa'di", "saadi"),
        ("qurtubi", "qurtubi"),
        ("maariful-quran", "maarif-ul-quran"),
    ])
    def test_normalize(self, raw, expected):
        assert normalize_tafsir_key(raw) == expected

    @pytest.mark.parametrize("raw", ["", "   ", "not-a-tafsir", None])
    def test_unknown_returns_none(self, raw):
        assert normalize_tafsir_key(raw) is None

    def test_default_keys_are_registered(self):
        for key in DEFAULT_TAFSIR_KEYS:
            assert key in TAFSIR_REGISTRY

    def test_resolve_defaults(self):
        assert resolve_requested_tafsirs(None) == list(DEFAULT_TAFSIR_KEYS)
        assert resolve_requested_tafsirs([]) == list(DEFAULT_TAFSIR_KEYS)

    def test_resolve_deduplicates(self):
        assert resolve_requested_tafsirs(["ibn-kathir", "Ibn Kathir"]) == ["ibn-kathir"]

    def test_resolve_drops_unknown_but_keeps_known(self):
        assert resolve_requested_tafsirs(["ibn-kathir", "nonsense"]) == ["ibn-kathir"]

    def test_resolve_all_unknown_raises(self):
        with pytest.raises(InvalidReference, match="Unknown tafsir"):
            resolve_requested_tafsirs(["nonsense", "gibberish"])


# ---------------------------------------------------------------------------
# HTML flattening and payload parsing
# ---------------------------------------------------------------------------


class TestPayloadParsing:
    def test_strip_html_flattens_blocks(self):
        assert strip_html("<h2>Title</h2><p>One.</p><p>Two.</p>") == "Title\nOne.\nTwo."

    def test_strip_html_unescapes_entities(self):
        assert strip_html("<p>Al-&#39;Asr &amp; time</p>") == "Al-'Asr & time"

    def test_strip_html_empty(self):
        assert strip_html("") == ""

    def test_strip_html_drops_footnote_markers(self):
        raw = 'By time,<sup foot_note="82311">1</sup>'
        assert strip_html(raw) == "By time,"

    def test_language_label_follows_the_edition_not_the_payload(self):
        """al-Tabari's Arabic text is never labelled 'english'.

        The source's ``translated_name.language_name`` describes the language
        the work's *name* was translated into, so it says "english" for an
        English-locale request even when the commentary is Arabic.
        """
        payload = dict(
            TABARI_103,
            translated_name={"name": "Tafsir al-Tabari", "language_name": "english"},
        )
        parsed = parse_tafsir_payload(TAFSIR_REGISTRY["tabari"], "ar", payload)
        assert parsed.language == "arabic"

    def test_parse_payload_uses_source_attribution(self):
        work = TAFSIR_REGISTRY["ibn-kathir"]
        parsed = parse_tafsir_payload(work, "en", IBN_KATHIR_103)
        assert parsed is not None
        # The name is the source's own label for the resource.
        assert parsed.name == "Ibn Kathir (Abridged)"
        assert parsed.language == "english"
        assert parsed.author == work.author
        assert "children of Adam" in parsed.text
        assert "<p>" not in parsed.text

    def test_parse_payload_records_multi_ayah_range(self):
        parsed = parse_tafsir_payload(TAFSIR_REGISTRY["ibn-kathir"], "en", IBN_KATHIR_103)
        assert parsed.verse_range == "103:1-3"

    def test_parse_payload_single_ayah_range(self):
        parsed = parse_tafsir_payload(TAFSIR_REGISTRY["saadi"], "ar", SAADI_103)
        assert parsed.verse_range == "103:2"

    def test_parse_payload_empty_text_returns_none(self):
        assert parse_tafsir_payload(TAFSIR_REGISTRY["qurtubi"], "ar", EMPTY_QURTUBI_103) is None


# ---------------------------------------------------------------------------
# Retrieval and degradation
# ---------------------------------------------------------------------------


class TestFetchTafsirs:
    def test_returns_attributed_entries(self):
        source = make_source()
        available, unavailable = run(fetch_tafsirs_for_ayah(
            AyahRef(surah=103, ayah=2), ["ibn-kathir", "saadi"], "en", source=source
        ))
        assert [t.key for t in available] == ["ibn-kathir", "saadi"]
        assert all(t.name and t.author and t.language for t in available)
        assert unavailable == []

    def test_missing_entry_degrades_to_unavailable(self):
        source = make_source()
        available, unavailable = run(fetch_tafsirs_for_ayah(
            AyahRef(surah=103, ayah=2), ["ibn-kathir", "qurtubi"], "en", source=source
        ))
        assert [t.key for t in available] == ["ibn-kathir"]
        assert [u.key for u in unavailable] == ["qurtubi"]
        assert "103:2" in unavailable[0].reason

    def test_empty_text_degrades_to_unavailable(self):
        source = make_source(tafsirs={("ar-tafseer-al-qurtubi", "103:2"): EMPTY_QURTUBI_103})
        available, unavailable = run(fetch_tafsirs_for_ayah(
            AyahRef(surah=103, ayah=2), ["qurtubi"], "en", source=source
        ))
        assert available == []
        assert "no commentary text" in unavailable[0].reason

    def test_language_fallback_labels_actual_language(self):
        """al-Sa'di has no English edition; the Arabic text is labelled Arabic."""
        source = make_source()
        available, _ = run(fetch_tafsirs_for_ayah(
            AyahRef(surah=103, ayah=2), ["saadi"], "en", source=source
        ))
        assert available[0].language == "arabic"
        assert source.tafsir_calls == [("ar-tafseer-al-saddi", "103:2")]

    def test_language_fallback_disabled_marks_unavailable(self):
        source = make_source()
        available, unavailable = run(fetch_tafsirs_for_ayah(
            AyahRef(surah=103, ayah=2),
            ["saadi"],
            "en",
            allow_language_fallback=False,
            source=source,
        ))
        assert available == []
        assert "Not available in 'en'" in unavailable[0].reason
        assert source.tafsir_calls == []

    def test_unknown_key_is_skipped(self):
        source = make_source()
        available, unavailable = run(fetch_tafsirs_for_ayah(
            AyahRef(surah=103, ayah=2), ["not-a-real-tafsir"], "en", source=source
        ))
        assert available == [] and unavailable == []

    def test_second_lookup_is_served_from_cache(self):
        source = make_source()
        ref = AyahRef(surah=103, ayah=2)
        run(fetch_tafsirs_for_ayah(ref, ["ibn-kathir"], "en", source=source))
        run(fetch_tafsirs_for_ayah(ref, ["ibn-kathir"], "en", source=source))
        assert len(source.tafsir_calls) == 1
        assert get_keyed_cache("tafsir").hits >= 1

    def test_cache_keys_do_not_collide_across_ayat(self):
        source = make_source(tafsirs={("en-tafisr-ibn-kathir", "103:3"): IBN_KATHIR_103})
        run(fetch_tafsirs_for_ayah(AyahRef(surah=103, ayah=2), ["ibn-kathir"], "en", source=source))
        run(fetch_tafsirs_for_ayah(AyahRef(surah=103, ayah=3), ["ibn-kathir"], "en", source=source))
        assert len(source.tafsir_calls) == 2


# ---------------------------------------------------------------------------
# Response assembly
# ---------------------------------------------------------------------------


class TestBuildResponse:
    def test_single_ayah_response(self):
        source = make_source()
        response = run(build_tafsir_response(
            TafsirRequest(reference="103:2", tafsirs=["ibn-kathir", "saadi"]), source
        ))
        assert response.reference == "103:2"
        assert len(response.ayat) == 1
        ayah = response.ayat[0]
        assert ayah.ayah == "103:2"
        assert ayah.surah_name == "Al-'Asr"
        assert ayah.translation == "Indeed, mankind is in loss,"
        assert [t.key for t in ayah.tafsirs] == ["ibn-kathir", "saadi"]
        assert response.disclaimer

    def test_every_entry_is_attributed(self):
        source = make_source()
        response = run(build_tafsir_response(TafsirRequest(reference="103:2"), source))
        for tafsir in response.ayat[0].tafsirs:
            assert tafsir.name.strip()
            assert tafsir.author.strip()
            assert tafsir.language.strip()
            assert tafsir.text.strip()

    def test_diverging_tafsirs_are_both_surfaced(self):
        """al-Tabari reports a disagreement al-Sa'di does not — both are kept."""
        source = make_source()
        response = run(build_tafsir_response(
            TafsirRequest(reference="103:2", tafsirs=["tabari", "saadi"]), source
        ))
        texts = {t.key: t.text for t in response.ayat[0].tafsirs}
        assert set(texts) == {"tabari", "saadi"}
        assert "differed" in texts["tabari"]
        assert "four described qualities" in texts["saadi"]

    def test_unavailable_tafsir_does_not_break_response(self):
        source = make_source()
        response = run(build_tafsir_response(
            TafsirRequest(reference="103:2", tafsirs=["ibn-kathir", "qurtubi"]), source
        ))
        ayah = response.ayat[0]
        assert [t.key for t in ayah.tafsirs] == ["ibn-kathir"]
        assert [u.key for u in ayah.unavailable] == ["qurtubi"]

    def test_missing_verse_text_degrades(self):
        source = FakeTafsirSource(
            tafsirs={("en-tafisr-ibn-kathir", "103:2"): IBN_KATHIR_103}
        )
        response = run(build_tafsir_response(
            TafsirRequest(reference="103:2", tafsirs=["ibn-kathir"]), source
        ))
        ayah = response.ayat[0]
        assert ayah.arabic is None and ayah.translation is None
        assert ayah.tafsirs[0].key == "ibn-kathir"

    def test_range_returns_one_entry_per_ayah(self):
        source = make_source()
        response = run(build_tafsir_response(
            TafsirRequest(reference="103:1-3", tafsirs=["ibn-kathir"]), source
        ))
        assert [a.ayah for a in response.ayat] == ["103:1", "103:2", "103:3"]

    def test_invalid_reference_raises(self):
        with pytest.raises(InvalidReference):
            run(build_tafsir_response(TafsirRequest(reference="2:300"), make_source()))


# ---------------------------------------------------------------------------
# Endpoint behaviour
# ---------------------------------------------------------------------------


class TestEndpoint:
    def test_valid_reference_returns_attributed_tafsirs(self):
        from tafsir import get_tafsir, set_source, QuranComTafsirSource

        set_source(make_source())
        try:
            response = run(get_tafsir(
                TafsirRequest(reference="103:2", tafsirs=["ibn-kathir", "saadi"])
            ))
        finally:
            set_source(QuranComTafsirSource())
        assert len(response.ayat[0].tafsirs) == 2

    @pytest.mark.parametrize("reference", ["2:300", "115:1", "0:1", "not a reference"])
    def test_invalid_reference_returns_400(self, reference):
        from fastapi import HTTPException

        from tafsir import get_tafsir, set_source, QuranComTafsirSource

        set_source(make_source())
        try:
            with pytest.raises(HTTPException) as exc:
                run(get_tafsir(TafsirRequest(reference=reference)))
        finally:
            set_source(QuranComTafsirSource())
        assert exc.value.status_code == 400
        assert exc.value.detail

    def test_sources_endpoint_lists_registry(self):
        from tafsir import list_tafsir_sources

        sources = run(list_tafsir_sources())
        keys = {s.key for s in sources}
        assert set(DEFAULT_TAFSIR_KEYS) <= keys
        for source in sources:
            assert source.name and source.author and source.languages


# ---------------------------------------------------------------------------
# Verse-explanation intent detection
# ---------------------------------------------------------------------------


class TestIntentDetection:
    @pytest.mark.parametrize("prompt,expected", [
        ("What does Surah al-Asr mean?", ["103:1", "103:2", "103:3"]),
        ("Explain 2:255", ["2:255"]),
        ("What is the tafsir of 2:255?", ["2:255"]),
        ("Explain surah al-baqarah 255", ["2:255"]),
        ("What does 103:1-2 mean?", ["103:1", "103:2"]),
        ("Give me the commentary on Al-Ikhlas 1", ["112:1"]),
    ])
    def test_detects_verse_questions(self, prompt, expected):
        assert [r.key for r in detect_ayah_references(prompt)] == expected

    @pytest.mark.parametrize("prompt", [
        "Hello, how are you?",
        "How do I perform wudu?",
        "What time is Maghrib in Lagos?",
        "",
        "Tell me about Surah al-Baqarah",  # no explanation cue
    ])
    def test_ignores_non_verse_questions(self, prompt):
        assert detect_ayah_references(prompt) == []

    @pytest.mark.parametrize("prompt", [
        # Names of Allah and personal names share spelling with surah names;
        # without the word "surah" an explicit ayah number is required.
        "What does ar-Rahman mean?",
        "What does the name Muhammad mean?",
        "Explain the meaning of Maryam as a name",
    ])
    def test_bare_names_are_not_read_as_surah_references(self, prompt):
        assert detect_ayah_references(prompt) == []

    def test_long_surah_without_ayah_takes_first_ayah_only(self):
        refs = detect_ayah_references("What does Surah al-Baqarah mean?")
        assert [r.key for r in refs] == ["2:1"]

    def test_out_of_range_reference_is_skipped_not_raised(self):
        assert detect_ayah_references("Explain 2:300") == []

    def test_reference_count_is_capped(self):
        refs = detect_ayah_references("Explain 2:1-100")
        assert len(refs) <= MAX_AYAT_PER_REQUEST


# ---------------------------------------------------------------------------
# Chat integration
# ---------------------------------------------------------------------------


class TestChatIntegration:
    def test_non_verse_prompt_returns_no_context(self):
        assert run(build_chat_tafsir_context("How do I make wudu?", source=make_source())) is None

    def test_verse_prompt_retrieves_real_tafsir(self):
        source = make_source()
        context = run(build_chat_tafsir_context("Explain 103:2", "en", source))
        assert context is not None
        assert context.references == ["103:2"]
        assert context.has_tafsir
        assert "children of Adam" in context.prompt_block

    def test_prompt_block_preserves_attribution(self):
        source = make_source()
        context = run(build_chat_tafsir_context("Explain 103:2", "en", source))
        block = context.prompt_block
        assert "Ibn Kathir (Abridged)" in block
        assert "Ibn Kathir (d. 774 AH)" in block
        assert "Al-Sa'di" in block or "Sa'di" in block
        # The unavailable work is named as unavailable, not silently dropped.
        assert "UNAVAILABLE" in block
        assert "Qurtubi" in block

    def test_system_context_carries_comparison_policy(self):
        source = make_source()
        context = run(build_chat_tafsir_context("Explain 103:2", "en", source))
        instruction = tafsir_system_context(context)
        assert "Attribute every explanatory claim" in instruction
        assert "differ" in instruction
        assert "Islam says" in instruction
        assert "RETRIEVED TAFSIR PASSAGES" in instruction

    def test_system_context_warns_when_nothing_retrieved(self):
        source = FakeTafsirSource()
        context = run(build_chat_tafsir_context("Explain 103:2", "en", source))
        assert context is not None
        assert not context.has_tafsir
        assert "no tafsir text could be retrieved" in tafsir_system_context(context)

    def test_summary_names_only_works_actually_retrieved(self):
        source = make_source()
        context = run(build_chat_tafsir_context("Explain 103:2", "en", source))
        info = summarize_tafsir_context(context)
        assert info.grounded is True
        assert info.references == ["103:2"]
        assert any("Ibn Kathir" in work for work in info.works_cited)
        assert all(" — " in work for work in info.works_cited)
        # Qurtubi had no entry, so it is reported as unavailable, never cited.
        assert not any("Qurtubi" in work for work in info.works_cited)
        assert any("Qurtubi" in item for item in info.unavailable)

    def test_summary_is_not_grounded_when_nothing_retrieved(self):
        context = run(build_chat_tafsir_context("Explain 103:2", "en", FakeTafsirSource()))
        info = summarize_tafsir_context(context)
        assert info.grounded is False
        assert info.works_cited == []

    def test_prompt_block_truncates_long_passages(self):
        long_payload = dict(IBN_KATHIR_103, text="<p>" + ("word " * 5000) + "</p>")
        source = make_source(tafsirs={("en-tafisr-ibn-kathir", "103:2"): long_payload})
        context = run(build_chat_tafsir_context("Explain 103:2", "en", source))
        block = build_tafsir_prompt_block(context.ayat, excerpt_chars=200)
        assert "excerpt truncated" in block
        assert len(block) < 2000


# ---------------------------------------------------------------------------
# Registry integrity
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_every_work_has_attribution_and_a_slug(self):
        for key, work in TAFSIR_REGISTRY.items():
            assert isinstance(work, TafsirWork)
            assert work.key == key
            assert work.name.strip() and work.author.strip()
            assert work.slugs, f"{key} has no source slug"

    def test_slugs_are_unique_across_works(self):
        seen = set()
        for work in TAFSIR_REGISTRY.values():
            for slug in work.slugs.values():
                assert slug not in seen, f"duplicate slug {slug}"
                seen.add(slug)
