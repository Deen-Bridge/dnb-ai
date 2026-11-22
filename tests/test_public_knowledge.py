"""Tests for the governed public-knowledge catalog (#848).

Offline and deterministic: records are synthetic, no network, no credentials,
and the FastAPI app is never imported.
"""

from __future__ import annotations

from dataclasses import fields, replace
from datetime import date
from typing import Any

import pytest

from public_knowledge import (
    MAX_QUERY_CHARS,
    MAX_QUERY_TERMS,
    PublicKnowledgeCatalog,
    PublicRecord,
    RejectedRecord,
    normalize_search_text,
    validate_record,
)

TODAY = date(2026, 11, 21)

PROVENANCE = {
    "source": "Deen Bridge events team",
    "license": "CC BY 4.0",
    "dataset_version": "1.0.0",
    "reviewed_by": "content-review",
    "last_reviewed": "2026-11-01",
}


def record(kind: str, title: str, summary: str, url: str, *, published: bool = True, **overrides: str) -> PublicRecord:
    values = {**PROVENANCE, **overrides}
    return PublicRecord(kind, title, summary, url, published, **values)


def catalog(*records: PublicRecord) -> PublicKnowledgeCatalog:
    return PublicKnowledgeCatalog(list(records), version="2026.11.21", today=TODAY)


# ---------------------------------------------------------------------------
# Publication boundary (existing behavior, now with provenance)
# ---------------------------------------------------------------------------


def test_search_returns_only_published_matching_records() -> None:
    cat = catalog(
        record("event", "Lagos Quran workshop", "Public weekend event", "/events/1"),
        record("event", "Draft workshop", "Private planning", "/events/2", published=False),
    )
    results = cat.search("quran workshop", kind="event")
    assert [item.url for item in results] == ["/events/1"]


def test_empty_queries_do_not_expose_catalog() -> None:
    cat = catalog(record("course", "Tajwid", "Course", "/courses/1"))
    assert cat.search("   ") == []
    assert cat.search("") == []
    assert cat.search("!!! ...") == []


def test_unpublished_records_are_dropped_not_kept() -> None:
    draft = record("event", "Secret draft", "Internal notes", "/events/9", published=False)
    cat = catalog(draft)
    assert cat.records == ()
    assert cat.rejected == ()
    assert cat.unpublished_count == 1
    assert cat.search("secret") == []


def test_kind_filter_is_exact() -> None:
    cat = catalog(
        record("event", "Tajwid evening", "Recitation practice", "/events/3"),
        record("course", "Tajwid course", "Recitation rules", "/courses/3"),
    )
    assert [r.url for r in cat.search("tajwid", kind="course")] == ["/courses/3"]
    assert cat.search("tajwid", kind="Course") == []
    assert [r.url for r in cat.search("tajwid")] == ["/events/3", "/courses/3"]


# ---------------------------------------------------------------------------
# Provenance: published records must carry it, or they fail closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"source": ""}, "missing source"),
        ({"license": "  "}, "missing license"),
        ({"reviewed_by": ""}, "missing reviewed_by"),
        ({"dataset_version": ""}, "dataset_version"),
        ({"dataset_version": "1.0 beta"}, "dataset_version"),
        ({"last_reviewed": "21/11/2026"}, "ISO date"),
        ({"last_reviewed": "2027-01-01"}, "in the future"),
    ],
)
def test_published_record_without_complete_provenance_is_rejected(override: dict[str, Any], reason: str) -> None:
    bad = record("course", "Fiqh of fasting", "Course", "/courses/7", **override)
    cat = catalog(bad)
    assert cat.search("fasting") == []
    [rejected] = cat.rejected
    assert rejected.url == "/courses/7"
    assert any(reason in item for item in rejected.reasons)


def test_rejection_withholds_title_and_summary() -> None:
    cat = catalog(record("course", "Private title", "Private summary", "/courses/8", source=""))
    [rejected] = cat.rejected
    assert isinstance(rejected, RejectedRecord)
    assert {f.name for f in fields(rejected)} == {"kind", "url", "reasons"}


@pytest.mark.parametrize(
    "url",
    ["javascript:alert(1)", "http://example.org/x", "//evil.example/x", "data:text/html,x", "", "events/1", "/a b"],
)
def test_unsafe_or_relative_urls_are_rejected(url: str) -> None:
    assert any("url" in reason for reason in validate_record(record("event", "T", "S", url), today=TODAY))


@pytest.mark.parametrize("url", ["/events/1", "/", "https://deenbridge.example/courses/1?lang=ar"])
def test_site_paths_and_https_urls_are_accepted(url: str) -> None:
    assert validate_record(record("event", "T", "S", url), today=TODAY) == []


def test_missing_kind_or_title_is_rejected() -> None:
    assert "missing kind" in validate_record(record("", "Title", "S", "/x"), today=TODAY)
    assert "missing title" in validate_record(record("event", " ", "S", "/x"), today=TODAY)


def test_duplicate_kind_and_url_keeps_the_first_record_only() -> None:
    first = record("event", "Quran circle", "Weekly", "/events/5")
    second = record("event", "Quran circle (copy)", "Weekly copy", "/events/5")
    cat = catalog(first, second)
    assert cat.records == (first,)
    assert "duplicate" in cat.rejected[0].reasons[0]


def test_results_carry_provenance_and_caveat() -> None:
    entry = record(
        "resource",
        "Narrator reliability notes",
        "Curated starter subset of classical rijal",
        "/resources/rijal",
        source="data/rijal/narrators.json",
        license="Bibliographic facts; no copyright claim",
        dataset_version="1",
        caveat="Advisory; not an exhaustive rijal dictionary.",
    )
    [result] = catalog(entry).search("rijal")
    assert result.source == "data/rijal/narrators.json"
    assert result.dataset_version == "1"
    assert result.reviewed_by == "content-review"
    assert result.caveat.startswith("Advisory")


def test_catalog_reports_its_version() -> None:
    assert catalog().version == "2026.11.21"
    assert PublicKnowledgeCatalog().version == "unversioned"


def test_record_order_is_stable() -> None:
    entries = [record("event", f"Halaqa {i}", "Weekly halaqa", f"/events/{i}") for i in range(5)]
    assert [r.url for r in catalog(*entries).search("halaqa")] == [f"/events/{i}" for i in range(5)]


# ---------------------------------------------------------------------------
# Arabic and English edge cases
# ---------------------------------------------------------------------------


def test_arabic_query_matches_with_and_without_tashkeel() -> None:
    cat = catalog(record("course", "دورة القُرْآن الكَريم", "تجويد وتلاوة", "/courses/ar-1"))
    assert [r.url for r in cat.search("القرآن")] == ["/courses/ar-1"]
    assert [r.url for r in cat.search("القُرْآنُ")] == ["/courses/ar-1"]
    assert [r.url for r in cat.search("القران الكريم")] == ["/courses/ar-1"]


def test_arabic_letter_variants_fold_together() -> None:
    cat = catalog(record("event", "محاضرة في المدرسة", "إلى الطلاب", "/events/ar-2"))
    # ta marbuta / ha and alef-with-hamza / bare alef and alef maqsura / ya
    assert [r.url for r in cat.search("المدرسه الي")] == ["/events/ar-2"]


def test_arabic_presentation_forms_and_tatweel_fold() -> None:
    cat = catalog(record("resource", "ذكر الله", "أذكار الصباح", "/resources/adhkar"))
    assert [r.url for r in cat.search("ﷲ")] == ["/resources/adhkar"]
    assert [r.url for r in cat.search("الصبـــاح")] == ["/resources/adhkar"]


def test_transliteration_spellings_fold_together() -> None:
    cat = catalog(record("course", "Ṣaḥīḥ al-Bukhārī study", "Qurʾān and sunnah", "/courses/7"))
    for query in ("sahih bukhari", "SAHIH", "Qur'an", "quran", "qurʾān"):
        assert [r.url for r in cat.search(query)] == ["/courses/7"], query


def test_mixed_arabic_english_query_requires_every_term() -> None:
    cat = catalog(
        record("event", "Tajwid workshop ورشة التجويد", "Lagos", "/events/mx"),
        record("event", "Tajwid revision", "Online", "/events/en"),
    )
    assert [r.url for r in cat.search("tajwid التجويد")] == ["/events/mx"]


def test_normalize_search_text_matches_examples() -> None:
    assert normalize_search_text("Ṣaḥīḥ") == "sahih"
    assert normalize_search_text("القُرْآنُ") == "القران"
    assert normalize_search_text("") == ""


# ---------------------------------------------------------------------------
# Failure states and bounds
# ---------------------------------------------------------------------------


def test_empty_catalog_searches_cleanly() -> None:
    cat = PublicKnowledgeCatalog()
    assert cat.search("anything") == []
    assert cat.records == ()


def test_overlong_query_raises() -> None:
    with pytest.raises(ValueError, match="characters"):
        catalog().search("a" * (MAX_QUERY_CHARS + 1))


def test_too_many_terms_raises() -> None:
    with pytest.raises(ValueError, match="terms"):
        catalog().search(" ".join(f"t{i}" for i in range(MAX_QUERY_TERMS + 1)))


def test_repeated_terms_count_once() -> None:
    cat = catalog(record("event", "Quran circle", "Weekly", "/events/q"))
    assert len(cat.search(" ".join(["quran"] * (MAX_QUERY_TERMS + 5)))) == 1


def test_records_are_immutable() -> None:
    entry = record("event", "Quran circle", "Weekly", "/events/q")
    with pytest.raises(AttributeError):
        entry.title = "changed"  # type: ignore[misc]
    assert replace(entry, title="copy").title == "copy"


# ---------------------------------------------------------------------------
# Freshness review queue
# ---------------------------------------------------------------------------


def test_needs_review_lists_stale_records_without_hiding_them() -> None:
    fresh = record("event", "Fresh", "S", "/events/f", last_reviewed="2026-11-01")
    stale = record("event", "Stale", "S", "/events/s", last_reviewed="2025-01-01")
    cat = catalog(fresh, stale)
    assert [r.url for r in cat.needs_review(as_of=TODAY, max_age_days=180)] == ["/events/s"]
    assert [r.url for r in cat.search("stale")] == ["/events/s"]


def test_needs_review_rejects_negative_age() -> None:
    with pytest.raises(ValueError):
        catalog().needs_review(as_of=TODAY, max_age_days=-1)
