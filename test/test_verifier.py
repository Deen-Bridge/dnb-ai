from verifier import (
    normalize_arabic,
    normalize_english,
    calculate_similarity,
    verify_quran_citation,
    verify_hadith_citation,
    extract_and_verify_all,
    VerificationStatus,
)


def test_normalize_arabic():
    # Test stripping diacritics (tashkeel) and unifying Alef forms
    raw_arabic = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
    normalized = normalize_arabic(raw_arabic)
    assert "ِ" not in normalized  # Kasra removed
    assert "ّ" not in normalized  # Shadda removed


def test_normalize_english():
    text = "In the name of Allah, the Entirely Merciful...!"
    normalized = normalize_english(text)
    assert normalized == "in the name of allah the entirely merciful"


def test_calculate_similarity():
    str1 = "In the name of Allah, the Entirely Merciful"
    str2 = "In the name of Allah the Entirely Merciful"
    similarity = calculate_similarity(str1, str2)
    assert similarity > 0.90


def test_verify_valid_quran_citation():
    # Surah 1:1 valid quote test
    quote = "In the name of Allah, the Entirely Merciful, the Especially Merciful."
    res = verify_quran_citation(1, 1, quote)
    assert res["status"] == VerificationStatus.VERIFIED
    assert res["source"] == "quran"


def test_verify_nonexistent_ayah():
    # Surah 1 only has 7 ayahs
    res = verify_quran_citation(1, 999)
    assert res["status"] == VerificationStatus.MISMATCH
    assert "does not exist" in res["reason"]


def test_verify_mismatched_quote():
    # Fabricated quote attributed to Surah 1:1
    wrong_quote = "This is a completely fabricated translation string that does not match."
    res = verify_quran_citation(1, 1, wrong_quote)
    assert res["status"] == VerificationStatus.MISMATCH
    assert "correct_text" in res


def test_verify_quran_citation_without_quote():
    # Citation provided without quoted text
    res = verify_quran_citation(1, 1)
    assert res["status"] == VerificationStatus.NOT_QUOTED


def test_verify_hadith_unverified_fallback():
    # Hadith citation should default to unverified when corpus is unavailable
    res = verify_hadith_citation("Bukhari", "123")
    assert res["status"] == VerificationStatus.UNVERIFIED
    assert res["source"] == "hadith"


def test_extract_and_verify_all():
    sample_text = (
        'As mentioned in Quran 1:1 "In the name of Allah, the Entirely Merciful, the Especially Merciful.", '
        'and also noted in Bukhari 42.'
    )
    results = extract_and_verify_all(sample_text)
    assert len(results) == 2

    # Quran result
    quran_res = next(r for r in results if r["source"] == "quran")
    assert quran_res["status"] == VerificationStatus.VERIFIED

    # Hadith result
    hadith_res = next(r for r in results if r["source"] == "hadith")
    assert hadith_res["status"] == VerificationStatus.UNVERIFIED
