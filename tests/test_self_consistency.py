"""Tests for self-consistency sampling (#55)."""

import pytest

from self_consistency import (
    Claim,
    SampleResult,
    SelfConsistencyResult,
    apply_consistency_policy,
    claims_match,
    compute_agreement,
    extract_claims,
    get_self_consistency_score,
    normalize_claim,
    run_self_consistency,
)


class TestClaimExtraction:
    """Tests for claim extraction from text."""

    def test_extract_simple_claims(self):
        text = "Salah is obligatory five times daily. The first prayer is Fajr."
        claims = extract_claims(text)
        assert len(claims) == 2
        assert any("obligatory" in c.text.lower() for c in claims)

    def test_extract_citation_claims(self):
        text = "Allah says in Quran 2:255 about His throne. Bukhari 1234 narrates this."
        claims = extract_claims(text)
        assert any(c.category == "citation" for c in claims)

    def test_extract_numerical_claims(self):
        text = "Zakat is 2.5% of savings. The nisab amount equals 85 grams of gold."
        claims = extract_claims(text)
        assert any(c.category == "numerical" for c in claims)

    def test_extract_religious_claims(self):
        text = "Music is haram according to some scholars. Eating halal food is mandatory."
        claims = extract_claims(text)
        assert any(c.category == "religious" for c in claims)

    def test_skip_questions(self):
        text = "Is prayer obligatory? Yes, it is fard."
        claims = extract_claims(text)
        # Should skip the question, only extract the statement
        assert len(claims) == 1
        assert "fard" in claims[0].text.lower()

    def test_skip_hedged_statements(self):
        text = "I think this might be correct. But salah is definitely obligatory."
        claims = extract_claims(text)
        # Should skip the hedged statement
        assert len(claims) == 1
        assert "obligatory" in claims[0].text.lower()

    def test_empty_input(self):
        assert extract_claims("") == []
        assert extract_claims("   ") == []
        assert extract_claims(None) == []


class TestClaimMatching:
    """Tests for semantic claim matching."""

    def test_normalize_claim(self):
        claim = "The Prophet (peace be upon him) said..."
        normalized = normalize_claim(claim)
        assert normalized == "the prophet peace be upon him said"

    def test_exact_match(self):
        assert claims_match("Salah is obligatory.", "Salah is obligatory.")

    def test_similar_claims(self):
        claim1 = "Prayer is obligatory five times daily."
        claim2 = "Daily prayer is obligatory, performed five times."
        assert claims_match(claim1, claim2, threshold=0.5)

    def test_different_claims(self):
        claim1 = "Salah is obligatory five times."
        claim2 = "Zakat is 2.5% of wealth."
        assert not claims_match(claim1, claim2)

    def test_empty_claims(self):
        assert not claims_match("", "Some claim")
        assert not claims_match("Some claim", "")


class TestAgreementScoring:
    """Tests for agreement computation across samples."""

    def test_perfect_agreement(self):
        claims = [Claim(text="Salah is obligatory.")]
        samples = [
            SampleResult(text="Answer 1", claims=claims),
            SampleResult(text="Answer 2", claims=claims),
            SampleResult(text="Answer 3", claims=claims),
        ]
        score, matrix, low = compute_agreement(samples)
        assert score == 1.0
        assert len(low) == 0

    def test_no_agreement(self):
        samples = [
            SampleResult(text="A", claims=[Claim(text="Claim A is true.")]),
            SampleResult(text="B", claims=[Claim(text="Claim B is true.")]),
            SampleResult(text="C", claims=[Claim(text="Claim C is true.")]),
        ]
        score, matrix, low = compute_agreement(samples)
        # Each claim appears in only 1/3 samples
        assert score < 0.5
        assert len(low) > 0

    def test_partial_agreement(self):
        common_claim = Claim(text="Common claim shared by all.")
        unique_claims = [
            Claim(text="Unique to sample one."),
            Claim(text="Unique to sample two."),
        ]
        samples = [
            SampleResult(text="A", claims=[common_claim, unique_claims[0]]),
            SampleResult(text="B", claims=[common_claim, unique_claims[1]]),
        ]
        score, matrix, low = compute_agreement(samples)
        # Common claim has 100% agreement, unique claims have 50%
        assert 0.5 < score < 1.0

    def test_single_sample(self):
        samples = [SampleResult(text="Only one", claims=[Claim(text="Single claim.")])]
        score, matrix, low = compute_agreement(samples)
        assert score == 1.0

    def test_empty_samples(self):
        samples = [
            SampleResult(text="A", claims=[]),
            SampleResult(text="B", claims=[]),
        ]
        score, matrix, low = compute_agreement(samples)
        assert score == 1.0


class TestSelfConsistencySampling:
    """Tests for the full self-consistency pipeline."""

    @pytest.mark.asyncio
    async def test_high_agreement_sampling(self):
        consistent_text = "Salah is obligatory five times daily."

        async def mock_generator(prompt: str, temperature: float) -> str:
            return consistent_text

        result = await run_self_consistency(
            prompt="Is prayer obligatory?",
            original_answer=consistent_text,
            generator=mock_generator,
        )

        assert result.score >= 0.8
        assert result.sample_count >= 2
        assert not result.low_agreement_claims

    @pytest.mark.asyncio
    async def test_low_agreement_sampling(self):
        counter = [0]

        async def mock_generator(prompt: str, temperature: float) -> str:
            counter[0] += 1
            # Each sample gives a completely different answer
            return f"Unique answer number {counter[0]} with different claims."

        result = await run_self_consistency(
            prompt="Some question",
            original_answer="Original unique answer with specific claims.",
            generator=mock_generator,
        )

        assert result.score < 0.8
        assert result.sample_count >= 2

    @pytest.mark.asyncio
    async def test_early_exit_high_agreement(self):
        async def mock_generator(prompt: str, temperature: float) -> str:
            return "Consistent answer across all samples."

        result = await run_self_consistency(
            prompt="Question",
            original_answer="Consistent answer across all samples.",
            generator=mock_generator,
        )

        # Should have triggered early exit due to high agreement
        if result.score >= 0.85:
            assert result.early_exit or result.sample_count <= 3

    @pytest.mark.asyncio
    async def test_generator_error_handling(self):
        async def failing_generator(prompt: str, temperature: float) -> str:
            raise ValueError("Simulated error")

        result = await run_self_consistency(
            prompt="Question",
            original_answer="Original answer.",
            generator=failing_generator,
        )

        # Should handle errors gracefully
        assert result.sample_count >= 1  # At least the original
        assert isinstance(result.score, float)

    @pytest.mark.asyncio
    async def test_high_stakes_uses_more_samples(self):
        sample_counts = []

        async def counting_generator(prompt: str, temperature: float) -> str:
            sample_counts.append(1)
            return "Sample answer."

        await run_self_consistency(
            prompt="Question",
            original_answer="Original.",
            generator=counting_generator,
            is_high_stakes=False,
        )
        normal_count = len(sample_counts)
        sample_counts.clear()

        await run_self_consistency(
            prompt="Question",
            original_answer="Original.",
            generator=counting_generator,
            is_high_stakes=True,
        )
        high_stakes_count = len(sample_counts)

        # High stakes should attempt more samples (unless early exit)
        assert high_stakes_count >= normal_count


class TestResponsePolicy:
    """Tests for the response policy application."""

    def test_high_agreement_no_modification(self):
        result = SelfConsistencyResult(
            score=0.9,
            sample_count=3,
            claim_count=5,
            low_agreement_claims=[],
        )
        answer, should_flag = apply_consistency_policy("Good answer.", result)
        assert answer == "Good answer."
        assert not should_flag

    def test_low_agreement_adds_warning(self):
        result = SelfConsistencyResult(
            score=0.3,
            sample_count=3,
            claim_count=5,
            low_agreement_claims=["Uncertain claim one", "Uncertain claim two"],
        )
        answer, should_flag = apply_consistency_policy("Original answer.", result)
        assert "Consistency notice" in answer
        assert "Uncertain claim one" in answer
        assert should_flag

    def test_moderate_agreement_may_flag(self):
        result = SelfConsistencyResult(
            score=0.55,
            sample_count=3,
            claim_count=5,
            low_agreement_claims=["Some uncertain claim"],
        )
        answer, should_flag = apply_consistency_policy("Answer.", result)
        # Moderate score below 0.6 should flag
        assert should_flag


class TestIntegration:
    """Integration tests with the confidence module interface."""

    @pytest.mark.asyncio
    async def test_get_self_consistency_score(self):
        async def mock_generator(prompt: str, temperature: float) -> str:
            return "Consistent answer."

        score = await get_self_consistency_score(
            prompt="Test question",
            original_answer="Consistent answer.",
            generator=mock_generator,
        )

        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_caching_works(self):
        call_count = [0]

        async def counting_generator(prompt: str, temperature: float) -> str:
            call_count[0] += 1
            return "Answer."

        # First call
        await get_self_consistency_score(
            prompt="Cached question",
            original_answer="Answer.",
            generator=counting_generator,
            context="context1",
        )
        first_call_count = call_count[0]

        # Second call with same prompt+context should use cache
        await get_self_consistency_score(
            prompt="Cached question",
            original_answer="Answer.",
            generator=counting_generator,
            context="context1",
        )

        # Should not have made additional generator calls
        assert call_count[0] == first_call_count
