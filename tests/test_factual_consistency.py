"""Tests for cross-session factual consistency subsystem (#Security).

Verifies factual claim extraction, long-term storage, cross-session contradiction
detection, attribution stability, ikhtilaf vs error differentiation, reconciliation,
and policy enforcement.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from consistency import (
    ClaimType,
    ConsistencyAction,
    ConsistencyAlertManager,
    ConsistencyEnforcer,
    ContradictionCategory,
    ContradictionSeverity,
    FactualClaim,
    InMemoryClaimStore,
    RulingType,
    detect_contradictions,
    extract_claims,
    get_alert_manager,
    get_core_position,
    get_ikhtilaf_entry,
    is_core_principle_violation,
)
from main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. Claim Extraction Tests
# ---------------------------------------------------------------------------


class TestClaimExtraction:
    def test_extract_ruling_and_polarity(self):
        text = "Drinking alcohol is strictly forbidden in Islam. Performing salah five times daily is obligatory."
        claims = extract_claims(text)
        assert len(claims) >= 2

        alcohol_claim = next(c for c in claims if "alcohol" in c.text.lower())
        assert alcohol_claim.ruling == RulingType.HARAM
        assert alcohol_claim.polarity is False
        assert alcohol_claim.topic == "dietary"

        salah_claim = next(c for c in claims if "salah" in c.text.lower() or "prayers" in c.text.lower())
        assert salah_claim.ruling in (RulingType.FARD, RulingType.WAJIB)
        assert salah_claim.polarity is True
        assert salah_claim.topic == "prayer"

    def test_extract_attribution_and_madhhab(self):
        text = (
            "According to the Hanafi school, touching a woman does not break wudu. "
            "Imam al-Shafi'i stated that skin contact invalidates purity."
        )
        claims = extract_claims(text)
        assert len(claims) >= 2

        hanafi_claim = next(c for c in claims if "hanafi" in c.text.lower())
        assert hanafi_claim.madhhab == "hanafi"
        assert "Hanafi" in (hanafi_claim.attribution or "")
        assert hanafi_claim.entity == "touching_opposite_gender_wudu"

        shafii_claim = next(c for c in claims if "shafi" in c.text.lower())
        assert shafii_claim.attribution == "al-Shafi'i"
        assert shafii_claim.entity == "touching_opposite_gender_wudu"

    def test_extract_conditions(self):
        text = "A traveler on a journey may shorten their prayers. If one eats unintentionally out of forgetfulness, fasting remains valid."
        claims = extract_claims(text)
        assert len(claims) >= 2

        traveler_claim = next(c for c in claims if "traveler" in c.text.lower())
        assert traveler_claim.condition == "traveler"

        forget_claim = next(
            c for c in claims if "forgetfulness" in c.text.lower() or "unintentionally" in c.text.lower()
        )
        assert forget_claim.condition == "forgetfulness"

    def test_extract_citations_and_numbers(self):
        text = "Zakat is 2.5% of qualifying wealth as ordained in Quran 9:60 and Sahih Bukhari 1454."
        claims = extract_claims(text)
        assert len(claims) >= 1
        claim = claims[0]
        assert "Quran 9:60" in claim.citations
        assert any("Bukhari" in cit for cit in claim.citations)
        assert claim.claim_type == ClaimType.NUMERICAL
        assert claim.topic == "zakat"

    def test_empty_or_trivial_text(self):
        assert extract_claims("") == []
        assert extract_claims("   \n  ") == []
        assert extract_claims("Hello?") == []


# ---------------------------------------------------------------------------
# 2. Claim Store Tests
# ---------------------------------------------------------------------------


class TestClaimStore:
    @pytest.mark.asyncio
    async def test_save_and_retrieve_by_topic_and_entity(self):
        store = InMemoryClaimStore()
        c1 = FactualClaim(
            claim_id="c1",
            text="Eating camel meat invalidates wudu.",
            topic="wudu_nullification",
            entity="camel_meat_wudu",
            ruling=RulingType.NULLIFIED,
            polarity=False,
            madhhab="hanbali",
        )
        c2 = FactualClaim(
            claim_id="c2",
            text="Eating camel meat does not break wudu.",
            topic="wudu_nullification",
            entity="camel_meat_wudu",
            ruling=RulingType.NOT_NULLIFIED,
            polarity=True,
            madhhab="hanafi",
        )
        await store.save_claim(c1)
        await store.save_claim(c2)

        topic_claims = await store.get_claims_by_topic("wudu_nullification")
        assert len(topic_claims) == 2

        entity_claims = await store.get_claims_by_entity("camel_meat_wudu")
        assert len(entity_claims) == 2

    @pytest.mark.asyncio
    async def test_user_claims_isolation_and_deletion(self):
        store = InMemoryClaimStore()
        c1 = FactualClaim(
            claim_id="u1_c1",
            text="User 1 asked about zakat rate 2.5%.",
            topic="zakat",
            user_id="user_123",
        )
        c2 = FactualClaim(
            claim_id="u2_c1",
            text="User 2 asked about fasting.",
            topic="fasting",
            user_id="user_456",
        )
        await store.save_claims([c1, c2])

        u1_claims = await store.get_claims_by_user("user_123")
        assert len(u1_claims) == 1
        assert u1_claims[0].claim_id == "u1_c1"

        deleted = await store.delete_user_claims("user_123")
        assert deleted == 1
        assert await store.get_claims_by_user("user_123") == []
        assert len(await store.get_claims_by_user("user_456")) == 1

    @pytest.mark.asyncio
    async def test_find_relevant_claims(self):
        store = InMemoryClaimStore()
        c1 = FactualClaim(
            claim_id="c1",
            text="Zakat on gold jewelry is mandatory in Hanafi.",
            topic="zakat",
            entity="zakat_jewelry",
        )
        c2 = FactualClaim(
            claim_id="c2",
            text="Five prayers are obligatory every single day.",
            topic="prayer",
            entity="daily_prayers_count",
        )
        await store.save_claims([c1, c2])

        results = await store.find_relevant_claims(query="gold jewelry zakat", topic="zakat")
        assert len(results) >= 1
        assert results[0].claim_id == "c1"


# ---------------------------------------------------------------------------
# 3. Core Knowledge Anchors & Ijma Tests
# ---------------------------------------------------------------------------


class TestCoreAnchors:
    def test_core_position_lookup(self):
        pos = get_core_position("tawhid_principles")
        assert pos is not None
        assert "Sole Creator" in pos.orthodox_position

    def test_core_principle_violation_detection(self):
        claim = FactualClaim(
            claim_id="v1",
            text="Some modern groups claim that polytheism is permissible and shirk is acceptable.",
            topic="aqeedah",
        )
        is_violation, reason = is_core_principle_violation(claim)
        assert is_violation is True
        assert reason is not None

    def test_orthodox_claim_passes(self):
        claim = FactualClaim(
            claim_id="v2",
            text="Allah is One and has no partners or equals.",
            topic="aqeedah",
        )
        is_violation, _ = is_core_principle_violation(claim)
        assert is_violation is False

    def test_ikhtilaf_entry_lookup(self):
        ikhtilaf = get_ikhtilaf_entry("touching_opposite_gender_wudu")
        assert ikhtilaf is not None
        assert "shafii" in ikhtilaf["positions"]
        assert "hanafi" in ikhtilaf["positions"]


# ---------------------------------------------------------------------------
# 4. Contradiction Detection Tests
# ---------------------------------------------------------------------------


class TestContradictionDetection:
    def test_direct_ruling_contradiction_without_ikhtilaf(self):
        hist = FactualClaim(
            claim_id="h1",
            text="Drinking wine is forbidden and haram.",
            topic="dietary",
            entity="prohibition_of_intoxicants",
            ruling=RulingType.HARAM,
            polarity=False,
        )
        cand = FactualClaim(
            claim_id="c1",
            text="Drinking wine is permissible and halal in small quantities.",
            topic="dietary",
            entity="prohibition_of_intoxicants",
            ruling=RulingType.HALAL,
            polarity=True,
        )
        contradictions = detect_contradictions([cand], [hist])
        assert len(contradictions) >= 1
        assert any(c.severity in (ContradictionSeverity.HIGH, ContradictionSeverity.CRITICAL) for c in contradictions)

    def test_legitimate_ikhtilaf_recognized(self):
        hist = FactualClaim(
            claim_id="h1",
            text="Touching one's wife invalidates wudu in the Shafi'i school.",
            topic="wudu_nullification",
            entity="touching_opposite_gender_wudu",
            ruling=RulingType.NULLIFIED,
            polarity=False,
            madhhab="shafii",
            attribution="Shafi'i school",
        )
        cand = FactualClaim(
            claim_id="c1",
            text="According to the Hanafi school, touching one's spouse does not break wudu.",
            topic="wudu_nullification",
            entity="touching_opposite_gender_wudu",
            ruling=RulingType.NOT_NULLIFIED,
            polarity=True,
            madhhab="hanafi",
            attribution="Hanafi school",
        )
        contradictions = detect_contradictions([cand], [hist])
        assert len(contradictions) >= 1
        assert contradictions[0].is_legitimate_variation is True
        assert contradictions[0].legitimate_reason == "madhhab_difference"

    def test_attribution_swap_detected(self):
        hist = FactualClaim(
            claim_id="h1",
            text="Imam Abu Hanifa held that sea creatures other than fish are disliked.",
            topic="dietary",
            entity="seafood_permissibility",
            attribution="Abu Hanifa",
        )
        cand = FactualClaim(
            claim_id="c1",
            text="Imam al-Shafi'i held that sea creatures other than fish are disliked.",
            topic="dietary",
            entity="seafood_permissibility",
            attribution="al-Shafi'i",
        )
        contradictions = detect_contradictions([cand], [hist])
        assert len(contradictions) >= 1
        attr_mismatch = next(c for c in contradictions if c.category == ContradictionCategory.ATTRIBUTION_MISMATCH)
        assert attr_mismatch.severity == ContradictionSeverity.MEDIUM
        assert "Scholarly attribution mismatch" in attr_mismatch.description

    def test_numerical_discrepancy_detected(self):
        hist = FactualClaim(
            claim_id="h1",
            text="Zakat is 2.5% of total annual savings.",
            topic="zakat",
            entity="zakat_rates_nisab",
            claim_type=ClaimType.NUMERICAL,
        )
        cand = FactualClaim(
            claim_id="c1",
            text="Zakat is 5% of total annual savings.",
            topic="zakat",
            entity="zakat_rates_nisab",
            claim_type=ClaimType.NUMERICAL,
        )
        contradictions = detect_contradictions([cand], [hist])
        assert len(contradictions) >= 1
        num_mismatch = next(c for c in contradictions if c.category == ContradictionCategory.NUMERICAL_DISCREPANCY)
        assert num_mismatch.severity == ContradictionSeverity.HIGH


# ---------------------------------------------------------------------------
# 5. Reconciliation & Policy Enforcement Tests
# ---------------------------------------------------------------------------


class TestReconciliationAndEnforcement:
    @pytest.mark.asyncio
    async def test_enforce_allow_on_consistent_response(self):
        store = InMemoryClaimStore()
        enforcer = ConsistencyEnforcer(store=store)

        resp = "Performing the five daily prayers is obligatory upon every sane adult Muslim."
        result = await enforcer.evaluate_response(response_text=resp)
        assert result.action == ConsistencyAction.ALLOW
        assert result.is_consistent is True
        assert result.final_text == resp

    @pytest.mark.asyncio
    async def test_enforce_reconciliation_on_ikhtilaf_variation(self):
        store = InMemoryClaimStore()
        # Seed prior session claim (Shafi'i ruling)
        h_claim = FactualClaim(
            claim_id="h_wudu",
            text="Touching non-mahram skin breaks wudu.",
            topic="wudu_nullification",
            entity="touching_opposite_gender_wudu",
            ruling=RulingType.NULLIFIED,
            polarity=False,
            madhhab="shafii",
        )
        await store.save_claim(h_claim)

        enforcer = ConsistencyEnforcer(store=store)
        cand_resp = "In the Hanafi school, touching your spouse does not break wudu."
        result = await enforcer.evaluate_response(
            response_text=cand_resp,
            madhhab="hanafi",
        )

        assert result.action == ConsistencyAction.RECONCILE
        assert "Scholarly Context & Reconciliation" in result.final_text
        assert "Hanafi" in result.final_text

    @pytest.mark.asyncio
    async def test_enforce_block_on_core_principle_violation(self):
        store = InMemoryClaimStore()
        enforcer = ConsistencyEnforcer(store=store)

        resp = "In some philosophies, polytheism is permissible and worshiping idols is valid in islam."
        result = await enforcer.evaluate_response(response_text=resp)

        assert result.action == ConsistencyAction.BLOCK
        assert "Authenticity & Consistency Notice" in result.final_text
        assert "diverged from established orthodox Islamic consensus" in result.final_text

    @pytest.mark.asyncio
    async def test_alerts_and_coherence_metrics(self):
        alert_mgr = get_alert_manager()
        alert_mgr.clear_alerts()

        store = InMemoryClaimStore()
        enforcer = ConsistencyEnforcer(store=store)

        # Trigger a block event
        await enforcer.evaluate_response("Polytheism is permissible in islam.")
        metrics = alert_mgr.get_metrics(total_claims_in_store=10)

        assert metrics.total_checks_performed >= 1
        assert metrics.violations_blocked >= 1
        alerts = alert_mgr.get_alerts()
        assert len(alerts) >= 1
        assert alerts[0].severity == ContradictionSeverity.CRITICAL


# ---------------------------------------------------------------------------
# 6. REST API Endpoints Tests
# ---------------------------------------------------------------------------


class TestConsistencyAPIEndpoints:
    def test_post_consistency_check(self):
        payload = {
            "text": "Fasting in Ramadan is mandatory for healthy adult Muslims.",
            "prompt": "Is fasting Ramadan required?",
        }
        res = client.post("/consistency/check", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["is_consistent"] is True
        assert data["action"] == "allow"

    def test_post_consistency_index_and_get_claims(self):
        payload = {
            "text": "Zakat on gold is 2.5% annually according to consensus.",
            "chat_id": "test_chat_001",
            "user_id": "test_user_001",
        }
        idx_res = client.post("/consistency/index", json=payload)
        assert idx_res.status_code == 200
        assert idx_res.json()["claims_indexed"] >= 1

        get_res = client.get("/consistency/claims", params={"user_id": "test_user_001"})
        assert get_res.status_code == 200
        claims_data = get_res.json()
        assert claims_data["total"] >= 1

    def test_get_alerts_and_coherence(self):
        res = client.get("/consistency/coherence")
        assert res.status_code == 200
        data = res.json()
        assert "consistency_rate" in data
        assert "cross_session_drift_score" in data

        alerts_res = client.get("/consistency/alerts")
        assert alerts_res.status_code == 200
        assert "alerts" in alerts_res.json()

    def test_get_core_positions(self):
        res = client.get("/consistency/core-positions")
        assert res.status_code == 200
        data = res.json()
        assert "core_positions" in data
        assert "ikhtilaf_map" in data
        assert "tawhid_principles" in data["core_positions"]

    def test_delete_user_claims(self):
        res = client.delete("/consistency/claims/test_user_001")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# 7. Multi-Session Flow & Temporal Consistency Tests
# ---------------------------------------------------------------------------


class TestCrossSessionConsistencyFlows:
    @pytest.mark.asyncio
    async def test_multi_session_perspective_evolution(self):
        store = InMemoryClaimStore()
        alert_manager = ConsistencyAlertManager()
        enforcer = ConsistencyEnforcer(store=store, alert_manager=alert_manager)

        # Session 1: User asks about seafood permissibility under Shafi'i
        sess1_text = "According to the Shafi'i school, all seafood including crab, lobster, and shrimp is halal."
        r1 = await enforcer.evaluate_response(
            response_text=sess1_text,
            chat_id="session_1",
            user_id="user_ahmed",
            madhhab="shafii",
        )
        assert r1.action == ConsistencyAction.ALLOW
        # Index claims from session 1
        await enforcer.index_claims(sess1_text, chat_id="session_1", user_id="user_ahmed")

        # Verify claim is indexed
        user_claims = await store.get_claims_by_user("user_ahmed")
        assert len(user_claims) >= 1

        # Session 2: Same user asks about Hanafi view on shellfish
        sess2_text = "In the Hanafi school, only fish is halal; shellfish and crab are disliked (makruh)."
        r2 = await enforcer.evaluate_response(
            response_text=sess2_text,
            chat_id="session_2",
            user_id="user_ahmed",
            madhhab="hanafi",
        )
        # Should recognize as legitimate ikhtilaf and reconcile with educational context
        assert r2.action == ConsistencyAction.RECONCILE
        assert "Scholarly Context & Reconciliation" in r2.final_text
        assert "Hanafi" in r2.final_text

        # Session 3: Model attempts a fabricated attribution (swapping scholars)
        sess3_text = "Imam al-Shafi'i held that shellfish is strictly prohibited and not halal."
        r3 = await enforcer.evaluate_response(
            response_text=sess3_text,
            chat_id="session_3",
            user_id="user_ahmed",
        )
        # Should detect contradiction/mismatch and warn or flag
        assert r3.action in (ConsistencyAction.WARN, ConsistencyAction.BLOCK)

        # Verify coherence metrics update
        metrics = enforcer.alert_manager.get_metrics(total_claims_in_store=len(user_claims))
        assert metrics.total_checks_performed == 3
        assert metrics.legitimate_variations_reconciled >= 1
