from __future__ import annotations

import pytest
from malay.models import MalayQueryRequest
from malay.processor import MalayProcessor

def test_malay_indonesian_processor() -> None:
    processor = MalayProcessor()
    request = MalayQueryRequest(text="Bagaimana cara mengerjakan shalat dan puasa?", dialect="indonesia", target_script="latin")
    response = processor.process(request)
    
    assert response.detected_dialect == "indonesia"
    assert response.detected_script == "latin"
    assert len(response.islamic_terms_identified) > 0
    assert "shalat" in response.normalized_text or "solat" in response.normalized_text
    assert "Assalamu'alaikum" in response.optimized_response

def test_jawi_transliteration() -> None:
    processor = MalayProcessor()
    request = MalayQueryRequest(text="solat puasa allah", target_script="jawi")
    response = processor.process(request)
    
    assert response.detected_script == "latin"
    assert response.transliterated_text is not None
    assert "صلاة" in response.transliterated_text
    assert "ڤواس" in response.transliterated_text
