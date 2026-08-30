from __future__ import annotations

from typing import Any
from malay.models import MalayQueryRequest, MalayQueryResponse
from malay.dialects import DialectHandler
from malay.terminology import TerminologyLexicon
from malay.loanwords import LoanwordRecognizer
from malay.cultural_context import CulturalContextManager
from malay.code_switching import CodeSwitchingHandler
from malay.generator import MalayContentGenerator
from malay.optimizer import QueryOptimizer
from malay.router import MalayRouter

class MalayProcessor:
    def __init__(self) -> None:
        self.lexicon = TerminologyLexicon()

    def detect_script(self, text: str) -> str:
        # Simple heuristic: Arabic characters imply Jawi
        arabic_chars = set("ابتثجحخدذرزسشصضطظعغفقكلمنهويأإآةىئؤإؤلاء")
        if any(c in arabic_chars for c in text):
            return "jawi"
        return "latin"

    def transliterate_to_jawi(self, text: str) -> str:
        # Basic demonstration transliteration mapping
        mapping = {
            "solat": "صلاة",
            "shalat": "صلاة",
            "puasa": "ڤواس",
            "allah": "الله",
            "muhammad": "محمد"
        }
        words = text.split()
        res = []
        for w in words:
            lw = w.lower().strip(",.?!;:()")
            res.append(mapping.get(lw, w))
        return " ".join(res)

    def process(self, request: MalayQueryRequest) -> MalayQueryResponse:
        script = self.detect_script(request.text)
        dialect = request.dialect or DialectHandler.detect_dialect(request.text)
        
        normalized = QueryOptimizer.optimize_query(request.text)
        normalized = DialectHandler.normalize_dialect(normalized, dialect)
        normalized = CodeSwitchingHandler.process_code_switching(normalized)
        
        terms = self.lexicon.identify_terms(normalized)
        loanwords = LoanwordRecognizer.recognize_loanwords(normalized)
        
        base_content = MalayContentGenerator.generate_islamic_content(normalized, dialect)
        enriched = CulturalContextManager.enrich_response(base_content, dialect)
        
        transliterated = None
        if request.target_script == "jawi" or script == "jawi":
            transliterated = self.transliterate_to_jawi(enriched)

        return MalayQueryResponse(
            original_text=request.text,
            normalized_text=normalized,
            detected_dialect=dialect,
            detected_script=script,
            transliterated_text=transliterated,
            islamic_terms_identified=terms,
            optimized_response=enriched,
            confidence=0.98
        )
