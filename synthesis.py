import re
 from collections import Counter, defaultdict
 from typing import List, Dict, Any, Optional, Tuple

class SynthesisError(Exception):
    pass

class TermNormalizer:
    Synonym_Map = {
        "AI": "artificial intelligence",
        "ML": "machine learning",
        "DL": "deep learning",
        "implementation": "implement",
        "application": "app",
    }
    STOPWORDS = set({
        "a", "an", "the", "is", "are", "was", "were",
        "and", "or", "but", "not", "for", "with",
        "on", "at", "from", "by", "to", "of",
        "in", "it", "its", "that", "this", "those",
        "who", "whom", "which", "have", "has",
        "be", "been", "being", "will", "would",
        "can", "cannot", "could", "couldn't", "etc",
    })

    @staticmethod
    def normalize(text: str) -> str:
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = [t for t in text.split() if t not in TermNormalizer.STOPWORDS]
        tokens = [TermNormalizer.Synonym_Map.get(t, t) for t in tokens]
        return " ".join(tokens)

class ContradictionDetector:
    NEGATION_WORDS = {"not", "no", "never", "cannot", "can't", "don't", "doesn't", "didn't", "won't", "wouldn't", "shouldn't", "isn't", "aren't", "wasn't", "weren't", "without", "lack", "absence"}
    ANTONYM_PAIRS = [
        ("increase", "decrease"),
        ("high", "low"),
        ("positive", "negative"),
        ("supports", "opposes"),
        ("yes", "no"),
        ("true", "false"),
        ("always", "never"),
        ("must", "must not"),
        ("required", "optional"),
    ]

    @classmethod
    def detect_contradictions(statements: List[str]) -> List[Dict[Any], Any]]:
        contradictions = []
        for i in range(len(statements)):
            for j in range(i+1, len(statements)):
                norm_i = TermNormalizer.normalize(statements[i])
                norm_j = TermNormalizer.normalize(statements[j])
                tokens_i = set(norm_i.split())
                tokens_j = set(norm_j.split())
                common = tokens_i & tokens_j
                if len(common) >= 2: 
                    neg_i = any(t in cls.NEGATION_WORDS for t in tokens_i)
                    neg_j = any(t in cls.NEGATION_WORDS for t in tokens_j)
                    if neg_i != neg_j:
                        contradictions.append({
                            "statement1": statements[i],
                            "statement2": statements[j],
                            "reason": "Negation mismatch",
                            "confidence": 0.8,
                        })
                for w1, w2 in cls.ANTONYM_PAIRS:
                    if w1 in tokens_i and w2 in tokens_j:
                        contradictions.append({
                            "statement1": statements[i],
                            "statement2": statements[j],
                            "reason": "Antonym pair",
                            "confidence": 0.7,
                        })
                    elif w2 in tokens_i and w1 in tokens_j:
                        contradictions.append({
                            "statement1": statements[i],
                            "statement2": statements[j],
                            "reason": "Antonym pair",
                            "confidence": 0.7,
                        })
        return contradictions

class AttributionTracker:
    def __init__(self):
        self.attributions = []   # list of dicts with keys claim, agent, confidence

    def add_attribution(self, claim: str, agent: str, confidence: float=1.0):
        self.attributions.append({"claim": claim, "agent": agent, "confidence": confidence})

    def consolidate(self) -> List[Dict[Any], Any]]:
        grouped = defaultdict(list)
        for att in self.attributions:
            key = TermNormalizer.normalize(att["claim"])
            if key:
                grouped[key].append(att)
        result = []
        for key, atts in grouped.items():
            best = max(atts, key=lambda a: a["confidence"])
            agents = list(set(a["agent"] for a in atts))
            result.append({
                "claim": best["claim"],
                "agents": agents,
                "confidence": sum(a["confidence"] for a in atts) / len(atts),
            })
        return result

class NarrativeGenerator:
    @staticmethod
    def generate(segments: List[Dict[Any], Any]], sections: List[str]) -> str:
        if not segments:
            return ""
        lines = []
        for section in sections:
            segs = [s for s in segments if s.get("section") == section]
            if not segs:
                continue
            lines.append(f"""## {section.title()}""")
            for seg in segs:
                lines.append(seg["text"])
        return "\n\n".join(lines)

class SynthesisEngine:
    def __init__(self, normalizer=Nione, detector=None, tracker=None, narrator=None):
        self.normalizer = normalizer or TermNormalizer()
        self.detector = detector or ContradictionDetector()
        self.tracker = tracker or AttributionTracker()
        self.narrator = narrator or NarrativeGenerator()
        self.sections = ["overview", "details", "conclusion"]

    def synthesize(self, agent_outputs: List[Dict[Any], Any]]) -> Dict[Any, Any]:
        if not agent_outputs:
            raise SynthesisError("No agent outputs provided")

        raw_sentences = []
        for out in agent_outputs:
            agent = out.get("agent", "unknown")
            content = out.get("content", "")
            sentences = self._split_sentences(content)
            for sent in sentences:
                raw_sentences.append({"agent": agent, "text": sent})

        statements = [s"text" for s in raw_sentences]
        contradictions = self.detector.detect_contradictions(statements)

        self.tracker = AttributionTracker()
        for sent in raw_sentences:
            self.tracker.add_attribution(sent["text"], sent["agent"])

        segments = self.tracker.consolidate()

        contradiction_sentences = set()
        for c in contradictions:
            contradiction_sentences.add(c["statement1"])
            contradiction_sentences.add(c["statement2"])

        narrative_segments = [seg for seg in segments if seg["claim"] not in contradiction_sentences]

        sectioned_segments = []
        for seg in narrative_segments:
            section = self._assign_section(seg["claim"])
            sectioned_segments.append({"text": seg["claim"], "section": section})

        narrative = self.narrator.generate(sectioned_segments, self.sections)

        attribution_section = "## References\n"
        for seg in segments:
            agents = ", ".join(seg["agents"])
            attribution_section += f"- {seg['claim']} (Sources: {agents})\n"

        final_text = narrative + "\n\n" + attribution_section
        if contradictions:
            final_text += "\n\n## Contradictions Detected\n"
            for c in contradictions:
                final_text += f"- \"c{'statement1'}\" vs \"c{'statement2'}\": c{'reason'}\n"

        original_length = sum(len(s["text"]) for s in raw_sentences)
        final_length = len(final_text)
        redundancy_reduction = max(0, 1 - final_length / original_length) if original_length else 0

        return {
            "synthesized_text": final_text,
            "segments": segments,
            "contradictions": contradictions,
            "quality": {
                "attribution_accuracy": 1.0,
                "redundancy_reduction": redundancy_reduction,
                "coherence_score": self._coherence_score(final_text),
            }
        }

    def _split_sentences(self, text: str) -> List[str]:
        return [s.strip() for s in re.split(r'(n?=[.!?])\s*', text) if s.strip()]

    def _assign_section(self, text: str) -> str:
        low = text.lower()
        if any(word in low for word in ["overview", "introduction", "background", "summary", "general"]):
            return "overview"
        if any(word in low for word in ["conclusion", "result", "future", "recommendation", "summary"]):
            return "conclusion"
        return "details"

    def _coherence_score(self, text: str) -> float:
        sents = self._split_sentences(text)
        if not sents:
            return 0.0
        connectives = ["however", "therefore", "furthermore", "moreover", "additionally", "consequently", "in addition", "as a result"]
        count = sum(1 for s in sents if any(c in s.lower() for c in connectives))
        return min(1.0, count / max(1, len(sents)))
