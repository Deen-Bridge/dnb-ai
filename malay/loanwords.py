from __future__ import annotations

class LoanwordRecognizer:
    ARABIC_LOANWORD_PREFIXES_OR_PATTERNS = [
        "ilmu", "makhluk", "syukur", "sabar", "tauhid", "iman", "takwa", "istighfar",
        "alhamdulillah", "insyaAllah", "bismillah", "qadar", "zakat", "puasa", "solat", "shalat"
    ]

    @classmethod
    def recognize_loanwords(cls, text: str) -> list[str]:
        words = text.split()
        found = []
        for w in words:
            lw = w.strip(",.?!;:()").lower()
            if lw in cls.ARABIC_LOANWORD_PREFIXES_OR_PATTERNS or lw.startswith("al-") or lw.endswith("ah") or lw.endswith("at"):
                if lw not in found:
                    found.append(lw)
        return found
