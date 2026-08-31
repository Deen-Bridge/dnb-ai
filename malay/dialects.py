from __future__ import annotations

class DialectHandler:
    @staticmethod
    def detect_dialect(text: str) -> str:
        lower = text.lower()
        indonesian_markers = ["shalat", "puasa", "uang", "rumah", "dengan", "tidak", "kalian", "bagaimana"]
        malaysian_markers = ["solat", "wang", "rumah", "dengan", "tidak", "awak", "kamu", "macam mana"]
        
        ind_score = sum(1 for m in indonesian_markers if m in lower)
        mal_score = sum(1 for m in malaysian_markers if m in lower)
        
        if "shalat" in lower or "uang" in lower:
            return "indonesia"
        if "solat" in lower or "wang" in lower:
            return "malaysia"
            
        return "indonesia" if ind_score >= mal_score else "malaysia"

    @staticmethod
    def normalize_dialect(text: str, target_dialect: str) -> str:
        words = text.split()
        normalized = []
        for w in words:
            lw = w.lower()
            if target_dialect == "malaysia":
                if lw == "shalat":
                    w = "solat"
                elif lw == "uang":
                    w = "wang"
            elif target_dialect == "indonesia":
                if lw == "solat":
                    w = "shalat"
                elif lw == "wang":
                    w = "uang"
            normalized.append(w)
        return " ".join(normalized)
