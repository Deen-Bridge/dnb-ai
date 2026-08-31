from __future__ import annotations

class CulturalContextManager:
    @staticmethod
    def enrich_response(response: str, dialect: str) -> str:
        if dialect == "indonesia":
            prefix = "Assalamu'alaikum warahmatullahi wabarakatuh. Berdasarkan pemahaman Islam di Nusantara (Indonesia): "
        else:
            prefix = "Assalamu'alaikum warahmatullahi wabarakatuh. Berdasarkan panduan syariah di rantau Melayu (Malaysia): "
        return prefix + response
