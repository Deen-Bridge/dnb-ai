from __future__ import annotations

class MalayRouter:
    @staticmethod
    def route_request(text: str) -> str:
        if any(term in text.lower() for term in ["shalat", "zakat", "haji", "puasa"]):
            return "fiqh_ibadat"
        return "general_islamic_qa"
