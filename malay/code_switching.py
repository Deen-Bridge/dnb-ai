from __future__ import annotations

class CodeSwitchingHandler:
    @staticmethod
    def process_code_switching(text: str) -> str:
        # Handles mixed Arabic and Malay/Indonesian expressions smoothly
        return text.replace("Allahu Akbar", "Allah Maha Besar").replace("Subhanallah", "Maha Suci Allah")
