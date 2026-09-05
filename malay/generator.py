from __future__ import annotations

class MalayContentGenerator:
    @staticmethod
    def generate_islamic_content(topic: str, dialect: str) -> str:
        if dialect == "indonesia":
            return f"Penjelasan mengenai {topic} dalam konteks ajaran Islam yang rahmatan lil 'alamin di Indonesia."
        else:
            return f"Penerangan mengenai {topic} menurut panduan Ahli Sunnah Wal Jamaah di Malaysia."
