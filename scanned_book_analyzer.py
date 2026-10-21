"""Scanned Islamic Book Page Layout & Citation Analyzer (#242).

Separates main body text (matn) from footnotes (hashiyah) and marginalia
(ta'liqat), extracts Quranic verses with surah/ayah references, parses Hadith
citations, and constructs a structured page representation.
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scanned-book", tags=["scanned-book"])


class ScannedTextBlock(BaseModel):
    block_id: str = Field(..., min_length=1, max_length=80)
    y_norm: float = Field(..., ge=0.0, le=1.0, description="Normalized vertical position (0=top, 1=bottom)")
    x_norm: float = Field(default=0.5, ge=0.0, le=1.0, description="Normalized horizontal center (0=left, 1=right)")
    font_scale: float = Field(default=1.0, gt=0.1, le=5.0)
    text: str = Field(..., min_length=1, max_length=10000)


class ScannedBookPageRequest(BaseModel):
    book_id: str = Field(..., min_length=1, max_length=120)
    page_number: int = Field(..., ge=1, le=10000)
    blocks: list[ScannedTextBlock] = Field(..., min_length=1, max_length=200)


class ClassifiedPageBlock(BaseModel):
    block_id: str
    zone: str  # "header", "main_text", "footnote", "marginalia"
    text: str


class ScannedBookPageResponse(BaseModel):
    book_id: str
    page_number: int
    main_text: str
    footnotes: list[str]
    marginalia: list[str]
    classified_blocks: list[ClassifiedPageBlock]
    quran_verses: list[str]
    hadith_citations: list[str]


_QURAN_VERSE_RE = re.compile(r"\b(\d{1,3}:\d{1,3})\b")
_HADITH_CITATION_RE = re.compile(
    r"\b(رواه\s+البخاري|رواه\s+مسلم|صحيح\s+البخاري|صحيح\s+مسلم|Sahih\s+(?:al-)?Bukhari|Sahih\s+Muslim)\b",
    re.IGNORECASE,
)


def analyze_scanned_book_page(req: ScannedBookPageRequest) -> ScannedBookPageResponse:
    classified: list[ClassifiedPageBlock] = []
    main_parts: list[str] = []
    footnotes: list[str] = []
    marginalia: list[str] = []
    quran_verses: list[str] = []
    hadith_citations: list[str] = []

    for blk in sorted(req.blocks, key=lambda b: (b.y_norm, -b.x_norm)):
        txt = blk.text.strip()
        if blk.x_norm < 0.15 or blk.x_norm > 0.85:
            zone = "marginalia"
            marginalia.append(txt)
        elif blk.y_norm >= 0.78 or blk.font_scale < 0.85 or re.match(r"^\(\d+\)|^\[\d+\]|^\d+\s*[-–]", txt):
            zone = "footnote"
            footnotes.append(txt)
        elif blk.y_norm <= 0.08:
            zone = "header"
        else:
            zone = "main_text"
            main_parts.append(txt)

        classified.append(ClassifiedPageBlock(block_id=blk.block_id, zone=zone, text=txt))

        for m in _QURAN_VERSE_RE.finditer(txt):
            ref = m.group(1)
            if ref not in quran_verses:
                quran_verses.append(ref)
        for hm in _HADITH_CITATION_RE.finditer(txt):
            cit = hm.group(1)
            if cit not in hadith_citations:
                hadith_citations.append(cit)

    return ScannedBookPageResponse(
        book_id=req.book_id,
        page_number=req.page_number,
        main_text="\n\n".join(main_parts),
        footnotes=footnotes,
        marginalia=marginalia,
        classified_blocks=classified,
        quran_verses=quran_verses,
        hadith_citations=hadith_citations,
    )


@router.post("/analyze", response_model=ScannedBookPageResponse)
async def analyze_scanned_book_endpoint(body: ScannedBookPageRequest) -> ScannedBookPageResponse:
    """Analyze a scanned Islamic book page into main matn, footnotes, marginalia, and citations."""
    return analyze_scanned_book_page(body)
