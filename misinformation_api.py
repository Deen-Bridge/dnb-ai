"""API router for the religious misinformation flagging system (#181).

Exposes endpoints to:
- Scan text for misinformation
- Validate quotations
- Query the misconception database
- Get correction suggestions
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from misinformation import (
    MisinfoScanResult,
    detect_misinformation,
    get_all_misconceptions,
    get_misconception_categories,
    get_misconceptions_by_category,
    suggest_correction,
    validate_quotation,
)

router = APIRouter(prefix="/misinformation", tags=["misinformation"])


# ---------------------------------------------------------------------------
# Request / Response models (defined here to keep the router self-contained)
# ---------------------------------------------------------------------------


class ScanRequest(BaseModel):
    text: str = Field(..., max_length=16000, description="Text to scan for religious misinformation.")


class QuotationValidationRequest(BaseModel):
    quoted_text: str = Field(..., max_length=4000, description="The quoted text to validate.")
    context: str = Field("quran", description="Context: 'quran' or 'hadith'.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/scan", response_model=MisinfoScanResult)
async def scan_text(body: ScanRequest, request: Request) -> MisinfoScanResult:
    """Scan a text for known religious misconceptions and misinformation.

    Returns flags with severity levels, corrections, and authoritative sources.
    Critical-severity findings will have `should_block` set to True.
    """
    result = detect_misinformation(body.text)
    return result


@router.post("/validate-quotation")
async def validate_quotation_endpoint(body: QuotationValidationRequest) -> dict:
    """Validate whether a quoted text shows signs of fabrication or misattribution."""
    match = validate_quotation(body.quoted_text, context=body.context)
    return match.model_dump()


@router.post("/corrections")
async def get_corrections(body: ScanRequest) -> dict:
    """Scan text and return correction suggestions for any misinformation found."""
    correction = suggest_correction(body.text)
    if correction is None:
        return {"found": False, "corrections": None}
    return {"found": True, "corrections": correction}


@router.get("/database")
async def list_misconceptions(category: str | None = None) -> dict:
    """List all misconceptions in the database, optionally filtered by category.

    Categories: tawheed, aqeedah, ahkam, rights, usul, tafsir
    """
    if category:
        items = get_misconceptions_by_category(category)
    else:
        items = get_all_misconceptions()
    return {
        "total": len(items),
        "category": category,
        "misconceptions": items,
    }


@router.get("/database/categories")
async def list_categories() -> dict:
    """List all misconception categories."""
    return {"categories": get_misconception_categories()}
