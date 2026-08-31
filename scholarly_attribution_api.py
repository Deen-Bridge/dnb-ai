"""API router for the Fabricated Scholarly Attribution Prevention System (#173).

Exposes endpoints to:
- Validate scholarly attributions in free text
- Validate a single scholar-opinion pair
- Query the scholar biography database
- Get attribution audit trail
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from scholarly_attribution import (
    AttributionValidationResult,
    get_scholars_by_school,
    get_scholars_list,
    validate_single_attribution,
    validate_scholarly_attribution,
)

router = APIRouter(prefix="/scholarly-attribution", tags=["scholarly-attribution"])


# -----------------------------------------------------------------------
# Request / Response models
# -----------------------------------------------------------------------


class ValidateTextRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=16000,
        description="Text containing scholarly attributions to validate.",
    )


class ValidateSingleRequest(BaseModel):
    scholar_name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Name of the scholar being attributed.",
    )
    opinion: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The opinion or position being attributed to the scholar.",
    )


# -----------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------


@router.post("/validate", response_model=AttributionValidationResult)
async def validate_attributions(body: ValidateTextRequest, request: Request) -> AttributionValidationResult:
    """Validate all scholarly attributions in the given text.

    Scans for fabricated opinions, misattributions, anachronisms,
    flattened nuances, and false consensus claims. Returns an audit
    trail of every validation decision.
    """
    result = validate_scholarly_attribution(body.text)
    return result


@router.post("/validate-single")
async def validate_single(body: ValidateSingleRequest) -> dict:
    """Validate a single scholar-opinion attribution pair.

    Use this to check one specific claim without parsing free text.
    """
    return validate_single_attribution(body.scholar_name, body.opinion)


@router.get("/scholars")
async def list_scholars(school: str | None = None) -> dict:
    """List scholars in the verified biography database.

    Optionally filter by school/madhhab.
    """
    if school:
        items = get_scholars_by_school(school)
    else:
        items = get_scholars_list()
    return {
        "total": len(items),
        "school": school,
        "scholars": items,
    }


@router.get("/scholars/{scholar_id}")
async def get_scholar(scholar_id: str) -> dict:
    """Get a single scholar's biography and known positions."""
    from scholarly_attribution import get_scholar_by_id

    scholar = get_scholar_by_id(scholar_id)
    if not scholar:
        from errors import APIException

        raise APIException(
            status_code=404,
            detail=f"Scholar '{scholar_id}' not found in the verified database.",
            hint="Use GET /scholarly-attribution/scholars to list all available scholars.",
        )
    return scholar.model_dump()
