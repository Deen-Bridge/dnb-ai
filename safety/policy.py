"""Loading and validation for the versioned content-safety policy."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import yaml


VALID_ACTIONS = {"allow", "allow_with_guidance", "refuse"}


@dataclass(frozen=True)
class PolicyCategory:
    id: str
    description: str
    action: str
    failure_action: str
    guidance: str
    refusal: str
    keywords: List[str]
    output_patterns: List[str]


@dataclass(frozen=True)
class Policy:
    version: str
    scholar_referral_disclaimer: str
    benign_near_miss_patterns: List[str]
    categories: Dict[str, PolicyCategory]


@lru_cache(maxsize=1)
def load_policy(path: str = "") -> Policy:
    policy_path = Path(path) if path else Path(__file__).with_name("policy.yaml")
    with policy_path.open(encoding="utf-8") as policy_file:
        raw = yaml.safe_load(policy_file)

    categories = {}
    for item in raw["categories"]:
        if item["action"] not in VALID_ACTIONS or item["failure_action"] not in VALID_ACTIONS:
            raise ValueError(f"Invalid action in policy category {item['id']}")
        category = PolicyCategory(
            id=item["id"],
            description=item["description"],
            action=item["action"],
            failure_action=item["failure_action"],
            guidance=item.get("guidance", ""),
            refusal=item.get("refusal", ""),
            keywords=item.get("keywords", []),
            output_patterns=item.get("output_patterns", []),
        )
        categories[category.id] = category

    return Policy(
        version=str(raw["version"]),
        scholar_referral_disclaimer=raw["scholar_referral_disclaimer"],
        benign_near_miss_patterns=raw.get("benign_near_miss_patterns", []),
        categories=categories,
    )
