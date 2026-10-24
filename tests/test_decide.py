"""Tests for src/decide.py — 5 hand-crafted perception scenarios."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.decide import decide
from src.schema import validate_row


def _obs(path="images/test/case_X/img_1.jpg", **kw):
    """Build a normalized image observation with sensible defaults."""
    base = {
        "machine_readable": True,
        "path": path,
        "object_type": "car",
        "object_matches_claim": True,
        "part_visible": "unknown",
        "part_clearly_inspectable": True,
        "issue_visible": "unknown",
        "issue_extent": "unclear",
        "quality": {
            "blurry": False,
            "cropped_or_obstructed": False,
            "low_light_or_glare": False,
            "wrong_angle": False,
        },
        "looks_manipulated": False,
        "looks_non_original": False,
        "one_line_observation": "test image",
    }
    base.update(kw)
    return base


def _claim_row(user_id="user_T", claim_object="car", user_claim="test claim"):
    return {
        "user_id": user_id,
        "image_paths": "images/test/case_X/img_1.jpg",
        "user_claim": user_claim,
        "claim_object": claim_object,
    }


def _extract(claimed_part="unknown", claimed_issue="unknown",
             claim_asserts_damage=True, depends_on_identity_or_side=False,
             text_instruction_present=False, claimed_object="car"):
    return {
        "claimed_object": claimed_object,
        "claimed_part": claimed_part,
        "claimed_issue": claimed_issue,
        "claim_asserts_damage": claim_asserts_damage,
        "depends_on_identity_or_side": depends_on_identity_or_side,
        "text_instruction_present": text_instruction_present,
    }


# ---- Case 1: CLEAR SUPPORTED ----

def test_clear_supported():
    row = _claim_row(claim_object="car")
    ext = _extract(claimed_part="door", claimed_issue="dent", claimed_object="car")
    obs = [_obs(
        object_type="car",
        object_matches_claim=True,
        part_visible="door",
        part_clearly_inspectable=True,
        issue_visible="dent",
        issue_extent="moderate",
    )]
    result = decide(row, ext, obs)

    assert result["claim_status"] == "supported"
    assert result["evidence_standard_met"] == "true"
    assert result["issue_type"] == "dent"
    assert result["object_part"] == "door"
    assert result["severity"] == "medium"
    assert "img_1" in result["supporting_image_ids"]
    assert validate_row(result) == []


# ---- Case 2: CLEAR CONTRADICTED ----

def test_clear_contradicted():
    row = _claim_row(claim_object="laptop")
    ext = _extract(
        claimed_object="laptop", claimed_part="screen",
        claimed_issue="crack", claim_asserts_damage=True,
    )
    obs = [_obs(
        object_type="laptop",
        object_matches_claim=True,
        part_visible="screen",
        part_clearly_inspectable=True,
        issue_visible="none",
        issue_extent="none",
    )]
    result = decide(row, ext, obs)

    assert result["claim_status"] == "contradicted"
    assert result["evidence_standard_met"] == "true"
    assert validate_row(result) == []


# ---- Case 3: INSUFFICIENT (quality) ----

def test_insufficient_quality():
    row = _claim_row(claim_object="package")
    ext = _extract(
        claimed_object="package", claimed_part="package_corner",
        claimed_issue="crushed_packaging", claim_asserts_damage=True,
    )
    obs = [_obs(
        object_type="package",
        object_matches_claim=True,
        part_visible="package_corner",
        part_clearly_inspectable=False,
        issue_visible="unknown",
        issue_extent="unclear",
        quality={
            "blurry": True,
            "cropped_or_obstructed": False,
            "low_light_or_glare": False,
            "wrong_angle": False,
        },
    )]
    result = decide(row, ext, obs)

    assert result["claim_status"] == "not_enough_information"
    assert result["evidence_standard_met"] == "false"
    assert "blurry_image" in result["risk_flags"]
    assert result["supporting_image_ids"] == "none"
    assert validate_row(result) == []


# ---- Case 4: WRONG OBJECT ----

def test_wrong_object():
    row = _claim_row(claim_object="car")
    ext = _extract(claimed_object="car", claimed_part="door", claimed_issue="dent")
    obs = [_obs(
        object_type="laptop",
        object_matches_claim=False,
        part_visible="screen",
        part_clearly_inspectable=True,
        issue_visible="crack",
        issue_extent="moderate",
    )]
    result = decide(row, ext, obs)

    assert "wrong_object" in result["risk_flags"]
    assert "claim_mismatch" in result["risk_flags"]
    assert result["claim_status"] == "contradicted"
    assert validate_row(result) == []


# ---- Case 5: HISTORY DOES NOT FLIP THE DECISION ----

def test_history_does_not_flip():
    row = _claim_row(claim_object="car")
    ext = _extract(claimed_part="door", claimed_issue="dent", claimed_object="car")
    obs = [_obs(
        object_type="car",
        object_matches_claim=True,
        part_visible="door",
        part_clearly_inspectable=True,
        issue_visible="dent",
        issue_extent="moderate",
    )]
    heavy_history = {
        "user_id": "user_T",
        "past_claim_count": "10",
        "accept_claim": "2",
        "manual_review_claim": "2",
        "rejected_claim": "6",
        "last_90_days_claim_count": "5",
        "history_flags": "frequent_claimant;escalation_history",
        "history_summary": "High-risk user with many rejected claims",
    }
    result = decide(row, ext, obs, history_row=heavy_history)

    assert result["claim_status"] == "supported", (
        "History must never flip claim_status"
    )
    assert "user_history_risk" in result["risk_flags"]
    assert validate_row(result) == []


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
