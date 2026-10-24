"""Tests for src/schema.py — one valid row and one violation per rule."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.schema import validate_row, write_output, OUTPUT_COLUMNS
import tempfile, csv

VALID_ROW = {
    "user_id": "user_001",
    "image_paths": "images/test/case_001/img_1.jpg",
    "user_claim": "The rear bumper has a dent.",
    "claim_object": "car",
    "evidence_standard_met": "true",
    "evidence_standard_met_reason": "Bumper visible in image.",
    "risk_flags": "none",
    "issue_type": "dent",
    "object_part": "rear_bumper",
    "claim_status": "supported",
    "claim_status_justification": "Image shows dent on bumper.",
    "supporting_image_ids": "img_1",
    "valid_image": "true",
    "severity": "medium",
}


def _with(overrides):
    row = dict(VALID_ROW)
    row.update(overrides)
    return row


def test_valid_row():
    assert validate_row(VALID_ROW) == []


def test_invalid_claim_status():
    errs = validate_row(_with({"claim_status": "approved"}))
    assert any("claim_status" in e for e in errs)


def test_invalid_issue_type():
    errs = validate_row(_with({"issue_type": "explosion"}))
    assert any("issue_type" in e for e in errs)


def test_invalid_severity():
    errs = validate_row(_with({"severity": "critical"}))
    assert any("severity" in e for e in errs)


def test_object_part_wrong_for_claim_object():
    errs = validate_row(_with({"claim_object": "car", "object_part": "screen"}))
    assert any("object_part" in e for e in errs)


def test_unknown_claim_object():
    errs = validate_row(_with({"claim_object": "bicycle", "object_part": "wheel"}))
    assert any("claim_object" in e for e in errs)


def test_valid_image_not_bool():
    errs = validate_row(_with({"valid_image": "yes"}))
    assert any("valid_image" in e for e in errs)


def test_evidence_standard_met_not_bool():
    errs = validate_row(_with({"evidence_standard_met": "True"}))
    assert any("evidence_standard_met" in e for e in errs)


def test_risk_flags_unknown_flag():
    errs = validate_row(_with({"risk_flags": "blurry_image;fake_flag"}))
    assert any("risk_flags" in e and "unknown" in e for e in errs)


def test_risk_flags_none_with_others():
    errs = validate_row(_with({"risk_flags": "none;blurry_image"}))
    assert any("none" in e and "must not appear" in e for e in errs)


def test_supporting_image_ids_spaces():
    errs = validate_row(_with({"supporting_image_ids": "img 1;img2"}))
    assert any("supporting_image_ids" in e for e in errs)


def test_supporting_image_ids_empty():
    errs = validate_row(_with({"supporting_image_ids": ""}))
    assert any("supporting_image_ids" in e for e in errs)


def test_supporting_image_ids_none_literal():
    assert validate_row(_with({"supporting_image_ids": "none"})) == []


def test_supporting_image_ids_multi():
    assert validate_row(_with({"supporting_image_ids": "img_1;img_2"})) == []


def test_write_output_valid(tmp_path):
    path = str(tmp_path / "out.csv")
    write_output([VALID_ROW], path)
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    assert list(rows[0].keys()) == OUTPUT_COLUMNS


def test_write_output_rejects_invalid(tmp_path):
    bad = _with({"claim_status": "yolo"})
    path = str(tmp_path / "out.csv")
    try:
        write_output([bad], path)
        assert False, "should have raised"
    except AssertionError as e:
        assert "claim_status" in str(e)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
