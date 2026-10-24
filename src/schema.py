"""Output schema constants and validation for the claims pipeline."""

import csv
import re

OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

CLAIM_STATUS = {"supported", "contradicted", "not_enough_information"}

ISSUE_TYPE = {
    "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
    "torn_packaging", "crushed_packaging", "water_damage", "stain", "none", "unknown",
}

SEVERITY = {"none", "low", "medium", "high", "unknown"}

OBJECT_PART = {
    "car": {
        "front_bumper", "rear_bumper", "door", "hood", "windshield", "side_mirror",
        "headlight", "taillight", "fender", "quarter_panel", "body", "unknown",
    },
    "laptop": {
        "screen", "keyboard", "trackpad", "hinge", "lid", "corner", "port", "base",
        "body", "unknown",
    },
    "package": {
        "box", "package_corner", "package_side", "seal", "label", "contents", "item",
        "unknown",
    },
}

RISK_FLAGS = {
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required",
}

_BOOL_VALUES = {"true", "false"}
_SUPPORTING_ID_RE = re.compile(r"^[A-Za-z0-9_]+$")


def validate_row(row: dict) -> list[str]:
    errors: list[str] = []

    if row.get("claim_status") not in CLAIM_STATUS:
        errors.append(f"claim_status: invalid value {row.get('claim_status')!r}")

    if row.get("issue_type") not in ISSUE_TYPE:
        errors.append(f"issue_type: invalid value {row.get('issue_type')!r}")

    if row.get("severity") not in SEVERITY:
        errors.append(f"severity: invalid value {row.get('severity')!r}")

    claim_object = row.get("claim_object", "")
    allowed_parts = OBJECT_PART.get(claim_object)
    if allowed_parts is None:
        errors.append(f"claim_object: unknown value {claim_object!r}")
    elif row.get("object_part") not in allowed_parts:
        errors.append(
            f"object_part: {row.get('object_part')!r} not valid for claim_object={claim_object!r}"
        )

    if row.get("valid_image") not in _BOOL_VALUES:
        errors.append(f"valid_image: must be 'true' or 'false', got {row.get('valid_image')!r}")

    if row.get("evidence_standard_met") not in _BOOL_VALUES:
        errors.append(
            f"evidence_standard_met: must be 'true' or 'false', got {row.get('evidence_standard_met')!r}"
        )

    risk_raw = row.get("risk_flags", "")
    tokens = [t.strip() for t in risk_raw.split(";") if t.strip()]
    bad_flags = [t for t in tokens if t not in RISK_FLAGS]
    if bad_flags:
        errors.append(f"risk_flags: unknown flag(s) {bad_flags}")
    if "none" in tokens and len(tokens) > 1:
        errors.append("risk_flags: 'none' must not appear alongside other flags")

    sid_raw = row.get("supporting_image_ids", "")
    if sid_raw != "none":
        parts = [p.strip() for p in sid_raw.split(";") if p.strip()]
        if not parts:
            errors.append("supporting_image_ids: empty value (use 'none' if no images)")
        for p in parts:
            if not _SUPPORTING_ID_RE.match(p):
                errors.append(f"supporting_image_ids: invalid id {p!r} (no spaces allowed)")
                break

    return errors


def write_output(rows: list[dict], path: str) -> None:
    for i, row in enumerate(rows):
        errs = validate_row(row)
        assert not errs, f"Row {i} (user_id={row.get('user_id')}) failed validation: {errs}"

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in OUTPUT_COLUMNS})
