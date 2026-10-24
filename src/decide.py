"""Decision layer — pure logic, no model calls."""

from pathlib import Path

from src.schema import OBJECT_PART, ISSUE_TYPE

EXTENT_TO_SEVERITY = {
    "none": "none",
    "minor": "low",
    "moderate": "medium",
    "severe": "high",
    "unclear": "unknown",
}

# ---------------------------------------------------------------------------
# Per-object requirement router
# ---------------------------------------------------------------------------

CAR_BODY_PARTS = {"front_bumper", "rear_bumper", "door", "hood", "fender", "quarter_panel", "body"}
CAR_GLASS_PARTS = {"windshield", "headlight", "taillight", "side_mirror"}
LAPTOP_SURFACE_PARTS = {"screen", "keyboard", "trackpad"}
LAPTOP_BODY_PARTS = {"hinge", "lid", "corner", "port", "base", "body"}
PACKAGE_EXTERIOR_PARTS = {"box", "package_corner", "package_side", "seal"}
PACKAGE_LABEL_PARTS = {"label"}
PACKAGE_CONTENT_PARTS = {"contents", "item"}


def _requirement_for(claim_object, part, issue, depends_on_identity):
    if claim_object == "car":
        if depends_on_identity:
            return ("REQ_CAR_IDENTITY_OR_SIDE",
                    "image set should show enough context to match the claimed vehicle and part")
        if part in CAR_GLASS_PARTS or issue in ("crack", "glass_shatter", "broken_part", "missing_part"):
            return ("REQ_CAR_GLASS_LIGHT_MIRROR",
                    "glass, light, mirror, or component should be visible clearly enough to inspect")
        if part in CAR_BODY_PARTS or issue in ("dent", "scratch"):
            return ("REQ_CAR_BODY_PANEL",
                    "car panel or bumper should be visible from an angle where surface marks can be assessed")
    elif claim_object == "laptop":
        if part in LAPTOP_SURFACE_PARTS:
            return ("REQ_LAPTOP_SCREEN_KEYBOARD_TRACKPAD",
                    "screen, keyboard, or trackpad should be visible clearly enough to inspect")
        if part in LAPTOP_BODY_PARTS:
            return ("REQ_LAPTOP_BODY_HINGE_PORT",
                    "hinge, lid, corner, body, or port should be visible")
    elif claim_object == "package":
        if part in PACKAGE_CONTENT_PARTS:
            return ("REQ_PACKAGE_CONTENTS",
                    "opened package and contents should be visible")
        if part in PACKAGE_LABEL_PARTS or issue in ("water_damage", "stain"):
            return ("REQ_PACKAGE_LABEL_OR_STAIN",
                    "affected surface or label should be visible")
        if part in PACKAGE_EXTERIOR_PARTS or issue in ("torn_packaging", "crushed_packaging"):
            return ("REQ_PACKAGE_EXTERIOR",
                    "package exterior should be visible to inspect packaging damage")
    return ("REQ_GENERAL_OBJECT_PART",
            "claimed object and relevant part should be visible clearly enough to inspect")


def _img_id(path: str) -> str:
    return Path(path).stem


# ---------------------------------------------------------------------------
# Part adjacency — suppress wrong_object_part for neighbouring parts
# ---------------------------------------------------------------------------

_ADJACENT_PARTS = {
    "car": [
        {"front_bumper", "hood", "headlight", "fender", "body"},
        {"rear_bumper", "taillight", "quarter_panel", "body"},
        {"door", "side_mirror", "fender", "quarter_panel", "body"},
    ],
    "laptop": [
        {"lid", "corner", "body", "base", "hinge"},
        {"keyboard", "trackpad"},
        {"port", "body", "base"},
    ],
    "package": [
        {"box", "package_corner", "package_side", "seal"},
        {"label", "box"},
        {"contents", "item"},
    ],
}


def _parts_grossly_different(claim_object: str, part_a: str, part_b: str) -> bool:
    if part_a == part_b:
        return False
    for group in _ADJACENT_PARTS.get(claim_object, []):
        if part_a in group and part_b in group:
            return False
    return True


# ---------------------------------------------------------------------------
# Part containment — one-directional: a generic part satisfies a claim about
# a specific sub-region it physically contains. Cars use exact match only;
# distinct sibling panels never satisfy each other.
# ---------------------------------------------------------------------------

_PART_CONTAINMENT = {
    "package": {
        "box": {"package_corner", "package_side", "seal", "label"},
    },
    "laptop": {
        "lid": {"corner"},
        "body": {"corner"},
        "base": {"corner"},
    },
}


def _part_satisfies_claim(claim_object: str, perceived_part: str, claimed_part: str) -> bool:
    if perceived_part == claimed_part:
        return True
    contains = _PART_CONTAINMENT.get(claim_object, {}).get(perceived_part, set())
    return claimed_part in contains


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_damage_suspect(obs: dict) -> bool:
    return obs.get("looks_manipulated", False) and not obs.get("looks_non_original", False)


def _is_fabricated(obs: dict) -> bool:
    return obs.get("looks_manipulated", False) and obs.get("looks_non_original", False)


def _add_history_flags(risk_flags: set, history_row: dict):
    h_flags = history_row.get("history_flags", "none")
    rejected = int(history_row.get("rejected_claim", "0") or "0")
    total = int(history_row.get("past_claim_count", "0") or "0")
    if h_flags != "none" or (total > 0 and rejected / total > 0.3):
        risk_flags.add("user_history_risk")


def _finalize(row_fields, risk_flags, claim_extract, history_row):
    if claim_extract.get("text_instruction_present", False):
        risk_flags.add("text_instruction_present")
    if history_row:
        _add_history_flags(risk_flags, history_row)
    if len(risk_flags) >= 3 or "possible_manipulation" in risk_flags:
        risk_flags.add("manual_review_required")
    row_fields["risk_flags"] = ";".join(sorted(risk_flags)) if risk_flags else "none"
    return row_fields


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def decide(claim_row, claim_extract, image_observations, history_row=None):
    """Return a full output row (dict matching OUTPUT_COLUMNS)."""
    claim_object = claim_row["claim_object"]
    claimed_part = claim_extract.get("claimed_part", "unknown")
    claimed_issue = claim_extract.get("claimed_issue", "unknown")
    depends_on_id = claim_extract.get("depends_on_identity_or_side", False)
    asserts_damage = claim_extract.get("claim_asserts_damage", False)

    _, req_text = _requirement_for(claim_object, claimed_part, claimed_issue, depends_on_id)

    readable = [o for o in image_observations if o.get("machine_readable", False)]
    risk_flags: set[str] = set()
    trustworthy: list[dict] = []

    # ----- phase 1: classify images, collect risk flags -----
    for obs in readable:
        q = obs.get("quality", {})
        if q.get("blurry"):
            risk_flags.add("blurry_image")
        if q.get("cropped_or_obstructed"):
            risk_flags.add("cropped_or_obstructed")
        if q.get("wrong_angle"):
            risk_flags.add("wrong_angle")

        # Rule 4: low_light_or_glare only when it impairs inspection
        if q.get("low_light_or_glare") and not obs.get("part_clearly_inspectable", False):
            risk_flags.add("low_light_or_glare")

        if obs.get("looks_manipulated"):
            risk_flags.add("possible_manipulation")

        # Rule 3: non_original_image only when BOTH flags confirm fabrication
        if obs.get("looks_manipulated") and obs.get("looks_non_original"):
            risk_flags.add("non_original_image")

        # Rule 1: claim_mismatch only for wrong object (not issue/part differences)
        if not obs.get("object_matches_claim", False):
            risk_flags.add("wrong_object")
            risk_flags.add("claim_mismatch")

        if not _is_fabricated(obs):
            trustworthy.append(obs)

    # Fix B: valid_image is false only when the set is genuinely unusable —
    # unreadable, or fabrication is corroborated (manipulated AND
    # non_original together). A lone looks_non_original is a documented
    # false-positive signal and must not invalidate the image on its own.
    all_fabricated = readable and all(_is_fabricated(o) for o in readable)
    valid_image = len(readable) > 0 and not all_fabricated

    base = {
        "user_id": claim_row["user_id"],
        "image_paths": claim_row["image_paths"],
        "user_claim": claim_row["user_claim"],
        "claim_object": claim_object,
    }

    # ----- Bug 5: unverifiable absence claims -----
    if claimed_issue == "missing_part" and claimed_part in ("contents", "item"):
        if asserts_damage:
            risk_flags.add("damage_not_visible")
        base.update(
            evidence_standard_met="false",
            evidence_standard_met_reason=(
                "The images cannot establish what should be present "
                "to verify the absence claim."
            ),
            issue_type="unknown",
            object_part=claimed_part,
            claim_status="not_enough_information",
            claim_status_justification=(
                "The claim is about missing contents but images "
                "cannot verify what should be present."
            ),
            supporting_image_ids="none",
            valid_image="true" if valid_image else "false",
            severity="unknown",
        )
        return _finalize(base, risk_flags, claim_extract, history_row)

    # ----- phase 2: split trustworthy by object match -----
    tw_match = [o for o in trustworthy if o.get("object_matches_claim", False)]
    tw_wrong = [o for o in trustworthy if not o.get("object_matches_claim", False)]

    # ----- phase 3: scan for part-level evidence -----
    # Exact-part evidence always outranks containment evidence: a direct shot
    # of the claimed part (damaged or clean) is more conclusive than a generic
    # container view that merely satisfies the claim by containment.
    supporting_ids: list[str] = []
    part_dam_exact: list[dict] = []
    part_clean_exact: list[dict] = []
    part_dam_contain: list[dict] = []
    part_clean_contain: list[dict] = []
    other_insp: list[dict] = []

    for obs in tw_match:
        part = obs.get("part_visible", "unknown")

        # Rule 2: wrong_object_part only for grossly different parts
        if (part not in ("unknown", "unclear")
                and claimed_part not in ("unknown", "unclear")
                and part != claimed_part
                and _parts_grossly_different(claim_object, part, claimed_part)):
            risk_flags.add("wrong_object_part")

        if not obs.get("part_clearly_inspectable", False):
            continue

        exact = (part == claimed_part
                 or claimed_part in ("unknown", "unclear")
                 or part in ("unknown", "unclear"))
        contained = (not exact) and _part_satisfies_claim(claim_object, part, claimed_part)

        issue = obs.get("issue_visible", "unknown")
        suspect = _is_damage_suspect(obs)
        has_damage = not suspect and issue not in ("none", "unknown")
        is_clean = issue == "none" or suspect

        if exact:
            if has_damage:
                part_dam_exact.append(obs)
            elif is_clean:
                part_clean_exact.append(obs)
        elif contained:
            if has_damage:
                part_dam_contain.append(obs)
            elif is_clean:
                part_clean_contain.append(obs)
        else:
            other_insp.append(obs)

    # ----- phase 4: determine evidence_met / claim_status / best_obs -----
    best_obs: dict | None = None

    if not readable:
        evidence_met = False
        evidence_reason = "No machine-readable images provided."
        claim_status = "not_enough_information"
        justification = evidence_reason

    elif tw_wrong and not tw_match:
        evidence_met = True
        evidence_reason = (
            "The submitted image clearly shows an object that does not "
            f"match the claimed {claim_object}."
        )
        claim_status = "contradicted"
        justification = "The image shows a different object than the claimed one."
        best_obs = tw_wrong[0]

    elif not trustworthy and readable:
        evidence_met = True
        evidence_reason = (
            "The submitted image appears non-original or digitally altered."
        )
        claim_status = "contradicted"
        justification = (
            "The submitted evidence appears fabricated or manipulated."
        )
        best_obs = readable[0]

    elif part_dam_exact or part_clean_exact or part_dam_contain or part_clean_contain:
        # Priority: exact-part damage > exact-part clean > containment damage
        # > containment clean. A direct shot of the claimed part always wins
        # over a generic container view.
        if part_dam_exact:
            best_obs = part_dam_exact[0]
            damaged = True
        elif part_clean_exact:
            best_obs = part_clean_exact[0]
            damaged = False
        elif part_dam_contain:
            best_obs = part_dam_contain[0]
            damaged = True
        else:
            best_obs = part_clean_contain[0]
            damaged = False

        pl = best_obs.get("part_visible", claimed_part).replace("_", " ")
        evidence_met = True
        evidence_reason = (
            f"The {pl} is visible and inspectable in the submitted image."
        )
        if damaged:
            supporting_ids.append(_img_id(best_obs["path"]))
            if asserts_damage:
                claim_status = "supported"
                iv = best_obs.get("issue_visible", "unknown")
                justification = f"Image shows {iv.replace('_', ' ')} on the {pl}."
            else:
                claim_status = "not_enough_information"
                justification = "The claim does not explicitly assert damage."
        else:
            if asserts_damage:
                claim_status = "contradicted"
                justification = (
                    "The claimed part is clearly visible but no damage is observed."
                )
            else:
                claim_status = "not_enough_information"
                justification = "The claim does not explicitly assert damage."

    elif other_insp:
        # Bug 1: the claimed part itself is not shown — only a different,
        # non-adjacent part is visible. Damage on a part the claim never
        # named cannot support the claim, no matter how clear that damage is.
        best_obs = other_insp[0]
        evidence_met = False
        evidence_reason = f"No image shows the {req_text}."
        claim_status = "not_enough_information"
        justification = (
            "Damage could not be clearly assessed from the submitted images."
        )

    else:
        evidence_met = False
        evidence_reason = f"No image shows the {req_text}."
        claim_status = "not_enough_information"
        justification = evidence_reason

    # ----- post-phase-4 risk flags -----

    # Rule 5: damage_not_visible — fires ONLY when the claimed part could not
    # be clearly assessed at all (no trustworthy image — exact or by
    # containment — of it was clearly inspectable). If the claimed part WAS
    # clearly inspectable and showed no damage, that's a contradiction, not a
    # visibility gap.
    is_wrong_object_path = tw_wrong and not tw_match
    is_fabrication_path = not trustworthy and readable
    has_part_evidence = (part_dam_exact or part_clean_exact
                         or part_dam_contain or part_clean_contain)
    if (asserts_damage and not has_part_evidence
            and not is_wrong_object_path and not is_fabrication_path):
        risk_flags.add("damage_not_visible")

    # Rule 1: claim_mismatch for identity-dependent claims
    if depends_on_id and not supporting_ids:
        risk_flags.add("claim_mismatch")

    # Bug 2: an unusable image set (valid_image=false) can never back a
    # 'supported' claim — downgrade for consistency.
    if not valid_image and claim_status == "supported":
        claim_status = "not_enough_information"
        evidence_met = False
        evidence_reason = (
            "The submitted image set is not usable as evidence "
            "(all readable images appear non-original)."
        )
        justification = (
            "The image set is not usable as evidence, so the claim "
            "cannot be confirmed."
        )
        supporting_ids = []

    # ----- derived output fields -----
    if best_obs:
        if is_wrong_object_path and best_obs in tw_wrong:
            object_part = "unknown"
            issue_type = "unknown"
            severity = "unknown"
        else:
            object_part = best_obs.get("part_visible", "unknown")
            if object_part in ("unclear",):
                object_part = claimed_part

            if _is_damage_suspect(best_obs):
                issue_type = "none"
                severity = "none"
            else:
                iv = best_obs.get("issue_visible", "unknown")
                issue_type = iv if iv != "unknown" else claimed_issue
                severity = EXTENT_TO_SEVERITY.get(
                    best_obs.get("issue_extent", "unclear"), "unknown"
                )
    else:
        object_part = claimed_part
        issue_type = claimed_issue
        severity = "unknown"

    supporting_str = ";".join(supporting_ids) if supporting_ids else "none"

    base.update(
        evidence_standard_met="true" if evidence_met else "false",
        evidence_standard_met_reason=evidence_reason,
        issue_type=issue_type,
        object_part=object_part,
        claim_status=claim_status,
        claim_status_justification=justification,
        supporting_image_ids=supporting_str,
        valid_image="true" if valid_image else "false",
        severity=severity,
    )
    return _finalize(base, risk_flags, claim_extract, history_row)
