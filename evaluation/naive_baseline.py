"""Naive baseline decision strategy — trusts the first image's raw damage signal.

No part matching, no fraud/manipulation filtering, no evidence gating.
Used only as a strategy-comparison baseline against src.decide.decide.
"""


def decide_naive(image_observations: list[dict]) -> str:
    """Return claim_status using only the first machine-readable image."""
    readable = [o for o in image_observations if o.get("machine_readable", False)]
    if not readable:
        return "not_enough_information"

    first = readable[0]
    issue = first.get("issue_visible", "unknown")

    if issue not in ("none", "unknown"):
        return "supported"
    if issue == "none":
        return "contradicted"
    return "not_enough_information"
