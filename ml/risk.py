"""Risk levels and the messages shown to the user (D-48, D-49).

- low:      probability below the model's screening threshold (80% recall, stored in metadata.json; 15.1% for the
            current model) -> not flagged
- elevated: from the threshold up to HIGH_RISK                                                -> flagged
- high:     HIGH_RISK (30%, about twice the average risk) and above                           -> flagged

The probabilities are calibrated (RESULTS.md, step 5): "30%" means about 30 of 100 people with similar answers
have prediabetes or diabetes. The messages recommend seeing a doctor, never a specific test, and never state a
diagnosis (D-07).
"""

HIGH_RISK = 0.30

DISCLAIMER = "This is a screening estimate based on survey answers, not a diagnosis."

MESSAGES = {
    "low": "Your estimated risk is low ({pct}). If you have symptoms or concerns, talk to your doctor.",
    "elevated": ("Your estimated risk is above average ({pct}). "
                 "Consider talking to your doctor about your diabetes risk."),
    "high": ("Your estimated risk is high ({pct}), about twice the average or more. "
             "We recommend that you see your doctor about your diabetes risk."),
}


def format_percent(probability):
    """'33%', or 'less than 1%' for very low risks (instead of a misleading '0%')."""
    return "less than 1%" if probability < 0.005 else f"{probability:.0%}"


def risk_level(probability, threshold, high_risk=HIGH_RISK):
    if probability >= high_risk:
        return "high"
    if probability >= threshold:
        return "elevated"
    return "low"


def assess(probability, threshold, high_risk=HIGH_RISK):
    """Turn one predicted probability into the result shown to the user."""
    level = risk_level(probability, threshold, high_risk)
    return {
        "probability": round(float(probability), 4),
        "risk_level": level,
        "flagged": level != "low",
        "message": MESSAGES[level].format(pct=format_percent(probability)),
        "disclaimer": DISCLAIMER,
    }
