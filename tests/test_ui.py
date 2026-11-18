import pytest

from api.main import create_app
from ml.validation import FEATURE_COLUMNS, validate_features
from tests.test_validation import PERSON
from ui.app import CHOICE_FIELDS, SECTIONS, bmi, build_payload, layout, marker_position, render_result

ANSWERS = {field: PERSON[field] for field in CHOICE_FIELDS}


def test_form_covers_every_feature_once():
    assert len(CHOICE_FIELDS) == len(set(CHOICE_FIELDS)) == 20
    assert set(CHOICE_FIELDS) | {"BMI"} == set(FEATURE_COLUMNS)


def test_option_codes_match_the_allowed_ranges():
    from ml.validation import FEATURE_RANGES
    for _, questions in SECTIONS:
        for field, _, options, kind in questions:
            if options:
                lo, hi = FEATURE_RANGES[field]
                assert sorted(o["value"] for o in options) == list(range(lo, hi + 1)), field


def test_bmi():
    assert bmi(180, 81) == pytest.approx(25.0)
    assert bmi(None, 80) is None and bmi(180, 0) is None


def test_complete_form_gives_a_valid_payload():
    payload, missing = build_payload(ANSWERS, 175, 96)
    assert missing == []
    assert payload["BMI"] == pytest.approx(31.3, abs=0.05)
    assert validate_features(payload).shape == (1, 21)


def test_unanswered_questions_are_listed():
    payload, missing = build_payload({**ANSWERS, "HighBP": None, "Age": None}, None, 80)
    assert "Age" in missing and "Height and weight" in missing
    assert any("high blood pressure" in m for m in missing)
    assert "HighBP" not in payload


def test_form_payload_is_accepted_by_the_api():
    payload, _ = build_payload(ANSWERS, 175, 96)
    response = create_app().test_client().post("/predict", json=payload)
    assert response.status_code == 200
    assert response.get_json()["risk_level"] in {"low", "elevated", "high"}


def test_bmi_error_is_explained_in_plain_language():
    from ui.app import friendly_error
    payload, _ = build_payload(ANSWERS, 250, 30)
    response = create_app().test_client().post("/predict", json=payload)
    assert response.status_code == 400
    message = friendly_error(response.get_json()["errors"][0], payload)
    assert "BMI of 4.8" in message and "height (in cm)" in message


@pytest.mark.parametrize("probability, position", [(0.0, 0), (0.15, 100 / 3), (0.30, 200 / 3), (1.0, 100)])
def test_marker_position(probability, position):
    assert marker_position(probability, threshold=0.15, high_risk=0.30) == pytest.approx(position)


def test_render_result_and_layout_build():
    body = {"probability": 0.33, "risk_level": "high", "message": "msg", "disclaimer": "disc"}
    assert render_result(body, 0.151, 0.30) is not None
    assert layout() is not None
