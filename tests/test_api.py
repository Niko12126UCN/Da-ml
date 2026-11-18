import pytest

from api.main import MAX_PEOPLE, create_app
from ml.risk import DISCLAIMER, HIGH_RISK, assess, risk_level
from tests.test_validation import PERSON

LOW_RISK_PERSON = {**PERSON, "HighBP": 0, "HighChol": 0, "BMI": 22, "GenHlth": 1, "PhysHlth": 0, "Age": 2,
                   "Income": 8}
HIGH_RISK_PERSON = {**PERSON, "BMI": 40, "GenHlth": 5, "DiffWalk": 1, "HeartDiseaseorAttack": 1, "Age": 11,
                    "PhysHlth": 30, "Income": 2}


@pytest.fixture(scope="module")
def client():
    return create_app().test_client()


# --- risk levels ---

@pytest.mark.parametrize("probability, level", [(0.05, "low"), (0.15, "low"), (0.151, "elevated"),
                                                (0.29, "elevated"), (HIGH_RISK, "high"), (0.9, "high")])
def test_risk_levels(probability, level):
    assert risk_level(probability, threshold=0.151) == level


def test_assess_wording():
    result = assess(0.42, threshold=0.151)
    assert result["risk_level"] == "high" and result["flagged"]
    assert "42%" in result["message"] and "doctor" in result["message"]
    assert result["disclaimer"] == DISCLAIMER
    assert not assess(0.05, threshold=0.151)["flagged"]
    assert "(less than 1%)" in assess(0.003, threshold=0.151)["message"]


# --- API ---

def test_health(client):
    body = client.get("/health").get_json()
    assert body["status"] == "ok" and body["model"].startswith("catboost_")


def test_model_info(client):
    body = client.get("/model").get_json()
    assert len(body["features"]) == 21
    assert body["feature_ranges"]["Age"] == [1, 13]
    assert 0 < body["threshold"] < body["high_risk"] == HIGH_RISK
    assert body["test_metrics"]["roc_auc"] > 0.8


def test_predict_one_person(client):
    response = client.post("/predict", json=PERSON)
    assert response.status_code == 200
    body = response.get_json()
    assert 0 <= body["probability"] <= 1
    assert body["risk_level"] in {"low", "elevated", "high"}
    assert body["message"] and body["disclaimer"] == DISCLAIMER


def test_predict_low_and_high_risk_examples(client):
    assert client.post("/predict", json=LOW_RISK_PERSON).get_json()["risk_level"] == "low"
    assert client.post("/predict", json=HIGH_RISK_PERSON).get_json()["risk_level"] == "high"


def test_predict_several_people(client):
    body = client.post("/predict", json=[LOW_RISK_PERSON, HIGH_RISK_PERSON]).get_json()
    levels = [r["risk_level"] for r in body["results"]]
    assert levels == ["low", "high"]


def test_invalid_input_returns_400_with_all_errors(client):
    response = client.post("/predict", json={**PERSON, "Age": 14, "HighBP": "yes"})
    assert response.status_code == 400
    errors = "\n".join(response.get_json()["errors"])
    assert "'Age'" in errors and "'HighBP' must be a number" in errors


def test_non_json_body_returns_400(client):
    response = client.post("/predict", data="HighBP=1", content_type="text/plain")
    assert response.status_code == 400
    assert "must be JSON" in response.get_json()["errors"][0]


def test_too_many_people_returns_400(client):
    response = client.post("/predict", json=[PERSON] * (MAX_PEOPLE + 1))
    assert response.status_code == 400
