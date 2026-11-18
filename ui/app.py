"""Web UI for the diabetes risk check (step 6 of PROJECT_PLAN.md).

A Dash page with the 21 questions in plain language. It sends the answers to the API (api/main.py) and shows the
risk level, percentage, message and disclaimer. The UI holds no model logic; cut-offs come from the API's /model.

Run from the project root, in the da-ml environment (the API must be running too):
    python -m api.main
    python -m ui.app        -> http://127.0.0.1:8050 (risk check), http://127.0.0.1:8050/dashboard (monitoring)
"""

import os
from datetime import datetime

import dash_bootstrap_components as dbc
import requests
from dash import Dash, Input, Output, State, dcc, html

from ui import dashboard

API_BASE = os.environ.get("DA_ML_API_URL", "http://127.0.0.1:5001")   # D-54
HOST, PORT = "127.0.0.1", 8050                                          # same as the taxi project (D-54)
MODEL_POLL_SECONDS = 10                                                 # how often the model label is refreshed (D-65)
YES_NO = [{"label": "Yes", "value": 1}, {"label": "No", "value": 0}]
INCOME_LABELS = ["Less than $10,000", "$10,000–15,000", "$15,000–20,000", "$20,000–25,000", "$25,000–35,000",
                 "$35,000–50,000", "$50,000–75,000", "$75,000 or more"]

# Questions in plain language, based on the BRFSS 2015 survey (EDA.ipynb section 3). Codes stay behind the scenes.
SECTIONS = [
    ("About you", [
        ("Sex", "Sex", [{"label": "Female", "value": 0}, {"label": "Male", "value": 1}], None),
        ("Age", "Age", [{"label": label, "value": code} for code, label in enumerate(
            ["18–24", "25–29", "30–34", "35–39", "40–44", "45–49", "50–54", "55–59", "60–64", "65–69", "70–74",
             "75–79", "80 or older"], start=1)], "dropdown"),
        ("Education", "Highest level of education", [{"label": label, "value": code} for code, label in enumerate(
            ["Never attended school or only kindergarten", "Grades 1–8", "Grades 9–11 (some high school)",
             "High school graduate", "Some college or technical school", "College graduate"], start=1)],
         "dropdown"),
        ("Income", "Household income per year (US dollars)",
         [{"label": label, "value": code} for code, label in enumerate(INCOME_LABELS, start=1)], "dropdown"),
    ]),
    ("Your health", [
        ("GenHlth", "In general, how would you say your health is?", [
            {"label": label, "value": code}
            for code, label in enumerate(["Excellent", "Very good", "Good", "Fair", "Poor"], start=1)], None),
        ("HighBP", "Has a doctor ever told you that you have high blood pressure?", YES_NO, None),
        ("HighChol", "Has a doctor ever told you that your cholesterol is high?", YES_NO, None),
        ("CholCheck", "Have you had your cholesterol checked in the past 5 years?", YES_NO, None),
        ("HeartDiseaseorAttack", "Have you ever had coronary heart disease or a heart attack?", YES_NO, None),
        ("Stroke", "Have you ever had a stroke?", YES_NO, None),
        ("DiffWalk", "Do you have serious difficulty walking or climbing stairs?", YES_NO, None),
        ("PhysHlth", "In the past 30 days, on how many days was your physical health not good "
                     "(illness or injury)?", None, "days"),
        ("MentHlth", "In the past 30 days, on how many days was your mental health not good "
                     "(stress, depression, emotional problems)?", None, "days"),
    ]),
    ("Lifestyle", [
        ("PhysActivity", "In the past 30 days, did you do any physical activity or exercise outside your job?",
         YES_NO, None),
        ("Fruits", "Do you eat fruit at least once a day?", YES_NO, None),
        ("Veggies", "Do you eat vegetables at least once a day?", YES_NO, None),
        ("Smoker", "Have you smoked at least 100 cigarettes (about 5 packs) in your life?", YES_NO, None),
        ("HvyAlcoholConsump", "Do you drink more than 14 drinks a week (men) or more than 7 a week (women)?",
         YES_NO, None),
    ]),
    ("Healthcare", [
        ("AnyHealthcare", "Do you have any kind of health care coverage (e.g. health insurance)?", YES_NO, None),
        ("NoDocbcCost", "In the past 12 months, did you need to see a doctor but couldn't because of the cost?",
         YES_NO, None),
    ]),
]
CHOICE_FIELDS = [field for _, questions in SECTIONS for field, *_ in questions]   # 20 answers; BMI comes from
LABELS = {field: label for _, questions in SECTIONS for field, label, *_ in questions}   # height and weight


def bmi(height_cm, weight_kg):
    """Body mass index from height in cm and weight in kg, or None if not (validly) entered."""
    if not height_cm or not weight_kg or height_cm <= 0 or weight_kg <= 0:
        return None
    return weight_kg / (height_cm / 100) ** 2


def build_payload(answers, height_cm, weight_kg):
    """Turn the form values into the JSON for POST /predict. Returns (payload, list of unanswered questions)."""
    missing = [LABELS[field] for field in CHOICE_FIELDS if answers.get(field) is None]
    value = bmi(height_cm, weight_kg)
    if value is None:
        missing.append("Height and weight")
    payload = {field: answers[field] for field in CHOICE_FIELDS if answers.get(field) is not None}
    if value is not None:
        payload["BMI"] = round(value, 1)
    return payload, missing


# --- layout ---

def question(field, label, options, kind):
    if kind == "dropdown":
        control = dcc.Dropdown(id=field, options=options, placeholder="Select…", clearable=False,
                               className="dropdown")
    elif kind == "days":
        control = dcc.Slider(id=field, min=0, max=30, step=1, value=0, marks={d: str(d) for d in (0, 10, 20, 30)},
                             tooltip={"placement": "top", "always_visible": True, "template": "{value} days"},
                             className="days-slider")
    else:
        control = dbc.RadioItems(id=field, options=options, value=None, inline=True, className="choice",
                                 inputClassName="btn-check", labelClassName="btn btn-choice",
                                 labelCheckedClassName="active")
    # Yes/No questions: question and buttons on one line on wide screens (they wrap on small screens)
    inline = options is YES_NO
    return html.Div([html.Div(label, className="question-label"), control],
                    className="question question-inline" if inline else "question")


def body_size():
    return html.Div([
        html.Div("Height and weight", className="question-label"),
        dbc.Row([
            dbc.Col(dbc.InputGroup([dbc.Input(id="height", type="number", placeholder="Height"),
                                    dbc.InputGroupText("cm")]), xs=6),
            dbc.Col(dbc.InputGroup([dbc.Input(id="weight", type="number", placeholder="Weight"),
                                    dbc.InputGroupText("kg")]), xs=6),
        ], className="g-2"),
        html.Div(id="bmi-text", className="hint"),
    ], className="question")


def section(title, questions):
    items = [question(*q) for q in questions]
    if title == "About you":
        items.append(body_size())
    return dbc.Card(dbc.CardBody([html.H2(title, className="section-title"), *items]), className="section")


DIAGNOSIS_OPTIONS = [{"label": "No", "value": 0}, {"label": "Prediabetes", "value": 1},
                     {"label": "Diabetes", "value": 2}, {"label": "Prefer not to say", "value": "skip"}]


def research_question():
    """Optional label for retraining (D-79): stored only with consent, never used for the prediction."""
    return dbc.Card(dbc.CardBody([
        html.H2("For research (optional)", className="section-title"),
        html.Div([html.Div("Has a doctor ever told you that you have diabetes or prediabetes?",
                           className="question-label"),
                  dbc.RadioItems(id="diagnosis", options=DIAGNOSIS_OPTIONS, value=None, inline=True,
                                 className="choice", inputClassName="btn-check", labelClassName="btn btn-choice",
                                 labelCheckedClassName="active"),
                  html.Div("Not used for your result. Only stored if you tick “Help improve the model”; it helps us "
                           "retrain the model on people like you.", className="hint")], className="question"),
    ]), className="section")


def result_placeholder():
    return html.Div([
        html.Div("Your result", className="result-kicker"),
        html.P("Answer the questions and press “Check my risk” to see your estimated risk.", className="muted"),
    ])


def model_label(info):
    """Small label with the live model (D-65). `info` is the API's /model answer, or None if it isn't reachable."""
    if info is None:
        return [html.Span(className="dot dot-off"), "Risk service not reachable"]
    parts = [html.Span(className="dot dot-on"), "Model ", html.Strong(info["model"])]
    if info.get("created"):
        parts.append(" · trained " + datetime.fromisoformat(info["created"]).strftime("%d %b %Y, %H:%M"))
    if info.get("test_metrics"):
        parts.append(f" · test ROC-AUC {info['test_metrics']['roc_auc']:.3f}")
    return parts


def layout():
    return dbc.Container([
        html.Header([
            html.Div(id="model-label", className="model-label", children=model_label(None)),
            dcc.Interval(id="model-poll", interval=MODEL_POLL_SECONDS * 1000),
            html.H1("Diabetes risk check"),
            html.P("21 short questions about your health and lifestyle. You get an estimate of your risk of "
                   "prediabetes or diabetes, and whether it's worth talking to your doctor. This is not a diagnosis.",
                   className="lead-text"),
        ], className="page-header"),
        dbc.Row([
            dbc.Col([section(title, questions) for title, questions in SECTIONS] + [research_question()], lg=7),
            dbc.Col(html.Div([
                dbc.Card(dbc.CardBody([
                    html.Div(id="result", children=result_placeholder()),
                    dbc.Checkbox(id="consent", value=False, className="consent",
                                 label="Help improve the model: store my answers and result anonymously for research "
                                       "(no name or IP address)."),
                    dbc.Button("Check my risk", id="submit", className="submit", size="lg"),
                ]), className="result-card"),
            ], className="sticky"), lg=5),
        ], className="g-4"),
        html.Footer("Model trained on the CDC BRFSS 2015 survey (253,680 US adults). The estimate describes people "
                    "with similar answers; it can't measure anything about your body.", className="page-footer"),
    ], className="page")


# --- result ---

_cutoffs = {}


def cutoffs(model_id):
    """Screening threshold and 'high' cut-off of the given model, from the API's /model. Cached per model, so they
    are fetched again when the API has switched to a new model (D-62)."""
    if _cutoffs.get("model") != model_id:
        info = requests.get(f"{API_BASE}/model", timeout=5).json()
        _cutoffs.clear()
        _cutoffs.update(model=info["model"], threshold=info["threshold"], high_risk=info["high_risk"])
    return _cutoffs["threshold"], _cutoffs["high_risk"]


def marker_position(probability, threshold, high_risk):
    """Position on the bar (0-100%). The three zones get equal width, so 'low' and 'elevated' stay readable."""
    if probability < threshold:
        return probability / threshold * 100 / 3
    if probability < high_risk:
        return 100 / 3 + (probability - threshold) / (high_risk - threshold) * 100 / 3
    return 200 / 3 + (probability - high_risk) / (1 - high_risk) * 100 / 3


KNOWN_DIAGNOSIS_NOTE = {
    1: "You told us a doctor has found prediabetes. This check is meant for people without a diagnosis; please follow "
       "your doctor's advice.",
    2: "You told us a doctor has diagnosed diabetes. This check is meant for people without a diagnosis; please follow "
       "your doctor's advice.",
}


def render_result(body, threshold, high_risk, diagnosis=None):
    level = body["risk_level"]
    probability = body["probability"]
    very_low = probability < 0.005
    stored_note = " · your answers were stored anonymously" if body.get("stored") else ""
    return html.Div([
        html.Div("Your result", className="result-kicker"),
        html.Div([html.Span("<1%" if very_low else f"{probability:.0%}", className="result-number"),
                  html.Span({"low": "Low", "elevated": "Elevated", "high": "High"}[level],
                            className=f"badge-level level-{level}")], className="result-headline"),
        html.Div([
            html.Div([html.Div(className="zone zone-low"), html.Div(className="zone zone-elevated"),
                      html.Div(className="zone zone-high")], className="zones"),
            html.Div(className="marker", style={"left": f"{marker_position(probability, threshold, high_risk):.1f}%"}),
        ], className="risk-bar"),
        html.Div([html.Span("Low"), html.Span(f"{threshold:.1%}"), html.Span("Elevated"),
                  html.Span(f"{high_risk:.0%}"), html.Span("High")], className="risk-scale"),
        html.P(body["message"], className="result-message"),
        *([dbc.Alert(KNOWN_DIAGNOSIS_NOTE[diagnosis], color="info", className="known-diagnosis")]
          if diagnosis in KNOWN_DIAGNOSIS_NOTE else []),
        html.P("Of 100 people with answers like yours, " + ("fewer than 1 has" if very_low else
               f"about {round(probability * 100)} have") + " prediabetes or diabetes.", className="muted"),
        html.P(body["disclaimer"], className="disclaimer"),
        html.Div(f"Model: {body['model']}" + stored_note, className="result-model"),
    ])


def friendly_error(error, payload):
    """Plain-language version of the API's validation errors. Only BMI can be out of range from this form;
    everything else is chosen from fixed options."""
    if error.startswith("'BMI'"):
        return (f"Your height and weight give a BMI of {payload['BMI']:.1f}, which is outside the range the model "
                "knows (12–98). Please check your height (in cm) and weight (in kg).")
    return error


def render_problems(title, problems):
    return html.Div([html.Div("Your result", className="result-kicker"),
                     dbc.Alert([html.Strong(title), html.Ul([html.Li(p) for p in problems])], color="warning",
                               className="problems")])


# --- app ---

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title="Diabetes risk check",
           meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
           suppress_callback_exceptions=True)   # each page's components only exist while that page is shown
app.layout = html.Div([
    dcc.Location(id="url"),
    html.Nav(dbc.Container([dcc.Link("Risk check", href="/", className="nav-link-item"),
                            dcc.Link("Monitoring", href="/dashboard", className="nav-link-item")],
                           className="topnav-inner"), className="topnav"),
    html.Div(id="page"),
])
dashboard.register(app, API_BASE)


@app.callback(Output("page", "children"), Input("url", "pathname"))
def show_page(pathname):
    """/ = the risk check, /dashboard = monitoring (D-77)."""
    return dashboard.layout() if pathname == "/dashboard" else layout()


@app.callback(Output("model-label", "children"), Input("model-poll", "n_intervals"))
def show_model(_):
    """Runs when the page loads and then every MODEL_POLL_SECONDS, so the label follows a model switch."""
    try:
        return model_label(requests.get(f"{API_BASE}/model", timeout=3).json())
    except (requests.RequestException, ValueError):
        return model_label(None)


@app.callback(Output("bmi-text", "children"), Input("height", "value"), Input("weight", "value"))
def show_bmi(height_cm, weight_kg):
    value = bmi(height_cm, weight_kg)
    return f"BMI {value:.1f}" if value else "Used to calculate your BMI."


@app.callback(Output("result", "children"), Input("submit", "n_clicks"),
              [State(field, "value") for field in CHOICE_FIELDS]
              + [State("height", "value"), State("weight", "value"), State("consent", "value"),
                 State("diagnosis", "value")],
              prevent_initial_call=True)
def check_risk(_, *values):
    answers = dict(zip(CHOICE_FIELDS, values[:-4]))
    height_cm, weight_kg, consent, diagnosis = values[-4:]
    payload, missing = build_payload(answers, height_cm, weight_kg)
    if missing:
        return render_problems("Please answer every question:", missing)
    params = {"store": "true", "source": "ui"} if consent else {}   # stored only with consent (D-68)
    if consent and diagnosis in (0, 1, 2):
        params["diagnosis"] = str(diagnosis)                         # optional label for retraining (D-79)
    try:
        response = requests.post(f"{API_BASE}/predict", json=payload, params=params, timeout=10)
        if response.status_code == 400:
            return render_problems("Some answers could not be used:",
                                   [friendly_error(e, payload) for e in response.json()["errors"]])
        response.raise_for_status()
        body = response.json()
        return render_result(body, *cutoffs(body["model"]), diagnosis=diagnosis)
    except requests.RequestException:
        return render_problems("The risk service is not available.",
                               [f"Is the API running? Start it with `python -m api.main` ({API_BASE})."])


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False)
