"""Real-time monitoring dashboard (step 11 of PROJECT_PLAN.md), served by the web UI at /dashboard.

Shows the API's /monitoring data (aggregates only, never individual answers) and refreshes every
REFRESH_SECONDS: overall status, key numbers, drift per question, predictions vs. the training reference, requests
per day and the model history. Chart colours were checked with the dataviz palette validator (D-78).
"""

from datetime import datetime

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import requests
from dash import Input, Output, dash_table, dcc, html

REFRESH_SECONDS = 10                     # D-77

# Chart colours (validated, D-78): live users vs. training reference; OK vs. invalid requests (status: critical)
LIVE, REFERENCE = "#2a78d6", "#eb6834"
OK, INVALID = "#2a78d6", "#d03b3b"
INK, MUTED, GRID, BASELINE = "#1f2933", "#65717e", "#e3e8ee", "#c3c2b7"
FONT = "system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
TABLE_CELL = {"fontFamily": FONT, "fontSize": "0.85rem", "padding": "8px 10px", "textAlign": "left"}
TABLE_HEADER = {"fontWeight": 600, "backgroundColor": "#f8fafc"}
STATUS_TEXT = {"stable": "Stable", "watch": "Watch", "drift": "Drift", "not enough data": "Not enough data",
               "no reference": "No reference"}
STATUS_ICON = {"stable": "✓", "watch": "!", "drift": "▲", "not enough data": "…", "no reference": "?"}
QUESTION_NAMES = {
    "HighBP": "High blood pressure", "HighChol": "High cholesterol", "CholCheck": "Cholesterol checked",
    "BMI": "BMI", "Smoker": "Smoker", "Stroke": "Stroke", "HeartDiseaseorAttack": "Heart disease / attack",
    "PhysActivity": "Physical activity", "Fruits": "Fruit daily", "Veggies": "Vegetables daily",
    "HvyAlcoholConsump": "Heavy drinking", "AnyHealthcare": "Health coverage", "NoDocbcCost": "No doctor (cost)",
    "GenHlth": "General health", "MentHlth": "Poor mental health days", "PhysHlth": "Poor physical health days",
    "DiffWalk": "Difficulty walking", "Sex": "Sex", "Age": "Age group", "Education": "Education", "Income": "Income",
}


# --- layout ---

def layout():
    return dbc.Container([
        html.Header([
            html.Div(id="dash-status", className="dash-status"),
            html.H1("Monitoring"),
            html.P("How the live model is used and whether today's users still look like the people it was trained "
                   "on. Updates every 10 seconds. Only counts and shares are shown, never individual answers.",
                   className="lead-text"),
        ], className="page-header"),
        dcc.Interval(id="dash-refresh", interval=REFRESH_SECONDS * 1000),
        html.Div(id="dash-body"),
    ], className="page")


def tile(label, value, note=None):
    return dbc.Col(dbc.Card(dbc.CardBody([
        html.Div(label, className="tile-label"), html.Div(value, className="tile-value"),
        html.Div(note or "", className="tile-note"),
    ]), className="section tile"), xs=6, md=4, lg=2)


def card(title, *children, note=None):
    items = [html.H2(title, className="section-title")]
    if note:
        items.append(html.P(note, className="muted card-note"))
    return dbc.Card(dbc.CardBody([*items, *children]), className="section")


def as_table(columns, rows):
    """The table-view twin of a chart (every value readable without hover or colour)."""
    return html.Details([
        html.Summary("Show as table", className="table-toggle"),
        dash_table.DataTable(columns=[{"name": c, "id": c} for c in columns], data=rows,
                             style_as_list_view=True, style_cell=TABLE_CELL, style_header=TABLE_HEADER),
    ])


def empty(message):
    return html.Div(message, className="empty-chart")


def base_figure(height):
    fig = go.Figure()
    fig.update_layout(height=height, margin=dict(l=8, r=16, t=8, b=8), paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(family=FONT, size=12, color=INK),
                      hoverlabel=dict(bgcolor="white", font_size=12), showlegend=False)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=BASELINE, linecolor=BASELINE, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=BASELINE, linecolor=BASELINE, tickfont=dict(color=MUTED))
    return fig


def drift_chart(features, thresholds):
    """PSI per question, largest first; one colour (length is the information), thresholds as reference lines."""
    rows = list(reversed(features))                      # plotly draws the first bar at the bottom
    fig = base_figure(height=26 * len(rows) + 60)
    fig.add_bar(x=[f["psi"] for f in rows], y=[QUESTION_NAMES.get(f["feature"], f["feature"]) for f in rows],
                orientation="h", marker=dict(color=LIVE, cornerradius=4), width=0.6,
                customdata=[STATUS_TEXT[f["status"]] for f in rows],
                # direct labels only where it matters: questions at "watch" or "drift"
                text=[f"{f['psi']:.3f} · {STATUS_TEXT[f['status']].lower()}" if f["status"] != "stable" else ""
                      for f in rows],
                textposition="outside", textfont=dict(color=INK, size=11), cliponaxis=False,
                hovertemplate="%{y}: PSI %{x:.3f} (%{customdata})<extra></extra>")
    for value, label, side in [(thresholds["watch"], "watch 0.1 ", "right"),
                               (thresholds["drift"], " drift 0.25", "left")]:
        fig.add_vline(x=value, line=dict(color=MUTED, width=1))
        fig.add_annotation(x=value, y=1, yref="paper", text=label, showarrow=False, yanchor="bottom", xanchor=side,
                           font=dict(size=11, color=MUTED))
    fig.update_xaxes(title=None, rangemode="tozero", range=[0, max(0.3, max(f["psi"] for f in rows) * 1.35)])
    fig.update_layout(margin=dict(l=8, r=16, t=24, b=8))
    return fig


def levels_chart(prediction):
    """Share per risk level: live users vs. the training reference (two series, legend + direct labels)."""
    levels = ["low", "elevated", "high"]
    names = ["Low", "Elevated", "High"]
    fig = base_figure(height=260)
    for label, key, colour in [("Live users", "levels", LIVE), ("Training reference", "levels_reference", REFERENCE)]:
        values = [prediction[key][lvl] for lvl in levels]
        fig.add_bar(x=names, y=values, name=label, marker=dict(color=colour, cornerradius=4),
                    text=[f"{v:.0%}" for v in values], textposition="outside", textfont=dict(color=INK, size=11),
                    hovertemplate=f"{label}, %{{x}}: %{{y:.1%}}<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08, showlegend=True,
                      legend=dict(orientation="h", y=1.12, x=0, font=dict(color=INK)), margin=dict(l=8, r=8, t=36, b=8))
    fig.update_yaxes(tickformat=".0%", rangemode="tozero", range=[0, 1])
    return fig


def requests_chart(daily):
    """Requests per day, OK and invalid stacked on one axis."""
    fig = base_figure(height=240)
    dates = [d["date"] for d in daily]
    gap = dict(color="white", width=1.5)                 # surface gap between stacked segments
    fig.add_bar(x=dates, y=[d["ok"] or None for d in daily], name="OK", marker=dict(color=OK, line=gap),
                hovertemplate="%{x}: %{y} OK<extra></extra>")
    fig.add_bar(x=dates, y=[d["invalid"] or None for d in daily], name="Invalid input",
                marker=dict(color=INVALID, line=gap),
                hovertemplate="%{x}: %{y} invalid<extra></extra>")
    fig.update_layout(barmode="stack", bargap=0.25, showlegend=True,
                      legend=dict(orientation="h", y=1.15, x=0, font=dict(color=INK), traceorder="normal"),
                      margin=dict(l=8, r=8, t=36, b=8))
    fig.update_yaxes(rangemode="tozero")
    fig.update_xaxes(tickformat="%d %b")
    return fig


def recommendation_card(rec):
    """'Retraining recommended' (drift) or 'Keep an eye on it' (watch), with reasons and what to run (D-81)."""
    if not rec or rec["level"] == "none":
        return None
    items = [html.Div([html.Span(STATUS_ICON["drift" if rec["level"] == "retrain" else "watch"],
                                 className="status-icon"), rec["title"]], className="rec-title"),
             html.Ul([html.Li(reason) for reason in rec["reasons"]], className="rec-reasons"),
             html.P(rec["data"], className="rec-data")]
    if rec["command"]:
        items.append(html.Div(["To retrain on the original dataset plus the labelled user rows, run ",
                               html.Code(rec["command"]), " (or “Train model (with user data)” in PyCharm). The new "
                               "model goes live only if it passes the quality check."], className="rec-command"))
    return html.Div(items, className=f"recommendation rec-{rec['level']}")


def status_badge(status, model):
    return [html.Span([html.Span(STATUS_ICON[status], className="status-icon"), STATUS_TEXT[status]],
                      className=f"status-badge status-{status.replace(' ', '-')}"),
            html.Span(["Live model ", html.Strong(model)], className="status-model")]


def body(data):
    r, v = data["report"], data["report"]["volume"]
    th = data["thresholds"]
    tiles = dbc.Row([
        tile("Requests, 24 h", f"{v['requests_24h']:,}", f"{v['requests_7d']:,} in 7 days"),
        tile("Requests, total", f"{v['requests']:,}", f"{v['people_predicted']:,} people predicted"),
        tile("Error rate", f"{v['error_rate']:.1%}", "invalid input"),
        tile("Opted in", f"{v['share_stored']:.0%}", "answers stored"),
        tile("Stored predictions", f"{r['stored_predictions']:,}",
             f"min. {r['min_predictions']} for drift · {data.get('labelled_user_rows', 0):,} with diagnosis"),
        tile("Mean risk", f"{r['prediction']['mean']:.1%}" if r["prediction"] else "–",
             f"reference {r['prediction']['mean_reference']:.1%}" if r["prediction"] else "needs stored predictions"),
    ], className="g-3 tiles")

    if r["status"] == "not enough data":
        drift = empty(f"Drift is judged from {r['min_predictions']} stored predictions of the live model on; "
                      f"there are {r['stored_predictions']} so far.")
        levels = empty("Shown once there are enough stored predictions.")
    elif r["status"] == "no reference":
        drift = levels = empty("The live model has no reference.json. Build it with "
                               "python -m ml.monitoring --build-reference <run_id> --data <csv>.")
    else:
        drift = html.Div([
            dcc.Graph(figure=drift_chart(r["features"], th), config={"displayModeBar": False}),
            as_table(["Question", "PSI", "Status"],
                     [{"Question": QUESTION_NAMES.get(f["feature"], f["feature"]), "PSI": f"{f['psi']:.3f}",
                       "Status": STATUS_TEXT[f["status"]]} for f in r["features"]]),
        ])
        p = r["prediction"]
        levels = html.Div([
            html.P([f"Prediction PSI {p['psi']:.3f} ({STATUS_TEXT[p['status']]}) · flagged ",
                    html.Strong(f"{p['share_flagged']:.1%}"), f" (reference {p['share_flagged_reference']:.1%})"],
                   className="muted"),
            dcc.Graph(figure=levels_chart(p), config={"displayModeBar": False}),
            as_table(["Risk level", "Live users", "Training reference"],
                     [{"Risk level": n, "Live users": f"{p['levels'][k]:.1%}",
                       "Training reference": f"{p['levels_reference'][k]:.1%}"}
                      for k, n in [("low", "Low"), ("elevated", "Elevated"), ("high", "High")]]),
        ])

    daily = data["daily_requests"]
    volume = (empty("No requests in the last 30 days.") if not any(d["ok"] or d["invalid"] for d in daily) else
              html.Div([dcc.Graph(figure=requests_chart(daily), config={"displayModeBar": False}),
                        as_table(["Date", "OK", "Invalid"],
                                 [{"Date": d["date"], "OK": d["ok"], "Invalid": d["invalid"]} for d in daily
                                  if d["ok"] or d["invalid"]])]))

    history = dash_table.DataTable(
        columns=[{"name": n, "id": i} for n, i in [("Model", "run_id"), ("Trained", "trained"), ("Data", "data_file"),
                                                   ("User rows", "user_rows"), ("Test ROC-AUC", "auc"),
                                                   ("Recall", "recall"), ("Precision", "precision"),
                                                   ("Deployed", "deployed")]],
        data=[{"run_id": ("● " if m["live"] else "") + m["run_id"],
               "trained": datetime.fromisoformat(m["created_at"]).strftime("%d %b %Y, %H:%M"),
               "data_file": m["data_file"], "user_rows": f"{m.get('user_rows', 0):,}",
               "auc": f"{m['test_roc_auc']:.4f}", "recall": f"{m['test_recall']:.1%}",
               "precision": f"{m['test_precision']:.1%}",
               "deployed": "live" if m["live"] else ("yes (forced)" if m["forced"] else "yes") if m["deployed"]
               else "no: " + "; ".join(m["failures"])} for m in data["model_history"]],
        style_as_list_view=True,
        style_cell={**TABLE_CELL, "whiteSpace": "normal", "height": "auto"},
        style_cell_conditional=[{"if": {"column_id": c}, "whiteSpace": "nowrap"} for c in ("run_id", "trained")],
        style_header=TABLE_HEADER,
        style_data_conditional=[{"if": {"filter_query": '{deployed} = "live"'}, "fontWeight": 600}])

    return html.Div([
        tiles,
        recommendation_card(data.get("recommendation")),
        dbc.Row([
            dbc.Col(card("Data drift per question", drift,
                         note="PSI between the stored answers and the training data. Below 0.1 stable, "
                              "0.1–0.25 watch, 0.25 and above drift."), lg=7),
            dbc.Col(card("Predictions", levels, note="Share of people per risk level, compared with the model's "
                                                     "predictions on its test data."), lg=5),
        ], className="g-4"),
        card("Requests per day (last 30 days)", volume, note="All requests, whether or not answers were stored."),
        card("Model history", history, note="Every training run, newest first. ● = live model."),
        html.Div(f"Updated {datetime.now().strftime('%H:%M:%S')}", className="updated"),
    ])


# --- callbacks ---

def register(app, api_base):
    @app.callback(Output("dash-body", "children"), Output("dash-status", "children"),
                  Input("dash-refresh", "n_intervals"))
    def refresh(_):
        try:
            data = requests.get(f"{api_base}/monitoring", timeout=5).json()
        except (requests.RequestException, ValueError):
            return (dbc.Alert("The risk service is not available. Is the API running? Start it with "
                              "python -m api.main.", color="warning"),
                    [html.Span([html.Span("…", className="status-icon"), "No connection"],
                               className="status-badge status-no-reference")])
        return body(data), status_badge(data["report"]["status"], data["report"]["model"])
