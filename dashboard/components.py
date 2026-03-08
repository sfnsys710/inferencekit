import plotly.express as px
from dash import dcc, html
from styles import (  # type: ignore[no-redef]
    CHART_LAYOUT,
    FILTER_DROPDOWN_STYLE,
    FILTER_INPUT_STYLE,
    FILTER_LABEL_STYLE,
    GREEN_YELLOW,
    RESET_BTN_STYLE,
    SECTION_STYLE,
)
from utils import dropdown_options, tag_options  # type: ignore[no-redef]

# ── Charts ────────────────────────────────────────────────────────────────────


def build_cin_ncorrect_chart(df):
    cin_df = df[df["doc"] == "cin"].copy()
    if cin_df.empty:
        return dcc.Graph(figure={})
    agg = (
        cin_df.groupby(["model", "prompt_version"], as_index=False)["correct_fields"]
        .mean()
        .rename(columns={"correct_fields": "avg_correct_fields"})
    )
    fig = px.bar(
        agg,
        x="avg_correct_fields",
        y="model",
        color="prompt_version",
        barmode="group",
        orientation="h",
        text_auto=".2f",  # type: ignore[arg-type]
        color_discrete_sequence=GREEN_YELLOW,
        labels={
            "model": "Model",
            "avg_correct_fields": "Avg Correct Fields",
            "prompt_version": "Prompt",
        },
        title="CIN — Average Correct Fields by Model & Prompt Version",
    )
    fig.update_layout(**CHART_LAYOUT, xaxis={"gridcolor": "#ebebeb", "range": [0, 15]})
    return dcc.Graph(figure=fig)


def build_cin_execution_time_chart(df):
    cin_df = df[df["doc"] == "cin"].copy()
    if cin_df.empty or "execution_time" not in cin_df.columns:
        return dcc.Graph(figure={})
    agg = (
        cin_df.groupby(["model", "prompt_version"], as_index=False)["execution_time"]
        .mean()
        .rename(columns={"execution_time": "avg_execution_time"})
    )
    fig = px.bar(
        agg,
        x="avg_execution_time",
        y="model",
        color="prompt_version",
        barmode="group",
        orientation="h",
        text_auto=".2f",  # type: ignore[arg-type]
        color_discrete_sequence=GREEN_YELLOW,
        labels={
            "model": "Model",
            "avg_execution_time": "Avg Execution Time (s)",
            "prompt_version": "Prompt",
        },
        title="CIN — Average Execution Time by Model & Prompt Version",
    )
    fig.update_layout(**CHART_LAYOUT, xaxis={"gridcolor": "#ebebeb", "ticksuffix": "s"})
    return dcc.Graph(figure=fig)


def build_cin_format_chart(df):
    cin_df = df[df["doc"] == "cin"].copy()
    if cin_df.empty:
        return dcc.Graph(figure={})
    agg = (
        cin_df.groupby(["model", "prompt_version"], as_index=False)["format_compliance"]
        .apply(lambda g: (g == "incorrect").mean() * 100)
        .rename(columns={"format_compliance": "pct_incorrect"})
    )
    fig = px.bar(
        agg,
        x="pct_incorrect",
        y="model",
        color="prompt_version",
        barmode="group",
        orientation="h",
        text_auto=".1f",  # type: ignore[arg-type]
        color_discrete_sequence=GREEN_YELLOW,
        labels={
            "model": "Model",
            "pct_incorrect": "% Format Incorrect",
            "prompt_version": "Prompt",
        },
        title="CIN — Format Incorrectness Rate by Model & Prompt Version",
    )
    fig.update_layout(
        **CHART_LAYOUT, xaxis={"gridcolor": "#ebebeb", "range": [0, 100], "ticksuffix": "%"}
    )
    return dcc.Graph(figure=fig)


# ── Filter panel ──────────────────────────────────────────────────────────────


def build_filter_panel(df) -> html.Div:
    n_min = (
        int(df["correct_fields"].min()) if "correct_fields" in df.columns and not df.empty else 0
    )
    n_max = (
        int(df["correct_fields"].max()) if "correct_fields" in df.columns and not df.empty else 15
    )

    return html.Div(
        style=SECTION_STYLE,
        children=[
            html.H4(
                "Filters",
                style={"marginTop": 0, "marginBottom": "16px", "color": "#4a7c59"},
            ),
            # Row 1: dropdowns
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(4, 1fr)",
                    "gap": "16px",
                    "marginBottom": "16px",
                },
                children=[
                    html.Div(
                        [
                            html.Label("Model", style=FILTER_LABEL_STYLE),
                            dcc.Dropdown(
                                id="filter-model",
                                options=dropdown_options(df["model"])
                                if "model" in df.columns
                                else [],
                                multi=True,
                                placeholder="All models",
                                style=FILTER_DROPDOWN_STYLE,
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Document", style=FILTER_LABEL_STYLE),
                            dcc.Dropdown(
                                id="filter-doc",
                                options=dropdown_options(df["doc"]) if "doc" in df.columns else [],
                                multi=True,
                                placeholder="All docs",
                                style=FILTER_DROPDOWN_STYLE,
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Prompt Version", style=FILTER_LABEL_STYLE),
                            dcc.Dropdown(
                                id="filter-prompt-version",
                                options=dropdown_options(df["prompt_version"])
                                if "prompt_version" in df.columns
                                else [],
                                multi=True,
                                placeholder="All versions",
                                style=FILTER_DROPDOWN_STYLE,
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Tags", style=FILTER_LABEL_STYLE),
                            dcc.Dropdown(
                                id="filter-tags",
                                options=tag_options(df),
                                multi=True,
                                placeholder="All tags",
                                style=FILTER_DROPDOWN_STYLE,
                            ),
                        ]
                    ),
                ],
            ),
            # Row 2: date, n_correct range, format correctness
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1.6fr 1fr 1fr",
                    "gap": "16px",
                    "alignItems": "start",
                },
                children=[
                    html.Div(
                        [
                            html.Label("Date Range", style=FILTER_LABEL_STYLE),
                            dcc.DatePickerRange(
                                id="filter-date",
                                display_format="YYYY-MM-DD",
                                style={"fontSize": "13px"},
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Correct Fields Range", style=FILTER_LABEL_STYLE),
                            html.Div(
                                style={"display": "flex", "gap": "8px", "alignItems": "center"},
                                children=[
                                    html.Span("Min", style={"fontSize": "12px", "color": "#888"}),
                                    dcc.Input(
                                        id="filter-ncorrect-min",
                                        type="number",
                                        placeholder=str(n_min),
                                        min=n_min,
                                        max=n_max,
                                        style=FILTER_INPUT_STYLE,
                                    ),
                                    html.Span("Max", style={"fontSize": "12px", "color": "#888"}),
                                    dcc.Input(
                                        id="filter-ncorrect-max",
                                        type="number",
                                        placeholder=str(n_max),
                                        min=n_min,
                                        max=n_max,
                                        style=FILTER_INPUT_STYLE,
                                    ),
                                ],
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Format Compliance", style=FILTER_LABEL_STYLE),
                            dcc.Dropdown(
                                id="filter-format-correctness",
                                options=[
                                    {"label": "Correct", "value": "correct"},
                                    {"label": "Incorrect", "value": "incorrect"},
                                ],
                                multi=True,
                                placeholder="All",
                                style=FILTER_DROPDOWN_STYLE,
                            ),
                        ]
                    ),
                ],
            ),
            # Reset button
            html.Div(
                style={"marginTop": "14px", "textAlign": "right"},
                children=[
                    html.Button(
                        "Reset Filters",
                        id="reset-filters",
                        n_clicks=0,
                        style=RESET_BTN_STYLE,
                    )
                ],
            ),
        ],
    )
