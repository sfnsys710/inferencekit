import pandas as pd
from components import (  # type: ignore[no-redef]
    build_cin_execution_time_chart,
    build_cin_format_chart,
    build_cin_ncorrect_chart,
    build_filter_panel,
)
from dash import Dash, Input, Output, State, dash_table, dcc, html
from llm import summarize_reasons  # type: ignore[no-redef]
from styles import (  # type: ignore[no-redef]
    BG_STYLE,
    BODY_STYLE,
    CODE_BOX_STYLE,
    CODE_COLUMNS_STYLE,
    DETAIL_HEADER_STYLE,
    FIELD_LABEL_STYLE,
    REASON_BOX_STYLE,
    SECTION_STYLE,
    SUMMARIZE_BTN_STYLE,
    SUMMARY_BOX_STYLE,
    SUMMARY_COLUMNS_STYLE,
    SUMMARY_LABEL_STYLE,
    TABLE_STYLE_CELL,
    TABLE_STYLE_HEADER,
)
from utils import DETAIL_COLS, DISPLAY_COLS, load_experiments, pretty_json  # type: ignore[no-redef]

app = Dash(__name__)

df = load_experiments()

table_columns = [{"name": c, "id": c} for c in DISPLAY_COLS if c in df.columns]

app.layout = html.Div(
    style=BG_STYLE,
    children=html.Div(
        style=BODY_STYLE,
        children=[
            html.H2("Evals Dashboard", style={"marginBottom": "24px"}),
            build_filter_panel(df),
            # ── Section 1: correct_fields ────────────────────────────────────
            html.Div(
                style=SECTION_STYLE,
                children=[
                    html.H3("CIN — Correct Fields", style={"marginBottom": "8px"}),
                    html.Div(id="chart-ncorrect", children=build_cin_ncorrect_chart(df)),
                ],
            ),
            # ── Section 2: format compliance ─────────────────────────────────
            html.Div(
                style=SECTION_STYLE,
                children=[
                    html.H3("CIN — Format Compliance", style={"marginBottom": "8px"}),
                    html.Div(id="chart-format", children=build_cin_format_chart(df)),
                ],
            ),
            # ── Section 3: execution time ─────────────────────────────────────
            html.Div(
                style=SECTION_STYLE,
                children=[
                    html.H3("CIN — Execution Time", style={"marginBottom": "8px"}),
                    html.Div(id="chart-exectime", children=build_cin_execution_time_chart(df)),
                ],
            ),
            # ── Section 4: Reason Summarizer ─────────────────────────────────
            html.Div(
                style=SECTION_STYLE,
                children=[
                    html.H3("Reason Summarizer", style={"marginBottom": "8px"}),
                    html.P(
                        "Apply your filters above, then click the button to summarize"
                        " the most common issues across the filtered results.",
                        style={"fontSize": "13px", "color": "#888", "marginBottom": "16px"},
                    ),
                    html.Button(
                        "Summarize Reasons",
                        id="summarize-btn",
                        n_clicks=0,
                        style=SUMMARIZE_BTN_STYLE,
                    ),
                    dcc.Loading(
                        type="circle",
                        color="#4a7c59",
                        style={"marginTop": "20px"},
                        children=html.Div(
                            style=SUMMARY_COLUMNS_STYLE,
                            children=[
                                html.Div(
                                    [
                                        html.Strong(
                                            "Correct Fields — Common Issues",
                                            style=SUMMARY_LABEL_STYLE,
                                        ),
                                        dcc.Markdown(
                                            id="summary-ncorrect",
                                            children="",
                                            style=SUMMARY_BOX_STYLE,
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Strong(
                                            "Format Compliance — Common Issues",
                                            style=SUMMARY_LABEL_STYLE,
                                        ),
                                        dcc.Markdown(
                                            id="summary-format",
                                            children="",
                                            style=SUMMARY_BOX_STYLE,
                                        ),
                                    ]
                                ),
                            ],
                        ),
                    ),
                ],
            ),
            # ── Section 5: Full results table ────────────────────────────────
            html.Div(
                style=SECTION_STYLE,
                children=[
                    html.H3("All Results", style={"marginBottom": "8px"}),
                    html.P(
                        "Click any row to inspect full details below.",
                        style={"fontSize": "13px", "color": "#888", "marginBottom": "12px"},
                    ),
                    dash_table.DataTable(
                        id="results-table",
                        columns=table_columns,  # type: ignore[arg-type]
                        data=df.to_dict("records"),  # type: ignore[arg-type]
                        filter_action="native",
                        sort_action="native",
                        page_size=25,
                        style_table={"overflowX": "auto"},
                        style_cell=TABLE_STYLE_CELL,
                        style_header=TABLE_STYLE_HEADER,
                        style_data_conditional=[  # type: ignore[arg-type]
                            {
                                "if": {"state": "active"},
                                "backgroundColor": "#e8f5e3",
                                "border": "1px solid #4a7c59",
                            }
                        ],
                        tooltip_duration=None,
                    ),
                ],
            ),
            # ── Section 6: Row detail panel ──────────────────────────────────
            html.Div(
                id="row-detail-panel",
                style={**SECTION_STYLE, "display": "none"},
                children=[],
            ),
        ],
    ),
)


# ── Callbacks ─────────────────────────────────────────────────────────────────


@app.callback(
    Output("filter-model", "value"),
    Output("filter-doc", "value"),
    Output("filter-prompt-version", "value"),
    Output("filter-tags", "value"),
    Output("filter-date", "start_date"),
    Output("filter-date", "end_date"),
    Output("filter-ncorrect-min", "value"),
    Output("filter-ncorrect-max", "value"),
    Output("filter-format-correctness", "value"),
    Input("reset-filters", "n_clicks"),
    prevent_initial_call=True,
)
def reset_filters(_):
    return None, None, None, None, None, None, None, None, None


@app.callback(
    Output("chart-ncorrect", "children"),
    Output("chart-format", "children"),
    Output("chart-exectime", "children"),
    Output("results-table", "data"),
    Input("filter-model", "value"),
    Input("filter-doc", "value"),
    Input("filter-prompt-version", "value"),
    Input("filter-tags", "value"),
    Input("filter-date", "start_date"),
    Input("filter-date", "end_date"),
    Input("filter-ncorrect-min", "value"),
    Input("filter-ncorrect-max", "value"),
    Input("filter-format-correctness", "value"),
)
def apply_filters(
    models,
    docs,
    prompt_versions,
    tags,
    start_date,
    end_date,
    ncorrect_min,
    ncorrect_max,
    format_compliance,
):
    filtered = df.copy()

    if models:
        filtered = filtered[filtered["model"].isin(models)]

    if docs:
        filtered = filtered[filtered["doc"].isin(docs)]

    if prompt_versions and "prompt_version" in filtered.columns:
        filtered = filtered[filtered["prompt_version"].isin(prompt_versions)]

    if tags and "tags" in filtered.columns:

        def _has_tag(row_tags) -> bool:
            if pd.isna(row_tags):
                return False
            return any(t in {s.strip() for s in str(row_tags).split(",")} for t in tags)

        filtered = filtered[filtered["tags"].apply(_has_tag)]

    if "timestamp" in filtered.columns:
        ts = pd.to_datetime(filtered["timestamp"], errors="coerce")
        if start_date:
            filtered = filtered[ts >= pd.to_datetime(start_date)]
        if end_date:
            filtered = filtered[ts <= pd.to_datetime(end_date) + pd.Timedelta(days=1)]

    if ncorrect_min is not None and "correct_fields" in filtered.columns:
        filtered = filtered[filtered["correct_fields"] >= ncorrect_min]

    if ncorrect_max is not None and "correct_fields" in filtered.columns:
        filtered = filtered[filtered["correct_fields"] <= ncorrect_max]

    if format_compliance and "format_compliance" in filtered.columns:
        filtered = filtered[filtered["format_compliance"].isin(format_compliance)]

    return (
        build_cin_ncorrect_chart(filtered),
        build_cin_format_chart(filtered),
        build_cin_execution_time_chart(filtered),
        filtered.to_dict("records"),  # type: ignore[return-value]
    )


@app.callback(
    Output("row-detail-panel", "style"),
    Output("row-detail-panel", "children"),
    Input("results-table", "active_cell"),
    State("results-table", "derived_virtual_data"),
)
def show_row_detail(active_cell, virtual_data):
    if not active_cell or not virtual_data:
        return {**SECTION_STYLE, "display": "none"}, []

    row = virtual_data[active_cell["row"]]
    row_id = row.get("id", "—")
    model = row.get("model", "—")
    doc = row.get("doc", "—")
    correct_fields = row.get("correct_fields", "—")

    available_detail = [c for c in DETAIL_COLS if c in row]

    reason_blocks = []
    for col in ["correct_fields_reason", "format_compliance_reason"]:
        if col not in row:
            continue
        label = col.replace("_", " ").title()
        reason_blocks.append(
            html.Div(
                [
                    html.Strong(label, style=FIELD_LABEL_STYLE),
                    html.Pre(str(row[col] or ""), style=REASON_BOX_STYLE),
                ]
            )
        )

    json_cols = [c for c in ["expected", "generated"] if c in available_detail]
    json_block = (
        html.Div(
            style=CODE_COLUMNS_STYLE,
            children=[
                html.Div(
                    [
                        html.Strong(col.title(), style=FIELD_LABEL_STYLE),
                        html.Pre(pretty_json(row.get(col, "")), style=CODE_BOX_STYLE),
                    ]
                )
                for col in json_cols
            ],
        )
        if json_cols
        else None
    )

    children = [
        html.H3("Row Detail", style={"marginBottom": "12px"}),
        html.Div(
            f"id: {row_id}  |  doc: {doc}  |  model: {model}  |  correct_fields: {correct_fields}",
            style=DETAIL_HEADER_STYLE,
        ),
        *reason_blocks,
        *([] if json_block is None else [json_block]),
    ]

    return {**SECTION_STYLE, "display": "block"}, children


@app.callback(
    Output("summary-ncorrect", "children"),
    Output("summary-format", "children"),
    Input("summarize-btn", "n_clicks"),
    State("results-table", "data"),
    prevent_initial_call=True,
)
def summarize_reasons_callback(n_clicks, table_data):
    if not table_data:
        return "No data to summarize.", "No data to summarize."

    ncorrect_reasons = [
        str(row["correct_fields_reason"])
        for row in table_data
        if row.get("correct_fields_reason") and str(row["correct_fields_reason"]).strip()
    ]
    format_reasons = [
        str(row["format_compliance_reason"])
        for row in table_data
        if row.get("format_compliance_reason") and str(row["format_compliance_reason"]).strip()
    ]

    ncorrect_summary = summarize_reasons(ncorrect_reasons, "correct_fields")
    format_summary = summarize_reasons(format_reasons, "format_compliance")

    return ncorrect_summary, format_summary


if __name__ == "__main__":
    app.run(debug=True)
