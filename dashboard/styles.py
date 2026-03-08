GREEN_YELLOW = ["#4a7c59", "#80b347", "#c5e063", "#f0f4a3"]

BG_STYLE = {
    "minHeight": "100vh",
    "backgroundImage": "radial-gradient(circle, #c8c8c8 1px, transparent 1px)",
    "backgroundSize": "24px 24px",
    "backgroundColor": "#f2f2f2",
    "padding": "40px 0",
}

BODY_STYLE = {
    "width": "80%",
    "margin": "0 auto",
    "fontFamily": "sans-serif",
}

SECTION_STYLE = {
    "marginBottom": "32px",
    "backgroundColor": "white",
    "borderRadius": "10px",
    "border": "1px solid #dde8d8",
    "boxShadow": "0 4px 16px rgba(0,0,0,0.08)",
    "padding": "24px 28px",
}

CHART_LAYOUT = {
    "plot_bgcolor": "white",
    "paper_bgcolor": "white",
    "legend_title_text": "Prompt version",
    "font": {"family": "sans-serif", "size": 13},
    "title_font_size": 15,
}

TABLE_STYLE_CELL = {
    "fontFamily": "monospace",
    "fontSize": "13px",
    "padding": "6px 12px",
    "textAlign": "left",
    "whiteSpace": "nowrap",
    "maxWidth": "300px",
    "overflow": "hidden",
    "textOverflow": "ellipsis",
}

DETAIL_HEADER_STYLE = {
    "marginBottom": "16px",
    "fontSize": "14px",
    "color": "#555",
    "fontFamily": "monospace",
}

REASON_BOX_STYLE = {
    "backgroundColor": "#f8f8f8",
    "border": "1px solid #dde8d8",
    "borderRadius": "6px",
    "padding": "12px 16px",
    "fontFamily": "monospace",
    "fontSize": "13px",
    "whiteSpace": "pre-wrap",
    "marginBottom": "16px",
    "maxHeight": "150px",
    "overflowY": "auto",
}

CODE_COLUMNS_STYLE = {
    "display": "grid",
    "gridTemplateColumns": "1fr 1fr",
    "gap": "16px",
    "marginTop": "8px",
}

CODE_BOX_STYLE = {
    "backgroundColor": "#1e1e1e",
    "color": "#d4d4d4",
    "border": "1px solid #333",
    "borderRadius": "6px",
    "padding": "14px 16px",
    "fontFamily": "monospace",
    "fontSize": "12px",
    "whiteSpace": "pre",
    "overflowX": "auto",
    "overflowY": "auto",
    "maxHeight": "400px",
}

TABLE_STYLE_HEADER = {
    "fontWeight": "bold",
    "backgroundColor": "#e8f5e3",
}

# ── Filter panel ──────────────────────────────────────────────────────────────

FILTER_LABEL_STYLE = {
    "fontSize": "11px",
    "fontWeight": "700",
    "color": "#666",
    "marginBottom": "5px",
    "display": "block",
    "textTransform": "uppercase",
    "letterSpacing": "0.6px",
}

FILTER_DROPDOWN_STYLE = {"fontSize": "13px"}

FILTER_INPUT_STYLE = {
    "width": "64px",
    "padding": "5px 8px",
    "border": "1px solid #ddd",
    "borderRadius": "4px",
    "fontSize": "13px",
    "fontFamily": "sans-serif",
    "color": "#333",
}

RESET_BTN_STYLE = {
    "backgroundColor": "white",
    "color": "#4a7c59",
    "border": "1px solid #4a7c59",
    "borderRadius": "4px",
    "padding": "6px 14px",
    "fontSize": "13px",
    "cursor": "pointer",
}

SUMMARIZE_BTN_STYLE = {
    "backgroundColor": "#4a7c59",
    "color": "white",
    "border": "none",
    "borderRadius": "4px",
    "padding": "8px 20px",
    "fontSize": "13px",
    "cursor": "pointer",
    "fontWeight": "600",
}

SUMMARY_COLUMNS_STYLE = {
    **CODE_COLUMNS_STYLE,  # type: ignore[misc]
    "marginTop": "20px",
}

SUMMARY_LABEL_STYLE = {
    "fontSize": "13px",
    "display": "block",
    "marginBottom": "8px",
    "color": "#4a7c59",
}

FIELD_LABEL_STYLE = {
    "fontSize": "13px",
    "display": "block",
    "marginBottom": "6px",
}

SUMMARY_BOX_STYLE = {
    "backgroundColor": "#f8f8f8",
    "border": "1px solid #dde8d8",
    "borderRadius": "6px",
    "padding": "14px 16px",
    "fontFamily": "sans-serif",
    "fontSize": "13px",
    "lineHeight": "1.6",
    "minHeight": "80px",
    "maxHeight": "300px",
    "overflowY": "auto",
    "color": "#333",
}
