from __future__ import annotations

DASHBOARD_CSS: str = """
<style>
    html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"],
    [data-testid="stHeader"], [data-testid="stMain"], .main, .block-container {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
    }
    .g-xtitle text, .g-ytitle text {
        fill: #0F172A !important;
        color: #0F172A !important;
    }

    [data-testid="stHeader"] {
        background-color: #FFFFFF !important;
        border-bottom: 1px solid #E2E8F0;
    }
    [data-testid="stCaptionContainer"],
    [data-testid="stCaption"],
    .stCaption {
        color: #64748B !important;
    }

    [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
        background-color: #F8FAFC !important;
    }
    [data-testid="stSidebar"] * { color: #0F172A !important; }
    [data-testid="stSidebar"] .stMarkdown h2 {
        font-size: 11px !important;
        color: #334155 !important;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-weight: 700;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        color: #0F172A !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #0F172A !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        padding: 6px 10px !important;
        margin-bottom: 4px !important;
        border-radius: 4px !important;
        width: 100%;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] {
        background-color: #DBEAFE !important;
        border-color: #2563EB !important;
    }
    [data-testid="stSidebar"] button[kind="primary"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] button[kind="primary"]:hover {
        background-color: #1D4ED8 !important;
    }
    button[kind="primary"] p { color: #FFFFFF !important; }

    h1, h2, h3, h4, h5, h6 { color: #0F172A !important; }
    h1 { font-size: 26px !important; }
    h3 {
        font-size: 16px !important;
        color: #1E293B !important;
        font-weight: 600 !important;
    }

    [data-testid="stPlotlyChart"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 12px;
    }
    hr { border-color: #E2E8F0 !important; }

    .metric-card {
        border-radius: 6px;
        padding: 14px 18px;
        border: 1px solid;
    }
    .metric-card-blue   { background: #EFF6FF !important; border-color: #DBEAFE !important; }
    .metric-card-orange { background: #FFF7ED !important; border-color: #FED7AA !important; }
    .metric-card-green  { background: #F0FDF4 !important; border-color: #BBF7D0 !important; }
    .metric-card-purple { background: #FAF5FF !important; border-color: #E9D5FF !important; }
    .metric-label  { color: #64748B !important; font-size: 12px; margin-bottom: 6px; }
    .metric-blue   { color: #1E40AF !important; font-size: 26px; font-weight: 700; }
    .metric-orange { color: #9A3412 !important; font-size: 26px; font-weight: 700; }
    .metric-green  { color: #15803D !important; font-size: 26px; font-weight: 700; }
    .metric-purple { color: #6B21A8 !important; font-size: 22px; font-weight: 700; }
    .metric-sub    { color: #64748B !important; font-size: 11px; margin-top: 6px; }
</style>
"""
