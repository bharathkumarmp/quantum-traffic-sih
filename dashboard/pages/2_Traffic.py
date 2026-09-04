# ============================================================
# Q-ROUTE
# 2_Traffic.py
# Traffic Monitoring & Scenario Analysis
# ============================================================

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# SERVICE IMPORTS
# ============================================================

from optimizer_service import (
    get_traffic_summary,
    get_traffic_records,
    get_scenario_info,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Q-ROUTE | Traffic",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LOAD PAGE-SPECIFIC CSS
# ============================================================

CSS_PATH = Path(__file__).resolve().parent / "2_Traffic.css"

if CSS_PATH.exists():
    css = CSS_PATH.read_text(encoding="utf-8")
    st.html(f"<style>{css}</style>")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def html(content):
    """Render HTML using Streamlit's HTML renderer."""
    st.html(content)


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def find_column(df, candidates):
    """Return the first matching column name."""
    for column in candidates:
        if column in df.columns:
            return column
    return None


def numeric_column(df, candidates, default=0.0):
    """Return a numeric pandas Series from the first matching column."""
    column = find_column(df, candidates)

    if column is None:
        return pd.Series(
            [default] * len(df),
            index=df.index,
            dtype=float,
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).fillna(default)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    html(
        """
        <div class="sidebar-brand">

            <div class="sidebar-logo">
                ⚛
            </div>

            <div>
                <div class="sidebar-title">
                    Q-ROUTE
                </div>

                <div class="sidebar-subtitle">
                    Intelligent Traffic Optimization
                </div>
            </div>

        </div>
        """
    )

    st.markdown("---")

    st.markdown(
        "### 🚦 Traffic Scenario"
    )

    scenario = st.selectbox(
        "Traffic Scenario",
        [
            "Normal",
            "Morning Peak",
            "Heavy Congestion",
            "Evening Peak",
            "Accident",
            "Road Closure",
        ],
        index=0,
    )

    st.markdown("---")

    st.markdown(
        """
        <div class="sidebar-note">
            Select a scenario to inspect current
            traffic conditions and road availability.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HEADER
# ============================================================

html(
    f"""
    <div class="traffic-header">

        <div class="traffic-header-left">

            <div class="traffic-status-row">

                <div class="status-ready">
                    <span class="status-dot"></span>
                    SYSTEM READY
                </div>

                <div class="scenario-badge">
                    🚦 {scenario}
                </div>

            </div>

            <div class="eyebrow">
                REAL-TIME TRAFFIC INTELLIGENCE
            </div>

            <h1>
                Traffic
                <span>Monitor</span>
            </h1>

            <p>
                Intelligent traffic monitoring for
                dynamic urban transportation networks
            </p>

        </div>

        <div class="traffic-signal-container">

            <div class="signal-pole"></div>

            <div class="signal-arm"></div>

            <div class="signal-box">

                <div class="signal-light signal-red"></div>
                <div class="signal-light signal-yellow"></div>
                <div class="signal-light signal-green"></div>

            </div>

            <div class="signal-road"></div>

        </div>

    </div>
    """
)


# ============================================================
# LOAD TRAFFIC DATA
# ============================================================

try:
    traffic_records = get_traffic_records()
except Exception as error:

    st.error(
        f"Unable to load traffic records: {error}"
    )

    traffic_records = []


# ============================================================
# CONVERT RECORDS TO DATAFRAME
# ============================================================

if isinstance(traffic_records, pd.DataFrame):

    records_df = traffic_records.copy()

elif isinstance(traffic_records, list):

    records_df = pd.DataFrame(
        traffic_records
    )

else:

    records_df = pd.DataFrame()


# ============================================================
# TRAFFIC SUMMARY
# ============================================================

try:
    traffic_summary = get_traffic_summary(
        scenario
    )
except Exception:
    traffic_summary = {}


if not isinstance(traffic_summary, dict):
    traffic_summary = {}


# ============================================================
# SCENARIO INFORMATION
# ============================================================

try:
    scenario_info = get_scenario_info(
        scenario
    )
except Exception:
    scenario_info = {}


if not isinstance(scenario_info, dict):
    scenario_info = {}


# ============================================================
# FILTER SCENARIO
# ============================================================

if (
    not records_df.empty
    and "scenario" in records_df.columns
):

    current_df = records_df[
        records_df["scenario"]
        .astype(str)
        .str.strip()
        .str.lower()
        ==
        scenario.strip().lower()
    ].copy()

else:

    current_df = records_df.copy()


# ============================================================
# EXTRACT METRICS
# ============================================================

if not current_df.empty:

    congestion_values = numeric_column(
        current_df,
        [
            "congestion",
            "Congestion",
            "congestion_level",
        ],
    )

    speed_values = numeric_column(
        current_df,
        [
            "traffic_speed",
            "Traffic Speed",
            "speed",
            "speed_kmph",
        ],
    )

    blocked_values = numeric_column(
        current_df,
        [
            "blocked",
            "Blocked",
            "is_blocked",
        ],
    )

    average_congestion = (
        float(congestion_values.mean())
        if len(congestion_values)
        else 0.0
    )

    average_speed = (
        float(speed_values.mean())
        if len(speed_values)
        else 0.0
    )

    blocked_edges = int(
        (blocked_values > 0).sum()
    )

    total_edges = len(current_df)

else:

    average_congestion = safe_float(
        traffic_summary.get(
            "average_congestion",
            0,
        )
    )

    average_speed = safe_float(
        traffic_summary.get(
            "average_speed",
            traffic_summary.get(
                "average_speed_kmph",
                0,
            ),
        )
    )

    blocked_edges = int(
        safe_float(
            traffic_summary.get(
                "blocked_edges",
                0,
            )
        )
    )

    total_edges = int(
        safe_float(
            traffic_summary.get(
                "total_edges",
                0,
            )
        )
    )


# ============================================================
# NORMALIZE CONGESTION
# ============================================================

if average_congestion > 1:
    average_congestion /= 100.0


# ============================================================
# TRAFFIC STATUS
# ============================================================

if blocked_edges > 0:

    traffic_status = "ROAD DISRUPTION"
    traffic_status_class = "status-warning"
    active_signal = "red"

elif average_congestion >= 0.70:

    traffic_status = "HEAVY CONGESTION"
    traffic_status_class = "status-warning"
    active_signal = "red"

elif average_congestion >= 0.40:

    traffic_status = "MODERATE TRAFFIC"
    traffic_status_class = "status-warning"
    active_signal = "yellow"

else:

    traffic_status = "TRAFFIC NORMAL"
    traffic_status_class = "status-ready"
    active_signal = "green"


# ============================================================
# UPDATE SIGNAL
# ============================================================

html(
    f"""
    <style>

    .signal-red {{
        opacity: {
            "1" if active_signal == "red" else "0.25"
        };
    }}

    .signal-yellow {{
        opacity: {
            "1" if active_signal == "yellow" else "0.25"
        };
    }}

    .signal-green {{
        opacity: {
            "1" if active_signal == "green" else "0.25"
        };
    }}

    </style>
    """
)


# ============================================================
# STATUS BAR
# ============================================================

html(
    f"""
    <div class="{traffic_status_class} traffic-status-main">

        <span class="status-dot"></span>

        {traffic_status}

        <span class="status-scenario">
            {scenario}
        </span>

    </div>
    """
)


# ============================================================
# TRAFFIC OVERVIEW
# ============================================================

html(
    """
    <div class="zone-header">

        <div class="zone-title">
            📊 Traffic Overview
        </div>

        <div class="zone-subtitle">
            Current traffic conditions across the network
        </div>

    </div>
    """
)


metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)


# ------------------------------------------------------------
# CONGESTION
# ------------------------------------------------------------

with metric_col1:

    html(
        f"""
        <div class="traffic-metric-card">

            <div class="traffic-metric-icon">
                🚦
            </div>

            <div class="traffic-metric-label">
                CONGESTION
            </div>

            <div class="traffic-metric-value">
                {average_congestion * 100:.1f}%
            </div>

            <div class="traffic-metric-description">
                Network average
            </div>

        </div>
        """
    )


# ------------------------------------------------------------
# BLOCKED EDGES
# ------------------------------------------------------------

with metric_col2:

    html(
        f"""
        <div class="traffic-metric-card">

            <div class="traffic-metric-icon">
                🚧
            </div>

            <div class="traffic-metric-label">
                BLOCKED EDGES
            </div>

            <div class="traffic-metric-value">
                {blocked_edges}
            </div>

            <div class="traffic-metric-description">
                Roads unavailable
            </div>

        </div>
        """
    )


# ------------------------------------------------------------
# SPEED
# ------------------------------------------------------------

with metric_col3:

    html(
        f"""
        <div class="traffic-metric-card">

            <div class="traffic-metric-icon">
                🏎️
            </div>

            <div class="traffic-metric-label">
                AVERAGE SPEED
            </div>

            <div class="traffic-metric-value">
                {average_speed:.1f}
            </div>

            <div class="traffic-metric-description">
                km/h
            </div>

        </div>
        """
    )


# ------------------------------------------------------------
# AVAILABLE ROADS
# ------------------------------------------------------------

with metric_col4:

    available_edges = max(
        total_edges - blocked_edges,
        0,
    )

    html(
        f"""
        <div class="traffic-metric-card">

            <div class="traffic-metric-icon">
                🛣️
            </div>

            <div class="traffic-metric-label">
                ROADS AVAILABLE
            </div>

            <div class="traffic-metric-value">
                {available_edges}
            </div>

            <div class="traffic-metric-description">
                of {total_edges} edges
            </div>

        </div>
        """
    )


# ============================================================
# CURRENT ROAD STATE
# ============================================================

html(
    """
    <div class="zone-header traffic-section-gap">

        <div class="zone-title">
            🛣️ Current Road State
        </div>

        <div class="zone-subtitle">
            Road availability under the selected scenario
        </div>

    </div>
    """
)


road_col1, road_col2 = st.columns(2)


with road_col1:

    html(
        f"""
        <div class="info-card">

            <div class="info-card-title">
                Roads Available
            </div>

            <div class="info-card-number green-number">
                {available_edges}
            </div>

            <div class="info-card-description">
                Available network edges
            </div>

            <div class="progress-track">

                <div
                    class="progress-fill"
                    style="width:{
                        (
                            available_edges / total_edges * 100
                        )
                        if total_edges > 0
                        else 100
                    }%;"
                ></div>

            </div>

        </div>
        """
    )


with road_col2:

    html(
        f"""
        <div class="info-card">

            <div class="info-card-title">
                Active Scenario
            </div>

            <div class="scenario-large">
                {scenario}
            </div>

            <div class="info-card-description">
                Current traffic condition
            </div>

            <div class="scenario-status-line">
                <span class="status-dot"></span>
                {traffic_status}
            </div>

        </div>
        """
    )


# ============================================================
# SCENARIO INFORMATION
# ============================================================

html(
    """
    <div class="zone-header traffic-section-gap">

        <div class="zone-title">
            ℹ️ Scenario Information
        </div>

        <div class="zone-subtitle">
            Details used by the Q-ROUTE traffic model
        </div>

    </div>
    """
)


source = scenario_info.get(
    "traffic_source",
    scenario_info.get(
        "source",
        "Simulated Traffic Data",
    ),
)

records_count = len(current_df)


info_col1, info_col2, info_col3 = st.columns(3)


with info_col1:

    html(
        f"""
        <div class="info-card small-card">

            <div class="info-card-title">
                Scenario
            </div>

            <div class="info-value">
                {scenario}
            </div>

            <div class="info-card-description">
                Active traffic scenario
            </div>

        </div>
        """
    )


with info_col2:

    html(
        f"""
        <div class="info-card small-card">

            <div class="info-card-title">
                Traffic Source
            </div>

            <div class="info-value">
                {source}
            </div>

            <div class="info-card-description">
                Data source
            </div>

        </div>
        """
    )


with info_col3:

    html(
        f"""
        <div class="info-card small-card">

            <div class="info-card-title">
                Average Congestion
            </div>

            <div class="info-value green-number">
                {average_congestion * 100:.1f}%
            </div>

            <div class="info-card-description">
                Network average
            </div>

        </div>
        """
    )


info_col4, info_col5, info_col6 = st.columns(3)


with info_col4:

    html(
        f"""
        <div class="info-card small-card">

            <div class="info-card-title">
                Average Speed
            </div>

            <div class="info-value">
                {average_speed:.1f} km/h
            </div>

            <div class="info-card-description">
                Current traffic speed
            </div>

        </div>
        """
    )


with info_col5:

    html(
        f"""
        <div class="info-card small-card">

            <div class="info-card-title">
                Blocked Edges
            </div>

            <div class="info-value red-number">
                {blocked_edges}
            </div>

            <div class="info-card-description">
                Road closures
            </div>

        </div>
        """
    )


with info_col6:

    html(
        f"""
        <div class="info-card small-card">

            <div class="info-card-title">
                Records
            </div>

            <div class="info-value">
                {records_count}
            </div>

            <div class="info-card-description">
                Traffic records
            </div>

        </div>
        """
    )


# ============================================================
# ROAD CONGESTION CHART
# ============================================================

html(
    """
    <div class="zone-header traffic-section-gap">

        <div class="zone-title">
            📈 Road Congestion
        </div>

        <div class="zone-subtitle">
            Congestion level across individual network edges
        </div>

    </div>
    """
)


if not current_df.empty:

    edge_column = find_column(
        current_df,
        [
            "edge_id",
            "Edge ID",
            "edge",
            "id",
        ],
    )

    if edge_column is None:

        current_df["_edge_id"] = [
            f"E{i + 1}"
            for i in range(
                len(current_df)
            )
        ]

        edge_column = "_edge_id"

    chart_congestion = numeric_column(
        current_df,
        [
            "congestion",
            "Congestion",
            "congestion_level",
        ],
    )

    chart_congestion = chart_congestion.clip(
        lower=0,
        upper=1,
    )

    chart_df = pd.DataFrame(
        {
            "Edge": current_df[
                edge_column
            ].astype(str),

            "Congestion": (
                chart_congestion * 100
            ),
        }
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=chart_df["Edge"],
            y=chart_df["Congestion"],
            text=[
                f"{value:.0f}%"
                for value in chart_df[
                    "Congestion"
                ]
            ],
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Congestion: %{y:.1f}%"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        height=400,
        margin=dict(
            l=30,
            r=30,
            t=30,
            b=30,
        ),
        yaxis=dict(
            title="Congestion (%)",
            range=[
                0,
                max(
                    100,
                    float(
                        chart_df[
                            "Congestion"
                        ].max()
                    ) * 1.15,
                ),
            ],
        ),
        xaxis=dict(
            title="Road Edge",
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

else:

    html(
        """
        <div class="empty-card">

            <div class="empty-card-icon">
                🚦
            </div>

            <div class="empty-card-title">
                No Road Data
            </div>

            <div class="empty-card-description">
                No road-level records are available
                for this scenario.
            </div>

        </div>
        """
    )


# ============================================================
# TRAFFIC DATA TABLE
# ============================================================

html(
    """
    <div class="zone-header traffic-section-gap">

        <div class="zone-title">
            🗂️ Traffic Data
        </div>

        <div class="zone-subtitle">
            Traffic records used by Q-ROUTE
        </div>

    </div>
    """
)


if not current_df.empty:

    st.dataframe(
        current_df,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "No traffic records available "
        "for the selected scenario."
    )


# ============================================================
# FOOTER
# ============================================================

html(
    """
    <div class="green-footer">

        <div class="footer-icon">
            🌱
        </div>

        <div>
            <strong>Smarter Routes</strong>
            <span> • </span>
            <strong>Greener Cities</strong>
            <span> • </span>
            <strong>Q-ROUTE</strong>
        </div>

    </div>
    """
)