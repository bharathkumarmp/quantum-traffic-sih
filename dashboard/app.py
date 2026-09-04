# ============================================================
# Q-ROUTE
# Intelligent Traffic Route Optimization
#
# dashboard/app.py
# ============================================================

import base64
from pathlib import Path
from textwrap import dedent

import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Q-ROUTE | Intelligent Traffic Route Optimization",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PATHS
# ============================================================

DASHBOARD_DIR = Path(__file__).resolve().parent

ASSETS_DIR = DASHBOARD_DIR / "assets"

BACKGROUND_IMAGE = (
    ASSETS_DIR / "traffic_background.png"
)

CSS_FILE = (
    DASHBOARD_DIR / "style.css"
)


# ============================================================
# HTML RENDERER
# ============================================================

def render_html(html: str):

    html = dedent(html).strip()

    if hasattr(st, "html"):

        st.html(html)

    else:

        st.markdown(
            html,
            unsafe_allow_html=True,
        )


# ============================================================
# LOAD CSS
# ============================================================

def load_css():

    if not CSS_FILE.exists():

        st.error(
            f"""
            style.css was not found.

            Expected location:

            {CSS_FILE}
            """
        )

        return

    try:

        with open(
            CSS_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            css = file.read()

        render_html(
            f"""
            <style>
            {css}
            </style>
            """
        )

    except Exception as error:

        st.error(
            f"Could not load style.css: {error}"
        )


load_css()


# ============================================================
# LOAD BACKGROUND IMAGE
# ============================================================

def load_background():

    if not BACKGROUND_IMAGE.exists():

        return None

    try:

        with open(
            BACKGROUND_IMAGE,
            "rb",
        ) as file:

            image_bytes = file.read()

        return base64.b64encode(
            image_bytes
        ).decode("utf-8")

    except Exception:

        return None


background_base64 = load_background()


# ============================================================
# BACKGROUND
# ============================================================

if background_base64:

    render_html(
        f"""
        <style>

        .stApp {{

            min-height: 100vh;

            background-image:

                linear-gradient(
                    90deg,

                    rgba(
                        2,
                        10,
                        19,
                        0.95
                    ) 0%,

                    rgba(
                        3,
                        12,
                        22,
                        0.87
                    ) 27%,

                    rgba(
                        3,
                        12,
                        22,
                        0.48
                    ) 58%,

                    rgba(
                        3,
                        12,
                        22,
                        0.12
                    ) 100%
                ),

                url(
                    "data:image/png;base64,{background_base64}"
                );

            background-size: cover;

            background-position: center;

            background-repeat: no-repeat;

            background-attachment: fixed;

        }}


        [data-testid="stAppViewContainer"] {{

            background:
                transparent !important;

        }}


        [data-testid="stAppViewContainer"]
        > .main {{

            background:
                transparent !important;

        }}


        [data-testid="stMain"] {{

            background:
                transparent !important;

        }}

        </style>
        """
    )

else:

    st.error(
        """
        Background image not found.

        Please put your image here:

        dashboard/assets/traffic_background.png
        """
    )


# ============================================================
# STREAMLIT UI CLEANUP
# ============================================================

render_html(
    """
    <style>

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    [data-testid="stSidebar"] {
        display: none;
    }

    [data-testid="stDecoration"] {
        display: none;
    }

    .block-container {

        max-width: 100% !important;

        padding-top: 0 !important;

        padding-bottom: 0 !important;

        padding-left: 0 !important;

        padding-right: 0 !important;

    }

    </style>
    """
)


# ============================================================
# HEADER
# ============================================================

render_html(
    """
    <div class="landing-header">


        <div class="brand">


            <div class="brand-icon">
                🚦
            </div>


            <div class="brand-name">
                Q-ROUTE
            </div>


        </div>


        <div class="header-status">


            <span class="status-dot"></span>


            <span>
                Intelligent Traffic System
            </span>


        </div>


    </div>
    """
)


# ============================================================
# HERO SECTION
# ============================================================

render_html(
    """
    <div class="hero-container">


        <div class="hero-content">


            <div class="hero-label">

                QUANTUM-INSPIRED
                TRAFFIC OPTIMIZATION

            </div>


            <h1 class="hero-title">

                Q-ROUTE

            </h1>


            <h2 class="hero-subtitle">

                Intelligent Traffic
                Route Optimization

            </h2>


            <div class="hero-line"></div>


            <p class="hero-description">

                Smart routing for dynamic

                <br>

                urban traffic

            </p>


            <p class="hero-text">

                Optimize vehicle routes using
                quantum-inspired metaheuristic
                optimization under changing
                traffic conditions.

            </p>


        </div>


    </div>
    """
)


# ============================================================
# GET STARTED BUTTON
# ============================================================

button_left, button_center, button_right = st.columns(
    [
        0.07,
        0.24,
        0.69,
    ]
)


with button_center:

    start = st.button(
        "GET STARTED  →",

        use_container_width=True,

        key="get_started",
    )


# ============================================================
# NAVIGATION
# ============================================================

if start:

    optimizer_page = (
        DASHBOARD_DIR
        / "pages"
        / "1_Optimizer.py"
    )

    if optimizer_page.exists():

        st.switch_page(
            "pages/1_Optimizer.py"
        )

    else:

        st.error(
            f"""
            Optimizer page not found.

            Expected:

            {optimizer_page}
            """
        )


# ============================================================
# FEATURE SECTION
#
# IMPORTANT:
# This section is AFTER the GET STARTED button.
# It is NOT absolute positioned.
# ============================================================

render_html(
    """
    <div class="bottom-info">


        <!-- ==================================================
             DYNAMIC TRAFFIC
             ================================================== -->

        <div class="info-item">


            <div class="info-icon traffic-icon">
                🚦
            </div>


            <div class="info-content">

                <b>
                    Dynamic Traffic
                </b>


                <small>
                    Real-time traffic scenarios
                </small>

            </div>


        </div>


        <!-- ==================================================
             QUANTUM INSPIRED
             ================================================== -->

        <div class="info-item">


            <div class="info-icon quantum-icon">
                ⚛️
            </div>


            <div class="info-content">

                <b>
                    Quantum-Inspired
                </b>


                <small>
                    QPSO-based optimization
                </small>

            </div>


        </div>


        <!-- ==================================================
             FLEET OPTIMIZATION
             ================================================== -->

        <div class="info-item">


            <div class="info-icon fleet-icon">
                🚚
            </div>


            <div class="info-content">

                <b>
                    Fleet Optimization
                </b>


                <small>
                    Constraint-aware routing
                </small>

            </div>


        </div>


        <!-- ==================================================
             BENCHMARKING
             ================================================== -->

        <div class="info-item">


            <div class="info-icon benchmark-icon">
                📊
            </div>


            <div class="info-content">

                <b>
                    Benchmarking
                </b>


                <small>
                    Compare optimization methods
                </small>

            </div>


        </div>


    </div>
    """
)


# ============================================================
# FOOTER
# ============================================================

render_html(
    """
    <div class="landing-footer">


        <span>
            Q-ROUTE
        </span>


        <span>
            Quantum-Inspired Intelligent
            Transportation Optimization
        </span>


    </div>
    """
)