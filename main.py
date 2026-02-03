import streamlit as st
from dashboard.components.data_uploader import upload
from dashboard.pages.analysis_page import render_analysis_page
from dashboard.pages.forecasting_page import render_forecasting_page
from dashboard.pages.regression_page import render_regression_page
import os
os.environ["STREAMLIT_SERVER_WATCH_FILE_BLACKLIST"] = ".*/__pycache__/.*,.*\\.pyc,.*\\.pyo,.*\\.pyd"
os.environ["STREAMLIT_SERVER_ENABLE_WATCHER"] = "false"  # Полное отключение при проблемах

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Montserrat', sans-serif;
        font-weight: 600;
        padding-top: 0rem !important;
    }

    .block-container {
        padding-top: 1rem !important;
    }

    .block {
        border: 1px solid #ccc;
        background-color: #e0e0e0;
        border-radius: 4px;
    }

    .stDataFrame {
        height: 250px !important;
        overflow-y: auto;
    }
    </style>
""", unsafe_allow_html=True)

header_cols = st.columns([2, 10])
with header_cols[0]:
    page = st.radio(
        "Выбор страницы",
        ["Прогнозирование", "Анализ данных", "Регрессионный анализ"]
    )

with header_cols[1]:
    df, outlier_percentage = upload()

if page == "Прогнозирование":
    render_forecasting_page(df, outlier_percentage)

elif page == "Анализ данных":
    render_analysis_page(df, outlier_percentage)

elif page == "Регрессионный анализ":
    render_regression_page(df, outlier_percentage)