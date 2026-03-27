import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gout_agent.ui import render_app


st.set_page_config(
    page_title="痛风健康分身",
    page_icon="",
    layout="wide",
)

render_app(PROJECT_ROOT)
