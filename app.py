import os
import json
import random
import html
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from textwrap import dedent

import streamlit as st
from rapidfuzz import fuzz
from google import genai


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="NZ Pharmacy OSCE Simulator",
    page_icon="💊",
    layout="wide"
)

# =========================
# Paths
# =========================
DATA_DIR = Path("data")
CASES_FILE = DATA_DIR / "cases.jsonl"
SCRIPTS_FILE = DATA_DIR / "mock_scripts.jsonl"
BEHAVIOUR_FILE = DATA_DIR / "patient_behaviour_rules.jsonl"
OVERLAY_FILE = DATA_DIR / "retrieval_overlay.jsonl"
ANSWER_KEY_FILE = DATA_DIR / "assessor_answer_key.jsonl"
SIM_RULES_FILE = DATA_DIR / "simulator_prompt_rules.jsonl"

DEFAULT_MODEL = "gemini-3-flash-preview"
MAX_HISTORY = 20


# =========================
# Styling
# =========================
st.markdown("""
<style>
#MainMenu, header, footer {visibility: hidden;}

.main-title {
    font-size: 2rem;
    font-weight: 700;
}

.sub-title {
    color: #6b7280;
}

.case-banner {
    background: #f7f7f8;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 14px;
    margin-bottom: 10px;
}

#chatbox {
    height: 65vh;
    overflow-y: auto;
    background: #fafafa;
    border-radius: 16px;
    padding: 12px;
}

.bubble {
    padding: 10px;
    border-radius: 14px;
    margin: 5px 0;
    max-width: 70%;
}

.user {
    background: black;
    color: white;
    float: right;
}

.assistant {
    background: #e5e7eb;
    float: left;
}

.clear { clear: both; }

.script-card {
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 15px;
    background: white;
}

.script-title {
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# =========================
# Helpers
# =========================
def get_client():
    key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    return genai.Client(api_key=key)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def escape_html(text):
    return html.escape(text).replace("\n", "<br>")


def render_chat(messages):
    html_block = ["<div id='chatbox'>"]
    for m in messages:
        if m["role"] == "system":
            continue
        cls = "user" if m["role"] == "user" else "assistant"
        html_block.append(f"<div class='bubble {cls}'>{escape_html(m['content'])}</div><div class='clear'></div>")
    html_block.append("</div>")
    st.markdown("\n".join(html_block), unsafe_allow_html=True)


def render_script(data):
    html_block = dedent(f"""
    <div class="script-card">
        <div class="script-title">Mock Prescription</div>
        <br>

        <b>Prescriber:</b> {data.get("prescriber_name")}<br>
        <b>Address:</b> {data.get("prescriber_address")}<br>
        <b>Phone:</b> {data.get("prescriber_phone")}<br>

        <hr>

        <b>Patient:</b> {data.get("patient_name")}<br>
        <b>Address:</b> {data.get("patient_address")}<br>
        <b>Date:</b> {data.get("date")}<br>

        <hr>

        <b>Medicine:</b> {data.get("medicine")}<br>
        <b>Strength:</b> {data.get("strength")}<br>
        <b>Directions:</b> {data.get("directions")}<br>
        <b>Quantity:</b> {data.get("quantity")}<br>
        <b>Repeats:</b> {data.get("repeats")}<br>
    </div>
    """)
    st.markdown(html_block, unsafe_allow_html=True)


# =========================
# Load data
# =========================
cases = load_jsonl(CASES_FILE)
scripts = load_jsonl(SCRIPTS_FILE)
script_map = {s["mock_script_id"]: s for s in scripts}
client = get_client()


# =========================
# Header
# =========================
st.markdown('<div class="main-title">NZ Pharmacy OSCE Simulator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Practice pharmacy OSCE interactions</div>', unsafe_allow_html=True)

if st.button("Clear cache"):
    st.cache_data.clear()
    st.rerun()


# =========================
# Case selection
# =========================
mode = st.radio("Choose mode", ["Manual", "Random"])

if mode == "Manual":
    case_label = st.selectbox("Pick case", [f"{c['case_id']} - {c['title']}" for c in cases])
    selected_case = cases[[f"{c['case_id']} - {c['title']}" for c in cases].index(case_label)]
else:
    selected_case = random.choice(cases)
    st.write(f"Random case: {selected_case['case_id']}")

mock_id = selected_case.get("mock_script_id")
selected_script = script_map.get(mock_id)


# =========================
# Buttons
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Start"):
        st.session_state.messages = [
            {"role": "assistant", "content": selected_case["patient_opening"]}
        ]

with col2:
    if st.button("Reset"):
        st.session_state.messages = []
        st.session_state.show_script = False

with col3:
    if st.button("Mock Script"):
        st.session_state.show_script = not st.session_state.get("show_script", False)


# =========================
# Show script
# =========================
if st.session_state.get("show_script"):
    if selected_script:
        render_script(selected_script["data"])
    else:
        st.warning("No script found")


# =========================
# Chat
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

render_chat(st.session_state.messages)

prompt = st.chat_input("Your response...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    if client:
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt
        )
        reply = response.text
    else:
        reply = "No API key"

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()


# =========================
# Download
# =========================
if st.session_state.messages:
    transcript = "\n".join(
        [f"{m['role']}: {m['content']}" for m in st.session_state.messages if m["role"] != "system"]
    )

    st.download_button(
        "Download transcript",
        transcript,
        file_name=f"chat_{datetime.now().strftime('%H%M%S')}.txt"
    )