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

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1rem;
    max-width: 1200px;
}

.main-title {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}

.sub-title {
    color: #6b7280;
    margin-bottom: 1rem;
}

.case-banner {
    background: #f7f7f8;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 14px 18px;
    margin-bottom: 12px;
}

.case-meta {
    font-size: 0.92rem;
    color: #4b5563;
    margin-top: 4px;
}

#chatbox {
    height: 66vh;
    overflow-y: auto;
    background: #fafafa;
    border: 1px solid #ececec;
    border-radius: 18px;
    padding: 16px 14px 70px 14px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
}

.bubble {
    display: inline-block;
    padding: 11px 14px;
    border-radius: 18px;
    margin: 6px 0;
    max-width: 72%;
    word-wrap: break-word;
    line-height: 1.45;
    font-size: 0.97rem;
}

.user {
    background: #111827;
    color: white;
    float: right;
    border-bottom-right-radius: 6px;
}

.assistant {
    background: #e5e7eb;
    color: #111827;
    float: left;
    border-bottom-left-radius: 6px;
}

.clear {
    clear: both;
}

.small-note {
    color: #6b7280;
    font-size: 0.85rem;
}

.feedback-box {
    background: #fcfcfc;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 16px;
    margin-top: 12px;
    white-space: pre-wrap;
}

.admin-box {
    background: #fffdf6;
    border: 1px solid #f3e8a6;
    border-radius: 14px;
    padding: 12px;
    margin-top: 10px;
}

.login-box {
    max-width: 460px;
    margin: 4rem auto 0 auto;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 24px;
}

.choice-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 18px;
    text-align: center;
    margin-bottom: 8px;
}

.choice-card-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 4px;
}

.choice-card-sub {
    color: #6b7280;
    font-size: 0.92rem;
}

.section-gap {
    margin-top: 12px;
    margin-bottom: 8px;
}

.script-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 18px;
    margin-top: 8px;
    margin-bottom: 12px;
}

.script-warning {
    font-size: 0.9rem;
    color: #92400e;
    margin-bottom: 14px;
}

.script-title {
    font-weight: 700;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


# =========================
# Helpers
# =========================
def get_gemini_client():
    api_key = None

    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    elif os.getenv("GEMINI_API_KEY"):
        api_key = os.getenv("GEMINI_API_KEY")
    elif "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    elif os.getenv("GOOGLE_API_KEY"):
        api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        return None

    return genai.Client(api_key=api_key)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_load_data():
    needed = [
        CASES_FILE,
        SCRIPTS_FILE,
        BEHAVIOUR_FILE,
        OVERLAY_FILE,
        ANSWER_KEY_FILE,
        SIM_RULES_FILE
    ]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        st.error("Missing required files:\n\n" + "\n".join(missing))
        st.stop()

    return (
        load_jsonl(str(CASES_FILE)),
        load_jsonl(str(SCRIPTS_FILE)),
        load_jsonl(str(BEHAVIOUR_FILE)),
        load_jsonl(str(OVERLAY_FILE)),
        load_jsonl(str(ANSWER_KEY_FILE)),
        load_jsonl(str(SIM_RULES_FILE)),
    )


def normalize_text(s: str) -> str:
    return (s or "").strip().lower()


def escape_html(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def find_overlay(case_id: str, overlays: List[Dict[str, Any]]) -> Dict[str, Any]:
    for o in overlays:
        if o.get("case_id") == case_id:
            return o
    return {}


def find_answer_key(case_id: str, answer_keys: List[Dict[str, Any]]) -> Dict[str, Any]:
    for a in answer_keys:
        if a.get("case_id") == case_id:
            return a
    return {}


def resolve_behaviour(persona: str, behaviours: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not persona:
        return behaviours[0] if behaviours else {}

    persona_norm = normalize_text(persona)
    best = None
    best_score = -1

    for b in behaviours:
        label = normalize_text(b.get("label", ""))
        score = fuzz.token_set_ratio(persona_norm, label)
        if score > best_score:
            best = b
            best_score = score

    return best or {}


def case_categories(case: Dict[str, Any], overlay: Dict[str, Any]) -> List[str]:
    cats = {"all_cases"}

    difficulty = normalize_text(case.get("difficulty", ""))
    if difficulty == "easy":
        cats.add("easy_cases")
    elif difficulty == "medium":
        cats.add("medium_cases")
    elif difficulty == "hard":
        cats.add("hard_cases")

    case_type = normalize_text(case.get("case_type", ""))
    if case_type == "counselling":
        cats.add("counselling_cases")
    elif case_type == "dispensing":
        cats.add("legality_cases")
    elif case_type == "hybrid":
        cats.add("counselling_cases")
        cats.add("legality_cases")

    tags = set(normalize_text(t) for t in case.get("tags", []))
    emotion_tags = set(normalize_text(t) for t in overlay.get("emotion_tags", []))
    legal_tags = set(normalize_text(t) for t in overlay.get("legal_tags", []))
    triage_tags = set(normalize_text(t) for t in overlay.get("triage_tags", []))
    title = normalize_text(case.get("title", ""))
    persona = normalize_text(case.get("patient_persona", ""))

    if "controlled drug" in title or any("controlled" in x for x in tags.union(legal_tags)):
        cats.add("controlled_drug_cases")

    if any(x in tags for x in ["parent", "paediatric", "child"]):
        cats.add("caregiver_cases")
        cats.add("parent_cases")

    if "elderly" in persona or "elderly" in title:
        cats.add("elderly_cases")

    if "low health literacy" in persona or "low_health_literacy" in emotion_tags:
        cats.add("low_health_literacy_cases")

    if any(x in tags for x in ["pregnancy", "women's health", "mental health", "emergency contraception"]):
        cats.add("sensitive_cases")

    if any("privacy" in x for x in emotion_tags):
        cats.add("sensitive_cases")

    if len(triage_tags) > 0:
        cats.add("triage_cases")

    if any("mental" in x for x in tags.union(triage_tags)):
        cats.add("mental_health_or_distress_cases")

    if any("copayment" in x or "co_payment" in x or "psc" in x or "csc" in x or "special_authority" in x
           for x in legal_tags.union(tags)):
        cats.add("funding_cases")

    if any(x in tags for x in ["affordability"]):
        cats.add("affordability_cases")

    if any(x in persona for x in ["angry", "annoyed", "impatient"]):
        cats.add("angry_or_annoyed_cases")

    if any(x in persona for x in ["distressed", "tearful", "panicky"]):
        cats.add("distressed_cases")

    if any(x in tags for x in ["no fixed abode", "visa", "corrections"]):
        cats.add("stigma_or_admin_sensitive_cases")
        cats.add("sensitive_cases")

    return sorted(list(cats))


def applicable_sim_rules(case: Dict[str, Any], overlay: Dict[str, Any], sim_rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cats = set(case_categories(case, overlay))
    picked = []
    for rule in sim_rules:
        applies = set(rule.get("applies_to", []))
        if "all_cases" in applies or len(cats.intersection(applies)) > 0:
            picked.append(rule)

    priority_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    picked.sort(key=lambda r: priority_rank.get(r.get("priority", "medium"), 2))
    return picked


def build_system_prompt(
    case: Dict[str, Any],
    overlay: Dict[str, Any],
    behaviour: Dict[str, Any],
    sim_rules: List[Dict[str, Any]],
    script_data: Dict[str, Any] | None = None
) -> str:
    rules_text = "\n".join(
        [f"- [{r.get('priority', 'medium').upper()}] {r.get('rule_text', '')}" for r in sim_rules[:40]]
    )

    script_section = "No mock prescription linked."

    if script_data:
        script_section = f"""
Prescriber: {script_data.get("prescriber_name", "")}
Prescriber address: {script_data.get("prescriber_address", "")}
Prescriber phone: {script_data.get("prescriber_phone", "")}
Patient name: {script_data.get("patient_name", "")}
Patient address: {script_data.get("patient_address", "")}
Date: {script_data.get("date", "")}
Medicine: {script_data.get("medicine", "")}
Strength: {script_data.get("strength", "")}
Directions: {script_data.get("directions", "")}
Quantity: {script_data.get("quantity", "")}
Repeats: {script_data.get("repeats", "")}
Notes: {script_data.get("notes", "")}
""".strip()

    return f"""
You are roleplaying as a New Zealand community pharmacy patient or caregiver in an OSCE style simulation.

Stay fully in character as the patient or caregiver.
Never act as the pharmacist, tutor, assessor, or legal expert.
Do not reveal hidden facts unless the pharmacist earns them through appropriate questioning, empathy, privacy, or explanation.
Use natural patient language.
Keep most answers short, around 1 to 3 sentences.
Do not state the hidden objective of the case.

CASE PROFILE
Case ID: {case.get("case_id", "")}
Title: {case.get("title", "")}
Difficulty: {case.get("difficulty", "")}
Case Type: {case.get("case_type", "")}
Opening statement: {case.get("patient_opening", "")}
Patient persona: {case.get("patient_persona", "")}
What the patient knows: {case.get("patient_knowledge", "")}
Hidden facts: {json.dumps(case.get("hidden_facts", []), ensure_ascii=False)}
Reveal rules: {json.dumps(case.get("reveal_rules", []), ensure_ascii=False)}
Background issue: {case.get("pharmacy_issue", "")}
Ideal patient behaviour: {case.get("ideal_patient_behaviour", "")}

SCRIPT DETAILS
{script_section}

RETRIEVAL OVERLAY
Intent tags: {json.dumps(overlay.get("intent_tags", []), ensure_ascii=False)}
Legal tags: {json.dumps(overlay.get("legal_tags", []), ensure_ascii=False)}
Triage tags: {json.dumps(overlay.get("triage_tags", []), ensure_ascii=False)}
Emotion tags: {json.dumps(overlay.get("emotion_tags", []), ensure_ascii=False)}
Reveal priority: {json.dumps(overlay.get("reveal_priority", []), ensure_ascii=False)}
Danger flags: {json.dumps(overlay.get("danger_flags", []), ensure_ascii=False)}

BEHAVIOUR LAYER
Behaviour label: {behaviour.get("label", "")}
Tone: {behaviour.get("tone", "")}
Default style: {behaviour.get("default_style", "")}
Reveal pattern: {behaviour.get("reveal_pattern", "")}
Challenge features: {json.dumps(behaviour.get("challenge_features", []), ensure_ascii=False)}
Simulator behaviour rule: {behaviour.get("simulator_rule", "")}

SIMULATOR RULES
{rules_text}

Final execution rules:
1. Reply only as the patient or caregiver.
2. Do not explain policy or legislation unless the patient would vaguely refer to past experience.
3. Do not praise the pharmacist or give hints.
4. If the pharmacist offers privacy, empathy, or a clear reason for questions, become more open.
5. If the pharmacist is blunt, judgmental, or confusing, become less open or more frustrated depending on persona.
6. Stay consistent with facts already revealed.
7. Sound natural, not like a checklist or textbook.
8. If a mock prescription is provided, keep your identity and prescription-related facts consistent with it unless the case is explicitly designed as a mismatch or wrong-patient scenario.
""".strip()


def build_feedback_prompt(case: Dict[str, Any], answer_key: Dict[str, Any], transcript: List[Dict[str, str]]) -> str:
    transcript_text = []
    for m in transcript:
        if m["role"] == "system":
            continue
        transcript_text.append(f"{m['role'].upper()}: {m['content']}")
    transcript_blob = "\n".join(transcript_text)

    return f"""
You are assessing a New Zealand community pharmacy OSCE roleplay.

Evaluate the pharmacist's performance based only on:
1. the case
2. the answer key
3. the transcript

CASE
Title: {case.get("title", "")}
Difficulty: {case.get("difficulty", "")}
Case type: {case.get("case_type", "")}
Issue: {case.get("pharmacy_issue", "")}

ANSWER KEY
Critical questions: {json.dumps(answer_key.get("critical_questions", []), ensure_ascii=False)}
Must do: {json.dumps(answer_key.get("must_do", []), ensure_ascii=False)}
Must not miss: {json.dumps(answer_key.get("must_not_miss", []), ensure_ascii=False)}
Pass cues: {json.dumps(answer_key.get("pass_cues", []), ensure_ascii=False)}
Fail cues: {json.dumps(answer_key.get("fail_cues", []), ensure_ascii=False)}

TRANSCRIPT
{transcript_blob}

Output in this exact format:

Overall rating: Pass / Borderline / Fail

Strengths:
- ...
- ...

Missed opportunities:
- ...
- ...

Legal / safety issues:
- ...
- ...

Suggested better questions:
- ...
- ...

Suggested better counselling / management:
- ...
- ...

One-paragraph summary:
...
""".strip()


def call_model(client, model: str, system_prompt: str, messages: List[Dict[str, str]]) -> str:
    trimmed = messages[-MAX_HISTORY:] if len(messages) > MAX_HISTORY else messages

    transcript_lines = []
    for m in trimmed:
        if m["role"] == "system":
            continue
        speaker = "Pharmacist" if m["role"] == "user" else "Patient"
        transcript_lines.append(f"{speaker}: {m['content']}")
    transcript_text = "\n".join(transcript_lines)

    prompt = f"""
{system_prompt}

Conversation so far:
{transcript_text}

Reply only as the patient or caregiver.
Keep the reply natural and concise.
"""

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    return response.text.strip() if response.text else "Sorry, I could not generate a response."


def generate_feedback(client, model: str, feedback_prompt: str) -> str:
    response = client.models.generate_content(
        model=model,
        contents=feedback_prompt,
    )
    return response.text.strip() if response.text else "No feedback returned."


def render_chat(messages: List[Dict[str, str]]):
    chat_html = ["<div id='chatbox'>"]
    for m in messages:
        if m["role"] == "system":
            continue
        css_class = "user" if m["role"] == "user" else "assistant"
        chat_html.append(f"<div class='bubble {css_class}'>{escape_html(m['content'])}</div><div class='clear'></div>")
    chat_html.append("</div>")
    st.markdown("\n".join(chat_html), unsafe_allow_html=True)


def conversation_transcript(messages: List[Dict[str, str]]) -> str:
    lines = []
    for m in messages:
        if m["role"] == "system":
            continue
        speaker = "Pharmacist" if m["role"] == "user" else "Patient"
        lines.append(f"{speaker}: {m['content']}")
    return "\n".join(lines)


def render_mock_script(script_data: Dict[str, Any]):
    script_type = script_data.get("script_type", "standard")

    title = "Standard prescription"
    if script_type == "controlled_drug":
        title = "Controlled drug prescription"
    elif script_type == "nzeps":
        title = "NZePS prescription"
    elif script_type == "emailed_non_nzeps":
        title = "Emailed non NZePS prescription"
    elif script_type == "faxed_controlled_drug":
        title = "Faxed controlled drug prescription"

    html_block = dedent(f"""
    <div class="script-card">
        <div class="script-title">{title}</div>
        <div class="script-warning">Training script only. Not a real prescription.</div>

        <div><strong>Prescriber:</strong> {html.escape(str(script_data.get("prescriber_name", "")))}</div>
        <div><strong>Prescriber address:</strong> {html.escape(str(script_data.get("prescriber_address", "")))}</div>
        <div><strong>Prescriber phone:</strong> {html.escape(str(script_data.get("prescriber_phone", "")))}</div>

        <hr style="margin:12px 0;">

        <div><strong>Patient:</strong> {html.escape(str(script_data.get("patient_name", "")))}</div>
        <div><strong>Patient address:</strong> {html.escape(str(script_data.get("patient_address", "")))}</div>
        <div><strong>Date:</strong> {html.escape(str(script_data.get("date", "")))}</div>

        <hr style="margin:12px 0;">

        <div><strong>Medicine:</strong> {html.escape(str(script_data.get("medicine", "")))}</div>
        <div><strong>Strength:</strong> {html.escape(str(script_data.get("strength", "")))}</div>
        <div><strong>Directions:</strong> {html.escape(str(script_data.get("directions", "")))}</div>
        <div><strong>Quantity:</strong> {html.escape(str(script_data.get("quantity", "")))}</div>
        <div><strong>Repeats:</strong> {html.escape(str(script_data.get("repeats", "")))}</div>
        <div><strong>Special notes:</strong> {html.escape(str(script_data.get("notes", "")))}</div>
    </div>
    """)

    st.markdown("### 📄 Mock prescription")
    st.markdown(html_block, unsafe_allow_html=True)


# =========================
# Load data and client
# =========================
cases, scripts, behaviours, overlays, answer_keys, sim_rules = safe_load_data()
script_map = {s["mock_script_id"]: s for s in scripts}
client = get_gemini_client()

# =========================
# Session state
# =========================
if "selected_case_id" not in st.session_state:
    st.session_state.selected_case_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "feedback_text" not in st.session_state:
    st.session_state.feedback_text = None

if "roleplay_started" not in st.session_state:
    st.session_state.roleplay_started = False

if "access_granted" not in st.session_state:
    st.session_state.access_granted = False

if "start_mode" not in st.session_state:
    st.session_state.start_mode = None

if "random_case_id" not in st.session_state:
    st.session_state.random_case_id = None

if "show_mock_script" not in st.session_state:
    st.session_state.show_mock_script = False

# =========================
# Access gate
# =========================
ACCESS_CODE = st.secrets.get("APP_ACCESS_CODE", "")

if ACCESS_CODE:
    if not st.session_state.access_granted:
        st.markdown("<div class='login-box'>", unsafe_allow_html=True)
        st.markdown("## Access required")
        entered_code = st.text_input("Enter access code", type="password")
        if st.button("Unlock app", use_container_width=True):
            if entered_code == ACCESS_CODE:
                st.session_state.access_granted = True
                st.rerun()
            else:
                st.error("Incorrect access code.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

# =========================
# Header
# =========================
st.markdown("<div class='main-title'>NZ Community Pharmacy OSCE Simulator</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Patient roleplay for counselling, dispensing, legal, funding, and controlled drug cases.</div>", unsafe_allow_html=True)

if st.button("Clear app cache"):
    st.rerun()

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.markdown("## 💊 Settings")

    if client is None:
        st.error("No Gemini key detected")
    else:
        st.success("Gemini key detected")

    model = st.text_input("Model", value=DEFAULT_MODEL)

    difficulty_filter = st.selectbox(
        "Difficulty",
        ["all", "easy", "medium", "hard"],
        index=0
    )

    admin_mode = st.toggle("Admin mode", value=False)

# =========================
# Main-screen case selection
# =========================
st.markdown("## Start your OSCE practice")
st.markdown("Choose whether to pick a case yourself or get a random one.")

filtered_cases = cases
if difficulty_filter != "all":
    filtered_cases = [c for c in cases if normalize_text(c.get("difficulty", "")) == difficulty_filter]

choice_col1, choice_col2 = st.columns(2)

with choice_col1:
    st.markdown("""
    <div class="choice-card">
        <div class="choice-card-title">🎯 Pick a case myself</div>
        <div class="choice-card-sub">Choose a specific case from the list</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Pick a case", use_container_width=True):
        st.session_state.start_mode = "manual"
        st.session_state.random_case_id = None
        st.session_state.roleplay_started = False
        st.session_state.messages = []
        st.session_state.feedback_text = None
        st.session_state.show_mock_script = False

with choice_col2:
    st.markdown("""
    <div class="choice-card">
        <div class="choice-card-title">🎲 Give me a random case</div>
        <div class="choice-card-sub">Get a surprise case based on the chosen difficulty</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Random case", use_container_width=True):
        st.session_state.start_mode = "random"
        st.session_state.roleplay_started = False
        st.session_state.messages = []
        st.session_state.feedback_text = None
        st.session_state.show_mock_script = False
        if filtered_cases:
            chosen = random.choice(filtered_cases)
            st.session_state.random_case_id = chosen["case_id"]
            st.session_state.selected_case_id = chosen["case_id"]

selected_case = None

if st.session_state.start_mode == "manual":
    if filtered_cases:
        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
        labels = [f"{c['case_id']} — {c['title']}" for c in filtered_cases]
        default_index = 0

        if st.session_state.selected_case_id:
            selected_indices = [i for i, c in enumerate(filtered_cases) if c["case_id"] == st.session_state.selected_case_id]
            if selected_indices:
                default_index = selected_indices[0]

        picked = st.selectbox("Select a case", labels, index=default_index)
        selected_case = filtered_cases[labels.index(picked)]
        if st.session_state.selected_case_id != selected_case["case_id"]:
            st.session_state.show_mock_script = False
            st.session_state.messages = []
            st.session_state.feedback_text = None
            st.session_state.roleplay_started = False
        st.session_state.selected_case_id = selected_case["case_id"]

elif st.session_state.start_mode == "random":
    if st.session_state.random_case_id:
        selected_case = next((c for c in cases if c["case_id"] == st.session_state.random_case_id), None)

        subcol1, subcol2 = st.columns([1, 3])
        with subcol1:
            if st.button("🔄 Randomise again", use_container_width=True):
                if filtered_cases:
                    chosen = random.choice(filtered_cases)
                    st.session_state.random_case_id = chosen["case_id"]
                    st.session_state.selected_case_id = chosen["case_id"]
                    st.session_state.roleplay_started = False
                    st.session_state.messages = []
                    st.session_state.feedback_text = None
                    st.session_state.show_mock_script = False
                    st.rerun()

if not selected_case and st.session_state.selected_case_id:
    selected_case = next((c for c in cases if c["case_id"] == st.session_state.selected_case_id), None)

if not selected_case:
    st.info("Choose how you want to start.")
    st.stop()

overlay = find_overlay(selected_case["case_id"], overlays)
answer_key = find_answer_key(selected_case["case_id"], answer_keys)
behaviour = resolve_behaviour(selected_case.get("patient_persona", ""), behaviours)
sim_rule_set = applicable_sim_rules(selected_case, overlay, sim_rules)

selected_script = None
if selected_case.get("mock_script_id"):
    selected_script = script_map.get(selected_case["mock_script_id"])

if not selected_script and selected_case.get("mock_script_enabled"):
    selected_script = {
        "data": {
            "script_type": "standard",
            "prescriber_name": "Training prescriber",
            "prescriber_address": "Training clinic",
            "prescriber_phone": "",
            "patient_name": "Training patient",
            "patient_address": "",
            "date": "",
            "medicine": selected_case.get("title", ""),
            "strength": "",
            "directions": "See case details",
            "quantity": "",
            "repeats": "",
            "notes": f"No linked mock script found for {selected_case.get('mock_script_id', '')}."
        }
    }

# =========================
# Case banner
# =========================
st.markdown(
    f"""
    <div class="case-banner">
        <div><strong>{selected_case['case_id']} — {selected_case['title']}</strong></div>
        <div class="case-meta">
            Difficulty: {selected_case.get('difficulty','')} &nbsp; • &nbsp;
            Type: {selected_case.get('case_type','')} &nbsp; • &nbsp;
            Persona: {selected_case.get('patient_persona','')}
        </div>
        <div class="case-meta" style="margin-top:8px;">
            Opening line: {html.escape(selected_case.get('patient_opening',''))}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# Action buttons
# =========================
b1, b2, b3, b4 = st.columns([1, 1, 1, 1])

with b1:
    if st.button("▶ Start roleplay", use_container_width=True):
        st.session_state.selected_case_id = selected_case["case_id"]
        st.session_state.messages = [
            {"role": "assistant", "content": selected_case["patient_opening"]}
        ]
        st.session_state.feedback_text = None
        st.session_state.roleplay_started = True
        st.rerun()

with b2:
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.messages = []
        st.session_state.feedback_text = None
        st.session_state.roleplay_started = False
        st.session_state.show_mock_script = False
        st.rerun()

with b3:
    if st.button("📄 Mock prescription", use_container_width=True):
        if selected_script:
            st.session_state.show_mock_script = not st.session_state.show_mock_script
        else:
            st.warning(
                f"No mock script found for case {selected_case.get('case_id')} "
                f"with mock_script_id={selected_case.get('mock_script_id')}"
            )

with b4:
    transcript = conversation_transcript(st.session_state.messages)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        "💾 Download transcript",
        data=transcript.encode("utf-8"),
        file_name=f"osce_transcript_{timestamp}.txt",
        mime="text/plain",
        use_container_width=True
    )

if st.session_state.show_mock_script and selected_script:
    render_mock_script(selected_script["data"])

# =========================
# Admin panel
# =========================
if admin_mode:
    with st.container():
        st.markdown("<div class='admin-box'>", unsafe_allow_html=True)
        st.markdown("**Admin / assessor panel**")
        st.write("**Hidden facts:**", selected_case.get("hidden_facts", []))
        st.write("**Reveal rules:**", selected_case.get("reveal_rules", []))
        st.write("**Pharmacy issue:**", selected_case.get("pharmacy_issue", ""))
        st.write("**Answer key:**", answer_key)
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Chat
# =========================
if not st.session_state.roleplay_started:
    st.markdown("<div class='small-note'>Click <strong>Start roleplay</strong> to begin the patient conversation.</div>", unsafe_allow_html=True)
else:
    render_chat(st.session_state.messages)

    prompt = st.chat_input("Type your response as the pharmacist...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        if client is None:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "The app is missing a Gemini API key."
            })
            st.rerun()

        system_prompt = build_system_prompt(
            case=selected_case,
            overlay=overlay,
            behaviour=behaviour,
            sim_rules=sim_rule_set,
            script_data=selected_script["data"] if selected_script and "data" in selected_script else None
        )

        with st.spinner("Patient is responding..."):
            try:
                reply = call_model(
                    client=client,
                    model=model,
                    system_prompt=system_prompt,
                    messages=st.session_state.messages
                )
            except Exception as e:
                reply = f"Error: {e}"

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

# =========================
# Feedback
# =========================
st.markdown("---")
st.subheader("Assessment")

if st.button("Generate assessor feedback"):
    if not st.session_state.messages:
        st.warning("No conversation to assess yet.")
    elif client is None:
        st.error("No Gemini key detected.")
    else:
        feedback_prompt = build_feedback_prompt(
            case=selected_case,
            answer_key=answer_key,
            transcript=st.session_state.messages
        )
        full_feedback_prompt = f"""
You are a strict but fair New Zealand community pharmacy OSCE assessor.

{feedback_prompt}
"""
        with st.spinner("Generating feedback..."):
            try:
                st.session_state.feedback_text = generate_feedback(
                    client=client,
                    model=model,
                    feedback_prompt=full_feedback_prompt
                )
            except Exception as e:
                st.session_state.feedback_text = f"Feedback generation failed: {e}"

if st.session_state.feedback_text:
    st.markdown(f"<div class='feedback-box'>{escape_html(st.session_state.feedback_text)}</div>", unsafe_allow_html=True)