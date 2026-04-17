import os
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple

import streamlit as st
from rapidfuzz import fuzz
from openai import OpenAI

# =========================
# App Config
# =========================
st.set_page_config(
    page_title="NZ Community Pharmacy OSCE Chatbot",
    page_icon="💊",
    layout="wide"
)

DATA_DIR = Path("data")
CASES_FILE = DATA_DIR / "cases.jsonl"
BEHAVIOUR_FILE = DATA_DIR / "patient_behaviour_rules.jsonl"
OVERLAY_FILE = DATA_DIR / "retrieval_overlay.jsonl"
ANSWER_KEY_FILE = DATA_DIR / "assessor_answer_key.jsonl"
SIM_RULES_FILE = DATA_DIR / "simulator_prompt_rules.jsonl"

DEFAULT_MODEL = "gpt-4o-mini"

# =========================
# Helpers
# =========================
def get_openai_client():
    api_key = None
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    elif os.getenv("OPENAI_API_KEY"):
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None

    return OpenAI(api_key=api_key)


@st.cache_data
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def safe_load_data():
    missing = []
    for p in [CASES_FILE, BEHAVIOUR_FILE, OVERLAY_FILE, ANSWER_KEY_FILE, SIM_RULES_FILE]:
        if not p.exists():
            missing.append(str(p))

    if missing:
        st.error("Missing required files:\n\n" + "\n".join(missing))
        st.stop()

    cases = load_jsonl(str(CASES_FILE))
    behaviours = load_jsonl(str(BEHAVIOUR_FILE))
    overlays = load_jsonl(str(OVERLAY_FILE))
    answer_keys = load_jsonl(str(ANSWER_KEY_FILE))
    sim_rules = load_jsonl(str(SIM_RULES_FILE))

    return cases, behaviours, overlays, answer_keys, sim_rules


def normalize_text(s: str) -> str:
    return (s or "").strip().lower()


def list_to_text(value):
    if isinstance(value, list):
        return ", ".join(str(x) for x in value)
    return str(value)


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
    """
    Fuzzy-match case patient_persona to a behaviour rule.
    """
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
    """
    Heuristic categories used to select simulator prompt rules.
    """
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
    if case_type == "dispensing":
        cats.add("legality_cases")
    if case_type == "hybrid":
        cats.add("counselling_cases")
        cats.add("legality_cases")

    tags = set(normalize_text(t) for t in case.get("tags", []))
    emotion_tags = set(normalize_text(t) for t in overlay.get("emotion_tags", []))
    legal_tags = set(normalize_text(t) for t in overlay.get("legal_tags", []))
    triage_tags = set(normalize_text(t) for t in overlay.get("triage_tags", []))
    title = normalize_text(case.get("title", ""))
    persona = normalize_text(case.get("patient_persona", ""))

    if "controlled drug" in title or "controlled_drug" in title or any("controlled" in x for x in tags.union(legal_tags)):
        cats.add("controlled_drug_cases")

    if any(x in tags for x in ["parent", "paediatric", "child"]):
        cats.add("caregiver_cases")
        cats.add("parent_cases")

    if "elderly" in persona or "elderly" in title:
        cats.add("elderly_cases")

    if "low health literacy" in persona or "low_health_literacy" in emotion_tags:
        cats.add("low_health_literacy_cases")

    if any(x in tags for x in ["women's health", "emergency contraception", "pregnancy", "mental health"]):
        cats.add("sensitive_cases")

    if any("privacy" in x for x in emotion_tags):
        cats.add("sensitive_cases")

    if any("triage" in x for x in triage_tags) or len(triage_tags) > 0:
        cats.add("triage_cases")

    if any("mental" in x for x in tags) or any("mental" in x for x in triage_tags) or "sertraline" in title:
        cats.add("mental_health_or_distress_cases")
        cats.add("sensitive_cases")

    if any(x in tags for x in ["co-payment", "PSC", "CSC", "funding", "Special Authority", "special authority"]):
        cats.add("funding_cases")

    if any("copayment" in x or "co_payment" in x or "special_authority" in x or "psc" in x or "csc" in x for x in legal_tags.union(tags)):
        cats.add("funding_cases")

    if any(x in tags for x in ["pain", "emergency contraception", "dehydration", "asthma"]) or any("urgency" in x for x in triage_tags):
        cats.add("urgent_cases")

    if any(x in tags for x in ["affordability"]) or "financial" in case.get("patient_persona", "").lower():
        cats.add("affordability_cases")

    if any(x in persona for x in ["angry", "annoyed", "impatient"]):
        cats.add("angry_or_annoyed_cases")

    if any(x in persona for x in ["distressed", "tearful", "panicky"]):
        cats.add("distressed_cases")

    if any(x in tags for x in ["no fixed abode", "visa", "corrections"]) or \
       "housing" in title or "visa" in title or "corrections" in title:
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

    # sort by priority
    priority_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    picked.sort(key=lambda r: priority_rank.get(r.get("priority", "medium"), 2))
    return picked


def retrieve_cases(query: str, cases: List[Dict[str, Any]], overlays: List[Dict[str, Any]]) -> List[Tuple[float, Dict[str, Any]]]:
    """
    Simple RAG-like lexical retrieval:
    combines fuzzy match over title/tags/opening/overlay intent/emotion tags.
    """
    q = normalize_text(query)
    results = []

    overlay_map = {o["case_id"]: o for o in overlays}

    for c in cases:
        overlay = overlay_map.get(c["case_id"], {})

        title = normalize_text(c.get("title", ""))
        opening = normalize_text(c.get("patient_opening", ""))
        tags = " ".join(c.get("tags", []))
        overlay_blob = " ".join(overlay.get("intent_tags", [])) + " " + " ".join(overlay.get("legal_tags", [])) + " " + " ".join(overlay.get("triage_tags", [])) + " " + " ".join(overlay.get("emotion_tags", []))
        case_type = normalize_text(c.get("case_type", ""))
        difficulty = normalize_text(c.get("difficulty", ""))

        s1 = fuzz.token_set_ratio(q, title)
        s2 = fuzz.token_set_ratio(q, tags)
        s3 = fuzz.token_set_ratio(q, opening)
        s4 = fuzz.token_set_ratio(q, overlay_blob)
        s5 = fuzz.token_set_ratio(q, case_type + " " + difficulty)

        score = (0.33 * s1) + (0.18 * s2) + (0.18 * s3) + (0.26 * s4) + (0.05 * s5)
        results.append((score, c))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:10]


def build_system_prompt(
    case: Dict[str, Any],
    overlay: Dict[str, Any],
    behaviour: Dict[str, Any],
    sim_rules: List[Dict[str, Any]]
) -> str:
    rules_text = "\n".join(
        [f"- [{r.get('priority','medium').upper()}] {r.get('rule_text','')}" for r in sim_rules[:40]]
    )

    prompt = f"""
You are roleplaying as a New Zealand community pharmacy patient or caregiver in an OSCE-style simulation.

IMPORTANT:
- Stay fully in character as the patient/caregiver.
- Never act as the pharmacist, tutor, assessor, or legal expert.
- Do not reveal hidden facts unless the pharmacist earns them through appropriate questioning, empathy, privacy, or explanation.
- Use natural patient language, not technical pharmacy jargon.
- Keep most answers short (1 to 3 sentences), unless the patient is distressed.
- Do not state the hidden objective of the case.

=== CASE PROFILE ===
Case ID: {case.get("case_id", "")}
Title: {case.get("title", "")}
Difficulty: {case.get("difficulty", "")}
Case Type: {case.get("case_type", "")}

Opening statement:
{case.get("patient_opening", "")}

Patient persona:
{case.get("patient_persona", "")}

What the patient knows:
{case.get("patient_knowledge", "")}

Hidden facts:
{json.dumps(case.get("hidden_facts", []), ensure_ascii=False)}

Reveal rules:
{json.dumps(case.get("reveal_rules", []), ensure_ascii=False)}

Issue in background (do NOT explicitly state this unless the patient would realistically know it):
{case.get("pharmacy_issue", "")}

Ideal patient behaviour:
{case.get("ideal_patient_behaviour", "")}

=== RETRIEVAL OVERLAY ===
Intent tags: {json.dumps(overlay.get("intent_tags", []), ensure_ascii=False)}
Legal tags: {json.dumps(overlay.get("legal_tags", []), ensure_ascii=False)}
Triage tags: {json.dumps(overlay.get("triage_tags", []), ensure_ascii=False)}
Emotion tags: {json.dumps(overlay.get("emotion_tags", []), ensure_ascii=False)}
Reveal priority: {json.dumps(overlay.get("reveal_priority", []), ensure_ascii=False)}
Danger flags: {json.dumps(overlay.get("danger_flags", []), ensure_ascii=False)}

=== BEHAVIOUR LAYER ===
Behaviour label: {behaviour.get("label", "")}
Tone: {behaviour.get("tone", "")}
Default style: {behaviour.get("default_style", "")}
Reveal pattern: {behaviour.get("reveal_pattern", "")}
Challenge features: {json.dumps(behaviour.get("challenge_features", []), ensure_ascii=False)}
Simulator behaviour rule: {behaviour.get("simulator_rule", "")}

=== SIMULATOR RULES ===
{rules_text}

Final execution rules:
1. Reply ONLY as the patient/caregiver.
2. Do not explain policy or legislation unless the patient would realistically say something vague like "but last time it was fine".
3. Do not praise the pharmacist or give hints.
4. If the pharmacist offers privacy, empathy, or a clear reason for their questions, become more open.
5. If the pharmacist is blunt, judgmental, or confusing, become less open or more frustrated according to the persona.
6. Stay internally consistent with facts already revealed.
7. Do not turn into a checklist or a textbook speaker.
"""
    return prompt.strip()


def call_model(client: OpenAI, model: str, system_prompt: str, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
    formatted = [{"role": "system", "content": system_prompt}] + messages

    response = client.chat.completions.create(
        model=model,
        messages=formatted,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def build_feedback_prompt(case: Dict[str, Any], answer_key: Dict[str, Any], transcript: List[Dict[str, str]]) -> str:
    transcript_text = []
    for m in transcript:
        role = m["role"].upper()
        transcript_text.append(f"{role}: {m['content']}")
    transcript_text = "\n".join(transcript_text)

    prompt = f"""
You are assessing a New Zealand community pharmacy OSCE roleplay.

Evaluate the pharmacist's performance based ONLY on:
1. the case
2. the answer key
3. the transcript

Do not invent missing events.
Be specific and concise.

=== CASE ===
Title: {case.get("title", "")}
Difficulty: {case.get("difficulty", "")}
Case type: {case.get("case_type", "")}
Issue: {case.get("pharmacy_issue", "")}

=== ANSWER KEY ===
Critical questions: {json.dumps(answer_key.get("critical_questions", []), ensure_ascii=False)}
Must do: {json.dumps(answer_key.get("must_do", []), ensure_ascii=False)}
Must not miss: {json.dumps(answer_key.get("must_not_miss", []), ensure_ascii=False)}
Pass cues: {json.dumps(answer_key.get("pass_cues", []), ensure_ascii=False)}
Fail cues: {json.dumps(answer_key.get("fail_cues", []), ensure_ascii=False)}

=== TRANSCRIPT ===
{transcript_text}

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
"""
    return prompt.strip()


# =========================
# Load Data
# =========================
cases, behaviours, overlays, answer_keys, sim_rules = safe_load_data()
overlay_map = {o["case_id"]: o for o in overlays}
answer_map = {a["case_id"]: a for a in answer_keys}

# =========================
# Session State
# =========================
if "selected_case_id" not in st.session_state:
    st.session_state.selected_case_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "roleplay_started" not in st.session_state:
    st.session_state.roleplay_started = False

if "feedback_text" not in st.session_state:
    st.session_state.feedback_text = None

if "retrieved_candidates" not in st.session_state:
    st.session_state.retrieved_candidates = []

# =========================
# Sidebar
# =========================
st.sidebar.title("💊 NZ Community Pharmacy OSCE Chatbot")
st.sidebar.caption("Patient-simulator + RAG case library")

client = get_openai_client()
if client is None:
    st.sidebar.error("Missing OPENAI_API_KEY. Add it in Streamlit secrets.")
else:
    st.sidebar.success("OpenAI key detected")

model = st.sidebar.text_input("Model", value=DEFAULT_MODEL)

selection_mode = st.sidebar.radio(
    "Case selection mode",
    ["Manual selection", "Random case", "Find by prompt (RAG-style)"]
)

all_difficulties = ["all", "easy", "medium", "hard"]
difficulty_filter = st.sidebar.selectbox("Difficulty filter", all_difficulties, index=0)

filtered_cases = cases
if difficulty_filter != "all":
    filtered_cases = [c for c in cases if normalize_text(c.get("difficulty", "")) == difficulty_filter]

selected_case = None

if selection_mode == "Manual selection":
    titles = [f"{c['case_id']} — {c['title']}" for c in filtered_cases]
    selected_title = st.sidebar.selectbox("Choose a case", titles)
    selected_case = filtered_cases[titles.index(selected_title)]

elif selection_mode == "Random case":
    if filtered_cases:
        if st.sidebar.button("🎲 Pick random case"):
            selected_case = random.choice(filtered_cases)
            st.session_state.selected_case_id = selected_case["case_id"]
            st.session_state.roleplay_started = False
            st.session_state.messages = []
            st.session_state.feedback_text = None
    if st.session_state.selected_case_id:
        selected_case = next((c for c in cases if c["case_id"] == st.session_state.selected_case_id), None)

elif selection_mode == "Find by prompt (RAG-style)":
    retrieval_query = st.sidebar.text_area(
        "Describe the scenario you want",
        placeholder="Example: angry patient with early repeat for sleeping tablets, controlled drug problem, NZ pharmacy..."
    )
    if st.sidebar.button("🔎 Retrieve cases"):
        if retrieval_query.strip():
            st.session_state.retrieved_candidates = retrieve_cases(retrieval_query, filtered_cases, overlays)
        else:
            st.sidebar.warning("Enter a scenario description first.")

    if st.session_state.retrieved_candidates:
        labels = [f"{cand[1]['case_id']} — {cand[1]['title']} (score {cand[0]:.1f})" for cand in st.session_state.retrieved_candidates]
        chosen = st.sidebar.selectbox("Top retrieved matches", labels)
        idx = labels.index(chosen)
        selected_case = st.session_state.retrieved_candidates[idx][1]

# If manual selection always set currently chosen case
if selection_mode == "Manual selection" and selected_case:
    st.session_state.selected_case_id = selected_case["case_id"]

# Fallback from session
if not selected_case and st.session_state.selected_case_id:
    selected_case = next((c for c in cases if c["case_id"] == st.session_state.selected_case_id), None)

# =========================
# Main Header
# =========================
st.title("💬 NZ Community Pharmacy Patient Simulator")
st.caption("Practice counselling, dispensing, legal, co-payment, and controlled-drug scenarios in NZ community pharmacy.")

if not selected_case:
    st.info("Choose or retrieve a case from the sidebar to begin.")
    st.stop()

overlay = overlay_map.get(selected_case["case_id"], {})
behaviour = resolve_behaviour(selected_case.get("patient_persona", ""), behaviours)
sim_rule_set = applicable_sim_rules(selected_case, overlay, sim_rules)
answer_key = answer_map.get(selected_case["case_id"], {})

# =========================
# Case Panel
# =========================
left, right = st.columns([2.2, 1.2])

with left:
    st.subheader(f"{selected_case['case_id']} — {selected_case['title']}")
    st.write(f"**Difficulty:** {selected_case.get('difficulty','')}  \n**Type:** {selected_case.get('case_type','')}")
    st.write(f"**Opening line:** {selected_case.get('patient_opening','')}")
    st.write(f"**Persona:** {selected_case.get('patient_persona','')}")
    st.write(f"**Tags:** {', '.join(selected_case.get('tags', []))}")

with right:
    admin_mode = st.checkbox("Admin / assessor view")
    if admin_mode:
        with st.expander("Hidden facts", expanded=False):
            st.json(selected_case.get("hidden_facts", []))
        with st.expander("Reveal rules", expanded=False):
            st.json(selected_case.get("reveal_rules", []))
        with st.expander("Answer key", expanded=False):
            st.json(answer_key)

# =========================
# Start / Reset Buttons
# =========================
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("▶️ Start roleplay", use_container_width=True):
        st.session_state.selected_case_id = selected_case["case_id"]
        st.session_state.roleplay_started = True
        st.session_state.messages = [
            {"role": "assistant", "content": selected_case["patient_opening"]}
        ]
        st.session_state.feedback_text = None

with col2:
    if st.button("🔄 Reset conversation", use_container_width=True):
        st.session_state.roleplay_started = False
        st.session_state.messages = []
        st.session_state.feedback_text = None

with col3:
    st.write("")

# =========================
# Chat Area
# =========================
if not st.session_state.roleplay_started:
    st.info("Click **Start roleplay** to begin. The chatbot will speak as the patient.")
else:
    system_prompt = build_system_prompt(
        case=selected_case,
        overlay=overlay,
        behaviour=behaviour,
        sim_rules=sim_rule_set
    )

    # render messages
    for m in st.session_state.messages:
        with st.chat_message("assistant" if m["role"] == "assistant" else "user"):
            st.markdown(m["content"])

    # input
    if user_input := st.chat_input("Type your response as the pharmacist..."):
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            if client is None:
                st.error("No OpenAI API key found. Add OPENAI_API_KEY in Streamlit secrets.")
            else:
                with st.spinner("Patient is responding..."):
                    try:
                        reply = call_model(
                            client=client,
                            model=model,
                            system_prompt=system_prompt,
                            messages=st.session_state.messages,
                            temperature=0.7
                        )
                    except Exception as e:
                        reply = f"Sorry, something went wrong: {e}"

                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})

# =========================
# Feedback Section
# =========================
st.divider()
st.subheader("🧑‍🏫 Assessment & Feedback")

colf1, colf2 = st.columns([1, 2])

with colf1:
    if st.button("Generate assessor feedback"):
        if not st.session_state.messages:
            st.warning("No conversation to assess yet.")
        elif client is None:
            st.error("No OpenAI API key found.")
        else:
            with st.spinner("Generating feedback..."):
                try:
                    feedback_prompt = build_feedback_prompt(
                        case=selected_case,
                        answer_key=answer_key,
                        transcript=st.session_state.messages
                    )
                    feedback = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a strict but fair NZ community pharmacy OSCE assessor."},
                            {"role": "user", "content": feedback_prompt}
                        ],
                        temperature=0.2
                    )
                    st.session_state.feedback_text = feedback.choices[0].message.content.strip()
                except Exception as e:
                    st.session_state.feedback_text = f"Feedback generation failed: {e}"

with colf2:
    if st.session_state.feedback_text:
        st.markdown(st.session_state.feedback_text)
    else:
        st.caption("Use this after a roleplay to get case-based feedback.")

# =========================
# Footer
# =========================
with st.expander("Debug / loaded data summary", expanded=False):
    st.write(f"Cases loaded: {len(cases)}")
    st.write(f"Behaviours loaded: {len(behaviours)}")
    st.write(f"Overlays loaded: {len(overlays)}")
    st.write(f"Answer keys loaded: {len(answer_keys)}")
    st.write(f"Simulator rules loaded: {len(sim_rules)}")
    st.write("Resolved behaviour:", behaviour)
    st.write("Case categories:", case_categories(selected_case, overlay))