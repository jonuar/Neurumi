"""
Neurumi — Artificial life with real ML.

Run:
    streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import torch
from pathlib import Path

from brain import NeurumiBrain, save_brain, load_brain
from state import NeurumiState, ACTION_EFFECTS, EMOTION_META
from trainer import NeurumiTrainer


# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Neurumi",
    page_icon="◉",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Global styles ────────────────────────────────────────────────────────────
# Injected once at the top. Scoped to Streamlit's markdown renderer.
# Button styles, typography, and layout tokens live here.

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #F7F5F0;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 680px; }

h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
    font-weight: 400;
    letter-spacing: -0.02em;
}

.creature-card {
    background: #FDFCFA;
    border: 1px solid #E8E4DC;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
}

.creature-orb {
    width: 120px;
    height: 120px;
    border-radius: 50%;
    margin: 0 auto 1rem;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 3rem;
    transition: all 0.6s ease;
}

.creature-name {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: #2C2C2A;
    margin: 0;
}

.creature-emotion {
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #888780;
    margin-top: 0.25rem;
}

.stButton > button {
    background: #FDFCFA !important;
    border: 1px solid #E8E4DC !important;
    border-radius: 12px !important;
    color: #2C2C2A !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1rem !important;
    width: 100% !important;
    transition: all 0.15s ease !important;
}

.stButton > button:hover {
    background: #F0EDE6 !important;
    border-color: #D3D1C7 !important;
    transform: translateY(-1px) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

.memory-item {
    font-size: 0.72rem;
    color: #5F5E5A;
    padding: 0.5rem 0.75rem;
    background: #F4F1EB;
    border-radius: 8px;
    margin-bottom: 0.35rem;
    line-height: 1.5;
    border-left: 2px solid #D3D1C7;
}

.memory-item.recent {
    border-left-color: #1D9E75;
    background: #EAF3DE;
    color: #3B6D11;
}

.metric-row {
    display: flex;
    justify-content: space-between;
    gap: 0.5rem;
    margin-bottom: 1rem;
}

.metric-box {
    flex: 1;
    background: #FDFCFA;
    border: 1px solid #E8E4DC;
    border-radius: 12px;
    padding: 0.75rem;
    text-align: center;
}

.metric-value {
    font-size: 1.1rem;
    font-weight: 400;
    color: #2C2C2A;
    font-family: 'DM Serif Display', serif;
}

.metric-label {
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #888780;
    margin-top: 0.15rem;
}

.emotion-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 400;
}

.section-header {
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #B4B2A9;
    margin-bottom: 0.6rem;
    margin-top: 1.25rem;
}

.scrollable-log {
    max-height: 140px;
    overflow-y: auto;
}

hr {
    border: none;
    border-top: 1px solid #E8E4DC;
    margin: 1.25rem 0;
}
</style>
""", unsafe_allow_html=True)


# ─── Session state ────────────────────────────────────────────────────────────
# st.session_state persists values across Streamlit reruns (each user interaction
# triggers a full script rerun). Without this, everything would reset on every click.

if "neurumi" not in st.session_state:
    if Path("neurumi_state.json").exists():
        st.session_state.neurumi = NeurumiState.load("neurumi_state.json")
    else:
        st.session_state.neurumi = NeurumiState(name="Neurumi")

if "brain" not in st.session_state:
    if Path("neurumi_brain.pt").exists():
        st.session_state.brain = load_brain("neurumi_brain.pt")
    else:
        st.session_state.brain = NeurumiBrain()

if "trainer" not in st.session_state:
    st.session_state.trainer = NeurumiTrainer(st.session_state.brain)

if "memory_log" not in st.session_state:
    loaded = Path("neurumi_state.json").exists()
    age = st.session_state.neurumi.age
    first_entry = f"Neurumi remembers. Epoch {age}." if loaded else "Neurumi wakes up for the first time."
    st.session_state.memory_log = [first_entry]

if "tick" not in st.session_state:
    st.session_state.tick = st.session_state.neurumi.age

# Local aliases for readability
neurumi: NeurumiState = st.session_state.neurumi
brain: NeurumiBrain = st.session_state.brain
trainer: NeurumiTrainer = st.session_state.trainer


# ─── Helpers ──────────────────────────────────────────────────────────────────

def add_memory(text: str, recent: bool = False):
    """Prepends a timestamped entry to the episodic memory log."""
    entry = {"text": f"[{neurumi.age:04d}] {text}", "recent": recent}
    st.session_state.memory_log.insert(0, entry)
    if len(st.session_state.memory_log) > 12:
        st.session_state.memory_log.pop()


def do_action(action: str):
    """
    Handles a player action:
    1. Trains the network against the expected effect of this action
    2. Applies the real effect to Neurumi's state
    3. Advances one tick (time passes)
    4. Persists state and model weights to disk
    """
    avg_loss = trainer.train_on_action(action, neurumi, steps=8)
    neurumi.apply_action_effect(ACTION_EFFECTS[action])
    neurumi.tick()
    st.session_state.tick = neurumi.age

    labels = {
        "feed":   "You fed her.",
        "play":   "You played with her.",
        "pet":    "You petted her.",
        "ignore": "You ignored her. Time passed.",
    }
    add_memory(f"{labels[action]} loss={avg_loss:.4f}", recent=(action != "ignore"))

    neurumi.save("neurumi_state.json")
    save_brain(brain, "neurumi_brain.pt")


def do_tick_only():
    """Advances time without player input. The network predicts the state delta."""
    deltas = trainer.infer(neurumi)
    neurumi.apply_deltas(deltas, scale=0.05)
    neurumi.tick()
    st.session_state.tick = neurumi.age
    add_memory("Free time. Network predicts and adjusts.")
    neurumi.save("neurumi_state.json")
    save_brain(brain, "neurumi_brain.pt")


def get_orb_style(emotion: str) -> str:
    """Returns inline CSS for the creature orb based on current emotion."""
    colors = {
        "happy":   ("background:#EAF3DE;", "border:2px solid #5DCAA5;"),
        "calm":    ("background:#E6F1FB;", "border:2px solid #378ADD;"),
        "curious": ("background:#EEEDFE;", "border:2px solid #7F77DD;"),
        "hungry":  ("background:#FAECE7;", "border:2px solid #D85A30;"),
        "scared":  ("background:#FAEEDA;", "border:2px solid #BA7517;"),
        "sleepy":  ("background:#F1EFE8;", "border:2px solid #888780;"),
        "lonely":  ("background:#FBEAF0;", "border:2px solid #D4537E;"),
    }
    bg, border = colors.get(emotion, colors["calm"])
    return f"{bg} {border}"


def build_drives_html(drives: list) -> str:
    """
    Builds a self-contained HTML string for the drive bars.

    Rendered via components.v1.html() instead of st.markdown() because
    Streamlit's Markdown parser escapes single quotes inside inline styles
    (e.g. font-family:'DM Mono'), which corrupts the output and renders
    raw HTML as plain text. components.v1.html() injects content into an
    iframe that bypasses the Markdown pipeline entirely.
    """
    rows = ""
    for label, value, color in drives:
        pct = round(value * 100)
        rows += f"""
        <div style="margin-bottom:14px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                <span style="font-size:11px;letter-spacing:1.5px;text-transform:uppercase;color:#888780;font-family:monospace;">{label}</span>
                <span style="font-size:11px;color:#B4B2A9;font-family:monospace;">{pct}</span>
            </div>
            <div style="background:#E8E4DC;border-radius:3px;height:5px;overflow:hidden;">
                <div style="width:{pct}%;height:100%;background:{color};border-radius:3px;"></div>
            </div>
        </div>
        """
    return f"""
    <div style="background:#FDFCFA;border:1px solid #E8E4DC;border-radius:16px;padding:20px 24px;font-family:monospace;">
        {rows}
    </div>
    """


# ─── UI ───────────────────────────────────────────────────────────────────────

emotion  = neurumi.get_emotion()
meta     = EMOTION_META[emotion]
wellness = neurumi.get_wellness()

# Minimal eyebrow label
st.markdown(
    '<p style="font-size:0.7rem;letter-spacing:0.18em;text-transform:uppercase;'
    'color:#B4B2A9;margin-bottom:0.25rem;">artificial life · ml</p>',
    unsafe_allow_html=True,
)

# ── Creature card ─────────────────────────────────────────────────────────────

orb_style = get_orb_style(emotion)

st.markdown(f"""
<div class="creature-card">
    <div class="creature-orb" style="{orb_style}">
        <span style="font-size:2.8rem;line-height:1;">{meta['emoji']}</span>
    </div>
    <p class="creature-name">{neurumi.name}</p>
    <p class="creature-emotion">
        <span class="emotion-badge" style="background:{meta['color']}22;color:{meta['color']};">
            {meta['label']}
        </span>
        &nbsp;·&nbsp; epoch {neurumi.age}
    </p>
</div>
""", unsafe_allow_html=True)

# ── Quick metrics ─────────────────────────────────────────────────────────────

last_loss    = trainer.get_last_loss()
interactions = len(trainer.loss_history)

st.markdown(f"""
<div class="metric-row">
    <div class="metric-box">
        <div class="metric-value">{round(wellness * 100)}</div>
        <div class="metric-label">Wellness</div>
    </div>
    <div class="metric-box">
        <div class="metric-value">{neurumi.age}</div>
        <div class="metric-label">Epochs</div>
    </div>
    <div class="metric-box">
        <div class="metric-value">{interactions}</div>
        <div class="metric-label">Trainings</div>
    </div>
    <div class="metric-box">
        <div class="metric-value">{last_loss:.4f}</div>
        <div class="metric-label">Last loss</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Internal state (drives) ───────────────────────────────────────────────────
# components.v1.html() renders into an iframe — no Markdown parsing, no escaping.
# height is set slightly larger than content to avoid iframe scrollbar.

st.markdown('<p class="section-header">Internal state</p>', unsafe_allow_html=True)

drive_config = [
    ("Hunger",    neurumi.hunger,    "#E24B4A"),
    ("Curiosity", neurumi.curiosity, "#7F77DD"),
    ("Affection", neurumi.affection, "#D4537E"),
    ("Energy",    neurumi.energy,    "#1D9E75"),
    ("Fear",      neurumi.fear,      "#BA7517"),
]

components.html(build_drives_html(drive_config), height=210)

# ── Actions ───────────────────────────────────────────────────────────────────

st.markdown('<p class="section-header">Interact</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🍎\nFeed", key="btn_feed"):
        do_action("feed")
        st.rerun()

with col2:
    if st.button("🎮\nPlay", key="btn_play"):
        do_action("play")
        st.rerun()

with col3:
    if st.button("✦\nPet", key="btn_pet"):
        do_action("pet")
        st.rerun()

with col4:
    if st.button("◌\nIgnore", key="btn_ignore"):
        do_action("ignore")
        st.rerun()

col_tick, col_reset = st.columns([3, 1])

with col_tick:
    if st.button("◎  Pass time  →", key="btn_tick"):
        do_tick_only()
        st.rerun()

with col_reset:
    if st.button("↺ Reset", key="btn_reset"):
        # Delete persisted files and wipe session state to start fresh
        Path("neurumi_state.json").unlink(missing_ok=True)
        Path("neurumi_brain.pt").unlink(missing_ok=True)
        for key in ["neurumi", "brain", "trainer", "memory_log", "tick"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# ── Loss chart ────────────────────────────────────────────────────────────────

if len(trainer.loss_history) > 2:
    st.markdown('<p class="section-header">Learning — loss per interaction</p>', unsafe_allow_html=True)
    st.line_chart(trainer.loss_history, height=100, use_container_width=True)
    st.markdown(
        '<p style="font-size:0.65rem;color:#B4B2A9;text-align:center;margin-top:-0.5rem;">'
        'Loss trends toward 0 as the network learns your interaction patterns.'
        '</p>',
        unsafe_allow_html=True,
    )

# ── Episodic memory log ───────────────────────────────────────────────────────

st.markdown('<p class="section-header">Episodic memory</p>', unsafe_allow_html=True)

log_html = '<div class="scrollable-log">'
for entry in st.session_state.memory_log:
    if isinstance(entry, dict):
        css_class = "memory-item recent" if entry.get("recent") else "memory-item"
        text = entry["text"]
    else:
        css_class = "memory-item"
        text = entry
    log_html += f'<div class="{css_class}">{text}</div>'
log_html += "</div>"

st.markdown(log_html, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    '<p style="font-size:0.62rem;color:#D3D1C7;text-align:center;letter-spacing:0.08em;">'
    'neurumi · feed-forward neural network · pytorch · streamlit'
    '</p>',
    unsafe_allow_html=True,
)