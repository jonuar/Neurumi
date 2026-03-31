"""
Neurumi — Artificial life with real ML.
Phase 1: Supervised learning  (AnimaBrain  — imitates player patterns)
Phase 2: Q-Learning / DQN     (QBrain      — discovers optimal actions by reward)

Run:
    streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

from brain     import AnimaBrain, save_brain, load_brain
from state     import NeurumiState, ACTION_EFFECTS, EMOTION_META
from trainer   import NeurumiTrainer
from q_brain   import QBrain, save_q_brain, load_q_brain
from q_trainer import DQNTrainer


# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Neurumi",
    page_icon="◉",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Global styles ────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #F7F5F0;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 680px; }

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

.memory-item.agent {
    border-left-color: #7F77DD;
    background: #EEEDFE;
    color: #3C3489;
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
}

.section-header {
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #B4B2A9;
    margin-bottom: 0.6rem;
    margin-top: 1.25rem;
}

.phase-label {
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 10px;
    display: inline-block;
    margin-bottom: 0.5rem;
}

.scrollable-log { max-height: 160px; overflow-y: auto; }

hr { border: none; border-top: 1px solid #E8E4DC; margin: 1.25rem 0; }
</style>
""", unsafe_allow_html=True)


# ─── Session state ────────────────────────────────────────────────────────────
# Streamlit reruns the full script on every interaction.
# session_state is the only mechanism to persist objects across reruns.

if "neurumi" not in st.session_state:
    if Path("neurumi_state.json").exists():
        st.session_state.neurumi = NeurumiState.load("neurumi_state.json")
    else:
        st.session_state.neurumi = NeurumiState()

# Phase 1 — supervised brain
if "brain" not in st.session_state:
    if Path("neurumi_brain.pt").exists():
        st.session_state.brain = load_brain("neurumi_brain.pt")
    else:
        st.session_state.brain = AnimaBrain()

if "trainer" not in st.session_state:
    st.session_state.trainer = NeurumiTrainer(st.session_state.brain)

# Phase 2 — Q-Learning brain
if "q_brain" not in st.session_state:
    if Path("neurumi_qbrain.pt").exists():
        st.session_state.q_brain = load_q_brain("neurumi_qbrain.pt")
    else:
        st.session_state.q_brain = QBrain()

if "dqn" not in st.session_state:
    st.session_state.dqn = DQNTrainer(st.session_state.q_brain)

if "memory_log" not in st.session_state:
    loaded = Path("neurumi_state.json").exists()
    age    = st.session_state.neurumi.age
    msg    = f"Neurumi remembers. Epoch {age}." if loaded else "Neurumi wakes up for the first time."
    st.session_state.memory_log = [msg]

# Local aliases
neurumi: NeurumiState = st.session_state.neurumi
trainer: NeurumiTrainer = st.session_state.trainer
dqn:     DQNTrainer   = st.session_state.dqn


# ─── Helpers ──────────────────────────────────────────────────────────────────

def add_memory(text: str, kind: str = "normal"):
    """
    kind = 'normal' | 'recent' | 'agent'
    'agent' entries are styled in purple to visually distinguish
    autonomous agent decisions from player interactions.
    """
    entry = {"text": f"[{neurumi.age:04d}] {text}", "kind": kind}
    st.session_state.memory_log.insert(0, entry)
    if len(st.session_state.memory_log) > 16:
        st.session_state.memory_log.pop()


def persist():
    """Saves Neurumi's state and both model weights to disk."""
    neurumi.save("neurumi_state.json")
    save_brain(st.session_state.brain, "neurumi_brain.pt")
    save_q_brain(st.session_state.q_brain, "neurumi_qbrain.pt")


# ── Phase 1 actions ───────────────────────────────────────────────────────────

def do_player_action(action: str):
    """
    Phase 1 player interaction:
    1. Train AnimaBrain to associate (current state → action effect)
    2. Apply the action's real effect to Neurumi's drives
    3. Tick time forward
    """
    avg_loss = trainer.train_on_action(action, neurumi, steps=8)
    neurumi.apply_action_effect(ACTION_EFFECTS[action])
    neurumi.tick()

    labels = {
        "feed":   "You fed her.",
        "play":   "You played with her.",
        "pet":    "You petted her.",
        "ignore": "You ignored her.",
    }
    kind = "normal" if action == "ignore" else "recent"
    add_memory(f"[P1] {labels[action]} loss={avg_loss:.4f}", kind=kind)
    persist()


def do_pass_time():
    """
    Phase 1 autonomous tick: AnimaBrain predicts state drift,
    applies it at low scale (0.05), then time ticks naturally.
    No training occurs here — pure inference.
    """
    deltas = trainer.infer(neurumi)
    neurumi.apply_deltas(deltas, scale=0.05)
    neurumi.tick()
    add_memory("[P1] Free time. Network predicts drift.")
    persist()


# ── Phase 2 actions ───────────────────────────────────────────────────────────

def do_agent_step(n: int = 1):
    """
    Phase 2 DQN autonomous step(s):
    The agent selects actions via epsilon-greedy, collects experience,
    and trains from the replay buffer. No player input involved.
    """
    last_result = None
    for _ in range(n):
        last_result = dqn.step(neurumi)

    if last_result:
        action  = last_result["action"]
        reward  = last_result["reward"]
        epsilon = last_result["epsilon"]
        buf     = last_result["buffer"]
        label   = f"[P2] Agent → {action} · r={reward:.3f} · ε={epsilon:.2f} · buf={buf}"
        kind    = "agent" if reward > 0.4 else "normal"
        add_memory(label, kind=kind)

    persist()


# ── Shared helpers ────────────────────────────────────────────────────────────

def do_reset():
    """Wipes all saved files and session state. Fresh start."""
    for f in ["neurumi_state.json", "neurumi_brain.pt", "neurumi_qbrain.pt"]:
        Path(f).unlink(missing_ok=True)
    for key in ["neurumi", "brain", "trainer", "q_brain", "dqn", "memory_log"]:
        st.session_state.pop(key, None)


def get_orb_style(emotion: str) -> str:
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
    Builds the drive bars as a self-contained HTML string.
    Rendered via components.v1.html() to bypass Streamlit's Markdown
    parser, which escapes single quotes inside inline styles and
    corrupts the output by rendering raw HTML as plain text.
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
        </div>"""
    return f"""<div style="background:#FDFCFA;border:1px solid #E8E4DC;border-radius:16px;padding:20px 24px;font-family:monospace;">{rows}</div>"""


def build_q_values_html(q_vals: dict, best: str, epsilon: float, steps: int, buf: int) -> str:
    """
    Renders Q-values as horizontal bars.
    The best action is highlighted in purple; others in gray.
    Bar width maps Q-values from [-1, 1] → [0%, 100%].
    """
    rows = ""
    for action, val in q_vals.items():
        pct    = min(100, max(0, int((val + 1) * 50)))  # [-1,1] → [0,100]
        color  = "#7F77DD" if action == best else "#D3D1C7"
        weight = "600" if action == best else "400"
        rows += f"""
        <div style="margin-bottom:10px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                <span style="font-size:11px;letter-spacing:1px;text-transform:uppercase;color:#888780;font-family:monospace;font-weight:{weight};">{action}</span>
                <span style="font-size:11px;color:#B4B2A9;font-family:monospace;">{val:.3f}</span>
            </div>
            <div style="background:#E8E4DC;border-radius:3px;height:4px;overflow:hidden;">
                <div style="width:{pct}%;height:100%;background:{color};border-radius:3px;"></div>
            </div>
        </div>"""
    return f"""
    <div style="background:#FDFCFA;border:1px solid #E8E4DC;border-radius:16px;padding:20px 24px;font-family:monospace;">
        <div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#B4B2A9;margin-bottom:14px;">
            Q-values &nbsp;·&nbsp; best action: <span style="color:#7F77DD;font-weight:600;">{best}</span>
        </div>
        {rows}
        <div style="font-size:10px;color:#D3D1C7;margin-top:12px;display:flex;gap:16px;">
            <span>ε={epsilon:.3f}</span>
            <span>steps={steps}</span>
            <span>buffer={buf}</span>
        </div>
    </div>"""


# ─── UI ───────────────────────────────────────────────────────────────────────

emotion  = neurumi.get_emotion()
meta     = EMOTION_META[emotion]
wellness = neurumi.get_wellness()

st.markdown(
    '<p style="font-size:0.7rem;letter-spacing:0.18em;text-transform:uppercase;'
    'color:#B4B2A9;margin-bottom:0.25rem;">artificial life · ml</p>',
    unsafe_allow_html=True,
)

# ── Creature card ─────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="creature-card">
    <div class="creature-orb" style="{get_orb_style(emotion)}">
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

# ── Metrics ───────────────────────────────────────────────────────────────────

p1_loss   = trainer.get_last_loss()
p2_loss   = dqn.get_last_loss()
avg_reward = dqn.get_avg_reward()

st.markdown(f"""
<div class="metric-row">
    <div class="metric-box">
        <div class="metric-value">{round(wellness * 100)}</div>
        <div class="metric-label">Wellness</div>
    </div>
    <div class="metric-box">
        <div class="metric-value">{neurumi.age}</div>
        <div class="metric-label">Epoch</div>
    </div>
    <div class="metric-box">
        <div class="metric-value">{p1_loss:.4f}</div>
        <div class="metric-label">P1 loss</div>
    </div>
    <div class="metric-box">
        <div class="metric-value">{avg_reward:.3f}</div>
        <div class="metric-label">Avg reward</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Internal state ────────────────────────────────────────────────────────────

st.markdown('<p class="section-header">Internal state</p>', unsafe_allow_html=True)

components.html(build_drives_html([
    ("Hunger",    neurumi.hunger,    "#E24B4A"),
    ("Curiosity", neurumi.curiosity, "#7F77DD"),
    ("Affection", neurumi.affection, "#D4537E"),
    ("Energy",    neurumi.energy,    "#1D9E75"),
    ("Fear",      neurumi.fear,      "#BA7517"),
]), height=210)

# ── Phase 1 — Player interaction ──────────────────────────────────────────────

st.markdown(
    '<p class="section-header">Phase 1 &nbsp;·&nbsp; '
    '<span style="background:#E6F1FB;color:#185FA5;padding:2px 8px;border-radius:10px;font-size:0.6rem;">'
    'supervised · you teach</span></p>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("🍎  Feed",   key="p1_feed"):   do_player_action("feed");   st.rerun()
with col2:
    if st.button("🎮  Play",   key="p1_play"):   do_player_action("play");   st.rerun()
with col3:
    if st.button("✦   Pet",    key="p1_pet"):    do_player_action("pet");    st.rerun()
with col4:
    if st.button("◌   Ignore", key="p1_ignore"): do_player_action("ignore"); st.rerun()

col_tick, col_reset = st.columns([3, 1])
with col_tick:
    if st.button("◎  Pass time  →", key="p1_tick"):
        do_pass_time()
        st.rerun()
with col_reset:
    if st.button("↺ Reset", key="btn_reset"):
        do_reset()
        st.rerun()

# Phase 1 loss chart
if len(trainer.loss_history) > 2:
    st.markdown(
        '<p style="font-size:0.65rem;color:#B4B2A9;margin-top:0.75rem;">P1 loss — decreases as the network learns your action patterns</p>',
        unsafe_allow_html=True,
    )
    st.line_chart(trainer.loss_history, height=80, use_container_width=True)

# ── Phase 2 — Autonomous agent ────────────────────────────────────────────────

st.markdown(
    '<p class="section-header">Phase 2 &nbsp;·&nbsp; '
    '<span style="background:#EEEDFE;color:#534AB7;padding:2px 8px;border-radius:10px;font-size:0.6rem;">'
    'Q-Learning · agent discovers</span></p>',
    unsafe_allow_html=True,
)

col_a1, col_a2 = st.columns(2)
with col_a1:
    if st.button("⚡  Agent step", key="p2_step"):
        do_agent_step(1)
        st.rerun()
with col_a2:
    if st.button("⚡ ×20  Train burst", key="p2_burst"):
        do_agent_step(20)
        st.rerun()

# Q-values panel — only shown once the buffer has started filling
if len(dqn.buffer) > 0:
    q_vals = dqn.get_q_values(neurumi)
    best   = max(q_vals, key=q_vals.get)
    components.html(
        build_q_values_html(q_vals, best, dqn.epsilon, dqn.steps, len(dqn.buffer)),
        height=230,
    )

# Phase 2 reward chart
if len(dqn.reward_history) > 2:
    st.markdown(
        '<p style="font-size:0.65rem;color:#B4B2A9;margin-top:0.75rem;">P2 reward — trends upward as the agent learns to keep Neurumi well</p>',
        unsafe_allow_html=True,
    )
    st.line_chart(dqn.reward_history, height=80, use_container_width=True)

# ── Episodic memory ───────────────────────────────────────────────────────────

st.markdown('<p class="section-header">Episodic memory</p>', unsafe_allow_html=True)

log_html = '<div class="scrollable-log">'
for entry in st.session_state.memory_log:
    if isinstance(entry, dict):
        kind = entry.get("kind", "normal")
        css  = f"memory-item {kind}" if kind != "normal" else "memory-item"
        text = entry["text"]
    else:
        css, text = "memory-item", entry
    log_html += f'<div class="{css}">{text}</div>'
log_html += "</div>"

st.markdown(log_html, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    '<p style="font-size:0.62rem;color:#D3D1C7;text-align:center;letter-spacing:0.08em;">'
    'neurumi · phase 1: supervised · phase 2: dqn · pytorch · streamlit'
    '</p>',
    unsafe_allow_html=True,
)