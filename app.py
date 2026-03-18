"""
NEURUMI — Vida artificial con aprendizaje real.

    streamlit run app.py
"""

import streamlit as st
import torch
import time
from pathlib import Path

from brain import NeurumiBrain, save_brain, load_brain
from state import NeurumiState, ACTION_EFFECTS, EMOTION_META
from trainer import NeurumiTrainer


# ─── Configuración de página ────────────────────────────────────────────────

st.set_page_config(
    page_title="NEURUMI",
    page_icon="◉",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Estilos ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400&display=swap');

/* Reset y base */
html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #F7F5F0;
}

/* Ocultar chrome de Streamlit */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 680px; }

/* Tipografía */
h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
    font-weight: 400;
    letter-spacing: -0.02em;
}

/* Criatura — el componente central */
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
    position: relative;
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

/* Drives */
.drive-container {
    background: #FDFCFA;
    border: 1px solid #E8E4DC;
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}

.drive-label {
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #888780;
    margin-bottom: 0.4rem;
}

/* Botones de acción */
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

/* Log de memoria */
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

/* Métricas */
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

/* Badge de emoción */
.emotion-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 400;
}

/* Section headers */
.section-header {
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #B4B2A9;
    margin-bottom: 0.6rem;
    margin-top: 1.25rem;
}

/* Loss indicator */
.loss-pill {
    display: inline-block;
    font-size: 0.65rem;
    font-family: 'DM Mono', monospace;
    color: #5F5E5A;
    background: #F0EDE6;
    border-radius: 20px;
    padding: 2px 8px;
    letter-spacing: 0.04em;
}

/* Wellness bar custom */
.wellness-bar-bg {
    background: #E8E4DC;
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
    margin-top: 6px;
}

.scrollable-log {
    max-height: 140px;
    overflow-y: auto;
}

/* Separador */
hr {
    border: none;
    border-top: 1px solid #E8E4DC;
    margin: 1.25rem 0;
}
</style>
""", unsafe_allow_html=True)


# ─── Session State — persiste entre reruns de Streamlit ──────────────────────
# st.session_state es como una variable global que Streamlit mantiene
# entre cada interacción del usuario. Sin esto, todo se reiniciaría
# con cada click.

if "neurumi" not in st.session_state:
    # Intentar cargar estado guardado
    if Path("neurumi_state.json").exists():
        st.session_state.neurumi = NeurumiState.load("neurumi_state.json")
    else:
        st.session_state.neurumi = NeurumiState()

if "brain" not in st.session_state:
    if Path("neurumi_brain.pt").exists():
        st.session_state.brain = load_brain("neurumi_brain.pt")
    else:
        st.session_state.brain = NeurumiBrain()

if "trainer" not in st.session_state:
    st.session_state.trainer = NeurumiTrainer(st.session_state.brain)

if "memory_log" not in st.session_state:
    loaded = Path("neurumi_state.json").exists()
    st.session_state.memory_log = [
        f"{'NEURUMI recuerda. Época ' + str(st.session_state.neurumi.age) if loaded else 'NEURUMI despierta por primera vez.'}"
    ]

if "tick" not in st.session_state:
    st.session_state.tick = st.session_state.neurumi.age

# Shortcuts locales para legibilidad
neurumi: NeurumiState = st.session_state.neurumi
brain: NeurumiBrain = st.session_state.brain
trainer: NeurumiTrainer = st.session_state.trainer


# ─── Helpers ─────────────────────────────────────────────────────────────────

def add_memory(text: str, recent: bool = False):
    """Agrega una entrada al log de memoria con timestamp de edad."""
    entry = {"text": f"[{neurumi.age:04d}] {text}", "recent": recent}
    st.session_state.memory_log.insert(0, entry)
    if len(st.session_state.memory_log) > 12:
        st.session_state.memory_log.pop()


def do_action(action: str):
    """
    Procesa una acción del jugador:
    1. Entrena la red con el efecto esperado de esta acción
    2. Aplica el efecto real al estado de NEURUMI
    3. Hace tick (pasa el tiempo)
    4. Guarda estado y pesos
    """
    # Entrenamiento: la red aprende qué debería pasar con esta acción
    avg_loss = trainer.train_on_action(action, neurumi, steps=8)

    # Efecto real sobre los drives
    neurumi.apply_action_effect(ACTION_EFFECTS[action])

    # Tick del tiempo
    neurumi.tick()
    st.session_state.tick = neurumi.age

    # Memoria
    action_labels = {
        "feed":   "Le diste de comer.",
        "play":   "Jugaste con ella.",
        "pet":    "La acariciaste.",
        "ignore": "La ignoraste. El tiempo pasó.",
    }
    add_memory(f"{action_labels[action]} loss={avg_loss:.4f}", recent=(action != "ignore"))

    # Persistir
    neurumi.save("neurumi_state.json")
    save_brain(brain, "neurumi_brain.pt")


def do_tick_only():
    """Solo pasa el tiempo, sin acción del jugador."""
    deltas = trainer.infer(neurumi)
    neurumi.apply_deltas(deltas, scale=0.05)
    neurumi.tick()
    st.session_state.tick = neurumi.age
    add_memory("Tiempo libre. La red predice y ajusta.")
    neurumi.save("neurumi_state.json")
    save_brain(brain, "neurumi_brain.pt")


def get_orb_style(emotion: str) -> str:
    """Genera el estilo de la orbe según la emoción actual."""
    colors = {
        "happy":   ("background: #EAF3DE;", "border: 2px solid #5DCAA5;"),
        "calm":    ("background: #E6F1FB;", "border: 2px solid #378ADD;"),
        "curious": ("background: #EEEDFE;", "border: 2px solid #7F77DD;"),
        "hungry":  ("background: #FAECE7;", "border: 2px solid #D85A30;"),
        "scared":  ("background: #FAEEDA;", "border: 2px solid #BA7517;"),
        "sleepy":  ("background: #F1EFE8;", "border: 2px solid #888780;"),
        "lonely":  ("background: #FBEAF0;", "border: 2px solid #D4537E;"),
    }
    bg, border = colors.get(emotion, colors["calm"])
    return f"{bg} {border}"


def drive_bar_html(label: str, value: float, color: str) -> str:
    """Genera HTML para una barra de drive."""
    pct = round(value * 100)
    return f"""
    <div style="margin-bottom: 0.75rem;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
            <span style="font-size:0.68rem; letter-spacing:0.1em; text-transform:uppercase; color:#888780;">{label}</span>
            <span style="font-size:0.68rem; color:#B4B2A9; font-family:'DM Mono',monospace;">{pct}</span>
        </div>
        <div style="background:#E8E4DC; border-radius:3px; height:5px; overflow:hidden;">
            <div style="width:{pct}%; height:100%; background:{color}; border-radius:3px; transition: width 0.5s ease;"></div>
        </div>
    </div>
    """


# ─── UI ───────────────────────────────────────────────────────────────────────

emotion = neurumi.get_emotion()
meta = EMOTION_META[emotion]
wellness = neurumi.get_wellness()

# Título mínimo
st.markdown(
    '<p style="font-size:0.7rem; letter-spacing:0.18em; text-transform:uppercase; '
    'color:#B4B2A9; margin-bottom:0.25rem;">vida artificial · ml</p>',
    unsafe_allow_html=True
)

# ── Criatura ──────────────────────────────────────────────────────────────────

orb_style = get_orb_style(emotion)

st.markdown(f"""
<div class="creature-card">
    <div class="creature-orb" style="{orb_style}">
        <span style="font-size:2.8rem; line-height:1;">{meta['emoji']}</span>
    </div>
    <p class="creature-name">{neurumi.name}</p>
    <p class="creature-emotion">
        <span class="emotion-badge" style="background:{meta['color']}22; color:{meta['color']};">
            {meta['label']}
        </span>
        &nbsp;·&nbsp; época {neurumi.age}
    </p>
</div>
""", unsafe_allow_html=True)


# ── Métricas rápidas ──────────────────────────────────────────────────────────

last_loss = trainer.get_last_loss()
interactions = len(trainer.loss_history)

st.markdown(f"""
<div class="metric-row">
    <div class="metric-box">
        <div class="metric-value">{round(wellness * 100)}</div>
        <div class="metric-label">Bienestar</div>
    </div>
    <div class="metric-box">
        <div class="metric-value">{neurumi.age}</div>
        <div class="metric-label">Épocas</div>
    </div>
    <div class="metric-box">
        <div class="metric-value">{interactions}</div>
        <div class="metric-label">Entrenamientos</div>
    </div>
    <div class="metric-box">
        <div class="metric-value">{last_loss:.4f}</div>
        <div class="metric-label">Último loss</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Drives ────────────────────────────────────────────────────────────────────

st.markdown('<p class="section-header">Estado interno</p>', unsafe_allow_html=True)

drive_config = [
    ("Hambre",    neurumi.hunger,    "#E24B4A"),
    ("Curiosidad", neurumi.curiosity, "#7F77DD"),
    ("Afecto",    neurumi.affection, "#D4537E"),
    ("Energía",   neurumi.energy,    "#1D9E75"),
    ("Miedo",     neurumi.fear,      "#BA7517"),
]

bars_html = '<div style="background:#FDFCFA; border:1px solid #E8E4DC; border-radius:16px; padding:1.25rem 1.5rem;">'
for label, value, color in drive_config:
    bars_html += drive_bar_html(label, value, color)
bars_html += "</div>"

st.markdown(bars_html, unsafe_allow_html=True)


# ── Acciones ──────────────────────────────────────────────────────────────────

st.markdown('<p class="section-header">Interacción</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🍎\nAlimentar", key="btn_feed"):
        do_action("feed")
        st.rerun()

with col2:
    if st.button("🎮\nJugar", key="btn_play"):
        do_action("play")
        st.rerun()

with col3:
    if st.button("✦\nAcariciar", key="btn_pet"):
        do_action("pet")
        st.rerun()

with col4:
    if st.button("◌\nIgnorar", key="btn_ignore"):
        do_action("ignore")
        st.rerun()

# Tick manual (tiempo libre)
col_tick, col_reset = st.columns([3, 1])
with col_tick:
    if st.button("◎  Pasar tiempo  →", key="btn_tick"):
        do_tick_only()
        st.rerun()

with col_reset:
    if st.button("↺ Reset", key="btn_reset"):
        # Borra archivos guardados y reinicia
        Path("neurumi_state.json").unlink(missing_ok=True)
        Path("neurumi_brain.pt").unlink(missing_ok=True)
        for key in ["neurumi", "brain", "trainer", "memory_log", "tick"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()


# ── Loss chart ────────────────────────────────────────────────────────────────

if len(trainer.loss_history) > 2:
    st.markdown('<p class="section-header">Aprendizaje — loss por interacción</p>', unsafe_allow_html=True)
    st.line_chart(
        trainer.loss_history,
        height=100,
        use_container_width=True,
    )
    st.markdown(
        '<p style="font-size:0.65rem; color:#B4B2A9; text-align:center; margin-top:-0.5rem;">'
        'El loss tiende a 0 conforme la red aprende tus patrones de interacción.'
        '</p>',
        unsafe_allow_html=True
    )


# ── Memoria episódica ─────────────────────────────────────────────────────────

st.markdown('<p class="section-header">Memoria episódica</p>', unsafe_allow_html=True)

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
    '<p style="font-size:0.62rem; color:#D3D1C7; text-align:center; letter-spacing:0.08em;">'
    'NEURUMI · red neuronal feed-forward · pytorch · streamlit'
    '</p>',
    unsafe_allow_html=True
)