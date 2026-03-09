import streamlit as st
import requests
import pandas as pd
import sys
import os

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from football_agent.db import (get_teams, get_years, get_standings, get_team_results, get_goals_scored,
                               get_teams_by_league)
from football_agent.config import get_league_options
from dotenv import load_dotenv
import concurrent.futures

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


@st.cache_data(ttl=3600)
def cached_league_options():
    return get_league_options()


@st.cache_data(ttl=3600)
def cached_teams_by_league(league_id: str):
    return get_teams_by_league(league_id=league_id)


def run_agent(query: str) -> str:
    try:
        return str(FootballAgent().crew().kickoff(inputs={"query": query}))
    except Exception as e:
        return f"Error: {e}"


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Football Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Fuerza el sidebar siempre visible */
    [data-testid="stSidebar"] {
        min-width: 250px !important;
        max-width: 250px !important;
        transform: none !important;
        visibility: visible !important;
    }

    /* Oculta el botón de colapsar sidebar */
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    [data-testid="stSidebar"] [data-testid="stSelectbox"] * {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    * { font-family: 'DM Sans', sans-serif; }

    .main { background-color: #f8f9fa; }

    .stApp { background-color: #f8f9fa; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0a0a0a;
        border-right: 1px solid #1a1a1a;
    }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    [data-testid="stSidebar"] .stRadio label { 
        font-size: 14px;
        padding: 8px 0;
        color: #aaaaaa !important;
    }
    [data-testid="stSidebar"] .stRadio [aria-checked="true"] + label {
        color: #ffffff !important;
    }

    /* Color del valor seleccionado en selectbox del sidebar */
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
        background-color: #1a1a1a !important;
        border-color: #333333 !important;
    }

    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div > div {
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] .stSelectbox input {
        color: #ffffff !important;
        background-color: #1a1a1a !important;
    }

    /* Cards */
    .card {
        background: #ffffff;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #eeeeee;
        margin-bottom: 16px;
    }

    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px 24px;
        border: 1px solid #eeeeee;
        text-align: center;
    }

    .metric-value {
        font-size: 32px;
        font-weight: 600;
        color: #0a0a0a;
        font-family: 'DM Mono', monospace;
    }

    .metric-label {
        font-size: 12px;
        color: #888888;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 4px;
    }

    /* Prob bars */
    .prob-bar-container { margin: 8px 0; }
    .prob-label {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        margin-bottom: 4px;
        color: #333;
    }
    .prob-bar {
        height: 6px;
        border-radius: 3px;
        background: #eeeeee;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 3px;
        background: #0a0a0a;
        transition: width 0.5s ease;
    }

    /* Section titles */
    .section-title {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #888888;
        margin-bottom: 16px;
    }

    /* Match header */
    .match-header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 24px;
        padding: 32px;
        background: #0a0a0a;
        border-radius: 12px;
        margin-bottom: 24px;
    }
    .team-name {
        font-size: 20px;
        font-weight: 600;
        color: #ffffff;
        text-align: center;
    }
    .vs-badge {
        font-size: 12px;
        color: #666666;
        font-family: 'DM Mono', monospace;
        letter-spacing: 0.2em;
    }

    /* Table */
    .standings-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
    }
    .standings-table th {
        text-align: left;
        padding: 8px 12px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #888888;
        border-bottom: 1px solid #eeeeee;
    }
    .standings-table td {
        padding: 10px 12px;
        border-bottom: 1px solid #f5f5f5;
        color: #0a0a0a;
    }
    .standings-table tr:hover td { background: #f8f9fa; }
    .pos-badge {
        display: inline-block;
        width: 24px;
        height: 24px;
        border-radius: 6px;
        background: #f0f0f0;
        text-align: center;
        line-height: 24px;
        font-size: 12px;
        font-weight: 600;
        font-family: 'DM Mono', monospace;
    }
    .pos-1 { background: #FFD700; color: #0a0a0a; }
    .pos-2 { background: #C0C0C0; color: #0a0a0a; }
    .pos-3 { background: #CD7F32; color: #ffffff; }
    .pos-rel { background: #ff4444; color: #ffffff; }

    /* Chat */
    .chat-message {
        padding: 12px 16px;
        border-radius: 10px;
        margin-bottom: 8px;
        font-size: 14px;
        line-height: 1.6;
    }
    .chat-user {
        background: #0a0a0a;
        color: #ffffff;
        margin-left: 40px;
    }
    .chat-agent {
        background: #ffffff;
        color: #0a0a0a;
        border: 1px solid #eeeeee;
        margin-right: 40px;
    }

    /* Hide streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CACHE
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600)
def cached_teams():
    return get_teams()


@st.cache_data(ttl=3600)
def cached_years():
    return get_years()


@st.cache_data(ttl=600)
def cached_standings(league, year):
    return get_standings(leagueId=league, year=year)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚽ Football Analytics")
    st.markdown("---")
    section = st.radio(
        "Navegación",
        ["Predicción", "Estadísticas", "Clasificación", "Agente"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    year_global = st.selectbox("Temporada", cached_years())

# ─────────────────────────────────────────────
# SECCIÓN: PREDICCIÓN
# ─────────────────────────────────────────────

if section == "Predicción":
    st.markdown("### Predicción de partido")
    st.markdown('<div class="section-title">Selecciona los equipos</div>', unsafe_allow_html=True)

    league_options = cached_league_options()
    league_name = st.selectbox("Liga", list(league_options.keys()), key="league_select_pred")
    league_id = league_options[league_name]

    # Filtrar equipos por liga
    teams = cached_teams_by_league(league_id=league_id)

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Local", teams, key="home")
    with col2:
        away_options = [t for t in teams if t != home_team]
        away_team = st.selectbox("Visitante", away_options, key="away")

    if st.button("Predecir", width="stretch"):
        with st.spinner("Calculando predicción..."):
            try:
                resp = requests.post(
                    "http://localhost:8001/predict",
                    json={"home_team": home_team, "away_team": away_team, "year": year_global,
                          "league_id": league_options[league_name]}
                )
                data = resp.json()

                # Match header
                st.markdown(f"""
                <div class="match-header">
                    <div class="team-name">{home_team.replace('-', ' ').title()}</div>
                    <div class="vs-badge">VS</div>
                    <div class="team-name">{away_team.replace('-', ' ').title()}</div>
                </div>
                """, unsafe_allow_html=True)

                # Resultado
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown('<div class="section-title">Resultado probable (cuota estimada)</div>', unsafe_allow_html=True)
                    probs = data["resultado"]["probabilities"]
                    predicted = data["resultado"]["predicted"]
                    odds = data["resultado"]["odds"]
                    labels = {"1": f"Victoria {home_team.replace('-', ' ').title()}",
                              "X": "Empate",
                              "2": f"Victoria {away_team.replace('-', ' ').title()}"}

                    for key, label in labels.items():
                        pct = probs[key]
                        odd_val = odds[key]
                        is_predicted = "●  " if key == predicted else "○  "
                        st.markdown(f"""
                        <div class="prob-bar-container">
                            <div class="prob-label">
                                <span>{is_predicted}{label} <span style="color:#888888;font-size:12px">({odd_val})</span></span>
                                <span style="font-family:'DM Mono',monospace">{pct:.1f}%</span>
                            </div>
                            <div class="prob-bar">
                                <div class="prob-fill" style="width:{pct}%"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="section-title">Over / Under goles</div>', unsafe_allow_html=True)
                    ou = data["over_under"]
                    thresholds = [("over_0_5", "0.5"), ("over_1_5", "1.5"), ("over_2_5", "2.5"), ("over_3_5", "3.5")]

                    cols = st.columns(4)
                    for i, (key, label) in enumerate(thresholds):
                        over_pct = ou[key]["over"]
                        under_pct = ou[key]["under"]
                        with cols[i]:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Over {label}</div>
                                <div class="metric-value">{over_pct:.0f}%</div>
                                <div class="metric-label" style="margin-top:8px">Under {under_pct:.0f}%</div>
                            </div>
                            """, unsafe_allow_html=True)



            except Exception as e:
                st.error(f"Error conectando con el servicio de predicción: {e}")
                st.info("Asegúrate de que el servidor ML está corriendo: `PYTHONPATH=src python ml/predict_old.py`")


# ─────────────────────────────────────────────
# SECCIÓN: ESTADÍSTICAS
# ─────────────────────────────────────────────

elif section == "Estadísticas":
    st.markdown("### Estadísticas de equipo")
    team = st.selectbox("Equipo", cached_teams())

    if team:
        col1, col2, col3 = st.columns(3)

        goals = get_goals_scored(team, year=year_global)
        if goals:
            g = goals[0]
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Goles marcados</div>
                    <div class="metric-value">{g.get('total_goals_scored', 0)}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Goles encajados</div>
                    <div class="metric-value">{g.get('total_goals_conceded', 0)}</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                diff = g.get('total_goals_scored', 0) - g.get('total_goals_conceded', 0)
                sign = "+" if diff >= 0 else ""
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Diferencia</div>
                    <div class="metric-value">{sign}{diff}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-title">Últimos partidos</div>', unsafe_allow_html=True)

        results = get_team_results(team, year=year_global, top_n=10)
        if results:
            rows = []
            for r in results:
                hg = int(r.get("home_goals") or 0)
                ag = int(r.get("away_goals") or 0)
                is_home = r["homeTeam"] == team
                gf = hg if is_home else ag
                gc = ag if is_home else hg
                score = f"{hg} - {ag}"
                result_icon = "✅" if r["points"] == 3 else ("➖" if r["points"] == 1 else "❌")
                rows.append({
                    "Jornada": int(r["Round"]),
                    "Partido": f"{r['homeTeam']} vs {r['awayTeam']}",
                    "Resultado": score,
                    "": result_icon,
                    "Pts": int(r["points"])
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ─────────────────────────────────────────────
# SECCIÓN: CLASIFICACIÓN
# ─────────────────────────────────────────────

elif section == "Clasificación":
    st.markdown("### Clasificación")

    league_options = cached_league_options()
    league_name = st.selectbox("Liga", list(league_options.keys()), key="league_select")
    league_id = league_options[league_name]

    data = cached_standings(league=league_id, year=year_global)

    if data:
        rows = ""
        for i, row in enumerate(data, 1):
            total = len(data)

            if i == 1:
                pos_class = "pos-1"
            elif i == 2:
                pos_class = "pos-2"
            elif i == 3:
                pos_class = "pos-3"
            elif i >= total - 2:
                pos_class = "pos-rel"
            else:
                pos_class = ""

            team_name = row['team'].replace('-', ' ').title()
            pts = int(row['points'])
            pj = int(row['played'])
            v = int(row['wins'])
            e = int(row['draws'])
            d = int(row['losses'])
            gf = int(row['goals_for'])
            gc = int(row['goals_against'])

            rows += (
                f"<tr>"
                f"<td><span class='pos-badge {pos_class}'>{i}</span></td>"
                f"<td><b>{team_name}</b></td>"
                f"<td>{pj}</td>"
                f"<td>{v}</td>"
                f"<td>{e}</td>"
                f"<td>{d}</td>"
                f"<td>{gf}</td>"
                f"<td>{gc}</td>"
                f"<td><b>{pts}</b></td>"
                f"</tr>"
            )

        html = (
            "<div class='card'>"
            "<table class='standings-table'>"
            "<thead><tr>"
            "<th>#</th><th>Equipo</th><th>PJ</th><th>V</th>"
            "<th>E</th><th>D</th><th>GF</th><th>GC</th><th>Pts</th>"
            "</tr></thead>"
            f"<tbody>{rows}</tbody>"
            "</table></div>"
        )

        st.markdown(html, unsafe_allow_html=True)
    else:
        st.info("No hay datos de clasificación para esta temporada.")


# ─────────────────────────────────────────────
# SECCIÓN: AGENTE
# ─────────────────────────────────────────────

elif section == "Agente":
    st.markdown("### Consulta al agente")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Mostrar historial
    for msg in st.session_state.chat_history:
        css_class = "chat-user" if msg["role"] == "user" else "chat-agent"
        icon = "→" if msg["role"] == "user" else "⚽"
        st.markdown(f"""
        <div class="chat-message {css_class}">
            <b>{icon}</b> {msg["content"]}
        </div>
        """, unsafe_allow_html=True)

    # Input
    query = st.chat_input("Pregunta algo sobre La Liga...")
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.spinner("El agente está pensando..."):
            try:
                resp = requests.post(
                    "http://localhost:8002/agent",
                    json={"query": query},
                    timeout=120
                )
                response = resp.json()["response"]
            except Exception as e:
                response = f"Error: {e}"

        st.session_state.chat_history.append({"role": "agent", "content": response})
        st.rerun()