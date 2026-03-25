import streamlit as st
import requests
import pandas as pd
import sys
import os
import html
import plotly.graph_objects as go

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from football_core.db import (get_teams, get_years, get_standings, get_team_results, get_goals_scored,
                              get_teams_by_league)
from football_core.config import get_league_options
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

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

    /* ════════════════════════════════════════════════════════════════════ */
    /* TONOS CÁLIDOS - FONDO PRINCIPAL */
    /* ════════════════════════════════════════════════════════════════════ */
    .main { background-color: #FDF6E3; }  /* Crema cálido */
    .stApp { background-color: #FDF6E3; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #2C3E50;  /* Azul oscuro cálido en vez de negro */
        border-right: 1px solid #34495E;
    }
    [data-testid="stSidebar"] * { color: #ECF0F1 !important; }
    [data-testid="stSidebar"] .stRadio label { 
        font-size: 14px;
        padding: 8px 0;
        color: #BDC3C7 !important;
    }
    [data-testid="stSidebar"] .stRadio [aria-checked="true"] + label {
        color: #E67E22 !important;  /* Naranja cálido */
    }

    /* Color del valor seleccionado en selectbox del sidebar */
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
        background-color: #34495E !important;
        border-color: #E67E22 !important;  /* Borde naranja */
    }

    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div > div {
        color: #ECF0F1 !important;
    }

    [data-testid="stSidebar"] .stSelectbox input {
        color: #ECF0F1 !important;
        background-color: #34495E !important;
    }

    /* ════════════════════════════════════════════════════════════════════ */
    /* SELECTBOXES/DROPDOWNS CON TONOS CÁLIDOS */
    /* ════════════════════════════════════════════════════════════════════ */
    .stSelectbox > div > div {
        background-color: #FFFBF0 !important;  /* Fondo marfil */
        border: 2px solid #F39C12 !important;  /* Borde ámbar */
        border-radius: 8px !important;
    }

    .stSelectbox label {
        color: #D35400 !important;  /* Naranja oscuro */
        font-weight: 600 !important;
    }

    /* ════════════════════════════════════════════════════════════════════ */
    /* BOTÓN PREDECIR CON COLOR SIDEBAR */
    /* ════════════════════════════════════════════════════════════════════ */
    .stButton > button {
        background-color: #2C3E50 !important;  /* Mismo color que sidebar */
        color: #ECF0F1 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 8px rgba(44, 62, 80, 0.3) !important;
    }

    .stButton > button:hover {
        background-color: #34495E !important;  /* Un tono más claro al hover */
        box-shadow: 0 6px 12px rgba(44, 62, 80, 0.5) !important;
        transform: translateY(-2px) !important;
    }

    /* ════════════════════════════════════════════════════════════════════ */
    /* CARDS CON TONOS CÁLIDOS */
    /* ════════════════════════════════════════════════════════════════════ */
    .card {
        background: #FFFBF0;  /* Marfil cálido */
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #F39C12;  /* Borde ámbar */
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(230, 126, 34, 0.1);
    }

    .metric-card {
        background: #FFFBF0;
        border-radius: 12px;
        padding: 20px 24px;
        border: 1px solid #F39C12;
        text-align: center;
        box-shadow: 0 2px 8px rgba(230, 126, 34, 0.1);
    }

    .metric-value {
        font-size: 32px;
        font-weight: 600;
        color: #D35400;  /* Naranja oscuro */
        font-family: 'DM Mono', monospace;
    }

    .metric-label {
        font-size: 12px;
        color: #7F8C8D;  /* Gris cálido */
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 4px;
    }

    /* ════════════════════════════════════════════════════════════════════ */
    /* CONFIDENCE BADGES CON TONOS CÁLIDOS */
    /* ════════════════════════════════════════════════════════════════════ */
    .confidence-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .confidence-very_high { background: #27AE60; color: white; }
    .confidence-high { background: #F39C12; color: white; }  /* Ámbar */
    .confidence-medium { background: #E67E22; color: white; }  /* Naranja */
    .confidence-low { background: #95A5A6; color: white; }

    /* ════════════════════════════════════════════════════════════════════ */
    /* BARRAS DE PROBABILIDAD */
    /* ════════════════════════════════════════════════════════════════════ */
    .prob-bar-container { margin: 8px 0; }
    .prob-label {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        margin-bottom: 4px;
        color: #2C3E50;
    }
    .prob-bar {
        height: 6px;
        border-radius: 3px;
        background: #ECF0F1;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 3px;
        background: linear-gradient(90deg, #E67E22 0%, #D35400 100%);  /* Gradiente naranja */
        transition: width 0.5s ease;
    }

    /* Section titles */
    .section-title {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #7F8C8D;
        margin-bottom: 16px;
    }

    /* ════════════════════════════════════════════════════════════════════ */
    /* MATCH HEADER CON TONOS CÁLIDOS */
    /* ════════════════════════════════════════════════════════════════════ */
    .match-header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 24px;
        padding: 32px;
        background: linear-gradient(135deg, #2C3E50 0%, #34495E 100%);
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(230, 126, 34, 0.2);
    }
    .team-name {
        font-size: 20px;
        font-weight: 600;
        color: #ECF0F1;
        text-align: center;
    }
    .vs-badge {
        font-size: 12px;
        color: #E67E22;  /* Naranja */
        font-family: 'DM Mono', monospace;
        letter-spacing: 0.2em;
    }

    /* ════════════════════════════════════════════════════════════════════ */
    /* TABLA DE CLASIFICACIÓN */
    /* ════════════════════════════════════════════════════════════════════ */
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
        color: #7F8C8D;
        border-bottom: 1px solid #F39C12;  /* Borde ámbar */
    }
    .standings-table td {
        padding: 10px 12px;
        border-bottom: 1px solid #ECF0F1;
        color: #2C3E50;
    }
    .standings-table tr:hover td { background: #FFF5E1; }  /* Hover cálido */
    .pos-badge {
        display: inline-block;
        width: 24px;
        height: 24px;
        border-radius: 6px;
        background: #ECF0F1;
        text-align: center;
        line-height: 24px;
        font-size: 12px;
        font-weight: 600;
        font-family: 'DM Mono', monospace;
    }
    .pos-1 { background: #FFD700; color: #2C3E50; }
    .pos-2 { background: #C0C0C0; color: #2C3E50; }
    .pos-3 { background: #CD7F32; color: #ffffff; }
    .pos-rel { background: #E74C3C; color: #ffffff; }

    /* Value bet indicator */
    .value-indicator {
        font-size: 12px;
        font-weight: 600;
        padding: 4px 8px;
        border-radius: 4px;
        display: inline-block;
        margin-left: 8px;
    }
    .value-positive { background: #d4edda; color: #27AE60; }
    .value-negative { background: #f8d7da; color: #E74C3C; }

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
def cached_years():
    return get_years()


@st.cache_data(ttl=3600)
def cached_standings(league, year):
    return get_standings(leagueId=league, year=year)


@st.cache_data(ttl=3600)
def cached_league_options():
    return get_league_options()


@st.cache_data(ttl=3600)
def cached_teams_by_league(league_id: str, season: str):
    return get_teams_by_league(league_id=league_id, season=season)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚽ Football Analytics")
    st.markdown("---")
    section = st.radio(
        "Navegación",
        ["Predicción", "Estadísticas", "Clasificación"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    year_global = st.selectbox("Temporada", cached_years())

# ─────────────────────────────────────────────
# SECCIÓN: PREDICCIÓN
# ─────────────────────────────────────────────

if section == "Predicción":
    st.markdown("### Predicción 1X2")
    st.markdown('<div class="section-title">Selecciona los equipos</div>', unsafe_allow_html=True)

    league_options = cached_league_options()
    league_name = st.selectbox("Liga", list(league_options.keys()), key="league_select_pred")
    league_id = league_options[league_name]

    # Filtrar equipos por liga
    teams = cached_teams_by_league(league_id=league_id, season=year_global)

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Local", teams, key="home")
    with col2:
        away_options = [t for t in teams if t != home_team]
        away_team = st.selectbox("Visitante", away_options, key="away")

    # BOTÓN DE PREDICCIÓN - Solo guarda en session_state
    if st.button("Predecir", type="primary", width='stretch'):
        with st.spinner("Calculando predicción..."):
            try:
                resp = requests.post(
                    "http://localhost:8001/predict",
                    json={
                        "home_team": home_team,
                        "away_team": away_team,
                        "year": year_global,
                        "league_id": league_id
                    },
                    timeout=30
                )
                # Guardar en session_state
                st.session_state['prediction_data'] = resp.json()
                st.session_state['home_team_display'] = home_team
                st.session_state['away_team_display'] = away_team

            except requests.exceptions.Timeout:
                st.error("⏱️ La predicción está tardando más de lo esperado. Intenta de nuevo.")
            except requests.exceptions.ConnectionError:
                st.error("❌ No se puede conectar con el servidor de predicción.")
                st.info("Asegúrate de que el servidor ML está corriendo: `PYTHONPATH=src python ml/predict.py`")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # MOSTRAR RESULTADOS - Si existen en session_state
    if 'prediction_data' in st.session_state:
        data = st.session_state['prediction_data']
        home_team_display = st.session_state['home_team_display']
        away_team_display = st.session_state['away_team_display']

        resultado = data["resultado"]
        probs = resultado["probabilities"]
        odds_modelo = resultado["odds"]
        confidence = resultado["confidence"]
        predicted = resultado["predicted"]

        # Match header
        st.markdown(f"""
        <div class="match-header">
            <div class="team-name">{home_team_display.replace('-', ' ').title()}</div>
            <div class="vs-badge">VS</div>
            <div class="team-name">{away_team_display.replace('-', ' ').title()}</div>
        </div>
        """, unsafe_allow_html=True)

        # Confianza
        confidence_class = f"confidence-{confidence['level']}"
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 24px;">
            <span class="confidence-badge {confidence_class}">{confidence['description']}</span>
            <div style="font-size: 12px; color: #888; margin-top: 8px;">
                Gap: {confidence['gap']:.1f}% | Elo diff: {confidence['elo_diff']:.0f}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ═══════════════════════════════════════════════════════════
        # RECOMENDACIÓN COMPACTA
        # ═══════════════════════════════════════════════════════════

        # Calcular EV para cada opción (usando valores típicos)
        bet365_default = {
            "1": 2.50 if predicted == "1" else 3.00,
            "X": 3.20,
            "2": 2.50 if predicted == "2" else 3.00
        }

        # ═══════════════════════════════════════════════════════════
        # GRÁFICOS Y CUOTAS
        # ═══════════════════════════════════════════════════════════

        labels_map = {
            "1": f"Local ({home_team_display.replace('-', ' ').title()})",
            "X": "Empate",
            "2": f"Visitante ({away_team_display.replace('-', ' ').title()})"
        }

        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown('<div class="section-title">Probabilidades del modelo</div>', unsafe_allow_html=True)

            # Gráfico de barras con Plotly
            categories = [labels_map["1"], labels_map["X"], labels_map["2"]]
            values = [probs["1"], probs["X"], probs["2"]]
            colors = ['#E67E22' if predicted == '1' else '#e5e7eb',
                      '#F39C12' if predicted == 'X' else '#e5e7eb',
                      '#E74C3C' if predicted == '2' else '#e5e7eb']

            fig = go.Figure(data=[
                go.Bar(
                    x=categories,
                    y=values,
                    marker_color=colors,
                    text=[f"{v:.1f}%" for v in values],
                    textposition='outside',
                    textfont=dict(size=14, family='DM Mono'),
                )
            ])

            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(
                    range=[0, 100],
                    showgrid=True,
                    gridcolor='#f0f0f0',
                    title=dict(
                        text='Probabilidad (%)',
                        font=dict(size=11, color='#888')
                    ),
                ),
                xaxis=dict(
                    showgrid=False,
                    title='',
                    tickfont=dict(size=12),
                ),
                font=dict(family='DM Sans'),
            )

            st.plotly_chart(fig, width='stretch')

        with col2:
            st.markdown('<div class="section-title">Cuotas del modelo</div>', unsafe_allow_html=True)

            for key in ["1", "X", "2"]:
                odd = odds_modelo[key]
                prob = probs[key]
                is_best = key == predicted
                label = labels_map[key].split('(')[0].strip().upper()

                # Fondo verde para la mejor opción
                if is_best:
                    bg_color = "#f0fdf4"
                    border_color = "#22c55e"
                    medal = "🥇 "
                else:
                    bg_color = "#ffffff"
                    border_color = "#eeeeee"
                    medal = ""

                st.markdown(f"""
                <div style="
                    background: {bg_color};
                    border: 2px solid {border_color};
                    border-radius: 12px;
                    padding: 20px 24px;
                    margin-bottom: 12px;
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 14px; font-weight: 500;">{label}{medal}</span>
                        <span style="font-size: 28px; font-weight: 600; font-family: 'DM Mono', monospace; color: #0a0a0a;">{odd:.2f}</span>
                    </div>
                    <div style="font-size: 12px; color: #888888; text-transform: uppercase; letter-spacing: 0.05em;">
                        PROBABILIDAD IMPLÍCITA MODELO: {prob:.2f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ═══════════════════════════════════════════════════════════
        # OVER/UNDER GOLES Y PARADAS
        # ═══════════════════════════════════════════════════════════

        if 'over_under_goals' in data:
            # Dividir en 2 columnas: Over/Under (izq) y Paradas (der)
            col_ou, col_saves = st.columns(2)

            with col_ou:
                st.markdown('<div class="section-title">⚽ Over/Under Goles</div>', unsafe_allow_html=True)

                ou = data['over_under_goals']

                # Fila 1: 0.5 y 1.5
                row1_col1, row1_col2 = st.columns(2)

                with row1_col1:
                    over_pct = ou["over_0_5"]["over"]
                    under_pct = ou["over_0_5"]["under"]
                    over_winner = over_pct > under_pct

                    st.markdown(f"""
                    <div style='border: 2px; border-radius: 12px; padding: 12px;'>
                        <div style='font-size: 11px; font-weight: 600; color: #D35400; margin-bottom: 8px; text-align: center; text-transform: uppercase; letter-spacing: 0.05em;'>
                            GOLES + 0.5
                        </div>
                        <div style='display: flex; gap: 6px;'>
                            <div style='flex: 1; background: {"#f0fdf4" if over_winner else "#ffffff"}; border: 2px solid {"#22c55e" if over_winner else "#e5e7eb"}; border-radius: 6px; padding: 8px; text-align: center;'>
                                <div style='font-size: 9px; color: #888;'>OVER</div>
                                <div style='font-size: 18px; font-weight: 700; color: {"#16a34a" if over_winner else "#6b7280"};'>{over_pct:.1f}%</div>
                            </div>
                            <div style='flex: 1; background: {"#fef2f2" if not over_winner else "#ffffff"}; border: 2px solid {"#ef4444" if not over_winner else "#e5e7eb"}; border-radius: 6px; padding: 8px; text-align: center;'>
                                <div style='font-size: 9px; color: #888;'>UNDER</div>
                                <div style='font-size: 18px; font-weight: 700; color: {"#dc2626" if not over_winner else "#6b7280"};'>{under_pct:.1f}%</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with row1_col2:
                    over_pct = ou["over_1_5"]["over"]
                    under_pct = ou["over_1_5"]["under"]
                    over_winner = over_pct > under_pct

                    st.markdown(f"""
                    <div style='border: 2px; border-radius: 12px; padding: 12px;'>
                        <div style='font-size: 11px; font-weight: 600; color: #D35400; margin-bottom: 8px; text-align: center; text-transform: uppercase; letter-spacing: 0.05em;'>
                            GOLES + 1.5
                        </div>
                        <div style='display: flex; gap: 6px;'>
                            <div style='flex: 1; background: {"#f0fdf4" if over_winner else "#ffffff"}; border: 2px solid {"#22c55e" if over_winner else "#e5e7eb"}; border-radius: 6px; padding: 8px; text-align: center;'>
                                <div style='font-size: 9px; color: #888;'>OVER</div>
                                <div style='font-size: 18px; font-weight: 700; color: {"#16a34a" if over_winner else "#6b7280"};'>{over_pct:.1f}%</div>
                            </div>
                            <div style='flex: 1; background: {"#fef2f2" if not over_winner else "#ffffff"}; border: 2px solid {"#ef4444" if not over_winner else "#e5e7eb"}; border-radius: 6px; padding: 8px; text-align: center;'>
                                <div style='font-size: 9px; color: #888;'>UNDER</div>
                                <div style='font-size: 18px; font-weight: 700; color: {"#dc2626" if not over_winner else "#6b7280"};'>{under_pct:.1f}%</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Fila 2: 2.5 y 3.5
                row2_col1, row2_col2 = st.columns(2)

                with row2_col1:
                    over_pct = ou["over_2_5"]["over"]
                    under_pct = ou["over_2_5"]["under"]
                    over_winner = over_pct > under_pct

                    st.markdown(f"""
                    <div style='border: 2px; border-radius: 12px; padding: 12px;'>
                        <div style='font-size: 11px; font-weight: 600; color: #D35400; margin-bottom: 8px; text-align: center; text-transform: uppercase; letter-spacing: 0.05em;'>
                            GOLES + 2.5
                        </div>
                        <div style='display: flex; gap: 6px;'>
                            <div style='flex: 1; background: {"#f0fdf4" if over_winner else "#ffffff"}; border: 2px solid {"#22c55e" if over_winner else "#e5e7eb"}; border-radius: 6px; padding: 8px; text-align: center;'>
                                <div style='font-size: 9px; color: #888;'>OVER</div>
                                <div style='font-size: 18px; font-weight: 700; color: {"#16a34a" if over_winner else "#6b7280"};'>{over_pct:.1f}%</div>
                            </div>
                            <div style='flex: 1; background: {"#fef2f2" if not over_winner else "#ffffff"}; border: 2px solid {"#ef4444" if not over_winner else "#e5e7eb"}; border-radius: 6px; padding: 8px; text-align: center;'>
                                <div style='font-size: 9px; color: #888;'>UNDER</div>
                                <div style='font-size: 18px; font-weight: 700; color: {"#dc2626" if not over_winner else "#6b7280"};'>{under_pct:.1f}%</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with row2_col2:
                    over_pct = ou["over_3_5"]["over"]
                    under_pct = ou["over_3_5"]["under"]
                    over_winner = over_pct > under_pct

                    st.markdown(f"""
                    <div style='border: 2px; border-radius: 12px; padding: 12px;'>
                        <div style='font-size: 11px; font-weight: 600; color: #D35400; margin-bottom: 8px; text-align: center; text-transform: uppercase; letter-spacing: 0.05em;'>
                            GOLES + 3.5
                        </div>
                        <div style='display: flex; gap: 6px;'>
                            <div style='flex: 1; background: {"#f0fdf4" if over_winner else "#ffffff"}; border: 2px solid {"#22c55e" if over_winner else "#e5e7eb"}; border-radius: 6px; padding: 8px; text-align: center;'>
                                <div style='font-size: 9px; color: #888;'>OVER</div>
                                <div style='font-size: 18px; font-weight: 700; color: {"#16a34a" if over_winner else "#6b7280"};'>{over_pct:.1f}%</div>
                            </div>
                            <div style='flex: 1; background: {"#fef2f2" if not over_winner else "#ffffff"}; border: 2px solid {"#ef4444" if not over_winner else "#e5e7eb"}; border-radius: 6px; padding: 8px; text-align: center;'>
                                <div style='font-size: 9px; color: #888;'>UNDER</div>
                                <div style='font-size: 18px; font-weight: 700; color: {"#dc2626" if not over_winner else "#6b7280"};'>{under_pct:.1f}%</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with col_saves:
                st.markdown('<div class="section-title">🧤 Paradas del Portero</div>', unsafe_allow_html=True)

                # Obtener datos de saves
                if 'saves' in data and data['saves']:
                    saves_home = data['saves'].get('home', {})
                    saves_away = data['saves'].get('away', {})

                    # Grid 2 columnas
                    col_home_saves, col_away_saves = st.columns(2)

                    with col_home_saves:
                        st.markdown(
                            f"<div style='text-align:center; font-weight:600; color:#D35400; margin-bottom:12px;'>🏠 {home_team_display.replace('-', ' ').title()}</div>",
                            unsafe_allow_html=True)

                        for threshold in ['over_0_5', 'over_1_5', 'over_2_5', 'over_3_5', 'over_4_5']:
                            prob = saves_home.get(threshold, 0) * 100
                            odd = 1 / (prob / 100) if prob > 0 else 0
                            label = threshold.replace('over_', '>').replace('_', '.')

                            st.markdown(f"""
                            <div style='display:flex; justify-content:space-between; align-items:center; padding:8px 12px; border-bottom:1px solid #ECF0F1;'>
                                <span style='color:#7F8C8D; font-size:13px;'>{label} paradas</span>
                                <div style='display:flex; gap:12px; align-items:center;'>
                                    <span style='color:#E67E22; font-weight:600; font-size:14px;'>{prob:.1f}%</span>
                                    <span style='color:#95A5A6; font-size:12px;'>({odd:.2f})</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                    with col_away_saves:
                        st.markdown(
                            f"<div style='text-align:center; font-weight:600; color:#D35400; margin-bottom:12px;'>✈️ {away_team_display.replace('-', ' ').title()}</div>",
                            unsafe_allow_html=True)

                        for threshold in ['over_0_5', 'over_1_5', 'over_2_5', 'over_3_5', 'over_4_5']:
                            prob = saves_away.get(threshold, 0) * 100
                            odd = 1 / (prob / 100) if prob > 0 else 0
                            label = threshold.replace('over_', '>').replace('_', '.')

                            st.markdown(f"""
                            <div style='display:flex; justify-content:space-between; align-items:center; padding:8px 12px; border-bottom:1px solid #ECF0F1;'>
                                <span style='color:#7F8C8D; font-size:13px;'>{label} paradas</span>
                                <div style='display:flex; gap:12px; align-items:center;'>
                                    <span style='color:#E67E22; font-weight:600; font-size:14px;'>{prob:.1f}%</span>
                                    <span style='color:#95A5A6; font-size:12px;'>({odd:.2f})</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("No hay datos de paradas disponibles")

            st.markdown("---")

        # ═══════════════════════════════════════════════════════════
        # CÓRNERS (ESTILO PARADAS: 6 COLUMNAS)
        # ═══════════════════════════════════════════════════════════

        if 'corners' in data and data['corners']:
            st.markdown('<div class="section-title">🚩 CÓRNERS</div>', unsafe_allow_html=True)

            corners_home = data['corners'].get('home', {})
            corners_away = data['corners'].get('away', {})

            # Grid 3 columnas sub cabeceras
            col1_subheader, col2_subheader, col3_subheader = st.columns(3)

            # Grid 6 columnas
            col1, col2, col3, col4, col5, col6 = st.columns(6)

            # Headers
            with col1_subheader:
                st.markdown(
                    "<div style='text-align:center; font-weight:600; color:#D35400; margin-bottom:12px;'>🏠 LOCAL </div>",
                    unsafe_allow_html=True)
            with col2_subheader:
                st.markdown(
                    "<div style='text-align:center; font-weight:600; color:#D35400; margin-bottom:12px;'>✈️ VISITANTE </div>",
                    unsafe_allow_html=True)
            with col3_subheader:
                st.markdown(
                    "<div style='text-align:center; font-weight:600; color:#D35400; margin-bottom:12px;'>📊 TOTAL </div>",
                    unsafe_allow_html=True)

            # Thresholds para cada columna
            col1_thresholds = ['over_2_5', 'over_3_5', 'over_4_5', 'over_5_5']
            col2_thresholds = ['over_6_5', 'over_7_5', 'over_8_5', 'over_9_5']

            # Columna 1: LOCAL 5.5-8.5
            with col1:
                for threshold in col1_thresholds:
                    prob = corners_home.get(threshold, 0) * 100
                    odd = 1 / (prob / 100) if prob > 0 else 0
                    label = threshold.replace('over_', '>').replace('_', '.')

                    st.markdown(f"""
                    <div style='display:flex; justify-content:space-between; align-items:center; padding:8px 12px; border-bottom:1px solid #ECF0F1; background:#FFF5E1; border-radius:4px; margin-bottom:4px;'>
                        <span style='color:#7F8C8D; font-size:13px;'>{label}</span>
                        <div style='display:flex; gap:8px; align-items:center;'>
                            <span style='color:#E67E22; font-weight:600; font-size:14px;'>{prob:.1f}%</span>
                            <span style='color:#95A5A6; font-size:11px;'>({odd:.2f})</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Columna 2: LOCAL 9.5-12.5
            with col2:
                for threshold in col2_thresholds:
                    prob = corners_home.get(threshold, 0) * 100
                    odd = 1 / (prob / 100) if prob > 0 else 0
                    label = threshold.replace('over_', '>').replace('_', '.')

                    st.markdown(f"""
                    <div style='display:flex; justify-content:space-between; align-items:center; padding:8px 12px; border-bottom:1px solid #ECF0F1; background:#FFF5E1; border-radius:4px; margin-bottom:4px;'>
                        <span style='color:#7F8C8D; font-size:13px;'>{label}</span>
                        <div style='display:flex; gap:8px; align-items:center;'>
                            <span style='color:#E67E22; font-weight:600; font-size:14px;'>{prob:.1f}%</span>
                            <span style='color:#95A5A6; font-size:11px;'>({odd:.2f})</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Columna 3: VISITANTE 5.5-8.5
            with col3:
                for threshold in col1_thresholds:
                    prob = corners_away.get(threshold, 0) * 100
                    odd = 1 / (prob / 100) if prob > 0 else 0
                    label = threshold.replace('over_', '>').replace('_', '.')

                    st.markdown(f"""
                    <div style='display:flex; justify-content:space-between; align-items:center; padding:8px 12px; border-bottom:1px solid #ECF0F1; background:#FFF5E1; border-radius:4px; margin-bottom:4px;'>
                        <span style='color:#2C3E50; font-size:13px;'>{label}</span>
                        <div style='display:flex; gap:8px; align-items:center;'>
                            <span style='color:#E67E22; font-weight:600; font-size:14px;'>{prob:.1f}%</span>
                            <span style='color:#95A5A6; font-size:11px;'>({odd:.2f})</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Columna 4: VISITANTE 9.5-12.5
            with col4:
                for threshold in col2_thresholds:
                    prob = corners_away.get(threshold, 0) * 100
                    odd = 1 / (prob / 100) if prob > 0 else 0
                    label = threshold.replace('over_', '>').replace('_', '.')

                    st.markdown(f"""
                    <div style='display:flex; justify-content:space-between; align-items:center; padding:8px 12px; border-bottom:1px solid #ECF0F1; background:#FFF5E1; border-radius:4px; margin-bottom:4px;'>
                        <span style='color:#2C3E50; font-size:13px;'>{label}</span>
                        <div style='display:flex; gap:8px; align-items:center;'>
                            <span style='color:#E67E22; font-weight:600; font-size:14px;'>{prob:.1f}%</span>
                            <span style='color:#95A5A6; font-size:11px;'>({odd:.2f})</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Columna 5: TOTAL 5.5-8.5 (suma local + visitante)
            with col5:
                for threshold in col1_thresholds:
                    home_prob = corners_home.get(threshold, 0) * 100
                    away_prob = corners_away.get(threshold, 0) * 100
                    total_prob = home_prob + away_prob
                    label = threshold.replace('over_', '>').replace('_', '.')

                    st.markdown(f"""
                    <div style='display:flex; justify-content:space-between; padding:8px 12px; border-bottom:1px solid #ECF0F1; background:#FFF5E1; border-radius:4px; margin-bottom:4px;'>
                        <span style='color:#7F8C8D; font-size:13px;'+{label} corners</span>
                        <span style='color:#D35400; font-weight:600; font-size:14px;'>{total_prob:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)

            # Columna 6: TOTAL 9.5-12.5 (suma local + visitante)
            with col6:
                for threshold in col2_thresholds:
                    home_prob = corners_home.get(threshold, 0) * 100
                    away_prob = corners_away.get(threshold, 0) * 100
                    total_prob = home_prob + away_prob
                    label = threshold.replace('over_', '>').replace('_', '.')

                    st.markdown(f"""
                    <div style='display:flex; justify-content:space-between; padding:8px 12px; border-bottom:1px solid #ECF0F1; background:#FFF5E1; border-radius:4px; margin-bottom:4px;'>
                        <span style='color:#7F8C8D; font-size:13px;' + {label} corners</span>
                        <span style='color:#D35400; font-weight:600; font-size:14px;'>{total_prob:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")

        # ═══════════════════════════════════════════════════════════
        # EXPANDERS
        # ═══════════════════════════════════════════════════════════

        # Comparación con Bet365 personalizada
        with st.expander("📊 Comparar con tus cuotas reales", expanded=False):
            st.markdown("**Introduce las cuotas de tu bookmaker:**")
            cols = st.columns(3)
            with cols[0]:
                bet365_1 = st.number_input("Cuota Local", min_value=1.01, value=float(bet365_default["1"]), step=0.05,
                                           key="bet365_1")
            with cols[1]:
                bet365_x = st.number_input("Cuota Empate", min_value=1.01, value=float(bet365_default["X"]), step=0.05,
                                           key="bet365_x")
            with cols[2]:
                bet365_2 = st.number_input("Cuota Visitante", min_value=1.01, value=float(bet365_default["2"]),
                                           step=0.05, key="bet365_2")

            bet365_odds = {"1": bet365_1, "X": bet365_x, "2": bet365_2}

            st.markdown("---")
            st.markdown("**Análisis de valor:**")

            for key in ["1", "X", "2"]:
                modelo = odds_modelo[key]
                bookmaker = bet365_odds[key]

                prob_modelo = (1 / modelo) * 100
                prob_bet365 = (1 / bookmaker) * 100
                diff_prob = prob_modelo - prob_bet365

                ev = (probs[key] / 100 * bookmaker) - 1
                ev_pct = ev * 100

                if diff_prob > 5:
                    indicator = '<span class="value-indicator value-positive">📈 VALUE BET</span>'
                elif diff_prob < -5:
                    indicator = '<span class="value-indicator value-negative">📉 SOBREVALORADA</span>'
                else:
                    indicator = '<span style="color: #888; font-size: 12px;">➡️ Alineada</span>'

                st.markdown(f"""
                <div style="padding: 12px; background: #f8f9fa; border-radius: 8px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span><b>{labels_map[key].split('(')[0].strip()}</b></span>
                        <span>{indicator}</span>
                    </div>
                    <div style="font-size: 13px; color: #666; margin-top: 4px;">
                        Modelo: {modelo:.2f} ({prob_modelo:.1f}%) vs Bookmaker: {bookmaker:.2f} ({prob_bet365:.1f}%)
                        <br>
                        <span style="font-weight: 600; color: {'#16a34a' if ev_pct > 0 else '#dc2626'};">
                            EV: {ev_pct:+.1f}% | Diferencia probabilidad: {diff_prob:+.1f}%
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Desglose técnico
        if "blend_info" in data:
            with st.expander("🔧 Desglose técnico del modelo", expanded=False):
                blend = data["blend_info"]

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Blend de modelos:**")
                    st.markdown(f"- Peso Elo: {blend['weight_elo'] * 100:.0f}%")
                    st.markdown(f"- Peso ML (Random Forest): {blend['weight_ml'] * 100:.0f}%")
                    st.markdown(f"- Diferencia Elo: {blend['elo_diff']:.1f}")

                with col2:
                    st.markdown("**Probabilidades Elo puro:**")
                    st.markdown(f"- Local: {blend['probas_elo'][0] * 100:.1f}%")
                    st.markdown(f"- Empate: {blend['probas_elo'][1] * 100:.1f}%")
                    st.markdown(f"- Visitante: {blend['probas_elo'][2] * 100:.1f}%")

                st.markdown("---")
                st.markdown("**Probabilidades ML (Random Forest):**")
                st.markdown(f"- Local: {blend['probas_ml'][0] * 100:.1f}%")
                st.markdown(f"- Empate: {blend['probas_ml'][1] * 100:.1f}%")
                st.markdown(f"- Visitante: {blend['probas_ml'][2] * 100:.1f}%")

# ─────────────────────────────────────────────
# SECCIÓN: ESTADÍSTICAS
# ─────────────────────────────────────────────

elif section == "Estadísticas":
    st.markdown("### Estadísticas de equipo")
    league_options = cached_league_options()
    league_name = st.selectbox("Liga", list(league_options.keys()), key="league_select_stats")
    league_id = league_options[league_name]

    teams = cached_teams_by_league(league_id=league_id, season=year_global)
    team = st.selectbox("Equipo", teams)

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
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)


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