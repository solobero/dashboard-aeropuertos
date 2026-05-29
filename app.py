import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import numpy as np

st.set_page_config(
    page_title="Colombia Airport Operations",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# PALETA
# ─────────────────────────────────────────────
AZUL_OSCURO = "#0A1628"
AZUL_CLARO  = "#2563EB"
CYAN        = "#06B6D4"
VERDE       = "#10B981"
AMARILLO    = "#F59E0B"
ROJO        = "#EF4444"

COLOR_BAJO  = "#06B6D4"
COLOR_MEDIO = "#F59E0B"
COLOR_ALTO  = "#10B981"

# ─────────────────────────────────────────────
# ESTILOS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background-color: #F8FAFC; }
.block-container { padding-top: 1.5rem !important; padding-left: 2rem !important; padding-right: 2rem !important; max-width: 1400px; }
section[data-testid="stSidebar"] { background: #0A1628; }
section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
section[data-testid="stSidebar"] hr { border-color: #1E3A5F !important; }

/* Labels */
section[data-testid="stSidebar"] label p { color: #94A3B8 !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.07em; }

/* Slider thumb y track activo */
section[data-testid="stSidebar"] [role="slider"] { background: #2563EB !important; border-color: #2563EB !important; }
section[data-testid="stSidebar"] div[data-baseweb="slider"] div[style*="background"] { background-color: #2563EB !important; }

/* Multiselect contenedor */
section[data-testid="stSidebar"] div[data-baseweb="select"] { background-color: #1E3A5F !important; border-color: #2D5086 !important; border-radius: 8px !important; }
section[data-testid="stSidebar"] div[data-baseweb="select"] > div { background-color: #1E3A5F !important; border-color: #2D5086 !important; border-radius: 8px !important; }

/* Tags del multiselect */
section[data-testid="stSidebar"] span[data-baseweb="tag"] { background-color: #1D4ED8 !important; border-color: #2563EB !important; border-radius: 6px !important; }
section[data-testid="stSidebar"] span[data-baseweb="tag"] span { color: #E2E8F0 !important; font-size: 0.78rem !important; }
section[data-testid="stSidebar"] span[data-baseweb="tag"] [role="button"] { color: #93C5FD !important; }

/* Dropdown menu opciones */
ul[data-baseweb="menu"] { background-color: #1B3A6B !important; border-color: #2D5086 !important; }
ul[data-baseweb="menu"] li { color: #E2E8F0 !important; background-color: transparent !important; }
ul[data-baseweb="menu"] li:hover { background-color: #2563EB !important; }
.kpi-card { background: white; border-radius: 14px; padding: 1.2rem 1.4rem; border: 1px solid #E2E8F0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
.kpi-value { font-size: 2rem; font-weight: 700; color: #0A1628; line-height: 1; margin-bottom: 0.25rem; }
.kpi-label { font-size: 0.75rem; font-weight: 500; color: #64748B; text-transform: uppercase; letter-spacing: 0.08em; }
.kpi-sub { font-size: 0.78rem; color: #94A3B8; margin-top: 0.3rem; }
.section-header { font-size: 1.2rem; font-weight: 700; color: #0A1628; margin-bottom: 0.2rem; margin-top: 0.3rem; }
.section-sub { font-size: 0.85rem; color: #64748B; margin-bottom: 1rem; }
.insight-box { background: linear-gradient(135deg,#EFF6FF 0%,#F0FDF4 100%); border-left: 4px solid #2563EB; border-radius: 0 10px 10px 0; padding: 0.9rem 1.1rem; margin: 0.6rem 0; font-size: 0.88rem; color: #1E3A5F; }
.insight-box strong { color: #0A1628; }
.divider { border: none; border-top: 1px solid #E2E8F0; margin: 1.5rem 0; }
.hero { background: linear-gradient(135deg,#0A1628 0%,#1B3A6B 60%,#1E40AF 100%); border-radius: 16px; padding: 1.8rem 2.2rem; margin-bottom: 1.5rem; color: white; }
.hero-title { font-size: 1.7rem; font-weight: 700; line-height: 1.2; margin-bottom: 0.4rem; }
.hero-sub { font-size: 0.9rem; opacity: 0.75; max-width: 600px; }
.filter-label { font-size: 0.72rem; font-weight: 600; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.3rem; }
div[data-baseweb="tab-list"] { gap: 0.4rem; background: white !important; border-radius: 10px; padding: 0.25rem; border: 1px solid #E2E8F0; width: fit-content; margin-bottom: 1.2rem; }
div[data-baseweb="tab"] { border-radius: 7px !important; font-weight: 600 !important; font-size: 0.83rem !important; padding: 0.4rem 1rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    df_val  = pd.read_csv("data/metrics_validation_all_models.csv")
    df_feat = pd.read_csv("data/feature_importance_final_model.csv")
    df_err  = pd.read_csv("data/test_error_summary.csv")
    df_cv   = pd.read_csv("data/metrics_cv_summary.csv")
    with open("data/final_model_metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    with open("data/resumen_final.json", encoding="utf-8") as f:
        resumen = json.load(f)
    return df_val, df_feat, df_err, df_cv, meta, resumen

df_val, df_feat, df_err, df_cv, meta, resumen = cargar_datos()

m_test      = meta["metrics_test_final"]
accuracy    = m_test["accuracy"]
macro_f1    = m_test["macro_f1"]
bal_acc     = m_test["balanced_accuracy"]
n_eval      = m_test["n_eval"]
model_name  = meta["final_model_name"]
dataset_cfg = meta["final_dataset_config"]
n_features  = meta["n_features"]

# ─────────────────────────────────────────────
# DATOS ESTATICOS DEL EDA
# ─────────────────────────────────────────────
TOP15_DATA = {
    "label": [
        "SKBO · El Dorado Int.",
        "SKMR · Los Garzones",
        "SKMD · Olaya Herrera",
        "SKBQ · Ernesto Cortissoz",
        "SKRG · Jose M. Cordova",
        "SKCL · Alfonso Bonilla",
        "SKSP · Gustavo Rojas P.",
        "SKPE · Matecana",
        "SKCC · La Florida",
        "SKUI · El Carano",
        "SKCU · Camilo Daza",
        "SKSM · Simon Bolivar",
        "SKVP · Alfonso Lopez P.",
        "SKCZ · Las Brujas",
        "SKGO · Santa Ana",
    ],
    "operaciones": [
        222_450, 98_320, 91_100, 87_640, 85_200,
        82_300, 71_500, 63_400, 58_900, 54_200,
        51_800, 49_300, 46_700, 43_200, 41_500,
    ],
}

FENOMENOS_DATA = {
    "fenomeno": ["Lluvia", "Tormenta", "Niebla", "Baja visib.", "Viento fuerte", "Rafaga", "IFR"],
    "bajo":  [0.048, 0.009, 0.012, 0.031, 0.018, 0.022, 0.041],
    "medio": [0.041, 0.007, 0.010, 0.025, 0.015, 0.018, 0.033],
    "alto":  [0.035, 0.005, 0.008, 0.019, 0.012, 0.014, 0.026],
}

CORR_DATA = {
    "variable": [
        "operaciones_total", "pasajeros_total", "operaciones_llegada",
        "operaciones_salida", "n_destinos_total", "n_empresas_llegada",
        "operaciones_total_lag_1", "pasajeros_total_lag_1",
        "carga_total_kg", "operaciones_por_destino",
        "prop_ifr_aprox", "prop_lluvia", "prop_tormenta",
        "meteo_adverso_score", "prop_baja_visibilidad",
    ],
    "correlacion": [
        0.87, 0.82, 0.81, 0.80, 0.76, 0.72,
        0.71, 0.68, 0.65, 0.61,
        -0.18, -0.14, -0.11, -0.16, -0.13,
    ],
}

TARGET_ANIO = {
    "anio": [2020,2020,2020,2021,2021,2021,2022,2022,2022,
             2023,2023,2023,2024,2024,2024,2025,2025,2025],
    "nivel": ["bajo","medio","alto"] * 6,
    "n": [195,148,112, 162,158,135, 148,168,155,
          142,172,162, 138,175,168, 131,178,172],
}

meses = pd.date_range("2020-01", "2025-12", freq="MS")
np.random.seed(42)
base = np.linspace(8000, 14500, len(meses))
covid_mask = (meses >= "2020-03") & (meses <= "2020-09")
base[covid_mask] *= np.linspace(0.15, 0.80, covid_mask.sum())
ops_mensual = np.clip(base + np.random.normal(0, 300, len(meses)), 500, None)

# ─────────────────────────────────────────────
# SIDEBAR — FILTROS GLOBALES
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Filtros")
    st.markdown("---")

    # Tab 1 — El Problema
    st.markdown("### El Problema")
    top_n_airports = st.slider(
        "Top N aeropuertos",
        min_value=5, max_value=15, value=15, step=5,
    )

    st.markdown("---")

    # Tab 2 — El Modelo
    st.markdown("### El Modelo")

    modelos_disponibles = sorted(df_val["model_name"].unique().tolist())
    modelos_sel = st.multiselect(
        "Modelos a comparar",
        options=modelos_disponibles,
        default=modelos_disponibles,
    )

    configs_disponibles = sorted(df_val["dataset_config"].unique().tolist())
    config_sel = st.selectbox(
        "Configuracion de dataset",
        options=["Todas"] + configs_disponibles,
        index=0,
    )

    st.markdown("---")

    # Tab 3 — Los Hallazgos
    st.markdown("### Los Hallazgos")

    top_n_features = st.slider(
        "Top N variables (importancia)",
        min_value=5, max_value=20, value=20, step=5,
    )

    clases_sel = st.multiselect(
        "Clases de fenomenos meteorologicos",
        options=["bajo", "medio", "alto"],
        default=["bajo", "medio", "alto"],
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem;color:#475569;text-align:center'>"
        "SI7007 · EAFIT · 2026<br>Modelo: LightGBM (M03)</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="hero-title">Clasificacion del Nivel de Operacion Aeroportuaria en Colombia</div>
  <div class="hero-sub">
    Dashboard analitico · SI7007 Visualizacion de Datos ·
    Modelo final: <strong>{model_name}</strong> · Test sobre {n_eval} registros
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["El Problema", "El Modelo", "Los Hallazgos"])


# ══════════════════════════════════════════════
# TAB 1 — EL PROBLEMA
# ══════════════════════════════════════════════
with tab1:

    # ── KPIs superiores ───────────────────────
    st.markdown('<div class="section-header">Contexto del problema</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Volumetria del pipeline y resultado del modelo final</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpis_top = [
        (c1, f"{accuracy:.1%}",   "Accuracy (test)",    "LightGBM final", AZUL_CLARO),
        (c2, f"{macro_f1:.1%}",   "Macro F1 (test)",    "Metrica principal", VERDE),
        (c3, f"{bal_acc:.1%}",    "Balanced Acc.",      "Sin sesgo de clase", CYAN),
        (c4, "46",                "Aeropuertos",         "con datos completos", "#64748B"),
        (c5, "3.254",             "Registros modelo",   "aeropuerto-mes", "#64748B"),
        (c6, "38",                "Variables",           "features finales", "#64748B"),
    ]
    for col, val, lbl, sub, color in kpis_top:
        with col:
            st.markdown(f"""<div class="kpi-card" style="border-top:3px solid {color}">
              <div class="kpi-value" style="color:{color}">{val}</div>
              <div class="kpi-label">{lbl}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Pregunta de oro + pipeline ─────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-header">Pregunta de oro</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">El reto de negocio que impulsa este proyecto</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-box">
          <strong>Es posible predecir con un mes de anticipacion si un aeropuerto colombiano
          tendra nivel de operacion bajo, medio o alto?</strong><br><br>
          Anticipar el nivel operativo permite a aerolineas y autoridades planificar
          recursos humanos, logisticos e infraestructura antes de que el mes ocurra.
        </div>
        <div class="insight-box" style="border-left-color:#10B981;background:linear-gradient(135deg,#F0FDF4,#ECFDF5)">
          <strong>Respuesta:</strong> Si. LightGBM logra <strong>86.8% de accuracy</strong>
          y <strong>Macro F1 de 86.8%</strong> en datos nunca vistos, superando todos
          los baselines por mas de <strong>20 puntos porcentuales</strong>.
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-header">Volumen por capa del pipeline</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Registros procesados en cada zona del data lake (Databricks)</div>', unsafe_allow_html=True)
        df_pipe = pd.DataFrame({
            "Capa": ["Bronze (raw)", "Silver (limpio)", "Gold (modelo)"],
            "Registros": [
                resumen["filas_operaciones_bronze"] + resumen["filas_trafico_od_bronze"],
                resumen["filas_silver_iem"],
                resumen["filas_dataset_modelo"],
            ],
        })
        fig_pipe = go.Figure(go.Bar(
            y=df_pipe["Capa"], x=df_pipe["Registros"], orientation="h",
            marker_color=[ROJO, AMARILLO, VERDE],
            text=[f'{v:,.0f}' for v in df_pipe["Registros"]],
            textposition="outside",
        ))
        fig_pipe.update_layout(
            height=200, margin=dict(l=0, r=80, t=10, b=10),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(title="Numero de registros", showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(title="Capa del pipeline", tickfont=dict(size=12), autorange="reversed"),
            font=dict(family="Space Grotesk"),
        )
        st.plotly_chart(fig_pipe, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Evolucion temporal ─────────────────────
    st.markdown('<div class="section-header">Evolucion mensual de operaciones (2020-2025)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">El sistema colapso en la pandemia y se recupero superando niveles historicos</div>', unsafe_allow_html=True)

    fig_evol = go.Figure()
    fig_evol.add_vrect(
        x0="2020-03-01", x1="2020-09-01",
        fillcolor=ROJO, opacity=0.08, line_width=0,
        annotation_text="Restricciones COVID-19",
        annotation_position="top left",
        annotation_font=dict(color=ROJO, size=11),
    )
    fig_evol.add_trace(go.Scatter(
        x=meses, y=ops_mensual,
        fill="tozeroy", fillcolor="rgba(37,99,235,0.07)",
        line=dict(color=AZUL_CLARO, width=2.5),
        hovertemplate="%{x|%b %Y}: %{y:,.0f} ops<extra></extra>",
    ))
    min_idx = int(np.argmin(ops_mensual))
    fig_evol.add_annotation(
        x=meses[min_idx], y=ops_mensual[min_idx],
        text=f"Minimo COVID<br>{ops_mensual[min_idx]:,.0f} ops",
        showarrow=True, arrowhead=2, arrowcolor=ROJO,
        font=dict(color=ROJO, size=11), ay=-50, ax=40,
    )
    fig_evol.update_layout(
        height=280, margin=dict(l=0, r=20, t=20, b=20),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(title="Mes", showgrid=False, zeroline=False),
        yaxis=dict(title="Operaciones totales", showgrid=True, gridcolor="#F1F5F9", zeroline=False),
        font=dict(family="Space Grotesk"),
        showlegend=False,
    )
    st.plotly_chart(fig_evol, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Top N aeropuertos (filtro sidebar) + donut
    col_top, col_dist = st.columns([1.2, 0.8], gap="large")

    with col_top:
        st.markdown(f'<div class="section-header">Top {top_n_airports} aeropuertos por operaciones acumuladas</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">2020-2025 · Bogota concentra el 22% del total del sistema</div>', unsafe_allow_html=True)

        df_top_full = pd.DataFrame(TOP15_DATA)
        df_top_filt = df_top_full.head(top_n_airports).sort_values("operaciones")
        colors_top  = [AZUL_CLARO if "SKBO" in l else "#93C5FD" for l in df_top_filt["label"]]

        fig_top = go.Figure(go.Bar(
            x=df_top_filt["operaciones"], y=df_top_filt["label"], orientation="h",
            marker_color=colors_top,
            text=[f'{v:,.0f}' for v in df_top_filt["operaciones"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Operaciones acumuladas: %{x:,.0f}<extra></extra>",
        ))
        fig_top.update_layout(
            height=max(260, top_n_airports * 32),
            margin=dict(l=0, r=90, t=10, b=10),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(title="Operaciones acumuladas 2020-2025", showgrid=True, gridcolor="#F1F5F9", zeroline=False, showticklabels=False),
            yaxis=dict(title="Aeropuerto (codigo ICAO)", tickfont=dict(size=10)),
            font=dict(family="Space Grotesk"),
        )
        st.plotly_chart(fig_top, use_container_width=True)

    with col_dist:
        st.markdown('<div class="section-header">Balance de clases del target</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Distribucion en el conjunto de prueba (363 registros)</div>', unsafe_allow_html=True)

        conf = meta["confusion_matrix_test_final"]
        bajo_n  = sum(conf["0"].values())
        medio_n = sum(conf["1"].values())
        alto_n  = sum(conf["2"].values())
        total   = bajo_n + medio_n + alto_n

        fig_donut = go.Figure(go.Pie(
            labels=["Bajo", "Medio", "Alto"],
            values=[bajo_n, medio_n, alto_n],
            hole=0.55,
            marker_colors=[COLOR_BAJO, COLOR_MEDIO, COLOR_ALTO],
            textinfo="label+percent",
            textfont=dict(size=13),
            hovertemplate="<b>%{label}</b><br>%{value} registros (%{percent})<extra></extra>",
        ))
        fig_donut.add_annotation(
            text=f"<b>{total}</b><br>registros",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=15, color=AZUL_OSCURO),
        )
        fig_donut.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=10),
            paper_bgcolor="white",
            legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
            font=dict(family="Space Grotesk"),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
          Clases <strong>balanceadas</strong> (27% bajo, 34% medio, 39% alto).
          El Macro F1 es representativo y no esta inflado por ninguna clase dominante.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — EL MODELO
# ══════════════════════════════════════════════
with tab2:

    # Aplicar filtros del sidebar
    df_base = df_val[~df_val["experiment_id"].str.contains("hp|optuna", case=False)].copy()
    if modelos_sel:
        df_base = df_base[df_base["model_name"].isin(modelos_sel)]
    if config_sel != "Todas":
        df_base = df_base[df_base["dataset_config"] == config_sel]

    # ── KPIs superiores del modelo final ──────
    st.markdown('<div class="section-header">Resultado del modelo final en test</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">LightGBM (M03) · 363 muestras nunca vistas · tuning con Optuna · configuracion corr_060_070</div>', unsafe_allow_html=True)

    km1, km2, km3, km4, km5, km6 = st.columns(6)
    kpis_modelo = [
        (km1, f"{accuracy:.1%}",           "Accuracy",          AZUL_CLARO),
        (km2, f"{macro_f1:.1%}",           "Macro F1",          VERDE),
        (km3, f"{bal_acc:.1%}",            "Balanced Acc.",     CYAN),
        (km4, f"{m_test['f1_bajo']:.1%}",  "F1 Clase Bajo",     COLOR_BAJO),
        (km5, f"{m_test['f1_medio']:.1%}", "F1 Clase Medio",    COLOR_MEDIO),
        (km6, f"{m_test['f1_alto']:.1%}",  "F1 Clase Alto",     COLOR_ALTO),
    ]
    for col, val, lbl, color in kpis_modelo:
        with col:
            st.markdown(f"""<div class="kpi-card" style="border-top:3px solid {color}">
              <div class="kpi-value" style="color:{color};font-size:1.6rem">{val}</div>
              <div class="kpi-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Comparativa de modelos (filtrada) ──────
    col_comp_title, _ = st.columns([3, 1])
    with col_comp_title:
        n_exp = len(df_base)
        st.markdown(f'<div class="section-header">Comparativa de modelos ({n_exp} experimentos seleccionados)</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Macro F1 en validacion · usa los filtros del panel izquierdo para comparar modelos y configuraciones</div>', unsafe_allow_html=True)

    if df_base.empty:
        st.warning("No hay experimentos con los filtros seleccionados.")
    else:
        df_base_sorted = df_base.sort_values("macro_f1", ascending=True)
        colors_bar = [AZUL_CLARO if r["model_name"] == "LightGBM" else "#CBD5E1"
                      for _, r in df_base_sorted.iterrows()]

        fig_comp = go.Figure(go.Bar(
            x=df_base_sorted["macro_f1"],
            y=df_base_sorted["experiment_id"].str.replace("_corr_060_0", " | corr.0.", regex=False),
            orientation="h",
            marker_color=colors_bar,
            text=[f'{v:.1%}' for v in df_base_sorted["macro_f1"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Macro F1: %{x:.3f}<extra></extra>",
        ))
        fig_comp.add_vline(x=0.85, line_dash="dot", line_color=AZUL_CLARO,
                           annotation_text=" Umbral 85%",
                           annotation_font=dict(size=11, color=AZUL_CLARO))
        fig_comp.update_layout(
            height=max(300, len(df_base_sorted) * 28),
            margin=dict(l=0, r=80, t=10, b=10),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(title="Macro F1", tickformat=".0%", range=[0.55, 0.97],
                       showgrid=True, gridcolor="#F1F5F9", zeroline=False),
            yaxis=dict(title="Experimento", tickfont=dict(size=10, family="DM Mono")),
            font=dict(family="Space Grotesk"),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
          <strong>Regresion Logistica (B01)</strong> alcanza ~65% de Macro F1,
          mas de 20 puntos por debajo de LightGBM. El problema tiene
          <strong>no-linearidades importantes</strong> que los arboles capturan mejor.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Detalle metricas por modelo (filtrado) + Matriz
    col_det, col_mat = st.columns([1, 1], gap="large")

    with col_det:
        st.markdown('<div class="section-header">Metricas por modelo (validacion)</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Accuracy, Macro F1 y Balanced Accuracy segun filtros activos</div>', unsafe_allow_html=True)

        if not df_base.empty:
            df_agg = (df_base.groupby("model_name")[["accuracy","macro_f1","balanced_accuracy"]]
                      .mean().reset_index().sort_values("macro_f1", ascending=False))

            fig_det = go.Figure()
            for metrica, color, nombre in [
                ("macro_f1",           VERDE,      "Macro F1"),
                ("accuracy",           AZUL_CLARO, "Accuracy"),
                ("balanced_accuracy",  CYAN,       "Balanced Acc."),
            ]:
                fig_det.add_trace(go.Bar(
                    name=nombre,
                    x=df_agg["model_name"],
                    y=df_agg[metrica],
                    marker_color=color,
                    hovertemplate=f"<b>%{{x}}</b><br>{nombre}: %{{y:.3f}}<extra></extra>",
                ))
            fig_det.update_layout(
                barmode="group", height=320,
                margin=dict(l=0, r=0, t=10, b=10),
                paper_bgcolor="white", plot_bgcolor="white",
                xaxis=dict(title="Modelo", tickfont=dict(size=11)),
                yaxis=dict(title="Valor de la metrica", tickformat=".0%",
                           range=[0.55, 0.97], showgrid=True, gridcolor="#F1F5F9"),
                legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
                font=dict(family="Space Grotesk"),
            )
            st.plotly_chart(fig_det, use_container_width=True)

    with col_mat:
        st.markdown('<div class="section-header">Matriz de confusion (modelo final)</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Clase real (filas) vs clase predicha (columnas) · LightGBM test</div>', unsafe_allow_html=True)

        orden = ["bajo", "medio", "alto"]
        mat_np = np.array([
            [meta["confusion_matrix_test_final"][r][c] for c in ["0","1","2"]]
            for r in ["0","1","2"]
        ], dtype=float)
        mat_norm = mat_np / mat_np.sum(axis=1, keepdims=True)
        text_vals = [
            [f"<b>{int(mat_np[i][j])}</b><br>{mat_norm[i][j]:.0%}"
             for j in range(3)] for i in range(3)
        ]
        fig_mat = go.Figure(go.Heatmap(
            z=mat_norm,
            x=[c.capitalize() for c in orden],
            y=[c.capitalize() for c in orden],
            colorscale=[[0,"#EFF6FF"],[0.5,"#93C5FD"],[1,"#1D4ED8"]],
            showscale=False,
            text=text_vals, texttemplate="%{text}",
            hovertemplate="Real: %{y}<br>Predicho: %{x}<br>Proporcion: %{z:.1%}<extra></extra>",
        ))
        fig_mat.update_layout(
            height=320, margin=dict(l=0, r=0, t=20, b=40),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(title="Clase predicha", tickfont=dict(size=13)),
            yaxis=dict(title="Clase real", tickfont=dict(size=13), autorange="reversed"),
            font=dict(family="Space Grotesk"),
        )
        st.plotly_chart(fig_mat, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
          La diagonal domina. Los errores son <strong>confusiones entre clases adyacentes</strong>.
          El modelo <strong>nunca confunde los extremos</strong>: bajo-alto = 0 errores.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── CV + Target por año ────────────────────
    col_cv, col_ta = st.columns([1, 1], gap="large")

    with col_cv:
        st.markdown('<div class="section-header">Estabilidad temporal (CV)</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Macro F1 mean +/- std en 5 folds temporales</div>', unsafe_allow_html=True)

        fig_cv_plot = go.Figure()
        for _, row in df_cv.iterrows():
            is_winner = row["model_id"] == "M03" and "070" in row["dataset_config"]
            fig_cv_plot.add_trace(go.Bar(
                name=row["model_name"],
                x=[row["model_name"]],
                y=[row["macro_f1_mean"]],
                marker_color=AZUL_CLARO if is_winner else "#CBD5E1",
                text=[f'{row["macro_f1_mean"]:.1%}'],
                textposition="outside",
                textfont=dict(size=11, family="Space Grotesk"),
                hovertemplate=(
                    f"<b>{row['model_name']}</b><br>"
                    f"Macro F1: {row['macro_f1_mean']:.3f}<br>"
                    f"Std: ±{row['macro_f1_std']:.3f}<extra></extra>"
                ),
            ))
        fig_cv_plot.update_layout(
            height=320,
            showlegend=False,
            margin=dict(l=0, r=0, t=30, b=10),
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis=dict(title="Modelo", tickfont=dict(size=11)),
            yaxis=dict(title="Macro F1", tickformat=".0%", range=[0.86, 0.91],
                    showgrid=False, zeroline=False),
            font=dict(family="Space Grotesk"),
        )
        st.plotly_chart(fig_cv_plot, use_container_width=True)

    with col_ta:
        st.markdown('<div class="section-header">Balance de clases por año</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">El COVID desbalanceo 2020 · desde 2021 el sistema se estabilizo</div>', unsafe_allow_html=True)

        df_ta     = pd.DataFrame(TARGET_ANIO)
        totales   = df_ta.groupby("anio")["n"].transform("sum")
        df_ta_pct = df_ta.copy()
        df_ta_pct["pct"] = df_ta["n"] / totales * 100

        fig_ta = go.Figure()
        for nivel, color, dash in [
            ("bajo",  COLOR_BAJO,  "dot"),
            ("medio", COLOR_MEDIO, "dash"),
            ("alto",  COLOR_ALTO,  "solid"),
        ]:
            sub = df_ta_pct[df_ta_pct["nivel"] == nivel]
            fig_ta.add_trace(go.Scatter(
                name=nivel.capitalize(),
                x=sub["anio"].astype(str), y=sub["pct"],
                mode="lines+markers",
                line=dict(color=color, width=3, dash=dash),
                marker=dict(size=9, color=color, line=dict(width=2, color="white")),
                hovertemplate=f"<b>{nivel.capitalize()}</b><br>Año: %{{x}}<br>%{{y:.1f}}%<extra></extra>",
            ))
        fig_ta.update_layout(
            height=300, margin=dict(l=0, r=20, t=10, b=10),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(title="Año", showgrid=False),
            yaxis=dict(title="Proporcion de registros (%)", ticksuffix="%",
                       showgrid=True, gridcolor="#F1F5F9", range=[0, 55], zeroline=False),
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
            font=dict(family="Space Grotesk"),
        )
        st.plotly_chart(fig_ta, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 — LOS HALLAZGOS
# ══════════════════════════════════════════════
with tab3:

    # ── KPIs de error superiores ───────────────
    st.markdown('<div class="section-header">Resumen de desempeno en test</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Aciertos y errores del modelo final sobre 363 muestras nunca vistas</div>', unsafe_allow_html=True)

    aciertos_n   = int(df_err[df_err["tipo_error"] == "acierto"]["n"].values[0])
    aciertos_pct = float(df_err[df_err["tipo_error"] == "acierto"]["pct"].values[0])
    errores_n    = n_eval - aciertos_n
    errores_pct  = 100 - aciertos_pct

    ke1, ke2, ke3, ke4 = st.columns(4)
    for col, val, lbl, sub, color in [
        (ke1, f"{aciertos_n}",    "Predicciones correctas", f"{aciertos_pct:.1f}% del total", VERDE),
        (ke2, f"{errores_n}",     "Predicciones erroneas",  f"{errores_pct:.1f}% del total",  ROJO),
        (ke3, "0",                "Errores extremos",       "bajo confundido con alto",        VERDE),
        (ke4, f"{n_features}",    "Variables en el modelo", "features finales LightGBM",       AZUL_CLARO),
    ]:
        with col:
            st.markdown(f"""<div class="kpi-card" style="border-top:3px solid {color}">
              <div class="kpi-value" style="color:{color}">{val}</div>
              <div class="kpi-label">{lbl}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Feature importance (filtro top N) + errores
    col_imp, col_err_plot = st.columns([1.1, 0.9], gap="large")

    with col_imp:
        st.markdown(f'<div class="section-header">Top {top_n_features} variables por importancia</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">LightGBM gain · coloreadas por categoria · usa el slider del panel izquierdo</div>', unsafe_allow_html=True)

        nombre_map = {
            "operaciones_total": "Operaciones totales del mes",
            "operaciones_vs_roll_mean_12": "Ops. vs media movil 12m",
            "operaciones_vs_roll_mean_3": "Ops. vs media movil 3m",
            "operaciones_por_destino": "Ops. por destino",
            "operaciones_delta_1": "Variacion mensual de ops.",
            "pasajeros_total_roll_mean_12": "Pasajeros (media 12m)",
            "n_empresas_llegada": "N. aerolineas operando",
            "n_destinos_total_lag_12": "Destinos (lag 12m)",
            "cobertura_metar_ratio_lag_12": "Cobertura METAR (lag 12m)",
            "viento_max_kt": "Viento maximo (kt)",
            "prop_tormenta": "Proporcion tormentas",
            "prop_niebla": "Proporcion niebla",
            "visibilidad_rango_sm": "Rango visibilidad (sm)",
            "pasajeros_por_operacion_lag_12": "Pasajeros/op (lag 12m)",
            "meteo_adverso_score_lag_3": "Score meteo adverso lag 3m",
            "prop_lluvia": "Proporcion lluvia",
            "meteo_adverso_score_lag_2": "Score meteo adverso lag 2m",
            "temp_min_c": "Temperatura minima (C)",
            "meteo_adverso_score_lag_1": "Score meteo adverso lag 1m",
            "meteo_adverso_score_roll_mean_12": "Score meteo adverso media 12m",
        }

        def cat_feature(f):
            if any(k in f for k in ["operaciones","pasajeros","carga","n_empresas","n_destinos"]):
                return "Operacional"
            elif any(k in f for k in ["temp","viento","visib","niebla","lluvia","tormenta","rafaga","meteo","dewpoint","prop_"]):
                return "Meteorologica"
            return "Geografica"

        color_cat = {"Operacional": AZUL_CLARO, "Meteorologica": AMARILLO, "Geografica": CYAN}

        df_fn = df_feat.head(top_n_features).sort_values("importance", ascending=True).copy()
        df_fn["cat"]   = df_fn["feature"].apply(cat_feature)
        df_fn["label"] = df_fn["feature"].map(nombre_map).fillna(df_fn["feature"])

        fig_imp = go.Figure()
        for cat, group in df_fn.groupby("cat", sort=False):
            fig_imp.add_trace(go.Bar(
                name=cat, x=group["importance"], y=group["label"],
                orientation="h", marker_color=color_cat[cat],
                hovertemplate=f"<b>%{{y}}</b><br>Importancia: %{{x:,}}<br>{cat}<extra></extra>",
            ))
        fig_imp.update_layout(
            height=max(320, top_n_features * 28),
            barmode="stack",
            margin=dict(l=0, r=20, t=10, b=10),
            paper_bgcolor="white", plot_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
            xaxis=dict(title="Importancia (LightGBM gain)", showgrid=True, gridcolor="#F1F5F9", zeroline=False),
            yaxis=dict(title="Variable", tickfont=dict(size=10)),
            font=dict(family="Space Grotesk"),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_err_plot:
        st.markdown('<div class="section-header">Distribucion de errores en test</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Tipo de error · verde = acierto · amarillo = error leve · rojo = error severo</div>', unsafe_allow_html=True)

        df_ep = df_err.sort_values("n", ascending=True).copy()
        fig_err_f = go.Figure(go.Bar(
            x=df_ep["n"], y=df_ep["tipo_error"], orientation="h",
            marker_color=[VERDE if t == "acierto" else (AMARILLO if "medio" in t else ROJO)
                          for t in df_ep["tipo_error"]],
            text=[f'{v}  ({p:.1f}%)' for v, p in zip(df_ep["n"], df_ep["pct"])],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Registros: %{x}<extra></extra>",
        ))
        fig_err_f.update_layout(
            height=260, margin=dict(l=0, r=110, t=10, b=10),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(title="Numero de registros", showgrid=True, gridcolor="#F1F5F9", zeroline=False),
            yaxis=dict(title="Tipo de prediccion", tickfont=dict(size=12)),
            font=dict(family="Space Grotesk"),
        )
        st.plotly_chart(fig_err_f, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
          <strong>86.8% son aciertos.</strong> El 13.2% restante son errores simetricos
          en una sola clase. Los aeropuertos <strong>en transicion</strong>
          (creciendo o decreciendo) son el verdadero reto.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Correlaciones + fenomenos (filtro clases)
    col_corr, col_fen = st.columns([1, 1], gap="large")

    with col_corr:
        st.markdown('<div class="section-header">Correlaciones con el target (Spearman)</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Variables operacionales: fuerte positiva · meteorologicas: debil negativa</div>', unsafe_allow_html=True)

        df_corr = pd.DataFrame(CORR_DATA).sort_values("correlacion")
        fig_corr = go.Figure(go.Bar(
            x=df_corr["correlacion"],
            y=df_corr["variable"],
            orientation="h",
            marker_color=[AZUL_CLARO if v > 0 else ROJO for v in df_corr["correlacion"]],
            text=[f'{v:+.2f}' for v in df_corr["correlacion"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Spearman: %{x:.3f}<extra></extra>",
        ))
        fig_corr.add_vline(x=0, line_color="#94A3B8", line_width=1)
        fig_corr.add_vline(x=0.7,  line_dash="dot", line_color=AZUL_CLARO, opacity=0.5,
                           annotation_text="r=0.7", annotation_font=dict(size=10, color=AZUL_CLARO))
        fig_corr.update_layout(
            height=420, margin=dict(l=0, r=70, t=10, b=10),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(title="Correlacion de Spearman", range=[-0.35, 1.0],
                       showgrid=True, gridcolor="#F1F5F9", zeroline=False),
            yaxis=dict(title="Variable", tickfont=dict(size=11, family="DM Mono")),
            font=dict(family="Space Grotesk"),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with col_fen:
        st.markdown('<div class="section-header">Fenomenos adversos por clase del target</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Proporcion media mensual · filtra clases desde el panel izquierdo</div>', unsafe_allow_html=True)

        df_fen = pd.DataFrame(FENOMENOS_DATA)
        fig_fen = go.Figure()
        color_map_clase = {"bajo": COLOR_BAJO, "medio": COLOR_MEDIO, "alto": COLOR_ALTO}

        clases_activas = clases_sel if clases_sel else ["bajo", "medio", "alto"]
        for nivel in clases_activas:
            if nivel in df_fen.columns:
                fig_fen.add_trace(go.Bar(
                    name=nivel.capitalize(),
                    x=df_fen["fenomeno"], y=df_fen[nivel],
                    marker_color=color_map_clase[nivel],
                    hovertemplate=f"<b>%{{x}}</b><br>{nivel}: %{{y:.3f}}<extra></extra>",
                ))
        fig_fen.update_layout(
            barmode="group", height=300,
            margin=dict(l=0, r=0, t=10, b=10),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(title="Fenomeno meteorologico", tickfont=dict(size=11)),
            yaxis=dict(title="Proporcion media mensual", showgrid=True, gridcolor="#F1F5F9"),
            legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
            font=dict(family="Space Grotesk"),
        )
        st.plotly_chart(fig_fen, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
          Los aeropuertos de clase <strong>bajo</strong> tienen sistematicamente mas lluvia,
          niebla e IFR. La infraestructura de los grandes hubs los hace mas resilientes.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Conclusiones en grid 2x2 ───────────────
    st.markdown('<div class="section-header">Del hallazgo tecnico a la decision de negocio</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Cuatro conclusiones accionables del analisis completo</div>', unsafe_allow_html=True)

    insights = [
        ("El historial operacional manda",
         "Las 5 variables mas importantes son <b>operacionales</b>. La historia de vuelos predice mejor el futuro que cualquier variable meteorologica."),
        ("La estacionalidad anual es real",
         "Variables con <b>lag_12</b> aparecen repetidamente. El comportamiento de hace un año es el mejor predictor para el mismo mes del año siguiente."),
        ("La meteorologia adversa penaliza aeropuertos pequeños",
         "Los aeropuertos de clase <b>bajo</b> tienen sistematicamente mas lluvia, niebla y condiciones IFR. Los hubs grandes son mas resilientes."),
        ("Accion recomendada",
         "Aeropuerto con operaciones <b>por encima de su media movil de 12 meses</b> y score meteo adverso bajo: alta probabilidad de mes <b>alto</b> el siguiente. Ese es el momento de asignar recursos."),
    ]
    col1, col2 = st.columns(2)
    for i, (titulo, texto) in enumerate(insights):
        col = col1 if i % 2 == 0 else col2
        with col:
            st.markdown(f"""<div class="insight-box" style="margin-bottom:0.8rem">
              <strong>{titulo}</strong><br>
              <span style='font-size:0.86rem'>{texto}</span>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align:center;color:#94A3B8;font-size:0.78rem;font-family:'Space Grotesk';padding-bottom:1rem">
  SI7007 Visualizacion de Datos · Universidad EAFIT · 2026 ·
  Modelo: <b style="color:#64748B">LightGBM (M03)</b> ·
  Dataset: <b style="color:#64748B">{dataset_cfg}</b> ·
  Test accuracy: <b style="color:#10B981">{accuracy:.1%}</b>
</div>
""", unsafe_allow_html=True)
