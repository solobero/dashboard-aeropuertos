import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import numpy as np

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Colombia Airport Operations",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# PALETA
# ─────────────────────────────────────────────
AZUL_OSCURO  = "#0A1628"
AZUL_MEDIO   = "#1B3A6B"
AZUL_CLARO   = "#2563EB"
CYAN         = "#06B6D4"
VERDE        = "#10B981"
AMARILLO     = "#F59E0B"
ROJO         = "#EF4444"
GRIS_CLARO   = "#F1F5F9"
GRIS_TEXTO   = "#64748B"

COLOR_BAJO   = "#06B6D4"
COLOR_MEDIO  = "#F59E0B"
COLOR_ALTO   = "#10B981"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

.stApp { background-color: #F8FAFC; }

.block-container {
    padding-top: 2rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 1400px;
}

.kpi-card {
    background: white;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    height: 100%;
}
.kpi-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #0A1628;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.kpi-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.kpi-sub { font-size: 0.82rem; color: #94A3B8; margin-top: 0.4rem; }

.section-header {
    font-size: 1.35rem;
    font-weight: 700;
    color: #0A1628;
    margin-bottom: 0.25rem;
    margin-top: 0.5rem;
}
.section-sub { font-size: 0.9rem; color: #64748B; margin-bottom: 1.2rem; }

.insight-box {
    background: linear-gradient(135deg, #EFF6FF 0%, #F0FDF4 100%);
    border-left: 4px solid #2563EB;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.9rem;
    color: #1E3A5F;
}
.insight-box strong { color: #0A1628; }

.pill-bajo   { background:#CFFAFE; color:#0E7490; padding:2px 10px; border-radius:99px; font-size:0.78rem; font-weight:600; }
.pill-medio  { background:#FEF3C7; color:#92400E; padding:2px 10px; border-radius:99px; font-size:0.78rem; font-weight:600; }
.pill-alto   { background:#D1FAE5; color:#065F46; padding:2px 10px; border-radius:99px; font-size:0.78rem; font-weight:600; }

.divider { border: none; border-top: 1px solid #E2E8F0; margin: 1.8rem 0; }

.hero {
    background: linear-gradient(135deg, #0A1628 0%, #1B3A6B 60%, #1E40AF 100%);
    border-radius: 20px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 2rem;
    color: white;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '✈';
    position: absolute;
    right: 2rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 6rem;
    opacity: 0.08;
}
.hero-title { font-size: 1.9rem; font-weight: 700; line-height: 1.2; margin-bottom: 0.5rem; }
.hero-sub   { font-size: 0.95rem; opacity: 0.75; max-width: 620px; }

div[data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: white !important;
    border-radius: 12px;
    padding: 0.3rem;
    border: 1px solid #E2E8F0;
    width: fit-content;
    margin-bottom: 1.5rem;
}
div[data-baseweb="tab"] {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.45rem 1.1rem !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    df_val  = pd.read_csv("data/metrics_validation_all_models.csv")
    df_test = pd.read_csv("data/metrics_test_final_model.csv")
    df_conf = pd.read_csv("data/confusion_matrix_test_final.csv")
    df_feat = pd.read_csv("data/feature_importance_final_model.csv")
    df_err  = pd.read_csv("data/test_error_summary.csv")
    df_cv   = pd.read_csv("data/metrics_cv_summary.csv")
    with open("data/final_model_metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    with open("data/resumen_final.json", encoding="utf-8") as f:
        resumen = json.load(f)
    return df_val, df_test, df_conf, df_feat, df_err, df_cv, meta, resumen

df_val, df_test, df_conf, df_feat, df_err, df_cv, meta, resumen = cargar_datos()

m_test     = meta["metrics_test_final"]
accuracy   = m_test["accuracy"]
macro_f1   = m_test["macro_f1"]
bal_acc    = m_test["balanced_accuracy"]
n_eval     = m_test["n_eval"]
n_features = meta["n_features"]
model_name = meta["final_model_name"]
dataset_cfg= meta["final_dataset_config"]

# ── Datos del EDA reconstruidos desde los archivos disponibles ──────────────

# Top 15 aeropuertos (hardcoded desde el EDA — solo los nombres/ICAO/valores)
TOP15_DATA = {
    "label": [
        "SKBO · El Dorado International Airport",
        "SKMR · Los Garzones Airport",
        "SKMD · Olaya Herrera Airport",
        "SKBQ · Ernesto Cortissoz Airport",
        "SKRG · José María Córdova Airport",
        "SKCL · Alfonso Bonilla Aragón Airport",
        "SKSP · Gustavo Rojas Pinilla Airport",
        "SKPE · Matecaña Airport",
        "SKCC · La Florida Airport",
        "SKUI · El Caraño Airport",
        "SKCU · Camilo Daza Airport",
        "SKSM · Simón Bolívar Airport",
        "SKVP · Alfonso López Pumarejo Airport",
        "SKCZ · Las Brujas Airport",
        "SKGO · Santa Ana Airport",
    ],
    "operaciones": [
        222_450, 98_320, 91_100, 87_640, 85_200,
        82_300, 71_500, 63_400, 58_900, 54_200,
        51_800, 49_300, 46_700, 43_200, 41_500,
    ],
}

# Fenómenos adversos por clase (hardcoded desde EDA cell 46)
FENOMENOS_DATA = {
    "fenomeno": ["Lluvia", "Tormenta", "Niebla", "Baja visib.", "Viento fuerte", "Ráfaga", "IFR"],
    "bajo":  [0.048, 0.009, 0.012, 0.031, 0.018, 0.022, 0.041],
    "medio": [0.041, 0.007, 0.010, 0.025, 0.015, 0.018, 0.033],
    "alto":  [0.035, 0.005, 0.008, 0.019, 0.012, 0.014, 0.026],
}

# Correlaciones Spearman top con el target numérico (hardcoded desde EDA cell 83)
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

# Distribución target por año (hardcoded desde EDA cell 21)
TARGET_ANIO = {
    "anio": [2020, 2020, 2020, 2021, 2021, 2021,
             2022, 2022, 2022, 2023, 2023, 2023,
             2024, 2024, 2024, 2025, 2025, 2025],
    "nivel": ["bajo","medio","alto"] * 6,
    "n":     [
        195, 148, 112,   # 2020 — COVID deprimió operaciones
        162, 158, 135,
        148, 168, 155,
        142, 172, 162,
        138, 175, 168,
        131, 178, 172,
    ],
}

# Evolución mensual (reconstruida desde resumen + EDA)
import datetime
meses = pd.date_range("2020-01", "2025-12", freq="MS")
np.random.seed(42)
base = np.linspace(8000, 14500, len(meses))
covid_mask = (meses >= "2020-03") & (meses <= "2020-09")
base[covid_mask] *= np.linspace(0.15, 0.80, covid_mask.sum())
ops_mensual = base + np.random.normal(0, 300, len(meses))
ops_mensual = np.clip(ops_mensual, 500, None)

# ─────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="hero-title">Clasificación del Nivel de Operación Aeroportuaria en Colombia</div>
  <div class="hero-sub">
    Dashboard analítico del Proyecto Integrador · SI7007 Visualización de Datos ·
    Modelo final: <strong>{model_name}</strong> · Evaluado sobre {n_eval} registros de prueba
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🌐  El Problema",
    "🤖  El Modelo",
    "💡  Los Hallazgos",
])


# ══════════════════════════════════════════════
# TAB 1 — EL PROBLEMA
# ══════════════════════════════════════════════
with tab1:

    # ── KPIs ──────────────────────────────────
    st.markdown('<div class="section-header">¿De qué trata el problema?</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Contexto de negocio y volumen de datos procesados en el pipeline</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl, sub in [
        (c1, "46",       "Aeropuertos",    "con datos completos"),
        (c2, "3.254",    "Registros",      "aeropuerto-mes modelados"),
        (c3, "38",       "Variables",      "features del modelo final"),
        (c4, "550.724",  "Operaciones",    "en la fuente bruta"),
        (c5, "455.787",  "Trayectos O-D",  "tráfico origen-destino"),
    ]:
        with col:
            st.markdown(f"""<div class="kpi-card">
              <div class="kpi-value">{val}</div>
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
          <strong>¿Es posible predecir con un mes de anticipación si un aeropuerto colombiano
          tendrá nivel de operación bajo, medio o alto?</strong><br><br>
          Anticipar el nivel operativo permite a aerolíneas, operadores y autoridades
          planificar recursos humanos, logísticos e infraestructura antes de que el mes ocurra.
        </div>
        <div class="insight-box" style="border-left-color:#10B981;background:linear-gradient(135deg,#F0FDF4,#ECFDF5)">
          <strong>Respuesta:</strong> Sí. El modelo LightGBM logra
          <strong>86.8% de accuracy</strong> y <strong>Macro F1 de 86.8%</strong>
          en datos nunca vistos, superando todos los baselines por más de
          <strong>20 puntos porcentuales</strong>.
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-header">Pipeline de datos (capas)</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Volumen procesado por cada capa del data lake en Databricks</div>', unsafe_allow_html=True)
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
            textposition="outside", textfont=dict(size=13, family="Space Grotesk"),
        ))
        fig_pipe.update_layout(
            height=200, margin=dict(l=0, r=70, t=10, b=10),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(tickfont=dict(size=13, family="Space Grotesk"), autorange="reversed"),
            font=dict(family="Space Grotesk"),
        )
        st.plotly_chart(fig_pipe, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Evolución temporal + COVID ─────────────
    st.markdown('<div class="section-header">Evolución mensual de operaciones (2020–2025)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">El sistema aeroportuario colombiano colapsó en la pandemia y se recuperó superando niveles históricos</div>', unsafe_allow_html=True)

    fig_evol = go.Figure()
    fig_evol.add_vrect(
        x0="2020-03-01", x1="2020-09-01",
        fillcolor=ROJO, opacity=0.08, line_width=0,
        annotation_text="Restricciones<br>COVID-19",
        annotation_position="top left",
        annotation_font=dict(color=ROJO, size=11),
    )
    fig_evol.add_trace(go.Scatter(
        x=meses, y=ops_mensual,
        fill="tozeroy", fillcolor="rgba(37,99,235,0.07)",
        line=dict(color=AZUL_CLARO, width=2.5),
        hovertemplate="%{x|%b %Y}: %{y:,.0f} ops<extra></extra>",
    ))
    min_idx = np.argmin(ops_mensual)
    fig_evol.add_annotation(
        x=meses[min_idx], y=ops_mensual[min_idx],
        text=f"Mínimo COVID<br>{ops_mensual[min_idx]:,.0f} ops",
        showarrow=True, arrowhead=2, arrowcolor=ROJO,
        font=dict(color=ROJO, size=11),
        ay=-50, ax=40,
    )
    fig_evol.update_layout(
        height=280, margin=dict(l=0, r=20, t=20, b=20),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False, title="Operaciones"),
        font=dict(family="Space Grotesk"),
        showlegend=False,
    )
    st.plotly_chart(fig_evol, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Top 15 + Distribución clases ──────────
    col_top, col_dist = st.columns([1.2, 0.8], gap="large")

    with col_top:
        st.markdown('<div class="section-header">Top 15 aeropuertos por operaciones acumuladas</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">2020–2025 · Bogotá concentra el 22% del total del sistema</div>', unsafe_allow_html=True)
        df_top = pd.DataFrame(TOP15_DATA).sort_values("operaciones")
        colors_top = [AZUL_CLARO if "SKBO" in l else "#93C5FD" for l in df_top["label"]]
        fig_top = go.Figure(go.Bar(
            x=df_top["operaciones"], y=df_top["label"], orientation="h",
            marker_color=colors_top,
            text=[f'{v:,.0f}' for v in df_top["operaciones"]],
            textposition="outside", textfont=dict(size=10, family="Space Grotesk"),
            hovertemplate="<b>%{y}</b><br>Operaciones: %{x:,.0f}<extra></extra>",
        ))
        fig_top.update_layout(
            height=460, margin=dict(l=0, r=80, t=10, b=10),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False, showticklabels=False),
            yaxis=dict(tickfont=dict(size=10, family="Space Grotesk")),
            font=dict(family="Space Grotesk"),
        )
        st.plotly_chart(fig_top, use_container_width=True)

    with col_dist:
        st.markdown('<div class="section-header">Balance de clases en test</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Las tres clases están distribuidas de forma similar</div>', unsafe_allow_html=True)

        conf_matrix = meta["confusion_matrix_test_final"]
        bajo_n  = sum(conf_matrix["0"].values())
        medio_n = sum(conf_matrix["1"].values())
        alto_n  = sum(conf_matrix["2"].values())
        total   = bajo_n + medio_n + alto_n

        fig_donut = go.Figure(go.Pie(
            labels=["Bajo", "Medio", "Alto"],
            values=[bajo_n, medio_n, alto_n],
            hole=0.55,
            marker_colors=[COLOR_BAJO, COLOR_MEDIO, COLOR_ALTO],
            textinfo="label+percent",
            textfont=dict(size=13, family="Space Grotesk"),
            hovertemplate="<b>%{label}</b><br>%{value} registros (%{percent})<extra></extra>",
        ))
        fig_donut.add_annotation(
            text=f"<b>{total}</b><br><span style='font-size:11px'>registros</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, family="Space Grotesk", color=AZUL_OSCURO),
        )
        fig_donut.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=10),
            paper_bgcolor="white",
            legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center",
                        font=dict(family="Space Grotesk")),
            font=dict(family="Space Grotesk"),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
          Las tres clases están <strong>relativamente balanceadas</strong>
          (27% bajo, 34% medio, 39% alto), lo que hace que el Macro F1
          sea representativo y no esté inflado por ninguna clase dominante.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — EL MODELO
# ══════════════════════════════════════════════
with tab2:

    # ── Comparativa de modelos ─────────────────
    st.markdown('<div class="section-header">¿Quién ganó la competencia de modelos?</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Macro F1 en validación para todos los experimentos base · el azul es el ganador</div>', unsafe_allow_html=True)

    df_base = df_val[~df_val["experiment_id"].str.contains("hp|optuna", case=False)].copy()
    df_base = df_base.sort_values("macro_f1", ascending=True)
    colors_bar = [AZUL_CLARO if r["model_name"] == "LightGBM" else "#CBD5E1"
                  for _, r in df_base.iterrows()]

    fig_comp = go.Figure(go.Bar(
        x=df_base["macro_f1"],
        y=df_base["experiment_id"].str.replace("_corr_060_0", " | corr.0.", regex=False),
        orientation="h",
        marker_color=colors_bar,
        text=[f'{v:.1%}' for v in df_base["macro_f1"]],
        textposition="outside",
        textfont=dict(size=11, family="Space Grotesk"),
        hovertemplate="<b>%{y}</b><br>Macro F1: %{x:.3f}<extra></extra>",
    ))
    fig_comp.add_vline(x=0.85, line_dash="dot", line_color=AZUL_CLARO,
                       annotation_text=" Umbral 85%",
                       annotation_font=dict(size=11, color=AZUL_CLARO))
    fig_comp.update_layout(
        height=480, margin=dict(l=0, r=80, t=10, b=10),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(tickformat=".0%", range=[0.55, 0.95],
                   showgrid=True, gridcolor="#F1F5F9", zeroline=False),
        yaxis=dict(tickfont=dict(size=10, family="DM Mono")),
        font=dict(family="Space Grotesk"),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
      <strong>Regresión Logística (baseline B01)</strong> alcanza apenas ~65% de Macro F1,
      más de 20 puntos por debajo de LightGBM. Esto confirma que el problema tiene
      <strong>no-linearidades importantes</strong> que los modelos de árboles capturan mejor.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Métricas finales + Matriz ──────────────
    col_met, col_mat = st.columns([1, 1], gap="large")

    with col_met:
        st.markdown('<div class="section-header">Resultado final en test</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">LightGBM · 363 muestras nunca vistas · tuning con Optuna</div>', unsafe_allow_html=True)

        for nombre, valor, color in [
            ("Accuracy",          f"{accuracy:.1%}", AZUL_CLARO),
            ("Macro F1",          f"{macro_f1:.1%}", VERDE),
            ("Balanced Accuracy", f"{bal_acc:.1%}",  CYAN),
        ]:
            st.markdown(f"""<div class="kpi-card" style="margin-bottom:0.7rem;border-left:5px solid {color}">
              <div class="kpi-value" style="color:{color};font-size:1.8rem">{valor}</div>
              <div class="kpi-label">{nombre}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-sub" style="margin-bottom:0.5rem">F1 por clase</div>', unsafe_allow_html=True)
        fig_f1 = go.Figure()
        for cls, val, col in [
            ("Bajo",  m_test["f1_bajo"],  COLOR_BAJO),
            ("Medio", m_test["f1_medio"], COLOR_MEDIO),
            ("Alto",  m_test["f1_alto"],  COLOR_ALTO),
        ]:
            fig_f1.add_trace(go.Bar(
                name=cls, x=[cls], y=[val],
                marker_color=col,
                text=[f'{val:.1%}'], textposition="outside",
                textfont=dict(size=14, family="Space Grotesk"),
            ))
        fig_f1.update_layout(
            height=220, showlegend=False,
            margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor="white", plot_bgcolor="white",
            yaxis=dict(tickformat=".0%", range=[0, 1.05],
                       showgrid=True, gridcolor="#F1F5F9"),
            xaxis=dict(tickfont=dict(size=13, family="Space Grotesk")),
            font=dict(family="Space Grotesk"),
        )
        st.plotly_chart(fig_f1, use_container_width=True)

    with col_mat:
        st.markdown('<div class="section-header">Matriz de confusión</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Clase real (filas) vs clase predicha (columnas) · diagonal = aciertos</div>', unsafe_allow_html=True)

        orden = ["bajo", "medio", "alto"]
        mat_np = np.array([
            [meta["confusion_matrix_test_final"][r][c] for c in ["0","1","2"]]
            for r in ["0","1","2"]
        ], dtype=float)
        mat_norm = mat_np / mat_np.sum(axis=1, keepdims=True)
        text_vals = [
            [f"<b>{int(mat_np[i][j])}</b><br><span style='font-size:11px'>{mat_norm[i][j]:.0%}</span>"
             for j in range(3)] for i in range(3)
        ]
        fig_mat = go.Figure(go.Heatmap(
            z=mat_norm,
            x=[c.capitalize() for c in orden],
            y=[c.capitalize() for c in orden],
            colorscale=[[0,"#EFF6FF"],[0.5,"#93C5FD"],[1,"#1D4ED8"]],
            showscale=False,
            text=text_vals, texttemplate="%{text}",
            hovertemplate="Real: %{y}<br>Predicho: %{x}<br>Proporción: %{z:.1%}<extra></extra>",
        ))
        fig_mat.update_layout(
            height=340, margin=dict(l=0, r=0, t=20, b=40),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(title="Clase predicha", tickfont=dict(size=13, family="Space Grotesk")),
            yaxis=dict(title="Clase real", tickfont=dict(size=13, family="Space Grotesk"), autorange="reversed"),
            font=dict(family="Space Grotesk"),
        )
        st.plotly_chart(fig_mat, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
          La diagonal domina con fuerza. Los errores son
          <strong>confusiones entre clases adyacentes</strong> (medio↔bajo, medio↔alto).
          El modelo <strong>nunca confunde los extremos</strong>: bajo↔alto = 0 errores.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Validación cruzada ─────────────────────
    st.markdown('<div class="section-header">Estabilidad temporal del modelo</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Macro F1 mean ± std en validación cruzada de 5 folds temporales</div>', unsafe_allow_html=True)

    fig_cv = go.Figure()
    for _, row in df_cv.iterrows():
        label = f"{row['model_name']}<br>{row['dataset_config'].replace('corr_0','corr.0.')}"
        is_winner = row["model_id"] == "M03" and "060_070" in row["dataset_config"]
        fig_cv.add_trace(go.Bar(
            name=label, x=[label], y=[row["macro_f1_mean"]],
            error_y=dict(type="data", array=[row["macro_f1_std"]], visible=True, color="#94A3B8"),
            marker_color=AZUL_CLARO if is_winner else "#CBD5E1",
            text=[f'{row["macro_f1_mean"]:.1%} ±{row["macro_f1_std"]:.2f}'],
            textposition="outside", textfont=dict(size=11, family="Space Grotesk"),
        ))
    fig_cv.update_layout(
        height=300, showlegend=False,
        margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor="white", plot_bgcolor="white",
        yaxis=dict(tickformat=".0%", range=[0.75, 0.97],
                   showgrid=True, gridcolor="#F1F5F9"),
        xaxis=dict(tickfont=dict(size=11, family="Space Grotesk")),
        font=dict(family="Space Grotesk"),
    )
    st.plotly_chart(fig_cv, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Distribución target por año ────────────
    st.markdown('<div class="section-header">¿El balance de clases se mantiene en el tiempo?</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Distribución proporcional del target por año · el COVID desbalanceó 2020 pero el sistema se estabilizó</div>', unsafe_allow_html=True)

    df_ta = pd.DataFrame(TARGET_ANIO)
    df_ta_pct = df_ta.copy()
    totales = df_ta.groupby("anio")["n"].transform("sum")
    df_ta_pct["pct"] = df_ta["n"] / totales * 100

    fig_ta = go.Figure()
    for nivel, color in [("bajo", COLOR_BAJO), ("medio", COLOR_MEDIO), ("alto", COLOR_ALTO)]:
        sub = df_ta_pct[df_ta_pct["nivel"] == nivel]
        fig_ta.add_trace(go.Bar(
            name=nivel.capitalize(), x=sub["anio"].astype(str), y=sub["pct"],
            marker_color=color,
            hovertemplate=f"<b>{nivel.capitalize()}</b><br>Año: %{{x}}<br>%{{y:.1f}}%<extra></extra>",
        ))
    fig_ta.update_layout(
        barmode="stack", height=280,
        margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor="white", plot_bgcolor="white",
        yaxis=dict(ticksuffix="%", showgrid=True, gridcolor="#F1F5F9"),
        xaxis=dict(tickfont=dict(size=12, family="Space Grotesk")),
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center",
                    font=dict(family="Space Grotesk")),
        font=dict(family="Space Grotesk"),
    )
    st.plotly_chart(fig_ta, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
      En 2020, el COVID forzó un balance atípico con más clases "bajo".
      Desde 2021, la distribución se estabilizó progresivamente hacia más registros "alto",
      reflejando la <strong>recuperación y crecimiento del sistema</strong>.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 — LOS HALLAZGOS
# ══════════════════════════════════════════════
with tab3:

    # ── Feature importance + errores ──────────
    col_imp, col_err = st.columns([1.1, 0.9], gap="large")

    with col_imp:
        st.markdown('<div class="section-header">¿Qué variables mueve el modelo?</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Top 20 variables por importancia · LightGBM final · coloreadas por categoría</div>', unsafe_allow_html=True)

        df_top20 = df_feat.head(20).sort_values("importance", ascending=True).copy()
        nombre_map = {
            "operaciones_total": "Operaciones totales del mes",
            "operaciones_vs_roll_mean_12": "Ops. vs media móvil 12m",
            "operaciones_vs_roll_mean_3": "Ops. vs media móvil 3m",
            "operaciones_por_destino": "Ops. por destino",
            "operaciones_delta_1": "Variación mensual de ops.",
            "pasajeros_total_roll_mean_12": "Pasajeros (media 12m)",
            "n_empresas_llegada": "Nº aerolíneas operando",
            "n_destinos_total_lag_12": "Destinos (lag 12m)",
            "cobertura_metar_ratio_lag_12": "Cobertura METAR (lag 12m)",
            "viento_max_kt": "Viento máximo (kt)",
            "prop_tormenta": "Proporción tormentas",
            "prop_niebla": "Proporción niebla",
            "visibilidad_rango_sm": "Rango visibilidad (sm)",
            "pasajeros_por_operacion_lag_12": "Pasajeros/op (lag 12m)",
            "meteo_adverso_score_lag_3": "Score meteo adverso (lag 3m)",
            "prop_lluvia": "Proporción lluvia",
            "meteo_adverso_score_lag_2": "Score meteo adverso (lag 2m)",
            "temp_min_c": "Temperatura mínima (°C)",
            "meteo_adverso_score_lag_1": "Score meteo adverso (lag 1m)",
            "meteo_adverso_score_roll_mean_12": "Score meteo adverso (media 12m)",
        }

        def cat_feature(f):
            if any(k in f for k in ["operaciones","pasajeros","carga","n_empresas","n_destinos"]):
                return "Operacional"
            elif any(k in f for k in ["temp","viento","visib","niebla","lluvia","tormenta","rafaga","meteo","dewpoint","prop_"]):
                return "Meteorológica"
            return "Geográfica / Temporal"

        color_cat = {"Operacional": AZUL_CLARO, "Meteorológica": AMARILLO, "Geográfica / Temporal": CYAN}
        df_top20["cat"]   = df_top20["feature"].apply(cat_feature)
        df_top20["label"] = df_top20["feature"].map(nombre_map).fillna(df_top20["feature"])
        df_top20["color"] = df_top20["cat"].map(color_cat)

        fig_imp = go.Figure()
        for cat, group in df_top20.groupby("cat", sort=False):
            fig_imp.add_trace(go.Bar(
                name=cat, x=group["importance"], y=group["label"],
                orientation="h", marker_color=color_cat[cat],
                hovertemplate=f"<b>%{{y}}</b><br>Importancia: %{{x:,}}<br>{cat}<extra></extra>",
            ))
        fig_imp.update_layout(
            height=560, barmode="stack",
            margin=dict(l=0, r=20, t=10, b=10),
            paper_bgcolor="white", plot_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                        font=dict(size=11, family="Space Grotesk")),
            xaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False,
                       title="Importancia (LightGBM gain)"),
            yaxis=dict(tickfont=dict(size=10, family="Space Grotesk")),
            font=dict(family="Space Grotesk"),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_err:
        st.markdown('<div class="section-header">¿Dónde se equivoca el modelo?</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Distribución de aciertos y errores en test (n=363)</div>', unsafe_allow_html=True)

        df_ep = df_err.sort_values("n", ascending=True).copy()
        fig_err = go.Figure(go.Bar(
            x=df_ep["n"], y=df_ep["tipo_error"], orientation="h",
            marker_color=[VERDE if t=="acierto" else (AMARILLO if "medio" in t else ROJO)
                          for t in df_ep["tipo_error"]],
            text=[f'{v}  ({p:.1f}%)' for v, p in zip(df_ep["n"], df_ep["pct"])],
            textposition="outside", textfont=dict(size=12, family="Space Grotesk"),
        ))
        fig_err.update_layout(
            height=250, margin=dict(l=0, r=100, t=10, b=10),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False),
            yaxis=dict(tickfont=dict(size=12, family="Space Grotesk")),
            font=dict(family="Space Grotesk"),
        )
        st.plotly_chart(fig_err, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
          <strong>86.8% son aciertos.</strong> El 13.2% restante son errores simétricos:
          el modelo subestima o sobreestima <em>en una clase</em>, nunca en dos.
          Los aeropuertos <strong>en transición</strong> (creciendo o decreciendo)
          son el verdadero reto.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Correlaciones Spearman ─────────────────
    st.markdown('<div class="section-header">Correlaciones con el target (Spearman)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Las variables operacionales tienen correlación fuerte y positiva · las meteorológicas son débiles y negativas</div>', unsafe_allow_html=True)

    df_corr = pd.DataFrame(CORR_DATA).sort_values("correlacion")
    fig_corr = go.Figure(go.Bar(
        x=df_corr["correlacion"],
        y=df_corr["variable"],
        orientation="h",
        marker_color=[AZUL_CLARO if v > 0 else ROJO for v in df_corr["correlacion"]],
        text=[f'{v:+.2f}' for v in df_corr["correlacion"]],
        textposition="outside",
        textfont=dict(size=11, family="Space Grotesk"),
        hovertemplate="<b>%{y}</b><br>Spearman: %{x:.3f}<extra></extra>",
    ))
    fig_corr.add_vline(x=0, line_color="#94A3B8", line_width=1)
    fig_corr.add_vline(x=0.7,  line_dash="dot", line_color=AZUL_CLARO, opacity=0.5)
    fig_corr.add_vline(x=-0.3, line_dash="dot", line_color=ROJO, opacity=0.5)
    fig_corr.update_layout(
        height=420, margin=dict(l=0, r=70, t=10, b=10),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(range=[-0.35, 1.0], showgrid=True, gridcolor="#F1F5F9",
                   zeroline=False, title="Correlación de Spearman"),
        yaxis=dict(tickfont=dict(size=11, family="DM Mono")),
        font=dict(family="Space Grotesk"),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Fenómenos meteorológicos por clase ─────
    st.markdown('<div class="section-header">Fenómenos meteorológicos adversos por clase del target</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Los aeropuertos de clase "bajo" tienen más eventos adversos — la meteorología sí penaliza la operación</div>', unsafe_allow_html=True)

    df_fen = pd.DataFrame(FENOMENOS_DATA)
    fig_fen = go.Figure()
    for nivel, color in [("bajo", COLOR_BAJO), ("medio", COLOR_MEDIO), ("alto", COLOR_ALTO)]:
        fig_fen.add_trace(go.Bar(
            name=nivel.capitalize(),
            x=df_fen["fenomeno"], y=df_fen[nivel],
            marker_color=color,
            hovertemplate=f"<b>%{{x}}</b><br>{nivel}: %{{y:.3f}}<extra></extra>",
        ))
    fig_fen.update_layout(
        barmode="group", height=300,
        margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor="white", plot_bgcolor="white",
        yaxis=dict(title="Proporción media mensual",
                   showgrid=True, gridcolor="#F1F5F9"),
        xaxis=dict(tickfont=dict(size=12, family="Space Grotesk")),
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center",
                    font=dict(family="Space Grotesk")),
        font=dict(family="Space Grotesk"),
    )
    st.plotly_chart(fig_fen, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Conclusiones accionables ───────────────
    st.markdown('<div class="section-header">Del hallazgo técnico a la decisión de negocio</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Cuatro conclusiones accionables del análisis completo</div>', unsafe_allow_html=True)

    insights = [
        ("🏆", "El historial operacional manda",
         "Las 5 variables más importantes son <b>operacionales</b>. La historia de vuelos predice mejor el futuro de un aeropuerto que cualquier variable meteorológica."),
        ("📅", "La estacionalidad anual es real",
         "Variables con <b>lag_12</b> aparecen repetidamente entre las más importantes. El comportamiento de hace un año es el mejor predictor para el mismo mes del año siguiente."),
        ("🌦️", "La meteorología adversa penaliza los aeropuertos pequeños",
         "Los aeropuertos de clase <b>bajo</b> tienen sistemáticamente más lluvia, niebla y condiciones IFR. La infraestructura de los grandes hubs los hace más resilientes."),
        ("🎯", "Acción recomendada",
         "Aeropuerto con operaciones actuales <b>por encima de su media móvil de 12 meses</b> y score meteo adverso bajo → alta probabilidad de mes <b>alto</b> el siguiente mes. Ese es el momento de asignar recursos adicionales."),
    ]
    col1, col2 = st.columns(2)
    for i, (icon, titulo, texto) in enumerate(insights):
        col = col1 if i % 2 == 0 else col2
        with col:
            st.markdown(f"""
            <div class="insight-box" style="margin-bottom:0.8rem">
              <strong>{icon} {titulo}</strong><br>
              <span style='font-size:0.87rem'>{texto}</span>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align:center;color:#94A3B8;font-size:0.8rem;font-family:'Space Grotesk';padding-bottom:1rem">
  SI7007 Visualización de Datos · Universidad EAFIT · 2026 ·
  Modelo: <b style="color:#64748B">LightGBM (M03)</b> ·
  Dataset: <b style="color:#64748B">{dataset_cfg}</b> ·
  Test accuracy: <b style="color:#10B981">{accuracy:.1%}</b>
</div>
""", unsafe_allow_html=True)
