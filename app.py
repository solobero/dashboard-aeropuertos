import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json

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
# PALETA Y ESTILOS
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
BLANCO       = "#FFFFFF"

COLOR_BAJO   = "#06B6D4"   # cyan
COLOR_MEDIO  = "#F59E0B"   # amber
COLOR_ALTO   = "#10B981"   # green

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* Fondo general */
.stApp {
    background-color: #F8FAFC;
}

/* Quitar padding arriba */
.block-container {
    padding-top: 2rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 1400px;
}

/* Tarjetas KPI */
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
.kpi-sub {
    font-size: 0.82rem;
    color: #94A3B8;
    margin-top: 0.4rem;
}

/* Encabezado de sección */
.section-header {
    font-size: 1.35rem;
    font-weight: 700;
    color: #0A1628;
    margin-bottom: 0.25rem;
    margin-top: 0.5rem;
}
.section-sub {
    font-size: 0.9rem;
    color: #64748B;
    margin-bottom: 1.2rem;
}

/* Insight destacado */
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

/* Pill de clase */
.pill-bajo   { background:#CFFAFE; color:#0E7490; padding:2px 10px; border-radius:99px; font-size:0.78rem; font-weight:600; }
.pill-medio  { background:#FEF3C7; color:#92400E; padding:2px 10px; border-radius:99px; font-size:0.78rem; font-weight:600; }
.pill-alto   { background:#D1FAE5; color:#065F46; padding:2px 10px; border-radius:99px; font-size:0.78rem; font-weight:600; }

/* Separador */
.divider { border: none; border-top: 1px solid #E2E8F0; margin: 1.8rem 0; }

/* Hero banner */
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
.hero-title {
    font-size: 1.9rem;
    font-weight: 700;
    line-height: 1.2;
    margin-bottom: 0.5rem;
}
.hero-sub {
    font-size: 0.95rem;
    opacity: 0.75;
    max-width: 620px;
}

/* Tab estilo custom */
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
# CARGA DE DATOS (con cache)
# ─────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    df_val   = pd.read_csv("data/metrics_validation_all_models.csv")
    df_test  = pd.read_csv("data/metrics_test_final_model.csv")
    df_conf  = pd.read_csv("data/confusion_matrix_test_final.csv")
    df_feat  = pd.read_csv("data/feature_importance_final_model.csv")
    df_err   = pd.read_csv("data/test_error_summary.csv")
    df_cv    = pd.read_csv("data/metrics_cv_summary.csv")

    with open("data/final_model_metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    with open("data/resumen_final.json", encoding="utf-8") as f:
        resumen = json.load(f)

    return df_val, df_test, df_conf, df_feat, df_err, df_cv, meta, resumen

df_val, df_test, df_conf, df_feat, df_err, df_cv, meta, resumen = cargar_datos()

# Extraer métricas clave
m_test      = meta["metrics_test_final"]
accuracy    = m_test["accuracy"]
macro_f1    = m_test["macro_f1"]
bal_acc     = m_test["balanced_accuracy"]
n_eval      = m_test["n_eval"]
n_features  = meta["n_features"]
model_name  = meta["final_model_name"]
dataset_cfg = meta["final_dataset_config"]


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
# NAVEGACIÓN POR PESTAÑAS
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

    st.markdown('<div class="section-header">¿De qué trata el problema?</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Contexto de negocio y volumen de datos procesados</div>', unsafe_allow_html=True)

    # KPIs de contexto
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, "46",      "Aeropuertos",       "con datos completos"),
        (c2, "3.254",   "Registros",         "aeropuerto-mes modelados"),
        (c3, "38",      "Variables",         "features del modelo final"),
        (c4, "550.724", "Operaciones",       "en la fuente bruta (bronze)"),
        (c5, "455.787", "Trayectos O-D",     "tráfico origen-destino"),
    ]
    for col, val, lbl, sub in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-value">{val}</div>
              <div class="kpi-label">{lbl}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-header">Pregunta de oro</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">El reto de negocio que impulsa este proyecto</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box">
          <strong>¿Es posible predecir con un mes de anticipación si un aeropuerto colombiano
          tendrá nivel de operación bajo, medio o alto?</strong><br><br>
          Anticipar el nivel operativo permite a aerolíneas, operadores y autoridades aeronáuticas
          planificar recursos humanos, logísticos y de infraestructura antes de que el mes ocurra,
          reduciendo costos de sobreoperación o suboperación.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box" style="border-left-color: #10B981; background: linear-gradient(135deg,#F0FDF4,#ECFDF5)">
          <strong>Respuesta:</strong> Sí. El modelo LightGBM logra
          <strong>86.8 % de accuracy</strong> y un
          <strong>Macro F1 de 86.8 %</strong> en datos que nunca vio,
          superando todos los baselines por más de 20 puntos porcentuales.
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-header">Arquitectura de datos (pipeline)</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Capas del data lake en Databricks</div>', unsafe_allow_html=True)

        pipeline_data = {
            "Capa": ["Bronze (raw)", "Silver (limpio)", "Gold (modelo)"],
            "Registros": [
                resumen["filas_operaciones_bronze"] + resumen["filas_trafico_od_bronze"],
                resumen["filas_silver_iem"],
                resumen["filas_dataset_modelo"],
            ],
            "Descripción": [
                "Operaciones aéreas + Tráfico O-D",
                "METAR históricos (IEM)",
                "Dataset final aeropuerto-mes",
            ],
        }
        df_pipeline = pd.DataFrame(pipeline_data)

        fig_pipe = go.Figure(go.Bar(
            y=df_pipeline["Capa"],
            x=df_pipeline["Registros"],
            orientation="h",
            marker_color=[ROJO, AMARILLO, VERDE],
            text=[f'{v:,.0f}' for v in df_pipeline["Registros"]],
            textposition="outside",
            textfont=dict(size=13, family="Space Grotesk"),
        ))
        fig_pipe.update_layout(
            height=220,
            margin=dict(l=0, r=60, t=10, b=10),
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(tickfont=dict(size=13, family="Space Grotesk"), autorange="reversed"),
            font=dict(family="Space Grotesk"),
        )
        st.plotly_chart(fig_pipe, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Distribución del target
    st.markdown('<div class="section-header">¿Qué tan balanceadas son las clases?</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Distribución de la variable objetivo en el conjunto de prueba (363 registros)</div>', unsafe_allow_html=True)

    conf_matrix = meta["confusion_matrix_test_final"]
    bajo_n  = sum(conf_matrix["0"].values())
    medio_n = sum(conf_matrix["1"].values())
    alto_n  = sum(conf_matrix["2"].values())
    total   = bajo_n + medio_n + alto_n

    col_b, col_m, col_a, col_txt = st.columns([1, 1, 1, 2])
    for col, label, n, pill_class, color in [
        (col_b, "Bajo",  bajo_n,  "pill-bajo",  COLOR_BAJO),
        (col_m, "Medio", medio_n, "pill-medio", COLOR_MEDIO),
        (col_a, "Alto",  alto_n,  "pill-alto",  COLOR_ALTO),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid {color}">
              <div class="kpi-value" style="color:{color}">{n}</div>
              <div class="kpi-label">Clase <span class="{pill_class}">{label}</span></div>
              <div class="kpi-sub">{n/total:.1%} del test set</div>
            </div>""", unsafe_allow_html=True)

    with col_txt:
        st.markdown("""
        <div class="insight-box" style="margin-top:0.2rem">
          Las tres clases están <strong>relativamente balanceadas</strong>
          (bajo 27%, medio 34%, alto 39%), lo que hace que el Macro F1 sea
          una métrica representativa y no esté inflada por una clase dominante.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — EL MODELO
# ══════════════════════════════════════════════
with tab2:

    st.markdown('<div class="section-header">¿Quién ganó la competencia de modelos?</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Macro F1 en validación para todos los experimentos base (sin tuning de hiperparámetros)</div>', unsafe_allow_html=True)

    # Filtrar solo experimentos base (sin optuna/hp)
    df_base = df_val[~df_val["experiment_id"].str.contains("hp|optuna", case=False)].copy()
    df_base = df_base.sort_values("macro_f1", ascending=True)

    # Colorear el ganador
    colors_bar = [
        "#2563EB" if row["model_name"] == "LightGBM" else "#CBD5E1"
        for _, row in df_base.iterrows()
    ]

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
    fig_comp.add_vline(
        x=0.85,
        line_dash="dot",
        line_color=AZUL_CLARO,
        annotation_text=" Umbral 85%",
        annotation_font_size=11,
        annotation_font_color=AZUL_CLARO,
    )
    fig_comp.update_layout(
        height=480,
        margin=dict(l=0, r=80, t=10, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(
            tickformat=".0%",
            range=[0.55, 0.95],
            showgrid=True,
            gridcolor="#F1F5F9",
            zeroline=False,
        ),
        yaxis=dict(tickfont=dict(size=10, family="DM Mono")),
        font=dict(family="Space Grotesk"),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
      <strong>Regresión Logística (baseline B01)</strong> alcanza apenas ~64–67 % de Macro F1 —
      más de 20 puntos por debajo de LightGBM.
      Esto confirma que el problema tiene <strong>no-linearidades importantes</strong>
      que los modelos basados en árboles capturan mejor.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Métricas finales + Matriz de confusión ──
    col_met, col_mat = st.columns([1, 1], gap="large")

    with col_met:
        st.markdown('<div class="section-header">Resultado final en test</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">LightGBM · 363 muestras nunca vistas · tuning con Optuna</div>', unsafe_allow_html=True)

        metricas_globales = [
            ("Accuracy",          f"{accuracy:.1%}",  AZUL_CLARO),
            ("Macro F1",          f"{macro_f1:.1%}",  VERDE),
            ("Balanced Accuracy", f"{bal_acc:.1%}",   CYAN),
        ]
        for nombre, valor, color in metricas_globales:
            st.markdown(f"""
            <div class="kpi-card" style="margin-bottom:0.7rem; border-left: 5px solid {color}; display:flex; align-items:center; gap:1rem;">
              <div>
                <div class="kpi-value" style="color:{color}; font-size:1.8rem">{valor}</div>
                <div class="kpi-label">{nombre}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-sub" style="margin-bottom:0.5rem">F1 por clase</div>', unsafe_allow_html=True)

        clases = ["bajo", "medio", "alto"]
        f1_vals = [m_test["f1_bajo"], m_test["f1_medio"], m_test["f1_alto"]]
        colors_f1 = [COLOR_BAJO, COLOR_MEDIO, COLOR_ALTO]

        fig_f1 = go.Figure()
        for cls, val, col in zip(clases, f1_vals, colors_f1):
            fig_f1.add_trace(go.Bar(
                name=cls.capitalize(),
                x=[cls.capitalize()],
                y=[val],
                marker_color=col,
                text=[f'{val:.1%}'],
                textposition="outside",
                textfont=dict(size=14, family="Space Grotesk", color="#0A1628"),
            ))
        fig_f1.update_layout(
            height=220,
            showlegend=False,
            margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor="white",
            plot_bgcolor="white",
            yaxis=dict(tickformat=".0%", range=[0, 1.05], showgrid=True, gridcolor="#F1F5F9"),
            xaxis=dict(tickfont=dict(size=13, family="Space Grotesk")),
            font=dict(family="Space Grotesk"),
        )
        st.plotly_chart(fig_f1, use_container_width=True)

    with col_mat:
        st.markdown('<div class="section-header">Matriz de confusión</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Clase real (filas) vs clase predicha (columnas)</div>', unsafe_allow_html=True)

        orden = ["bajo", "medio", "alto"]
        label_map = {"0": "bajo", "1": "medio", "2": "alto"}
        id_map    = {"bajo": "0", "medio": "1", "alto": "2"}

        matriz = [[0]*3 for _ in range(3)]
        for i, real in enumerate(["0","1","2"]):
            for j, pred in enumerate(["0","1","2"]):
                matriz[i][j] = meta["confusion_matrix_test_final"][real][pred]

        import numpy as np
        mat_np = np.array(matriz, dtype=float)
        row_sums = mat_np.sum(axis=1, keepdims=True)
        mat_norm = mat_np / row_sums

        text_vals = [[f"<b>{int(mat_np[i][j])}</b><br><span style='font-size:11px'>{mat_norm[i][j]:.0%}</span>"
                      for j in range(3)] for i in range(3)]

        fig_mat = go.Figure(go.Heatmap(
            z=mat_norm,
            x=[c.capitalize() for c in orden],
            y=[c.capitalize() for c in orden],
            colorscale=[[0, "#EFF6FF"], [0.5, "#93C5FD"], [1, "#1D4ED8"]],
            showscale=False,
            text=text_vals,
            texttemplate="%{text}",
            hovertemplate="Real: %{y}<br>Predicho: %{x}<br>Proporción: %{z:.1%}<extra></extra>",
        ))
        fig_mat.update_layout(
            height=340,
            margin=dict(l=0, r=0, t=20, b=40),
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis=dict(title="Clase predicha", tickfont=dict(size=13, family="Space Grotesk")),
            yaxis=dict(title="Clase real", tickfont=dict(size=13, family="Space Grotesk"), autorange="reversed"),
            font=dict(family="Space Grotesk"),
        )
        st.plotly_chart(fig_mat, use_container_width=True)

        st.markdown("""
        <div class="insight-box" style="margin-top:-0.5rem">
          La diagonal domina con fuerza. Los errores restantes son
          <strong>confusiones entre clases adyacentes</strong> (medio↔bajo, medio↔alto),
          nunca saltos extremos (bajo↔alto: 0 errores). El modelo
          <strong>nunca confunde los extremos</strong>.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Validación cruzada
    st.markdown('<div class="section-header">Estabilidad del modelo (validación cruzada)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Mean ± Std del Macro F1 en 5 folds temporales</div>', unsafe_allow_html=True)

    fig_cv = go.Figure()
    for _, row in df_cv.iterrows():
        label = f"{row['model_name']}<br>{row['dataset_config'].replace('corr_0','corr.0.')}"
        is_winner = row["model_id"] == "M03" and "060_070" in row["dataset_config"]
        color = AZUL_CLARO if is_winner else "#CBD5E1"
        fig_cv.add_trace(go.Bar(
            name=label,
            x=[label],
            y=[row["macro_f1_mean"]],
            error_y=dict(type="data", array=[row["macro_f1_std"]], visible=True, color="#94A3B8"),
            marker_color=color,
            text=[f'{row["macro_f1_mean"]:.1%} ±{row["macro_f1_std"]:.2f}'],
            textposition="outside",
            textfont=dict(size=11, family="Space Grotesk"),
        ))
    fig_cv.update_layout(
        height=300,
        showlegend=False,
        margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        yaxis=dict(tickformat=".0%", range=[0.75, 0.97], showgrid=True, gridcolor="#F1F5F9"),
        xaxis=dict(tickfont=dict(size=11, family="Space Grotesk")),
        font=dict(family="Space Grotesk"),
    )
    st.plotly_chart(fig_cv, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 — LOS HALLAZGOS
# ══════════════════════════════════════════════
with tab3:

    col_imp, col_err = st.columns([1.1, 0.9], gap="large")

    with col_imp:
        st.markdown('<div class="section-header">¿Qué variables mueve el modelo?</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Top 20 variables por importancia · LightGBM final</div>', unsafe_allow_html=True)

        df_top20 = df_feat.head(20).sort_values("importance", ascending=True).copy()

        # Categorizar variables para color
        def categorizar(feature):
            if "operaciones" in feature or "pasajeros" in feature or "carga" in feature or "n_empresas" in feature or "n_destinos" in feature:
                return "Operacional"
            elif any(k in feature for k in ["temp","viento","visib","niebla","lluvia","tormenta","rafaga","meteo","dewpoint","prop_"]):
                return "Meteorológica"
            else:
                return "Geográfica / Temporal"

        df_top20["categoria"] = df_top20["feature"].apply(categorizar)
        color_map = {"Operacional": AZUL_CLARO, "Meteorológica": AMARILLO, "Geográfica / Temporal": CYAN}
        df_top20["color"] = df_top20["categoria"].map(color_map)

        # Limpiar nombres
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
        df_top20["feature_label"] = df_top20["feature"].map(nombre_map).fillna(df_top20["feature"])

        fig_imp = go.Figure()
        for cat, group in df_top20.groupby("categoria", sort=False):
            fig_imp.add_trace(go.Bar(
                name=cat,
                x=group["importance"],
                y=group["feature_label"],
                orientation="h",
                marker_color=color_map[cat],
                hovertemplate=f"<b>%{{y}}</b><br>Importancia: %{{x:,}}<br>Categoría: {cat}<extra></extra>",
            ))
        fig_imp.update_layout(
            height=560,
            barmode="stack",
            margin=dict(l=0, r=20, t=10, b=10),
            paper_bgcolor="white",
            plot_bgcolor="white",
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.01,
                xanchor="left", x=0,
                font=dict(size=11, family="Space Grotesk"),
            ),
            xaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False, title="Importancia (LightGBM gain)"),
            yaxis=dict(tickfont=dict(size=10, family="Space Grotesk")),
            font=dict(family="Space Grotesk"),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_err:
        st.markdown('<div class="section-header">¿Dónde se equivoca el modelo?</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Distribución de aciertos y errores en test (n=363)</div>', unsafe_allow_html=True)

        df_err_plot = df_err.sort_values("n", ascending=True).copy()
        color_err = []
        for t in df_err_plot["tipo_error"]:
            if t == "acierto":
                color_err.append(VERDE)
            elif "medio" in t:
                color_err.append(AMARILLO)
            else:
                color_err.append(ROJO)

        fig_err = go.Figure(go.Bar(
            x=df_err_plot["n"],
            y=df_err_plot["tipo_error"],
            orientation="h",
            marker_color=color_err,
            text=[f'{v}  ({p:.1f}%)' for v, p in zip(df_err_plot["n"], df_err_plot["pct"])],
            textposition="outside",
            textfont=dict(size=12, family="Space Grotesk"),
            hovertemplate="<b>%{y}</b><br>Registros: %{x}<extra></extra>",
        ))
        fig_err.update_layout(
            height=260,
            margin=dict(l=0, r=100, t=10, b=10),
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False, title="Nº registros"),
            yaxis=dict(tickfont=dict(size=12, family="Space Grotesk")),
            font=dict(family="Space Grotesk"),
        )
        st.plotly_chart(fig_err, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
          <strong>86.8 % son aciertos.</strong>
          El 13.2 % restante son errores simétricos: el modelo
          <em>subestima o sobreestima en una clase</em>, nunca en dos.
          El patrón sugiere que los aeropuertos en transición
          (creciendo o decreciendo) son el verdadero reto.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Conclusión del análisis</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Del hallazgo técnico a la decisión de negocio</div>', unsafe_allow_html=True)

        insights = [
            ("🏆", "El operacional manda",
             "Las 5 variables más importantes son <b>operacionales</b>. La historia de vuelos de un aeropuerto predice mejor su futuro que la meteorología."),
            ("🌦️", "La meteorología sí importa, pero en agregado",
             "Variables como <b>score meteo adverso</b> y <b>proporción de tormentas</b> aparecen en el top 20, pero su peso es complementario."),
            ("📅", "El lag de 12 meses es clave",
             "Varias features con <b>lag_12</b> tienen alta importancia: la <b>estacionalidad anual</b> es un patrón real en la aviación colombiana."),
            ("🎯", "Acción recomendada",
             "Aeropuertos con operaciones actuales <b>por encima de su media móvil 12m</b> y score meteo bajo → alta probabilidad de mes <b>alto</b> el siguiente mes."),
        ]
        for icon, titulo, texto in insights:
            st.markdown(f"""
            <div class="insight-box" style="margin-bottom:0.6rem">
              <strong>{icon} {titulo}</strong><br>
              <span style='font-size:0.87rem'>{texto}</span>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align:center; color:#94A3B8; font-size:0.8rem; font-family:'Space Grotesk'; padding-bottom:1rem">
  SI7007 Visualización de Datos · Universidad EAFIT · 2026 ·
  Modelo: <b style="color:#64748B">LightGBM (M03)</b> ·
  Dataset: <b style="color:#64748B">{dataset_cfg}</b> ·
  Test accuracy: <b style="color:#10B981">{accuracy:.1%}</b>
</div>
""", unsafe_allow_html=True)
