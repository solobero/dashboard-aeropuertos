import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import numpy as np

st.set_page_config(
    page_title="Operacion Aeroportuaria Colombia",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# PALETA
# ─────────────────────────────────────────────
CIELO       = "#0EA5E9"
NOCHE       = "#0C1A2E"
RADAR       = "#10B981"
ALERTA      = "#F59E0B"
PELIGRO     = "#EF4444"
INSTRUMENTO = "#6366F1"

COLOR_BAJO  = ALERTA
COLOR_MEDIO = CIELO
COLOR_ALTO  = RADAR

COLOR_LARGE  = "#E63946"
COLOR_MEDIUM = "#F77F00"
COLOR_SMALL  = "#0077B6"

# ─────────────────────────────────────────────
# ESTILOS — fondo blanco
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #FFFFFF; }
.block-container {
    padding-top: 0 !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    max-width: 100% !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #F8FAFC;
    border-right: 1px solid #E2E8F0;
}
section[data-testid="stSidebar"] * { color: #1E293B !important; }
section[data-testid="stSidebar"] hr { border-color: #E2E8F0 !important; }
section[data-testid="stSidebar"] label p {
    color: #64748B !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: white !important;
    border-color: #E2E8F0 !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] span[data-baseweb="tag"] {
    background-color: #0EA5E9 !important;
    border-color: #0EA5E9 !important;
    border-radius: 6px !important;
}
section[data-testid="stSidebar"] span[data-baseweb="tag"] span {
    color: white !important;
}

/* Card aeropuerto */
.airport-card {
    background: white;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    margin: 0.5rem 0;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.airport-card-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 0.1rem;
}
.airport-card-icao {
    font-size: 0.72rem;
    font-weight: 600;
    color: #0EA5E9;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.08em;
}
.airport-stat {
    display: flex;
    justify-content: space-between;
    padding: 0.26rem 0;
    border-bottom: 1px solid #F1F5F9;
    font-size: 0.8rem;
}
.airport-stat-label { color: #64748B; }
.airport-stat-value { color: #0F172A; font-weight: 600; }

/* Tabs */
div[data-baseweb="tab-list"] {
    gap: 0.3rem;
    background: #F8FAFC !important;
    border-radius: 10px;
    padding: 0.2rem;
    border: 1px solid #E2E8F0;
    width: fit-content;
    margin-bottom: 0;
}
div[data-baseweb="tab"] {
    border-radius: 7px !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 0.38rem 0.9rem !important;
    color: #64748B !important;
}
div[data-baseweb="tab"][aria-selected="true"] {
    background: #0EA5E9 !important;
    color: white !important;
}

/* Tab panel */
.stTabs [data-baseweb="tab-panel"] {
    padding: 1.2rem 2rem 2rem 2rem !important;
    background: #FFFFFF;
}

/* Cards metricas */
.kpi-card {
    background: white;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.kpi-value {
    font-size: 1.8rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.2rem;
}
.kpi-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
.kpi-sub { font-size: 0.72rem; color: #94A3B8; margin-top: 0.2rem; }

/* Encabezados */
.section-q {
    font-size: 1.15rem;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 0.15rem;
}
.section-sub {
    font-size: 0.82rem;
    color: #64748B;
    margin-bottom: 0.8rem;
}

/* Respuesta */
.answer-box {
    background: #F0F9FF;
    border-left: 3px solid #0EA5E9;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin: 0.6rem 0;
    font-size: 0.84rem;
    color: #1E293B;
}
.answer-box.verde {
    background: #F0FDF4;
    border-left-color: #10B981;
}
.answer-box.alerta {
    background: #FFFBEB;
    border-left-color: #F59E0B;
}
.answer-box strong { color: #0F172A; }

.divider { border: none; border-top: 1px solid #F1F5F9; margin: 1.2rem 0; }
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

m_test     = meta["metrics_test_final"]
accuracy   = m_test["accuracy"]
macro_f1   = m_test["macro_f1"]
bal_acc    = m_test["balanced_accuracy"]
n_eval     = m_test["n_eval"]
model_name = meta["final_model_name"]
n_features = meta["n_features"]

# ─────────────────────────────────────────────
# DATOS DE AEROPUERTOS
# ─────────────────────────────────────────────
AIRPORT_GEO = pd.DataFrame([
    {"icao":"SKBO","name":"El Dorado",      "ciudad":"Bogota",         "lat":4.70, "lon":-74.15,"type":"large_airport", "ops":222450,"temp_c":13.2,"viento_kt":4.1,"prop_lluvia":0.031,"prop_ifr":0.038,"visib_sm":8.9,"nivel":"alto"},
    {"icao":"SKRG","name":"J.M. Cordova",   "ciudad":"Rionegro",       "lat":6.16, "lon":-75.42,"type":"large_airport", "ops":85200, "temp_c":17.4,"viento_kt":5.8,"prop_lluvia":0.045,"prop_ifr":0.041,"visib_sm":8.1,"nivel":"alto"},
    {"icao":"SKCL","name":"A. Bonilla",      "ciudad":"Cali",           "lat":3.54, "lon":-76.38,"type":"large_airport", "ops":82300, "temp_c":24.1,"viento_kt":6.2,"prop_lluvia":0.038,"prop_ifr":0.029,"visib_sm":8.4,"nivel":"alto"},
    {"icao":"SKBQ","name":"E. Cortissoz",    "ciudad":"Barranquilla",   "lat":10.89,"lon":-74.78,"type":"large_airport", "ops":87640, "temp_c":27.8,"viento_kt":8.4,"prop_lluvia":0.021,"prop_ifr":0.018,"visib_sm":9.2,"nivel":"alto"},
    {"icao":"SKCG","name":"R. Nunez",        "ciudad":"Cartagena",      "lat":10.44,"lon":-75.51,"type":"large_airport", "ops":72100, "temp_c":27.2,"viento_kt":7.9,"prop_lluvia":0.024,"prop_ifr":0.021,"visib_sm":9.0,"nivel":"alto"},
    {"icao":"SKMD","name":"Olaya Herrera",   "ciudad":"Medellin",       "lat":6.22, "lon":-75.59,"type":"medium_airport","ops":91100, "temp_c":22.3,"viento_kt":3.8,"prop_lluvia":0.052,"prop_ifr":0.044,"visib_sm":7.8,"nivel":"alto"},
    {"icao":"SKMR","name":"Los Garzones",    "ciudad":"Monteria",       "lat":8.82, "lon":-75.83,"type":"medium_airport","ops":98320, "temp_c":28.4,"viento_kt":5.1,"prop_lluvia":0.029,"prop_ifr":0.022,"visib_sm":8.8,"nivel":"alto"},
    {"icao":"SKSP","name":"G. Rojas P.",     "ciudad":"San Andres",     "lat":12.58,"lon":-81.71,"type":"medium_airport","ops":71500, "temp_c":27.6,"viento_kt":9.2,"prop_lluvia":0.018,"prop_ifr":0.015,"visib_sm":9.4,"nivel":"alto"},
    {"icao":"SKPE","name":"Matecana",         "ciudad":"Pereira",        "lat":4.81, "lon":-75.74,"type":"medium_airport","ops":63400, "temp_c":21.8,"viento_kt":4.4,"prop_lluvia":0.048,"prop_ifr":0.039,"visib_sm":7.9,"nivel":"medio"},
    {"icao":"SKBG","name":"Palonegro",        "ciudad":"Bucaramanga",    "lat":7.13, "lon":-73.18,"type":"large_airport", "ops":58200, "temp_c":25.1,"viento_kt":6.7,"prop_lluvia":0.033,"prop_ifr":0.028,"visib_sm":8.5,"nivel":"medio"},
    {"icao":"SKCU","name":"Camilo Daza",      "ciudad":"Cucuta",         "lat":7.93, "lon":-72.51,"type":"large_airport", "ops":51800, "temp_c":28.6,"viento_kt":5.3,"prop_lluvia":0.027,"prop_ifr":0.023,"visib_sm":8.7,"nivel":"medio"},
    {"icao":"SKSM","name":"Simon Bolivar",    "ciudad":"Santa Marta",    "lat":11.12,"lon":-74.23,"type":"large_airport", "ops":49300, "temp_c":27.1,"viento_kt":8.1,"prop_lluvia":0.019,"prop_ifr":0.016,"visib_sm":9.3,"nivel":"medio"},
    {"icao":"SKVP","name":"A. Lopez P.",      "ciudad":"Valledupar",     "lat":10.44,"lon":-73.25,"type":"large_airport", "ops":46700, "temp_c":29.2,"viento_kt":7.4,"prop_lluvia":0.022,"prop_ifr":0.019,"visib_sm":9.1,"nivel":"medio"},
    {"icao":"SKUI","name":"El Carano",        "ciudad":"Quibdo",         "lat":5.69, "lon":-76.64,"type":"medium_airport","ops":54200, "temp_c":26.8,"viento_kt":2.9,"prop_lluvia":0.089,"prop_ifr":0.071,"visib_sm":6.2,"nivel":"bajo"},
    {"icao":"SKFL","name":"G. Artunduaga",    "ciudad":"Florencia",      "lat":1.59, "lon":-75.56,"type":"medium_airport","ops":22800, "temp_c":25.4,"viento_kt":2.1,"prop_lluvia":0.071,"prop_ifr":0.058,"visib_sm":6.8,"nivel":"bajo"},
    {"icao":"SKPS","name":"Tres de Mayo",     "ciudad":"Puerto Asis",    "lat":0.51, "lon":-76.50,"type":"small_airport", "ops":12300, "temp_c":24.9,"viento_kt":1.8,"prop_lluvia":0.078,"prop_ifr":0.062,"visib_sm":6.5,"nivel":"bajo"},
    {"icao":"SKAQ","name":"El Alcaravan",     "ciudad":"Arauca",         "lat":7.08, "lon":-70.74,"type":"medium_airport","ops":28400, "temp_c":29.7,"viento_kt":4.8,"prop_lluvia":0.035,"prop_ifr":0.029,"visib_sm":8.3,"nivel":"bajo"},
    {"icao":"SKNV","name":"Benito Salas",     "ciudad":"Neiva",          "lat":2.95, "lon":-75.29,"type":"medium_airport","ops":32400, "temp_c":27.3,"viento_kt":5.9,"prop_lluvia":0.041,"prop_ifr":0.034,"visib_sm":7.7,"nivel":"bajo"},
    {"icao":"SKVV","name":"Vanguardia",       "ciudad":"Villavicencio",  "lat":4.17, "lon":-73.61,"type":"medium_airport","ops":35100, "temp_c":25.8,"viento_kt":3.7,"prop_lluvia":0.055,"prop_ifr":0.045,"visib_sm":7.4,"nivel":"bajo"},
])

# Evolucion mensual
meses = pd.date_range("2020-01", "2025-12", freq="MS")
np.random.seed(42)
base  = np.linspace(9000, 15000, len(meses))
covid = (meses >= "2020-03") & (meses <= "2020-09")
base[covid] *= np.linspace(0.12, 0.75, covid.sum())
ops_mensual = np.clip(base + np.random.normal(0, 280, len(meses)), 400, None)

# Fenomenos por clase
FENOMENOS = pd.DataFrame({
    "fenomeno": ["Lluvia","Tormenta","Niebla","Baja visib.","Viento fuerte","IFR"],
    "bajo":     [0.048, 0.009, 0.012, 0.031, 0.018, 0.041],
    "medio":    [0.041, 0.007, 0.010, 0.025, 0.015, 0.033],
    "alto":     [0.035, 0.005, 0.008, 0.019, 0.012, 0.026],
})

# ─────────────────────────────────────────────
# ESTADO
# ─────────────────────────────────────────────
if "aeropuerto_sel" not in st.session_state:
    st.session_state.aeropuerto_sel = None

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:0.8rem 0 0.4rem 0'>
      <div style='font-size:0.62rem;color:#94A3B8;text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:0.3rem'>SI7007 · EAFIT · 2026</div>
      <div style='font-size:1rem;font-weight:700;color:#0F172A;line-height:1.3'>
        Operacion Aeroportuaria<br>Colombia
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style='font-size:0.7rem;color:#0EA5E9;text-transform:uppercase;
                letter-spacing:0.08em;margin-bottom:0.4rem;font-weight:600'>
      Pregunta central
    </div>
    <div style='font-size:0.82rem;color:#475569;line-height:1.5'>
      ¿Las condiciones meteorologicas de un aeropuerto colombiano
      permiten anticipar su nivel de operacion el proximo mes?
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Filtro color del mapa
    color_mapa = st.radio(
        "Color del mapa por",
        options=["Tipo de aeropuerto", "Nivel de operacion"],
        index=0,
    )

    nivel_sel = st.multiselect(
        "Nivel de operacion",
        options=["bajo","medio","alto"],
        default=["bajo","medio","alto"],
    )

    sel_nombre = st.selectbox(
        "Ver detalle de aeropuerto",
        ["Ninguno"] + sorted(AIRPORT_GEO["name"].tolist()),
    )
    st.session_state.aeropuerto_sel = sel_nombre if sel_nombre != "Ninguno" else None

    st.markdown("---")

    # Card aeropuerto
    if st.session_state.aeropuerto_sel:
        row = AIRPORT_GEO[AIRPORT_GEO["name"] == st.session_state.aeropuerto_sel].iloc[0]
        badge_bg    = {"bajo":"#FEF3C7","medio":"#E0F2FE","alto":"#D1FAE5"}[row["nivel"]]
        badge_color = {"bajo":"#92400E","medio":"#075985","alto":"#065F46"}[row["nivel"]]
        st.markdown(f"""
        <div class="airport-card">
          <div class="airport-card-icao">{row["icao"]}</div>
          <div class="airport-card-title">{row["name"]}</div>
          <div style="font-size:0.75rem;color:#64748B;margin-bottom:0.5rem">{row["ciudad"]}</div>
          <span style="background:{badge_bg};color:{badge_color};padding:2px 10px;
                       border-radius:99px;font-size:0.72rem;font-weight:700">
            Nivel {row["nivel"]}
          </span>
          <div style="margin-top:0.7rem">
            <div class="airport-stat">
              <span class="airport-stat-label">Ops. acumuladas</span>
              <span class="airport-stat-value">{int(row["ops"]):,}</span>
            </div>
            <div class="airport-stat">
              <span class="airport-stat-label">Prop. lluvia</span>
              <span class="airport-stat-value">{row["prop_lluvia"]:.1%}</span>
            </div>
            <div class="airport-stat">
              <span class="airport-stat-label">Prop. IFR</span>
              <span class="airport-stat-value">{row["prop_ifr"]:.1%}</span>
            </div>
            <div class="airport-stat">
              <span class="airport-stat-label">Viento medio</span>
              <span class="airport-stat-value">{row["viento_kt"]:.1f} kt</span>
            </div>
            <div class="airport-stat">
              <span class="airport-stat-label">Visibilidad</span>
              <span class="airport-stat-value">{row["visib_sm"]:.1f} sm</span>
            </div>
            <div class="airport-stat" style="border:none">
              <span class="airport-stat-label">Temperatura</span>
              <span class="airport-stat-value">{row["temp_c"]:.1f} °C</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='font-size:0.78rem;color:#94A3B8;text-align:center;
                    padding:1rem;border:1px dashed #E2E8F0;border-radius:10px'>
          Selecciona un aeropuerto para ver su perfil meteorologico
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.7rem;color:#94A3B8;line-height:2'>
      Modelo: <b style='color:#0EA5E9'>LightGBM</b><br>
      Accuracy: <b style='color:#10B981'>{accuracy:.1%}</b><br>
      Macro F1: <b style='color:#10B981'>{macro_f1:.1%}</b>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAPA HERO — estilo carto-positron
# ─────────────────────────────────────────────
df_mapa = AIRPORT_GEO[AIRPORT_GEO["nivel"].isin(nivel_sel)].copy() if nivel_sel else AIRPORT_GEO.copy()

if color_mapa == "Tipo de aeropuerto":
    color_map_disc = {
        "large_airport":  COLOR_LARGE,
        "medium_airport": COLOR_MEDIUM,
        "small_airport":  COLOR_SMALL,
    }
    label_map_disc = {
        "large_airport":  "Hub principal",
        "medium_airport": "Regional",
        "small_airport":  "Local",
    }
    grupos = [
        ("large_airport",  "Hub principal", COLOR_LARGE),
        ("medium_airport", "Regional",      COLOR_MEDIUM),
        ("small_airport",  "Local",         COLOR_SMALL),
    ]
    col_campo = "type"
else:
    color_map_disc = {
        "bajo":  COLOR_BAJO,
        "medio": COLOR_MEDIO,
        "alto":  COLOR_ALTO,
    }
    label_map_disc = {
        "bajo":  "Nivel bajo",
        "medio": "Nivel medio",
        "alto":  "Nivel alto",
    }
    grupos = [
        ("bajo",  "Nivel bajo",  COLOR_BAJO),
        ("medio", "Nivel medio", COLOR_MEDIO),
        ("alto",  "Nivel alto",  COLOR_ALTO),
    ]
    col_campo = "nivel"

fig_mapa = go.Figure()

for key, label, color in grupos:
    sub = df_mapa[df_mapa[col_campo] == key]
    if sub.empty:
        continue
    # Resaltar seleccionado
    sizes = sub["ops"].apply(lambda x: max(10, min(24, x/7500 + 9)))
    line_widths = [3 if row["name"] == st.session_state.aeropuerto_sel else 1 for _, row in sub.iterrows()]
    line_colors = ["white" if row["name"] == st.session_state.aeropuerto_sel else "rgba(255,255,255,0.6)" for _, row in sub.iterrows()]

    fig_mapa.add_trace(go.Scattermapbox(
        lat=sub["lat"],
        lon=sub["lon"],
        mode="markers",
        name=label,
        marker=dict(
            size=sizes,
            color=color,
            opacity=0.92,
        ),
        text=sub["name"],
        customdata=sub[["icao","ciudad","ops","prop_lluvia","viento_kt","nivel","type"]].values,
        hovertemplate=(
            "<b>%{text}</b> (%{customdata[0]})<br>"
            "%{customdata[1]}<br>"
            "──────────────────<br>"
            "Nivel predominante: <b>%{customdata[5]}</b><br>"
            "Tipo: %{customdata[6]}<br>"
            "Operaciones acum.: <b>%{customdata[2]:,}</b><br>"
            "Prop. lluvia: <b>%{customdata[3]:.1%}</b><br>"
            "Viento medio: <b>%{customdata[4]:.1f} kt</b>"
            "<extra></extra>"
        ),
    ))

fig_mapa.update_layout(
    height=580,
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor="white",
    mapbox=dict(
        style="carto-positron",
        center=dict(lat=4.5709, lon=-74.2973),
        zoom=5.0,
    ),
    legend=dict(
        orientation="h",
        y=0.02, x=0.5, xanchor="center",
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="#E2E8F0",
        borderwidth=1,
        font=dict(family="Inter", size=12, color="#1E293B"),
    ),
    font=dict(family="Inter"),
)

st.plotly_chart(fig_mapa, use_container_width=True, config={"displayModeBar": False})

# ─────────────────────────────────────────────
# TABS DE ANALISIS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "Meteorologia y Operacion",
    "Variables del modelo",
    "Comparativa de modelos",
])

# ══════════════════════════════════════════════
# TAB 1 — METEOROLOGIA Y OPERACION
# ══════════════════════════════════════════════
with tab1:

    col_var, col_fen = st.columns([1, 1], gap="large")

    with col_var:
        st.markdown('<div class="section-q">¿Los aeropuertos con peor clima operan menos?</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Valor promedio de la variable meteorologica segun nivel de operacion del mes siguiente</div>', unsafe_allow_html=True)

        METEO_VARS = {
            "Proporcion de lluvia mensual": {"bajo":0.048,"medio":0.041,"alto":0.035,"fmt":".1%","titulo":"Prop. lluvia"},
            "Proporcion condiciones IFR":   {"bajo":0.041,"medio":0.033,"alto":0.026,"fmt":".1%","titulo":"Prop. IFR"},
            "Proporcion de tormentas":      {"bajo":0.009,"medio":0.007,"alto":0.005,"fmt":".1%","titulo":"Prop. tormenta"},
            "Proporcion de niebla":         {"bajo":0.012,"medio":0.010,"alto":0.008,"fmt":".1%","titulo":"Prop. niebla"},
            "Viento medio (kt)":            {"bajo":5.8,  "medio":5.3,  "alto":4.9, "fmt":".1f","titulo":"Viento medio (kt)"},
            "Visibilidad media (sm)":       {"bajo":7.4,  "medio":8.1,  "alto":8.9, "fmt":".1f","titulo":"Visibilidad (sm)"},
            "Temperatura media (C)":        {"bajo":19.8, "medio":21.2, "alto":22.9,"fmt":".1f","titulo":"Temperatura (C)"},
        }

        var_meteo = st.selectbox("Variable meteorologica", options=list(METEO_VARS.keys()), key="var_m")
        d = METEO_VARS[var_meteo]
        niveles_a = [n for n in ["bajo","medio","alto"] if n in nivel_sel] or ["bajo","medio","alto"]

        fig_meteo = go.Figure()
        for nivel in niveles_a:
            fig_meteo.add_trace(go.Bar(
                name=nivel.capitalize(),
                x=[nivel.capitalize()],
                y=[d[nivel]],
                marker_color={"bajo":COLOR_BAJO,"medio":COLOR_MEDIO,"alto":COLOR_ALTO}[nivel],
                text=[f'{d[nivel]:{d["fmt"]}}'],
                textposition="outside",
                textfont=dict(size=14),
            ))
        ymin = min(d[n] for n in niveles_a) * 0.87
        ymax = max(d[n] for n in niveles_a) * 1.16
        fig_meteo.update_layout(
            height=300, showlegend=False,
            margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(title="Nivel de operacion", tickfont=dict(color="#64748B")),
            yaxis=dict(title=d["titulo"], range=[ymin, ymax],
                       showgrid=True, gridcolor="#F1F5F9", zeroline=False,
                       tickfont=dict(color="#64748B")),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig_meteo, use_container_width=True)

        st.markdown("""
        <div class="answer-box">
          <strong>Si.</strong> Los aeropuertos con nivel bajo tienen consistentemente
          mas lluvia, mas IFR y menor visibilidad. La meteorologia adversa
          es un factor penalizador sistematico aunque no causal directo.
        </div>
        """, unsafe_allow_html=True)

    with col_fen:
        st.markdown('<div class="section-q">Frecuencia de fenomenos adversos por nivel</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Proporcion media mensual de cada fenomeno segun clase de operacion</div>', unsafe_allow_html=True)

        fig_fen = go.Figure()
        for nivel in (nivel_sel or ["bajo","medio","alto"]):
            if nivel in FENOMENOS.columns:
                fig_fen.add_trace(go.Bar(
                    name=nivel.capitalize(),
                    x=FENOMENOS["fenomeno"],
                    y=FENOMENOS[nivel],
                    marker_color={"bajo":COLOR_BAJO,"medio":COLOR_MEDIO,"alto":COLOR_ALTO}[nivel],
                    hovertemplate="<b>%{x}</b><br>" + nivel + ": %{y:.3f}<extra></extra>",
                ))
        fig_fen.update_layout(
            barmode="group", height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(title="Fenomeno meteorologico", tickfont=dict(color="#64748B", size=11)),
            yaxis=dict(title="Proporcion media mensual",
                       showgrid=True, gridcolor="#F1F5F9", zeroline=False,
                       tickfont=dict(color="#64748B")),
            legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center",
                        font=dict(size=11)),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig_fen, use_container_width=True)

        st.markdown("""
        <div class="answer-box alerta">
          <strong>Quibdo y Florencia</strong> son los casos mas claros:
          lluvia sistematicamente alta, condiciones IFR frecuentes
          y nivel bajo estructural. Los hubs del Caribe tienen
          mucho viento pero poca lluvia y operan en nivel alto.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — VARIABLES DEL MODELO
# ══════════════════════════════════════════════
with tab2:

    col_imp, col_cm = st.columns([1.1, 0.9], gap="large")

    with col_imp:
        st.markdown('<div class="section-q">¿Que variables usa el modelo para predecir?</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Top 15 por importancia (LightGBM gain) · azul = operacional · naranja = meteorologica · verde = geografica</div>', unsafe_allow_html=True)

        nombre_map = {
            "operaciones_total":            "Operaciones del mes",
            "operaciones_vs_roll_mean_12":  "Ops. vs media 12m",
            "operaciones_vs_roll_mean_3":   "Ops. vs media 3m",
            "operaciones_por_destino":      "Ops. por destino",
            "operaciones_delta_1":          "Variacion mensual",
            "pasajeros_total_roll_mean_12": "Pasajeros media 12m",
            "n_empresas_llegada":           "Aerolineas operando",
            "n_destinos_total_lag_12":      "Destinos lag 12m",
            "cobertura_metar_ratio_lag_12": "Cobertura METAR 12m",
            "viento_max_kt":                "Viento maximo (kt)",
            "prop_tormenta":                "Prop. tormentas",
            "prop_niebla":                  "Prop. niebla",
            "visibilidad_rango_sm":         "Rango visibilidad",
            "meteo_adverso_score_lag_3":    "Score meteo adverso 3m",
            "prop_lluvia":                  "Prop. lluvia",
        }

        def cat_f(f):
            if any(k in f for k in ["operaciones","pasajeros","carga","n_empresas","n_destinos"]):
                return "Operacional"
            elif any(k in f for k in ["temp","viento","visib","niebla","lluvia","tormenta","rafaga","meteo","dewpoint","prop_"]):
                return "Meteorologica"
            return "Geografica"

        color_cat = {"Operacional": CIELO, "Meteorologica": ALERTA, "Geografica": RADAR}
        df_fn = df_feat.head(15).copy()
        df_fn["cat"]   = df_fn["feature"].apply(cat_f)
        df_fn["label"] = df_fn["feature"].map(nombre_map).fillna(df_fn["feature"])
        df_fn["ord"]   = df_fn["cat"].map({"Geografica":0,"Meteorologica":1,"Operacional":2})
        df_fn = df_fn.sort_values(["ord","importance"], ascending=[True,True])

        fig_imp = go.Figure()
        for cat in ["Geografica","Meteorologica","Operacional"]:
            g = df_fn[df_fn["cat"] == cat]
            if g.empty: continue
            fig_imp.add_trace(go.Bar(
                name=cat, x=g["importance"], y=g["label"],
                orientation="h", marker_color=color_cat[cat],
                hovertemplate=f"<b>%{{y}}</b><br>Importancia: %{{x:,}}<br>{cat}<extra></extra>",
            ))
        fig_imp.update_layout(
            height=420, barmode="stack",
            margin=dict(l=0, r=20, t=10, b=0),
            paper_bgcolor="white", plot_bgcolor="white",
            legend=dict(orientation="h", y=1.03, x=0, font=dict(size=11)),
            xaxis=dict(title="Importancia (LightGBM gain)",
                       showgrid=True, gridcolor="#F1F5F9", zeroline=False,
                       tickfont=dict(color="#64748B")),
            yaxis=dict(title="Variable",
                       tickfont=dict(size=10, color="#64748B"),
                       categoryorder="array",
                       categoryarray=df_fn["label"].tolist()),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_cm:
        st.markdown('<div class="section-q">Matriz de confusion del modelo final</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Clase real (filas) vs clase predicha (columnas) · LightGBM en test</div>', unsafe_allow_html=True)

        mat = np.array([
            [meta["confusion_matrix_test_final"][r][c] for c in ["0","1","2"]]
            for r in ["0","1","2"]
        ], dtype=float)
        mat_norm = mat / mat.sum(axis=1, keepdims=True)
        textos = [
            [f"<b>{int(mat[i][j])}</b><br>{mat_norm[i][j]:.0%}" for j in range(3)]
            for i in range(3)
        ]
        fig_cm = go.Figure(go.Heatmap(
            z=mat_norm,
            x=["Bajo","Medio","Alto"],
            y=["Bajo","Medio","Alto"],
            colorscale=[[0,"#F0F9FF"],[0.5,"#7DD3FC"],[1,"#0369A1"]],
            showscale=False,
            text=textos, texttemplate="%{text}",
            hovertemplate="Real: %{y}<br>Predicho: %{x}<br>%{z:.1%}<extra></extra>",
        ))
        fig_cm.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=10, b=30),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(title="Clase predicha", tickfont=dict(color="#64748B", size=12)),
            yaxis=dict(title="Clase real", tickfont=dict(color="#64748B", size=12), autorange="reversed"),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown(f"""
        <div class="answer-box verde">
          <strong>Accuracy {accuracy:.1%} · Macro F1 {macro_f1:.1%}</strong><br><br>
          El modelo nunca confunde los extremos (bajo↔alto = 0 errores).
          Los errores ocurren solo entre clases adyacentes, que corresponden
          a aeropuertos en transicion operacional.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 — COMPARATIVA DE MODELOS
# ══════════════════════════════════════════════
with tab3:

    st.markdown('<div class="section-q">¿Por que LightGBM?</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Macro F1 promedio en validacion por modelo · azul = ganador</div>', unsafe_allow_html=True)

    modelos_disponibles = sorted(df_val["model_name"].unique().tolist())
    modelos_sel = st.multiselect(
        "Modelos a mostrar",
        options=modelos_disponibles,
        default=modelos_disponibles,
        key="mod_sel",
    )

    df_base = df_val[~df_val["experiment_id"].str.contains("hp|optuna", case=False)].copy()
    if modelos_sel:
        df_base = df_base[df_base["model_name"].isin(modelos_sel)]
    df_agg = (df_base.groupby("model_name")["macro_f1"]
              .mean().reset_index().sort_values("macro_f1", ascending=True))

    fig_mod = go.Figure(go.Bar(
        x=df_agg["macro_f1"],
        y=df_agg["model_name"],
        orientation="h",
        marker_color=[CIELO if m == "LightGBM" else "#CBD5E1" for m in df_agg["model_name"]],
        text=[f'{v:.1%}' for v in df_agg["macro_f1"]],
        textposition="outside",
        textfont=dict(size=12),
        hovertemplate="<b>%{y}</b><br>Macro F1: %{x:.3f}<extra></extra>",
    ))
    fig_mod.add_vline(x=0.85, line_dash="dot", line_color=CIELO, opacity=0.6,
                      annotation_text=" 85%",
                      annotation_font=dict(size=10, color=CIELO))
    fig_mod.update_layout(
        height=300,
        margin=dict(l=0, r=70, t=10, b=0),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(title="Macro F1", tickformat=".0%",
                   range=[0.55, 0.97],
                   showgrid=True, gridcolor="#F1F5F9",
                   zeroline=False, tickfont=dict(color="#64748B")),
        yaxis=dict(title="Modelo", tickfont=dict(color="#64748B", size=11)),
        font=dict(family="Inter"),
    )
    st.plotly_chart(fig_mod, use_container_width=True)

    st.markdown("""
    <div class="answer-box">
      <strong>LightGBM supera a la Regresion Logistica por mas de 20 puntos</strong>
      de Macro F1. El problema tiene no-linearidades importantes:
      las interacciones entre variables operacionales y meteorologicas
      que solo los modelos basados en arboles capturan bien.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;color:#94A3B8;font-size:0.72rem;
            font-family:'Inter';padding:0.8rem 2rem 1.2rem 2rem;
            border-top:1px solid #F1F5F9;margin-top:1rem">
  SI7007 Visualizacion de Datos · Universidad EAFIT · 2026 ·
  LightGBM · Accuracy <b style="color:#10B981">{accuracy:.1%}</b> ·
  Macro F1 <b style="color:#10B981">{macro_f1:.1%}</b>
</div>
""", unsafe_allow_html=True)
