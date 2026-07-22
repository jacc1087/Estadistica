import os
import urllib.request

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ── Configuración de página ─────────────────────────────────────
st.set_page_config(page_title="Estadística NBA", page_icon="🏀", layout="wide")

CSV_URL = "https://raw.githubusercontent.com/jacc1087/Estadistica/main/nba_players.csv"

POSICIONES = {
    "C": "Pívot", "PF": "Ala-Pívot", "SF": "Alero",
    "SG": "Escolta", "PG": "Base",
}

RENAME_MAP = {
    "lg": "league", "pos": "position", "g": "games", "gs": "games_started", "mp": "minutes_played",
    "fg": "field_goals", "fga": "field_goals_attempts", "fg_percent": "field_goals_percent",
    "x3p": "3p", "x3pa": "3p_attempts", "x3p_percent": "3p_percent",
    "x2p": "2p", "x2pa": "2p_attempts", "x2p_percent": "2p_percent",
    "e_fg_percent": "effective_percent", "ft": "Free_Throws", "fta": "Free_Throws_Attempts",
    "orb": "offensive_rebounds", "drb": "deffensive_rebounds", "trb": "total_rebounds",
    "ast": "assists", "stl": "steals", "blk": "blocks", "tov": "turnovers",
    "pf": "personal_fouls", "pts": "points",
}


@st.cache_data(show_spinner="Cargando y preparando los datos...")
def load_data():
    if not os.path.exists("nba_players.csv"):
        urllib.request.urlretrieve(CSV_URL, "nba_players.csv")

    df = pd.read_csv("nba_players.csv")
    df = df.rename(columns=RENAME_MAP)
    df = df[(df["league"] == "NBA") & (df["minutes_played"] >= 500)]

    cleaning = df.drop(columns=["season", "league", "player_id", "age", "team", "games_started"], errors="ignore")

    data = cleaning.groupby("player").agg({
        "player": "first",
        "position": "first",
        "games": "sum",
        "minutes_played": "sum",
        "total_rebounds": "sum",
        "assists": "sum",
        "steals": "sum",
        "blocks": "sum",
        "turnovers": "sum",
        "personal_fouls": "sum",
        "points": "sum",
        "trp_dbl": "sum",
        "field_goals_percent": "mean",
        "3p_percent": "mean",
        "2p_percent": "mean",
        "effective_percent": "mean",
        "ft_percent": "mean",
    }).round(4)

    for col in ["minutes_played", "total_rebounds", "steals", "blocks", "turnovers", "personal_fouls", "trp_dbl"]:
        data[col] = data[col].fillna(0).astype("int64")

    data = data.dropna(subset=["3p_percent"]).reset_index(drop=True)
    return data


df = load_data()

# ── Cabecera ─────────────────────────────────────────────────────
st.title("🏀 Estadística para Data Science — Jugadores NBA")
st.caption("Análisis descriptivo, correlación y regresión sobre estadísticas históricas de la NBA (desde 1947)")

# ── Sidebar: filtros ─────────────────────────────────────────────
with st.sidebar:
    st.header("Filtros")
    posiciones_disponibles = sorted(df["position"].dropna().unique())
    pos_labels = [f"{p} — {POSICIONES.get(p, p)}" for p in posiciones_disponibles]
    seleccion_labels = st.multiselect("Posición", pos_labels, default=pos_labels)
    posiciones_sel = [p.split(" — ")[0] for p in seleccion_labels] or posiciones_disponibles

    min_partidos = st.slider(
        "Partidos jugados (mínimo)",
        min_value=0, max_value=int(df["games"].max()),
        value=0, step=10,
    )

df_filtrado = df[(df["position"].isin(posiciones_sel)) & (df["games"] >= min_partidos)]

st.markdown(f"**{len(df_filtrado)}** jugadores cumplen los filtros seleccionados.")

# ── KPIs ─────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Jugadores analizados", f"{len(df_filtrado):,}")
c2.metric("Puntos — mediana", f"{df_filtrado['points'].median():,.0f}")
c3.metric("Efectividad media", f"{df_filtrado['effective_percent'].mean():.1%}")
c4.metric("Máx. anotador", df_filtrado.sort_values("points", ascending=False)["player"].iloc[0] if len(df_filtrado) else "—")

tab_resumen, tab_comparativas, tab_regresion, tab_buscar = st.tabs(
    ["📊 Resumen", "🔍 Comparativas", "📈 Regresión", "🕵️ Buscar jugador"]
)

# ── TAB 1: Resumen ───────────────────────────────────────────────
with tab_resumen:
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            df_filtrado, x="points", nbins=50,
            title="Distribución de puntos totales en carrera",
            labels={"points": "Puntos acumulados"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "La distribución está muy sesgada a la derecha: la mayoría de jugadores "
            "no supera los 2.500 puntos en toda su carrera."
        )
    with col2:
        fig = px.histogram(
            df_filtrado, x="effective_percent", nbins=50,
            title="Distribución del porcentaje de tiro efectivo",
            labels={"effective_percent": "Effective %"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Esta distribución es mucho más simétrica, cercana a una campana de Gauss.")

    st.subheader("🏆 Top 10 máximos anotadores")
    top10 = df_filtrado.sort_values("points", ascending=False).head(10)[["player", "position", "games", "points", "effective_percent"]]
    st.dataframe(top10, use_container_width=True, hide_index=True)

    st.subheader("🎯 Top 10 más efectivos")
    top10_eff = df_filtrado.sort_values("effective_percent", ascending=False).head(10)[["player", "position", "games", "effective_percent", "points"]]
    st.dataframe(top10_eff, use_container_width=True, hide_index=True)
    st.caption("El top de efectividad está dominado por pívots (C): tiros cercanos al aro con pocos intentos totales.")

# ── TAB 2: Comparativas ──────────────────────────────────────────
with tab_comparativas:
    st.subheader("Matriz de correlación")
    numeric_cols = [
        "games", "minutes_played", "total_rebounds", "assists", "steals", "blocks",
        "turnovers", "personal_fouls", "points", "field_goals_percent", "3p_percent",
        "2p_percent", "effective_percent", "ft_percent",
    ]
    corr = df_filtrado[numeric_cols].corr(numeric_only=True)
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Correlación entre variables numéricas",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Relación entre dos variables")
    col1, col2 = st.columns(2)
    with col1:
        var_x = st.selectbox("Eje X", numeric_cols, index=numeric_cols.index("minutes_played"))
    with col2:
        var_y = st.selectbox("Eje Y", numeric_cols, index=numeric_cols.index("points"))

    fig = px.scatter(
        df_filtrado, x=var_x, y=var_y, color="position",
        hover_name="player", opacity=0.6,
        title=f"{var_y} vs {var_x}",
    )
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 3: Regresión ─────────────────────────────────────────────
with tab_regresion:
    st.subheader("Prueba tu propio modelo de regresión lineal")
    st.caption("Elige qué variables usar para predecir el target. El modelo se entrena al vuelo sobre los datos filtrados.")

    target = st.selectbox("Variable a predecir (target)", ["points", "effective_percent"])
    predictoras_disponibles = [c for c in numeric_cols if c != target]
    predictoras = st.multiselect(
        "Variables predictoras",
        predictoras_disponibles,
        default=["minutes_played"] if target == "points" else ["2p_percent", "3p_percent", "ft_percent"],
    )

    if len(predictoras) == 0:
        st.info("Selecciona al menos una variable predictora.")
    elif len(df_filtrado) < 20:
        st.warning("Muy pocos datos con los filtros actuales para entrenar un modelo fiable.")
    else:
        datos_modelo = df_filtrado.dropna(subset=predictoras + [target])
        X = datos_modelo[predictoras]
        y = datos_modelo[target]

        modelo = LinearRegression()
        modelo.fit(X, y)
        y_pred = modelo.predict(X)

        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        c1, c2 = st.columns(2)
        c1.metric("R²", f"{r2:.3f}")
        c2.metric("RMSE", f"{rmse:,.2f}")

        coefs = pd.DataFrame({"Variable": predictoras, "Coeficiente": modelo.coef_}).sort_values(
            "Coeficiente", key=abs, ascending=False
        )
        st.dataframe(coefs, use_container_width=True, hide_index=True)

        if len(predictoras) == 1:
            fig = px.scatter(
                datos_modelo, x=predictoras[0], y=target, opacity=0.5,
                trendline="ols", trendline_color_override="red",
                title=f"Regresión: {target} ~ {predictoras[0]}",
            )
            st.plotly_chart(fig, use_container_width=True)

# ── TAB 4: Buscar jugador ────────────────────────────────────────
with tab_buscar:
    st.subheader("Compara un jugador con la media de su posición")
    jugador = st.selectbox(
        "Elige un jugador",
        sorted(df["player"].unique()),
        index=None,
        placeholder="Escribe un nombre para buscar...",
    )

    if jugador:
        fila = df[df["player"] == jugador].iloc[0]
        media_posicion = df[df["position"] == fila["position"]].mean(numeric_only=True)

        st.markdown(f"### {jugador} — {POSICIONES.get(fila['position'], fila['position'])}")

        comparar_cols = ["points", "assists", "total_rebounds", "steals", "blocks", "effective_percent"]
        comparativa = pd.DataFrame({
            "Estadística": comparar_cols,
            jugador: [fila[c] for c in comparar_cols],
            f"Media ({fila['position']})": [media_posicion[c] for c in comparar_cols],
        })

        fig = px.bar(
            comparativa.melt(id_vars="Estadística", var_name="Serie", value_name="Valor"),
            x="Estadística", y="Valor", color="Serie", barmode="group",
            title=f"{jugador} vs. media de su posición",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Busca un jugador en el desplegable de arriba para ver su comparativa.")
