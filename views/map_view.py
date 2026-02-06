import folium
from folium.plugins import Draw, Fullscreen
from streamlit_folium import st_folium
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import streamlit as st

def inject_draw_css():
    st.markdown("""
    <style>
    .leaflet-draw-actions { z-index: 10000 !important; }
    .leaflet-draw-toolbar { z-index: 10000 !important; }
    .leaflet-top, .leaflet-right { z-index: 10000 !important; }
    .leaflet-control { z-index: 10000 !important; }
    </style>
    """, unsafe_allow_html=True)

def render_colored_map(df, color_by="Dia", key=None, editable=False):
    if color_by not in df.columns:
        st.warning(f"⚠️ La columna '{color_by}' no existe en el DataFrame.")
        return None

    m = folium.Map(location=[df['Latitud'].mean(), df['Longitud'].mean()], zoom_start=12)
    Fullscreen().add_to(m)

    categorias_unicas = sorted(df[color_by].dropna().unique())
    cmap = cm.get_cmap('tab20', len(categorias_unicas))
    colores_map = {cat: mcolors.to_hex(cmap(i)) for i, cat in enumerate(categorias_unicas)}

    # Normalizar nombres de columnas para encontrar "Contrato"
    normalized_cols = {c.lower().replace(" ", ""): c for c in df.columns}
    col_contrato = next((original for norm, original in normalized_cols.items() if "contrato" in norm), None)

    for cat, color in colores_map.items():
        subset = df[df[color_by] == cat]
        cantidad = len(subset)
        # Ajuste: mostrar días desde 1 en la leyenda y popup
        grupo = folium.FeatureGroup(name=f"{color_by} {int(cat)+1} ({cantidad})")
        for _, row in subset.iterrows():
            contrato_text = f"Contrato: {row[col_contrato]}" if col_contrato and pd.notna(row[col_contrato]) else "Contrato: Sin dato"
            popup_text = f"{color_by}: {int(cat)+1}<br>{contrato_text}"
            folium.CircleMarker(
                [row['Latitud'], row['Longitud']],
                radius=6,
                color=color,
                fill=True,
                popup=popup_text
            ).add_to(grupo)
        grupo.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
 
    if editable:
        Draw(
            export=True,
            position="topleft",
            draw_options={
                "polyline": False,
                "polygon": {
                    "allowIntersection": False,
                    "showArea": True,
                    "shapeOptions": {"color": "#97009c"},
                    "repeatMode": False,
                    "finishOnDoubleClick": True
                },
                "circle": False,
                "rectangle": False,
                "marker": False,
                "circlemarker": False
            },
            edit_options={"edit": True, "remove": True}
        ).add_to(m)

        inject_draw_css()

    return st_folium(m, width=700, height=500, key=key)

def render_map(df):
    """
    Mapa inicial con todos los puntos en azul.
    Al hacer clic en un punto se muestra el día y el contrato.
    """
    m = folium.Map(location=[df['Latitud'].mean(), df['Longitud'].mean()], zoom_start=12)
    Fullscreen().add_to(m)

    # Normalizar nombres de columnas para encontrar "Contrato"
    normalized_cols = {c.lower().replace(" ", ""): c for c in df.columns}
    col_contrato = next((original for norm, original in normalized_cols.items() if "contrato" in norm), None)

    for _, row in df.iterrows():
        contrato_text = f"Contrato: {row[col_contrato]}" if col_contrato and pd.notna(row[col_contrato]) else "Contrato: Sin dato"
        dia_val = int(row['Dia']) + 1 if "Dia" in df.columns and pd.notna(row['Dia']) else None
        dia_text = f"Dia: {dia_val}" if dia_val is not None else "Dia: Sin asignar"
        popup_text = f"{dia_text}<br>{contrato_text}"


        folium.CircleMarker(
            [row['Latitud'], row['Longitud']],
            radius=5,
            color="blue",
            fill=True,
            popup=popup_text
        ).add_to(m)

    # 👉 Este mapa NO necesita Draw
    return st_folium(m, width=700, height=500)
