import io
from gtts import gTTS
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch

# --- CONFIGURACIÓN DE PÁGINA (TEMA TERMINATOR) ---
st.set_page_config(
    page_title="SYSTEM T-800 // TARGETING HUD",
    page_icon="🤖",
    layout="wide",
)

# Estilos CSS estilo Cyberdyne Systems HUD
st.markdown(
    """
    <style>
    /* Fondo oscuro y tipografía estilo terminal cibernética */
    .stApp {
        background-color: #050505;
        color: #ff3333;
        font-family: 'Courier New', Courier, monospace;
    }
    h1, h2, h3, h4, span, label {
        color: #ff2222 !important;
        font-family: 'Courier New', Courier, monospace !important;
        text-shadow: 0 0 8px #ff0000;
    }
    .stSidebar {
        background-color: #0f0000 !important;
        border-right: 2px solid #ff0000;
    }
    /* Botones y sliders personalizados estilo cibernético */
    .stButton>button {
        background-color: #330000 !important;
        color: #ff0000 !important;
        border: 2px solid #ff0000 !important;
        box-shadow: 0 0 10px #ff0000;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ff0000 !important;
        color: #000000 !important;
    }
    .hud-box {
        border: 2px solid #ff0000;
        padding: 15px;
        background-color: rgba(255, 0, 0, 0.1);
        box-shadow: 0 0 15px #ff0000;
        margin-bottom: 15px;
        border-radius: 5px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO

        model = YOLO("yolov5su.pt")
        return model
    except Exception as e:
        st.error(f"SYSTEM FAILURE: Error al inicializar T-800 Vision ({str(e)})")
        return None


# Lista de clases consideradas "Organismos Vivos" en COCO Dataset
ORGANISMOS_VIVOS = {
    "person",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}

# Encabezado HUD Terminator
st.title("🤖 CYBERDYNE SYSTEMS // MODEL T-800")
st.caption("SYSTEM STATUS: ONLINE // TACTICAL VISUAL SCANNER v800.42")

with st.spinner("Cargando matriz de visión táctica YOLOv5..."):
    model = load_model()

if model:
    with st.sidebar:
        st.title("⚙️ PANEL DE CONTROL T-800")
        st.subheader("PARAMETROS DE ESCANEO")
        conf_threshold = st.slider("SENSIBILIDAD OPTICA", 0.0, 1.0, 0.25, 0.01)
        iou_threshold = st.slider("UMBRAL IOU", 0.0, 1.0, 0.45, 0.01)
        max_det = st.number_input(
            "OBJETIVOS MAXIMOS", 10, 2000, 1000, 10
        )

    st.markdown("### 👁️ SENSOR DE OPTICA Y CAMARA TACTICA")
    picture = st.camera_input("INICIAR CAPTURA DE OBJETIVO", key="camera")

    if picture:
        bytes_data = picture.getvalue()

        # Decodificación y conversión BGR para el pipeline
        pil_img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        np_img = np.array(pil_img)[..., ::-1]

        with st.spinner("ANALIZANDO ENTIDAD DETECTADA..."):
            try:
                results = model(
                    np_img,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    max_det=int(max_det),
                )
            except Exception as e:
                st.error(f"ANALYSIS ERROR: {str(e)}")
                st.stop()

        result = results[0]
        boxes = result.boxes
        annotated = result.plot()
        annotated_rgb = annotated[:, :, ::-1]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔴 VISUALIZACION TACTICA HUD")
            st.image(annotated_rgb, use_container_width=True)

        with col2:
            st.subheader("⚡ ESTADO DE OBJETIVOS AMBIENTALES")

            if boxes is not None and len(boxes) > 0:
                label_names = model.names
                category_count = {}
                category_conf = {}

                has_living_organism = False
                has_object = False

                for box in boxes:
                    cat = int(box.cls.item())
                    conf = float(box.conf.item())
                    class_name = label_names[cat]

                    # Clasificación de entidad (Vivo vs Inerte)
                    if class_name in ORGANISMOS_VIVOS:
                        has_living_organism = True
                    else:
                        has_object = True

                    category_count[cat] = category_count.get(cat, 0) + 1
                    category_conf.setdefault(cat, []).append(conf)

                # --- LÓGICA TERMINATOR: MENSAJES DE ESTADO Y ALERTA DE VOZ ---
                mensaje_alerta = ""

                if has_living_organism and has_object:
                    mensaje_alerta = "¡ALERTA! DETECTANDO ORGANISMO VIVO Y OBJETOS AMBIENTALES."
                    st.markdown(
                        f"""
                        <div class="hud-box">
                            <h2 style="color: #ff0000; margin:0;">🚨 AMENAZA COMBINADA</h2>
                            <p style="font-size:18px; color: #ff6666;">DETECTANDO ORGANISMO VIVO Y OBJETOS</p>
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )
                elif has_living_organism:
                    mensaje_alerta = "¡ATENCION! DETECTANDO ORGANISMO VIVO."
                    st.markdown(
                        f"""
                        <div class="hud-box">
                            <h2 style="color: #ff0000; margin:0;">⚠️ TARGET ACQUIRED</h2>
                            <p style="font-size:18px; color: #ff6666;">DETECTANDO ORGANISMO VIVO</p>
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    mensaje_alerta = "SECTOR DESPEJADO. DETECTANDO OBJETOS INERTES."
                    st.markdown(
                        f"""
                        <div class="hud-box">
                            <h2 style="color: #00ff00; margin:0;">🛡️ OBJETOS INERTES</h2>
                            <p style="font-size:18px; color: #66ff66;">DETECTANDO OBJETOS</p>
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )

                # Sintetizador de voz T-800 con gTTS
                try:
                    tts = gTTS(text=mensaje_alerta, lang="es", slow=False)
                    audio_fp = io.BytesIO()
                    tts.write_to_fp(audio_fp)
                    audio_fp.seek(0)
                    st.audio(audio_fp, format="audio/mp3", autoplay=True)
                except Exception:
                    pass

                # Tabla cibernética de objetivos
                data = [
                    {
                        "TIPO DE AMENAZA / ENTIDAD": label_names[cat].upper(),
                        "CLASIFICACION": (
                            "ORGANISMO VIVO 🧬"
                            if label_names[cat] in ORGANISMOS_VIVOS
                            else "OBJETO INERTE 📦"
                        ),
                        "CANTIDAD": count,
                        "PRECISION CIBERNETICA": (
                            f"{np.mean(category_conf[cat])*100:.1f}%"
                        ),
                    }
                    for cat, count in category_count.items()
                ]

                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
                st.bar_chart(
                    df.set_index("TIPO DE AMENAZA / ENTIDAD")["CANTIDAD"]
                )
            else:
                st.info("SISTEMA SCANNER: NO SE DETECTARON AMENAZAS EN EL SECTOR.")
                st.caption(
                    "Ajuste los parámetros de sensibilidad óptica en el panel lateral."
                )
else:
    st.error("SYSTEM CRITICAL FAILURE: T-800 Vision offline.")
    st.stop()

st.markdown("---")
st.caption(
    "CYBERDYNE SYSTEMS © 2026 // ALL RIGHTS RESERVED // TERMINATOR T-800 VISION"
)
