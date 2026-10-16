import os
import subprocess
import sys


def _running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:
        return False


if __name__ == "__main__" and not _running_in_streamlit():
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)],
            check=False,
        )
    except KeyboardInterrupt:
        pass
    sys.exit(0)


import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Detector de Objetos", page_icon="🔍", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stAppDeployButton"],
    [data-testid="stMainMenuItem-print"],
    [data-testid="stMainMenuItem-autoRerun"] {
        display: none;
    }

    [data-testid="stMainMenuItem-rerun"] [data-testid="stMainMenuItemLabel"] {
        font-size: 0;
    }

    [data-testid="stMainMenuItem-rerun"] [data-testid="stMainMenuItemLabel"]::after {
        content: "Recarregar";
        font-size: 0.875rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔍 Detector de Objetos em Python")
st.write("Identifique objetos em imagens salvas, usando a webcam ou um stream remoto.")


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


with st.spinner("Carregando modelo YOLOv8..."):
    model = load_model()

st.sidebar.header("⚙️ Configurações")
confidence_percent = st.sidebar.slider(
    "Nível de Confiança Mínimo",
    min_value=0,
    max_value=100,
    value=50,
    step=1,
    format="%d%%",
    help="Valores maiores exigem mais confiança para exibir uma detecção.",
)
confidence = confidence_percent / 100

source_option = st.sidebar.radio(
    "Escolha a Fonte da Imagem:", ("Upload de Arquivo", "Webcam", "Acesso Remoto")
)

st.success(f"Modelo carregado com {len(model.names)} classes. Confiança mínima: {confidence_percent}%")

if source_option == "Upload de Arquivo":
    st.subheader("📁 Upload de Imagem")
    st.info("🚧 Em construção (Etapa 2).")
elif source_option == "Webcam":
    st.subheader("📷 Capturar da Webcam")
    st.info("🚧 Em construção (Etapa 3).")
else:
    st.subheader("🔗 Acesso Remoto à Câmera")
    st.info("🚧 Em construção (Etapas 4 e 5).")
