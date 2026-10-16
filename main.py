import os
import subprocess
import sys


# -------------------------------------------------------------
# LANÇADOR: permite rodar com "python main.py" ou com
# "streamlit run main.py". Os dois funcionam.
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# -------------------------------------------------------------
st.set_page_config(page_title="Detector de Objetos", page_icon="🔍", layout="wide")

st.title("🔍 Detector de Objetos em Python")
st.write("Identifique objetos em imagens salvas, usando a webcam ou um stream remoto.")


# -------------------------------------------------------------
# MODELO
# -------------------------------------------------------------
@st.cache_resource
def load_model():
    # Baixa automaticamente o yolov8n.pt na primeira execução
    return YOLO("yolov8n.pt")


with st.spinner("Carregando modelo YOLOv8..."):
    model = load_model()

# -------------------------------------------------------------
# BARRA LATERAL
# -------------------------------------------------------------
st.sidebar.header("⚙️ Configurações")
confidence = st.sidebar.slider(
    "Nível de Confiança Mínimo",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Ajuste para aceitar detecções mais ou menos precisas.",
)

source_option = st.sidebar.radio(
    "Escolha a Fonte da Imagem:", ("Upload de Arquivo", "Webcam", "Acesso Remoto")
)

# -------------------------------------------------------------
# ÁREA PRINCIPAL (as fontes serão implementadas nas próximas etapas)
# -------------------------------------------------------------
st.success(f"Modelo carregado com {len(model.names)} classes. Confiança mínima: {confidence:.2f}")

if source_option == "Upload de Arquivo":
    st.subheader("📁 Upload de Imagem")
    st.info("🚧 Em construção (Etapa 2).")
elif source_option == "Webcam":
    st.subheader("📷 Capturar da Webcam")
    st.info("🚧 Em construção (Etapa 3).")
else:
    st.subheader("🔗 Acesso Remoto à Câmera")
    st.info("🚧 Em construção (Etapas 4 e 5).")
