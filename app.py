import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="Detector de Objetos", page_icon="🔍", layout="wide"
)

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
st.write("Identifique objetos em imagens salvas ou usando a sua webcam.")


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


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
    "Escolha a Fonte da Imagem:", ("Upload de Arquivo", "Webcam")
)

if source_option == "Upload de Arquivo":
    st.subheader("📁 Upload de Imagem")
    uploaded_file = st.file_uploader(
        "Selecione uma imagem do seu computador:",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original")
            st.image(image, use_container_width=True)

        results = model.predict(source=image, conf=confidence)

        res_plotted = results[0].plot()
        res_image = Image.fromarray(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))

        with col2:
            st.subheader("Objetos Identificados")
            st.image(res_image, use_container_width=True)

        st.subheader("📋 Resumo dos Objetos Detectados")
        boxes = results[0].boxes
        if len(boxes) > 0:
            detected_classes = [
                model.names[int(box.cls[0])] for box in boxes
            ]
            unique_classes = set(detected_classes)

            for obj in unique_classes:
                count = detected_classes.count(obj)
                st.write(f"- **{obj.capitalize()}**: {count} encontrado(s)")
        else:
            st.warning(
                "Nenhum objeto identificado. Tente diminuir o nível de confiança na barra lateral."
            )

elif source_option == "Webcam":
    st.subheader("📷 Capturar da Webcam")
    img_file_buffer = st.camera_input("Tire uma foto para analisar:")

    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(
            np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR
        )

        results = model.predict(source=cv2_img, conf=confidence)

        res_plotted = results[0].plot()
        res_image = Image.fromarray(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))

        st.subheader("Resultado da Detecção")
        st.image(res_image, use_container_width=True)

        st.subheader("📋 Resumo dos Objetos Detectados")
        boxes = results[0].boxes
        if len(boxes) > 0:
            detected_classes = [
                model.names[int(box.cls[0])] for box in boxes
            ]
            unique_classes = set(detected_classes)

            for obj in unique_classes:
                count = detected_classes.count(obj)
                st.write(f"- **{obj.capitalize()}**: {count} encontrado(s)")
        else:
            st.warning(
                "Nenhum objeto identificado. Tente diminuir o nível de confiança na barra lateral."
            )
