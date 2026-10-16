import tempfile
import threading
import uuid
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from radar import VehicleRadar

try:
    import av
    from aiortc.contrib.media import MediaRecorder
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    video_dependencies_available = True
except ImportError:
    av = None
    MediaRecorder = None
    WebRtcMode = None
    webrtc_streamer = None
    video_dependencies_available = False

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


@st.cache_resource
def load_inference_lock():
    return threading.Lock()


def make_video_callback(confidence, radar_enabled=False, radar_distance_m=5.0):
    inference_lock = load_inference_lock()
    radar = VehicleRadar()

    def process_frame(frame):
        image = frame.to_ndarray(format="bgr24")
        with inference_lock:
            if radar_enabled:
                result = model.track(
                    image,
                    conf=confidence,
                    classes=[1, 2, 3],
                    persist=True,
                    verbose=False,
                )[0]
            else:
                result = model.predict(image, conf=confidence, verbose=False)[0]
        annotated_image = result.plot()

        if radar_enabled:
            height, width = annotated_image.shape[:2]
            left_line = int(width * 0.35)
            right_line = int(width * 0.65)
            cv2.line(
                annotated_image,
                (left_line, 0),
                (left_line, height),
                (0, 255, 255),
                2,
            )
            cv2.line(
                annotated_image,
                (right_line, 0),
                (right_line, height),
                (0, 255, 255),
                2,
            )

            if result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().tolist()
                vehicle_names = {
                    1: "Bicicleta",
                    2: "Carro",
                    3: "Motocicleta",
                }

                for box, track_id in zip(result.boxes, track_ids):
                    class_id = int(box.cls[0])
                    object_confidence = float(box.conf[0])
                    x1, y1, x2, _ = box.xyxy[0].int().cpu().tolist()
                    center_x = (x1 + x2) / 2
                    speed_kmh = radar.update(
                        track_id,
                        center_x,
                        left_line,
                        right_line,
                        radar_distance_m,
                    )
                    label = (
                        f"{vehicle_names[class_id]} "
                        f"{object_confidence * 100:.0f}%"
                    )
                    if speed_kmh is not None:
                        label += f" | {speed_kmh:.1f} km/h"
                    cv2.putText(
                        annotated_image,
                        label,
                        (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

        return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

    return process_frame


model = load_model()
inference_lock = load_inference_lock()

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
radar_enabled = st.sidebar.checkbox("Modo Radar", value=False)
radar_distance_m = 5.0
if radar_enabled:
    radar_distance_m = st.sidebar.number_input(
        "Distância real entre as linhas (m)",
        min_value=0.1,
        max_value=100.0,
        value=5.0,
        step=0.5,
    )
    st.sidebar.caption(
        "Informe a distância real entre as duas linhas verticais do vídeo."
    )

upload_tab, webcam_tab, recording_tab = st.tabs(
    ["Upload de Arquivo", "Webcam", "Gravação"]
)

with upload_tab:
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

        with inference_lock:
            results = model.predict(source=image, conf=confidence, verbose=False)

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

with webcam_tab:
    photo_tab, realtime_tab = st.tabs(["Foto", "Em tempo real"])

    with photo_tab:
        st.subheader("📷 Capturar da Webcam")
        img_file_buffer = st.camera_input("Tire uma foto para analisar:")

        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(
                np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR
            )

            with inference_lock:
                results = model.predict(
                    source=cv2_img, conf=confidence, verbose=False
                )

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

    with realtime_tab:
        st.subheader("Identificação contínua")
        if video_dependencies_available:
            if radar_enabled:
                st.info("O Radar acompanha bicicletas, carros e motocicletas.")
            st.info("Inicie a câmera e permita o acesso solicitado pelo navegador.")
            webrtc_streamer(
                key="vision-ai-realtime",
                mode=WebRtcMode.SENDRECV,
                video_frame_callback=make_video_callback(
                    confidence,
                    radar_enabled=radar_enabled,
                    radar_distance_m=radar_distance_m,
                ),
                media_stream_constraints={"video": True, "audio": False},
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                },
                async_processing=True,
            )
        else:
            st.warning(
                "Vídeo contínuo requer streamlit-webrtc e aiortc instalados."
            )

with recording_tab:
    st.subheader("Gravação de vídeo")
    if video_dependencies_available:
        st.write("Grave um vídeo pela webcam e baixe o arquivo após encerrar a captura.")

        recording_id = st.session_state.setdefault("recording_id", uuid.uuid4().hex)
        recording_path = Path(tempfile.gettempdir()) / f"vision_ai_{recording_id}.webm"

        def create_recorder():
            return MediaRecorder(str(recording_path), format="webm")

        recording_context = webrtc_streamer(
            key="vision-ai-recording",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": True, "audio": False},
            in_recorder_factory=create_recorder,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
        )

        if (
            not recording_context.state.playing
            and recording_path.is_file()
            and recording_path.stat().st_size > 0
        ):
            recording_bytes = recording_path.read_bytes()
            st.video(recording_bytes, format="video/webm")
            st.download_button(
                "Baixar gravação",
                data=recording_bytes,
                file_name="gravacao.webm",
                mime="video/webm",
            )
    else:
        st.warning("Gravação requer streamlit-webrtc e aiortc instalados.")
