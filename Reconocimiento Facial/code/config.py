"""Configuración y utilidades compartidas por todos los scripts del proyecto.

Todas las rutas son relativas a la ubicación del proyecto, así que funciona
igual en cualquier PC / carpeta. Se puede cambiar el comportamiento sin editar
código mediante variables de entorno:

    RF_CAMERA         índice de cámara (0, 1...) o URL RTSP. Por defecto 0.
    RF_ARDUINO_URL    URL del Arduino. Por defecto http://10.50.4.205
    RF_ARDUINO_TOKEN  clave compartida con el Arduino (ver arduino.ino).
"""
from __future__ import annotations

import json
import os
import platform
import re
from pathlib import Path

# ----------------------------------------------------------------------------
# Rutas
# ----------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent
BASE_DIR = CODE_DIR.parent
DATA_DIR = BASE_DIR / "Data"
REGISTRO_DIR = BASE_DIR / "Registro"
MODELS_DIR = BASE_DIR / "Models"
MODEL_PATH = MODELS_DIR / "modelo.xml"
META_PATH = MODELS_DIR / "modelo.json"
LOCAL_CASCADE = CODE_DIR / "haarcascade_frontalface_default.xml"

# ----------------------------------------------------------------------------
# Parámetros
# ----------------------------------------------------------------------------
FACE_SIZE = 150                 # lado (px) de los rostros guardados/entrenados
CAPTURE_TOTAL = 400             # fotos a capturar por persona
CAPTURE_INTERVAL = 0.3          # segundos entre fotos
MIN_FACE_PX = 80                # rostros más pequeños se ignoran
CONFIRM_FRAMES = 3              # frames seguidos con la misma persona para abrir
DOOR_COOLDOWN = 5               # segundos entre aperturas / registros
DOOR_TIMEOUT = 3                # timeout HTTP hacia el Arduino
MIN_IMAGES_WARNING = 20         # avisa si una persona tiene menos fotos

# Umbral de distancia por defecto según algoritmo (menor distancia = más parecido)
DEFAULT_THRESHOLDS = {"fisher": 200.0, "lbph": 65.0, "eigen": 5700.0}

ARDUINO_URL = os.environ.get("RF_ARDUINO_URL", "http://10.50.4.205").rstrip("/")
ARDUINO_TOKEN = os.environ.get("RF_ARDUINO_TOKEN", "")

_cam = os.environ.get("RF_CAMERA", "0").strip()
CAMERA_SOURCE = int(_cam) if _cam.isdigit() else _cam

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_NAME_RE = re.compile(r"^\w[\w \-]{0,49}$", re.UNICODE)


# ----------------------------------------------------------------------------
# Personas / nombres
# ----------------------------------------------------------------------------
def safe_name(name: str) -> str | None:
    """Devuelve el nombre limpio si es válido como carpeta, o None.

    Evita nombres como '..\\..\\Windows' que escaparían de la carpeta Data.
    """
    name = (name or "").strip()
    return name if _NAME_RE.match(name) else None


def list_people() -> list[str]:
    """Personas registradas (subcarpetas de Data), en orden alfabético."""
    if not DATA_DIR.is_dir():
        return []
    return sorted(
        p.name for p in DATA_DIR.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def list_images(person_dir: Path) -> list[Path]:
    return sorted(
        f for f in person_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )


# ----------------------------------------------------------------------------
# Modelo
# ----------------------------------------------------------------------------
def load_meta() -> dict:
    """Metadatos del modelo entrenado (algoritmo, etiquetas, umbral...).

    Si no existe modelo.json (modelos antiguos) se asume Fisherfaces sin
    ecualización y las personas en orden alfabético, como hacía la v1.
    """
    if META_PATH.exists():
        with open(META_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {
        "algorithm": "fisher",
        "people": list_people(),
        "equalize": False,
        "threshold": DEFAULT_THRESHOLDS["fisher"],
        "face_size": FACE_SIZE,
    }


def make_recognizer(algorithm: str):
    import cv2 as cv

    factories = {
        "fisher": cv.face.FisherFaceRecognizer_create,
        "lbph": cv.face.LBPHFaceRecognizer_create,
        "eigen": cv.face.EigenFaceRecognizer_create,
    }
    if algorithm not in factories:
        raise ValueError(f"Algoritmo desconocido: {algorithm}")
    return factories[algorithm]()


def preprocess_face(gray, face_size: int = FACE_SIZE, equalize: bool = False):
    """Normaliza un recorte en gris: tamaño fijo y (opcional) ecualización."""
    import cv2 as cv

    if gray.shape[0] != face_size or gray.shape[1] != face_size:
        gray = cv.resize(gray, (face_size, face_size), interpolation=cv.INTER_CUBIC)
    if equalize:
        gray = cv.equalizeHist(gray)
    return gray


# ----------------------------------------------------------------------------
# Cámara y detector
# ----------------------------------------------------------------------------
def open_camera(width: int = 640, height: int = 480):
    """Abre la cámara configurada. Devuelve None si no se pudo abrir."""
    import cv2 as cv

    if isinstance(CAMERA_SOURCE, int):
        backend = cv.CAP_DSHOW if platform.system() == "Windows" else cv.CAP_ANY
        cap = cv.VideoCapture(CAMERA_SOURCE, backend)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    else:  # URL RTSP / archivo de video
        cap = cv.VideoCapture(CAMERA_SOURCE, cv.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def load_cascade():
    """Carga el clasificador Haar (copia local, o la incluida en OpenCV)."""
    import cv2 as cv

    path = LOCAL_CASCADE if LOCAL_CASCADE.exists() else (
        Path(cv.data.haarcascades) / "haarcascade_frontalface_default.xml"
    )
    cascade = cv.CascadeClassifier(str(path))
    return None if cascade.empty() else cascade


def detect_faces(gray, cascade):
    return cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5,
        minSize=(MIN_FACE_PX, MIN_FACE_PX),
    )
