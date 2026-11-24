"""Reconocimiento facial en vivo; abre la puerta (Arduino) si reconoce a alguien.

Uso:  python FacialRecognizer.py        (ESC para salir)
"""
import sys
import threading
import time

import cv2 as cv
import requests

import config


def abrir_puerta() -> None:
    """Pide al Arduino que abra. Corre en un hilo para no congelar el video."""
    params = {"token": config.ARDUINO_TOKEN} if config.ARDUINO_TOKEN else None
    try:
        r = requests.get(f"{config.ARDUINO_URL}/open", params=params,
                         timeout=config.DOOR_TIMEOUT)
        print(f"Arduino respondió {r.status_code}: {r.text.strip()}")
    except requests.RequestException as e:
        print(f"Error al contactar al Arduino: {e}")


def guardar_registro(frame_color, box, nombre: str) -> None:
    x, y, w, h = box
    config.REGISTRO_DIR.mkdir(parents=True, exist_ok=True)
    archivo = config.REGISTRO_DIR / f"{nombre}_{int(time.time())}.jpg"
    cv.imwrite(str(archivo), frame_color[y:y + h, x:x + w])
    print(f"Registro guardado: {archivo.name}")


def main() -> int:
    if not config.MODEL_PATH.exists():
        print("ERROR: no hay modelo entrenado. Use 'Entrenar modelo' primero.")
        return 1

    meta = config.load_meta()
    people = meta["people"]
    threshold = float(meta["threshold"])
    face_size = int(meta.get("face_size", config.FACE_SIZE))
    equalize = bool(meta.get("equalize", False))

    recognizer = config.make_recognizer(meta["algorithm"])
    recognizer.read(str(config.MODEL_PATH))
    print(f"Modelo {meta['algorithm']} cargado. Personas: {people}")

    cascade = config.load_cascade()
    if cascade is None:
        print("ERROR: no se pudo cargar el clasificador Haar.")
        return 1

    cap = config.open_camera()
    if cap is None:
        print("ERROR: no se pudo abrir la cámara (revise RF_CAMERA / conexión).")
        return 1

    print(f"Arduino: {config.ARDUINO_URL}  |  ESC para salir.")
    last_action = 0.0
    streak_name, streak = None, 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No se pudo leer el frame de la cámara.")
                break

            raw = frame.copy()  # sin rectángulos, para los registros
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = config.detect_faces(gray, cascade)
            cooldown_ok = time.time() - last_action >= config.DOOR_COOLDOWN
            seen_known = None

            for (x, y, w, h) in faces:
                rostro = config.preprocess_face(gray[y:y + h, x:x + w], face_size, equalize)
                label, dist = recognizer.predict(rostro)
                conocido = dist < threshold and 0 <= label < len(people)

                if conocido:
                    nombre = people[label]
                    seen_known = nombre
                    color = (0, 255, 0)
                    texto = f"{nombre} ({dist:.0f})"
                else:
                    nombre = "Desconocido"
                    color = (0, 0, 255)
                    texto = f"Desconocido ({dist:.0f})"
                cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv.putText(frame, texto, (x, y - 8), cv.FONT_HERSHEY_SIMPLEX,
                           0.7, color, 2, cv.LINE_AA)

                if not cooldown_ok:
                    continue
                if conocido:
                    # Exige varios frames seguidos para evitar falsos positivos.
                    streak = streak + 1 if streak_name == nombre else 1
                    streak_name = nombre
                    if streak >= config.CONFIRM_FRAMES:
                        print(f"Acceso concedido: {nombre} (distancia {dist:.1f})")
                        threading.Thread(target=abrir_puerta, daemon=True).start()
                        guardar_registro(raw, (x, y, w, h), nombre)
                        last_action = time.time()
                        streak_name, streak = None, 0
                else:
                    print(f"Persona desconocida (distancia {dist:.1f})")
                    guardar_registro(raw, (x, y, w, h), "PersonaDesconocida")
                    last_action = time.time()

            if seen_known is None:
                streak_name, streak = None, 0

            cv.imshow("Reconocimiento facial", frame)
            if cv.waitKey(1) == 27:
                break
    finally:
        cap.release()
        cv.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
