"""Captura rostros de una persona con la cámara y los guarda en Data/<nombre>.

Uso:  python Faces.py <nombre>
"""
import sys
import time

import cv2 as cv

import config


def main() -> int:
    name = config.safe_name(sys.argv[1] if len(sys.argv) > 1 else "")
    if not name:
        print("Nombre no válido. Use letras, números, espacios, '-' o '_' (máx. 50).")
        return 1

    person_dir = config.DATA_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)

    # Continuar la numeración si la persona ya tenía fotos (no sobrescribir).
    existing = [
        int(p.stem.split("_")[1]) for p in person_dir.glob("rostro_*.jpg")
        if p.stem.split("_")[1].isdigit()
    ]
    next_id = max(existing, default=-1) + 1
    print(f"Carpeta: {person_dir}")

    cascade = config.load_cascade()
    if cascade is None:
        print("ERROR: no se pudo cargar el clasificador Haar.")
        return 1

    cap = config.open_camera()
    if cap is None:
        print("ERROR: no se pudo abrir la cámara (revise RF_CAMERA / conexión).")
        return 1

    print(f"Capturando {config.CAPTURE_TOTAL} fotos. Presione ESC para salir.")
    saved = 0
    last_save = 0.0
    try:
        while saved < config.CAPTURE_TOTAL:
            ok, frame = cap.read()
            if not ok:
                print("No se pudo leer el frame de la cámara.")
                break

            h0, w0 = frame.shape[:2]
            if w0 != 640:
                frame = cv.resize(frame, (640, int(h0 * 640 / w0)))
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = config.detect_faces(gray, cascade)

            if len(faces):
                # Solo el rostro más grande: evita guardar fondos/otras personas.
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if time.time() - last_save >= config.CAPTURE_INTERVAL:
                    rostro = cv.resize(
                        gray[y:y + h, x:x + w],
                        (config.FACE_SIZE, config.FACE_SIZE),
                        interpolation=cv.INTER_CUBIC,
                    )
                    cv.imwrite(str(person_dir / f"rostro_{next_id}.jpg"), rostro)
                    next_id += 1
                    saved += 1
                    last_save = time.time()
                    if saved % 50 == 0:
                        print(f"  {saved}/{config.CAPTURE_TOTAL}")

            cv.putText(frame, f"{name}: {saved}/{config.CAPTURE_TOTAL}  (ESC=salir)",
                       (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv.imshow("Captura de rostros", frame)
            if cv.waitKey(1) == 27:
                break
    finally:
        cap.release()
        cv.destroyAllWindows()

    print(f"Capturadas {saved} fotos para '{name}'. Ahora entrene el modelo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
