"""Entrena el modelo con las fotos de Data/ y lo guarda en Models/.

Uso:  python ModelsFaceRecognizer.py [--algo fisher|lbph|eigen]

 - fisher (por defecto): necesita al menos 2 personas.
 - lbph: funciona con 1 persona y suele ser más robusto a cambios de luz.
"""
import argparse
import json
import os
import sys

import cv2 as cv
import numpy as np

import config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["fisher", "lbph", "eigen"], default="fisher")
    parser.add_argument("--no-equalize", action="store_true",
                        help="no ecualizar el histograma (menos robusto a la luz)")
    args = parser.parse_args()
    equalize = not args.no_equalize

    people = config.list_people()
    print("Personas encontradas:", people or "ninguna")

    faces, labels, kept_people = [], [], []
    for name in people:
        images = config.list_images(config.DATA_DIR / name)
        count = 0
        for path in images:
            img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  (omitida, no se pudo leer) {path.name}")
                continue
            faces.append(config.preprocess_face(img, config.FACE_SIZE, equalize))
            labels.append(len(kept_people))
            count += 1
        if count == 0:
            print(f"  '{name}': sin imágenes válidas, se omite.")
            continue
        kept_people.append(name)
        aviso = "  <-- pocas fotos, el reconocimiento será poco fiable" \
            if count < config.MIN_IMAGES_WARNING else ""
        print(f"  {name}: {count} imágenes{aviso}")

    if not kept_people:
        print("ERROR: no hay imágenes para entrenar. Capture rostros primero.")
        return 1
    if args.algo in ("fisher",) and len(kept_people) < 2:
        print("ERROR: Fisherfaces necesita al menos 2 personas distintas.\n"
              "       Capture otra persona o use:  --algo lbph")
        return 1

    print(f"Entrenando ({args.algo}) con {len(faces)} imágenes...")
    recognizer = config.make_recognizer(args.algo)
    recognizer.train(faces, np.array(labels))

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_model = config.MODELS_DIR / "modelo.tmp.xml"
    recognizer.write(str(tmp_model))
    os.replace(tmp_model, config.MODEL_PATH)  # escritura atómica

    meta = {
        "algorithm": args.algo,
        "people": kept_people,
        "equalize": equalize,
        "threshold": config.DEFAULT_THRESHOLDS[args.algo],
        "face_size": config.FACE_SIZE,
    }
    with open(config.META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Modelo guardado en {config.MODEL_PATH}")
    print(f"Umbral de reconocimiento: {meta['threshold']} (editable en modelo.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
