# RF1 · Sistema de Reconocimiento Facial

Captura rostros, entrena un modelo (OpenCV) y reconoce personas en vivo desde la cámara.
Si reconoce a alguien, pide a un Arduino con relé que abra la puerta.

## Estructura

```
Reconocimiento Facial/
├── main.py                  Interfaz gráfica (ejecutar este)
├── code/
│   ├── config.py            Rutas y parámetros compartidos
│   ├── Faces.py             Captura de rostros → Data/<nombre>/
│   ├── ModelsFaceRecognizer.py   Entrenamiento → Models/
│   ├── FacialRecognizer.py  Reconocimiento en vivo + apertura de puerta
│   └── arduino/arduino.ino  Firmware del Arduino (Ethernet + relé)
├── Data/                    Fotos por persona (no se versiona)
├── Models/                  modelo.xml + modelo.json (algoritmo, personas, umbral)
└── Registro/                Capturas de accesos y desconocidos (no se versiona)
```

## Instalación

```bash
pip install -r requirements.txt
cd "Reconocimiento Facial"
python main.py
```

En Linux, `tkinter` puede requerir `sudo apt install python3-tk`.

## Uso

1. **Captura de cara**: escriba un nombre y capture ~400 fotos (ESC para parar). Mueva un poco la cabeza y varíe la luz.
2. **Entrenar modelo**: tras añadir o borrar personas.
3. **Reconocimiento facial**: ESC para salir.

Fisherfaces necesita **al menos 2 personas**. Con una sola, entrene con LBPH:
`python code/ModelsFaceRecognizer.py --algo lbph`.

## Configuración (variables de entorno, opcionales)

| Variable | Por defecto | Descripción |
|---|---|---|
| `RF_CAMERA` | `0` | Índice de cámara o URL RTSP |
| `RF_ARDUINO_URL` | `http://10.50.4.205` | Dirección del Arduino |
| `RF_ARDUINO_TOKEN` | (vacío) | Clave compartida con el Arduino |

El umbral de reconocimiento se puede ajustar en `Models/modelo.json` (`threshold`; menor = más estricto).

## Arduino

El PC llama a `GET /open?token=...`. Para proteger la puerta, defina el mismo `TOKEN` en
`arduino.ino` y `RF_ARDUINO_TOKEN` en el PC. La IP del sketch debe coincidir con `RF_ARDUINO_URL`.

## Privacidad

Las fotos de `Data/` y `Registro/` son datos biométricos: están excluidas de git (`.gitignore`).
