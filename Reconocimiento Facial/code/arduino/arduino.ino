#include <SPI.h>
#include <Ethernet.h>

// ---- Configuración de red (debe coincidir con RF_ARDUINO_URL en el PC) ----
byte mac[] = { 0xDE, 0xAD, 0xBE, 0xEF, 0xFE, 0xED };
IPAddress ip(10, 50, 4, 205);   // IP fija del Arduino (cambiar según su red)
EthernetServer server(80);

// ---- Relé ----
const int relayPin = 7;
const unsigned long OPEN_TIME_MS = 3000;  // tiempo que permanece abierta la puerta

// ---- Seguridad ----
// Si no está vacío, la petición debe incluir ?token=<TOKEN>.
// Use letras y números y el mismo valor en la variable RF_ARDUINO_TOKEN del PC.
const char TOKEN[] = "";

bool relayOn = false;
unsigned long relayStart = 0;

void setup() {
  pinMode(relayPin, OUTPUT);
  digitalWrite(relayPin, LOW);  // El relé empieza apagado

  Ethernet.begin(mac, ip);
  server.begin();

  Serial.begin(9600);
  Serial.println(F("Servidor listo..."));
}

bool tokenValido(const String &requestLine) {
  if (strlen(TOKEN) == 0) return true;
  String needle = String("token=") + TOKEN;
  int i = requestLine.indexOf(needle);
  if (i < 0) return false;
  char next = requestLine.charAt(i + needle.length());
  return next == ' ' || next == '&' || next == 0;  // coincidencia exacta
}

void responder(EthernetClient &client, int code, const __FlashStringHelper *msg) {
  client.print(F("HTTP/1.1 "));
  client.print(code);
  client.println(code == 200 ? F(" OK") : F(" Error"));
  client.println(F("Content-Type: text/plain"));
  client.println(F("Connection: close"));
  client.println();
  client.println(msg);
}

void loop() {
  // Cierra el relé sin bloquear el servidor (antes se usaba delay(3000)).
  if (relayOn && millis() - relayStart >= OPEN_TIME_MS) {
    digitalWrite(relayPin, LOW);
    relayOn = false;
    Serial.println(F("Puerta cerrada."));
  }

  EthernetClient client = server.available();
  if (!client) return;

  String requestLine = "";
  bool firstLineDone = false;
  bool lineBlank = true;
  unsigned long t0 = millis();

  while (client.connected() && millis() - t0 < 2000) {
    if (!client.available()) continue;
    char c = client.read();

    if (!firstLineDone) {
      if (c == '\n') firstLineDone = true;
      else if (c != '\r' && requestLine.length() < 120) requestLine += c;
    }

    if (c == '\n' && lineBlank) {  // fin de las cabeceras HTTP: responder
      Serial.println(requestLine);
      if (!requestLine.startsWith("GET /open")) {
        responder(client, 404, F("No encontrado."));
      } else if (!tokenValido(requestLine)) {
        responder(client, 403, F("Token invalido."));
      } else {
        digitalWrite(relayPin, HIGH);
        relayOn = true;
        relayStart = millis();
        responder(client, 200, F("Puerta abierta."));
      }
      break;
    }
    if (c == '\n') lineBlank = true;
    else if (c != '\r') lineBlank = false;
  }
  delay(1);
  client.stop();
}
