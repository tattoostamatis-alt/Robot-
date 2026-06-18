#include <Wire.h>
#include <Adafruit_BNO08x.h>
#include <esp_task_wdt.h>

// -1: do NOT use a hardware reset pin. Empirically (2026-06-18), driving
// the BNO085 RST line from a GPIO here lets the chip ACK on I2C and answer
// single command/response transactions (prodId query, sh2_setSensorConfig)
// completely normally, but it then never streams a single continuous
// sensor report afterward (confirmed with a minimal unmodified Adafruit
// example sketch, swapping only this one #define). Letting the library
// fall back to its I2C soft-reset packet (sent in i2chal_open) instead
// fixed it immediately — reports started flowing right away.
#define BNO08X_RESET -1
#define BNO08X_I2C_ADDR 0x4B
#define SDA_PIN 21
#define SCL_PIN 22

// If no sensor event arrives for this long, something has wedged (the I2C
// bus or the sensor hub itself) — force a full bus recovery + reinit
// instead of staying silently stuck forever.
#define STALL_TIMEOUT_MS 3000

Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;
bool bno08x_ok = false;
unsigned long last_event_ms = 0;

float q_i = 0, q_j = 0, q_k = 0, q_real = 1;
float gx = 0, gy = 0, gz = 0;
float ax = 0, ay = 0, az = 0;

void setReports() {
  bno08x.enableReport(SH2_ROTATION_VECTOR);
  bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED);
  bno08x.enableReport(SH2_LINEAR_ACCELERATION);
}

// Manually clocks SCL up to 9 times to release a slave that's holding SDA
// low after an interrupted transfer — standard I2C bus-recovery procedure.
// Needed because nothing else here resets the bus's electrical state (no
// hardware RST pin is used, see BNO08X_RESET above).
void i2cBusRecovery() {
  pinMode(SDA_PIN, INPUT);
  pinMode(SCL_PIN, OUTPUT);
  for (int i = 0; i < 9; i++) {
    digitalWrite(SCL_PIN, LOW);
    delayMicroseconds(5);
    digitalWrite(SCL_PIN, HIGH);
    delayMicroseconds(5);
  }
  pinMode(SDA_PIN, INPUT);
}

bool initBno() {
  i2cBusRecovery();
  Wire.begin();
  for (int attempt = 0; attempt < 5; attempt++) {
    if (bno08x.begin_I2C(BNO08X_I2C_ADDR)) {
      // Library default of 1MHz is too fast for jumper-wire connections.
      // 400kHz still wasn't reliable either — the bus would consistently
      // lock up ~13s into streaming (reproduced identically across
      // repeated runs), well after init succeeded, suggesting marginal
      // signal integrity rather than an init-time problem. Standard-mode
      // 100kHz is far more tolerant of the capacitance/noise on these
      // jumper wires.
      Wire.setClock(100000);
      setReports();
      return true;
    }
    delay(300);
  }
  return false;
}

void setup() {
  Serial.begin(115200);
  delay(2000);  // BNO085 needs ~650ms after power-up before it answers on I2C
  // initBno() (and the ESP32 Wire/I2C driver underneath it) has been
  // observed to hang indefinitely — not just fail — when the BNO085's
  // I2C state machine is wedged after a software-only (EN/DTR) reset of
  // the ESP32 itself; only cutting the board's actual power has reliably
  // cleared it. Subscribing to the TWDT (already initialized by the
  // arduino-esp32 core at boot, 5s/panic by default — see sdkconfig)
  // before this risky call means a true hang now force-reboots the chip
  // instead of staying silently dead forever, requiring someone to
  // notice and physically unplug/replug the USB cable.
  esp_task_wdt_add(NULL);
  bno08x_ok = initBno();
  esp_task_wdt_reset();
  last_event_ms = millis();
  Serial.println(bno08x_ok ? "OK: BNO08x init succeeded" : "ERR: BNO08x init failed");
}

void loop() {
  esp_task_wdt_reset();

  if (!bno08x_ok) {
    delay(500);
    bno08x_ok = initBno();
    esp_task_wdt_reset();
    last_event_ms = millis();
    return;
  }

  if (bno08x.wasReset()) {
    setReports();
  }

  if (bno08x.getSensorEvent(&sensorValue)) {
    last_event_ms = millis();
    switch (sensorValue.sensorId) {
      case SH2_ROTATION_VECTOR:
        q_i = sensorValue.un.rotationVector.i;
        q_j = sensorValue.un.rotationVector.j;
        q_k = sensorValue.un.rotationVector.k;
        q_real = sensorValue.un.rotationVector.real;
        Serial.print("IMU,");
        Serial.print(q_real, 5); Serial.print(',');
        Serial.print(q_i, 5);    Serial.print(',');
        Serial.print(q_j, 5);    Serial.print(',');
        Serial.print(q_k, 5);    Serial.print(',');
        Serial.print(gx, 5);     Serial.print(',');
        Serial.print(gy, 5);     Serial.print(',');
        Serial.print(gz, 5);     Serial.print(',');
        Serial.print(ax, 5);     Serial.print(',');
        Serial.print(ay, 5);     Serial.print(',');
        Serial.println(az, 5);
        break;
      case SH2_GYROSCOPE_CALIBRATED:
        gx = sensorValue.un.gyroscope.x;
        gy = sensorValue.un.gyroscope.y;
        gz = sensorValue.un.gyroscope.z;
        break;
      case SH2_LINEAR_ACCELERATION:
        ax = sensorValue.un.linearAcceleration.x;
        ay = sensorValue.un.linearAcceleration.y;
        az = sensorValue.un.linearAcceleration.z;
        break;
    }
  }

  if (millis() - last_event_ms > STALL_TIMEOUT_MS) {
    Serial.println("WARN: no IMU data, recovering bus and reinitializing");
    bno08x_ok = initBno();
    esp_task_wdt_reset();
    last_event_ms = millis();
  }
}
