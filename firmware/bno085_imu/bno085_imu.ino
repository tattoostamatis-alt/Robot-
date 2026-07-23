// ESP32 + BNO085 (Adafruit BNO08x lib) -> streams "IMU,qw,qi,qj,qk,gx,gy,gz,ax,ay,az"
// over Serial at 115200, matching the format imu_node.py expects.
//
// BNO08X_RESET stays -1 on purpose: handing the chip's RESET pin to the
// library previously made getSensorEvent() stall forever (~3s timeout, never
// recovering). We own the pin ourselves instead (BNO_RESET_PIN) and pulse it
// only on the recovery path -- the library never touches it.
#include <Adafruit_BNO08x.h>

#define BNO08X_RESET -1
#define SDA_PIN 21
#define SCL_PIN 22

// BNO085 nRESET -> ESP32 GPIO18 (wired 2026-07-23). Active low.
//
// Why this exists: the BNO085's protocol state can wedge in a way that keeps
// it ACKing its I2C address -- a bare bus scan still reports FOUND 0x4B on
// every pass -- while begin_I2C() fails forever. Nothing on the ESP32 side
// cleared that before: ESP.restart() reboots us but leaves the sensor powered
// and still wedged, which is why the only recoveries that ever worked were
// accidental power cycles (pulling a dupont cuts 3V3) or a reflash. Four
// outages in three days were all this, not the loose wiring they looked like.
// With nRESET on a GPIO the firmware can clear it unaided.
#define BNO_RESET_PIN 18

Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;

float gx = 0, gy = 0, gz = 0;
float ax = 0, ay = 0, az = 0;
float qw = 1, qi = 0, qj = 0, qk = 0;

unsigned long lastReportMs = 0;
const unsigned long STALL_MS = 1500;  // no report for this long -> assume hung I2C bus

// The rotation vector can stream fine while the gyro report silently never
// starts (setReports() retries can all fail on the flaky bus and nothing
// re-tries afterwards, so gx/gy/gz stay 0 forever and the EKF yaw-rate input
// is a constant 0 that fights every turn). Watch the gyro on its own: if it
// stays silent while other reports flow, re-enable it, then reboot as a last
// resort.
unsigned long lastGyroMs = 0;
const unsigned long GYRO_STALL_MS = 3000;
int gyroReenableAttempts = 0;

// Per-report rate counters, printed every 5s as a "DIAG,..." line (ignored by
// imu_node) so a healthy 100Hz gyro can be told apart from one that only
// trickles in stale updates.
unsigned long diagWindowStartMs = 0;
unsigned int rotCount = 0, gyroCount = 0;

// False until begin_I2C() has succeeded. Every bno08x.* call must be guarded
// by this: the library dereferences internal state that only exists after a
// successful init, so calling e.g. wasReset() on a failed init panics the
// ESP32 (Guru Meditation / LoadProhibited) instead of retrying.
bool imuReady = false;
unsigned long lastInitRetryMs = 0;
const unsigned long INIT_RETRY_MS = 2000;

void setReports() {
  // GAME_ROTATION_VECTOR = gyro + accel fusion WITHOUT the magnetometer.
  // The magnetometer-fused SH2_ROTATION_VECTOR gave an "absolute" heading
  // that was corrupted indoors by the Roomba's DC motors and nearby metal,
  // so the yaw jumped/drifted unpredictably -> EKF odom->base_link rotated
  // -> the LiDAR scan no longer lined up with the mapped walls and AMCL
  // could not stay converged ("the compass doesn't work"). For SLAM/AMCL we
  // do NOT need true north -- only a stable, low-drift relative heading, and
  // AMCL corrects the slow gyro yaw drift via scan matching. The game
  // rotation vector yaw=0 is an arbitrary direction each boot, which is fine
  // because AMCL computes map->odom anyway.
  //
  // Enable reports ONE AT A TIME with a settle delay and verify the return
  // value. Enabling several reports back-to-back silently drops some of them
  // over the flaky I2C bus -- that is why SH2_GYROSCOPE_CALIBRATED never
  // streamed (gx/gy/gz stuck at 0, so the EKF's yaw-rate input was a constant
  // 0 that fought every turn). We now request only the two reports the EKF
  // consumes: absolute yaw (game rotation vector) + yaw-rate (gyro).
  // SH2_LINEAR_ACCELERATION is dropped -- imu0_config leaves ax/ay/az off, so
  // it was pure bus/serial load; ax/ay/az are just streamed as 0.
  bool okRot = false, okGyro = false;
  for (int i = 0; i < 5 && !okRot; i++) {
    okRot = bno08x.enableReport(SH2_GAME_ROTATION_VECTOR, 10000);   // 100Hz
    delay(50);
  }
  for (int i = 0; i < 5 && !okGyro; i++) {
    okGyro = bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, 10000);  // 100Hz
    delay(50);
  }
  Serial.print("Reports enabled: rot="); Serial.print(okRot);
  Serial.print(" gyro="); Serial.println(okGyro);
}

// Try both BNO085 addresses. Which one the chip answers on is set by its
// ADR/DI pin, so a replacement module can land on 0x4A even though every
// board used here so far has been 0x4B.
// Hold nRESET low, release, then wait for the sensor to come back. The
// datasheet wants ~90ms before it answers I2C; 300ms is deliberately generous
// because this only runs when we are already failing. Harmless no-op if the
// pin is not wired yet -- the chip simply never sees it.
void pulseReset() {
  digitalWrite(BNO_RESET_PIN, LOW);
  delay(20);
  digitalWrite(BNO_RESET_PIN, HIGH);
  delay(300);
}

bool initBno() {
  const uint8_t addrs[] = {0x4B, 0x4A};
  for (int attempt = 1; attempt <= 4; attempt++) {
    // Clear a wedged sensor before each round rather than retrying a handshake
    // that will keep failing for exactly as long as the chip stays powered.
    pulseReset();
    for (uint8_t addr : addrs) {
      Serial.print("Attempt "); Serial.print(attempt);
      Serial.print(": begin_I2C(0x"); Serial.print(addr, HEX); Serial.println(")...");
      if (bno08x.begin_I2C(addr, &Wire)) {
        Serial.print("BNO085 Found at 0x"); Serial.println(addr, HEX);
        return true;
      }
      delay(300);
    }
  }
  return false;
}

void setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println("Starting BNO085...");

  // Release reset before touching the bus. Drive it HIGH explicitly: left as a
  // floating input the sensor could sit held in reset and look "dead".
  pinMode(BNO_RESET_PIN, OUTPUT);
  digitalWrite(BNO_RESET_PIN, HIGH);

  Wire.begin(SDA_PIN, SCL_PIN);
  delay(300);

  imuReady = initBno();
  if (imuReady) {
    setReports();
  } else {
    Serial.println("ERR: BNO085 init failed, will keep retrying in loop()");
  }
  lastReportMs = millis();
  lastGyroMs = millis();
  lastInitRetryMs = millis();
}

void loop() {
  // Init failed (bad wiring, unpowered chip): retry the I2C handshake on a
  // timer and touch nothing else. Rebooting instead would spin the ESP32 in
  // a ~1s reset loop that never prints anything useful, and falling through
  // to the bno08x.* calls below would panic outright.
  if (!imuReady) {
    if (millis() - lastInitRetryMs > INIT_RETRY_MS) {
      lastInitRetryMs = millis();
      Serial.println("Retrying BNO085 init...");
      imuReady = initBno();
      if (imuReady) {
        setReports();
        lastReportMs = millis();
        lastGyroMs = millis();
      }
    }
    return;
  }

  // The chip/bus connection here is flaky (no hardware fix in use) --
  // a manual Wire.end()/bit-bang bus recovery was tried but corrupts
  // the Adafruit_BNO08x library's internal state and crashes (Guru
  // Meditation/LoadProhibited). A full clean reboot is slower per
  // cycle (~1-3s) but never crashes and reliably re-syncs everything,
  // so it's the safe choice for "retry forever without touching wiring".
  if (millis() - lastReportMs > STALL_MS) {
    Serial.println("Stall detected, rebooting to recover...");
    delay(50);
    ESP.restart();
  }

  if (millis() - diagWindowStartMs >= 5000) {
    float secs = (millis() - diagWindowStartMs) / 1000.0;
    Serial.print("DIAG,rotHz="); Serial.print(rotCount / secs, 1);
    Serial.print(",gyroHz="); Serial.println(gyroCount / secs, 1);
    rotCount = 0; gyroCount = 0;
    diagWindowStartMs = millis();
  }

  if (bno08x.wasReset()) {
    Serial.println("BNO085 reset, re-enabling reports");
    setReports();
    lastGyroMs = millis();
  }

  if (millis() - lastGyroMs > GYRO_STALL_MS) {
    if (gyroReenableAttempts < 3) {
      gyroReenableAttempts++;
      Serial.print("Gyro silent, re-enabling reports (attempt ");
      Serial.print(gyroReenableAttempts); Serial.println("/3)");
      setReports();
      lastGyroMs = millis();  // fresh window before judging again
    } else {
      Serial.println("Gyro still silent after re-enables, rebooting...");
      delay(50);
      ESP.restart();
    }
  }

  if (bno08x.getSensorEvent(&sensorValue)) {
    lastReportMs = millis();
    switch (sensorValue.sensorId) {
      case SH2_GAME_ROTATION_VECTOR:
        rotCount++;
        qw = sensorValue.un.gameRotationVector.real;
        qi = sensorValue.un.gameRotationVector.i;
        qj = sensorValue.un.gameRotationVector.j;
        qk = sensorValue.un.gameRotationVector.k;
        break;
      case SH2_GYROSCOPE_CALIBRATED:
        gx = sensorValue.un.gyroscope.x;
        gy = sensorValue.un.gyroscope.y;
        gz = sensorValue.un.gyroscope.z;
        lastGyroMs = millis();
        gyroReenableAttempts = 0;
        gyroCount++;
        break;
      case SH2_LINEAR_ACCELERATION:
        ax = sensorValue.un.linearAcceleration.x;
        ay = sensorValue.un.linearAcceleration.y;
        az = sensorValue.un.linearAcceleration.z;
        break;
    }

    Serial.print("IMU,");
    Serial.print(qw, 6); Serial.print(",");
    Serial.print(qi, 6); Serial.print(",");
    Serial.print(qj, 6); Serial.print(",");
    Serial.print(qk, 6); Serial.print(",");
    Serial.print(gx, 6); Serial.print(",");
    Serial.print(gy, 6); Serial.print(",");
    Serial.print(gz, 6); Serial.print(",");
    Serial.print(ax, 6); Serial.print(",");
    Serial.print(ay, 6); Serial.print(",");
    Serial.println(az, 6);
  }
}
