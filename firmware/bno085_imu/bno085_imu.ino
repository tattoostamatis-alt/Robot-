// ESP32 + BNO085 (Adafruit BNO08x lib) -> streams "IMU,qw,qi,qj,qk,gx,gy,gz,ax,ay,az"
// over Serial at 115200, matching the format imu_node.py expects.
//
// BNO08X_RESET must stay -1 (no hardware reset pin wired) -- using the
// chip's RESET pin here previously caused getSensorEvent() to stall
// forever (~3s timeout, never recovering). The library's I2C soft-reset
// works correctly instead.
#include <Adafruit_BNO08x.h>

#define BNO08X_RESET -1
#define SDA_PIN 21
#define SCL_PIN 22

Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;

float gx = 0, gy = 0, gz = 0;
float ax = 0, ay = 0, az = 0;
float qw = 1, qi = 0, qj = 0, qk = 0;

unsigned long lastReportMs = 0;
const unsigned long STALL_MS = 1500;  // no report for this long -> assume hung I2C bus

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

void setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println("Starting BNO085...");

  Wire.begin(SDA_PIN, SCL_PIN);
  delay(300);

  bool ok = false;
  for (int attempt = 1; attempt <= 8 && !ok; attempt++) {
    Serial.print("Attempt "); Serial.print(attempt); Serial.println(": begin_I2C(0x4B)...");
    ok = bno08x.begin_I2C(0x4B, &Wire);
    if (!ok) delay(300);
  }
  if (ok) {
    Serial.println("BNO085 Found!");
    setReports();
  } else {
    Serial.println("ERR: BNO085 init failed, will keep retrying in loop()");
  }
  lastReportMs = millis();
}

void loop() {
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

  if (bno08x.wasReset()) {
    Serial.println("BNO085 reset, re-enabling reports");
    setReports();
  }

  if (bno08x.getSensorEvent(&sensorValue)) {
    lastReportMs = millis();
    switch (sensorValue.sensorId) {
      case SH2_GAME_ROTATION_VECTOR:
        qw = sensorValue.un.gameRotationVector.real;
        qi = sensorValue.un.gameRotationVector.i;
        qj = sensorValue.un.gameRotationVector.j;
        qk = sensorValue.un.gameRotationVector.k;
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
