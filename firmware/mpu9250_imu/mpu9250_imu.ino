#include <Wire.h>

// Swapped in to replace an earlier IMU that had a well-documented I2C
// clock-stretching SDA/SCL setup-time violation, causing the bus to lock
// up ~15-20s into streaming on this jumper-wire setup, with no
// pure-firmware fix available short of an extra pull-up resistor.
//
// Upgraded to full 6DOF Madgwick sensor fusion (gyro+accel -> quaternion)
// so this behaves like a real AHRS (roll/pitch corrected by gravity, not
// just gyro-integrated) instead of the earlier yaw-only-from-gyro scheme.
// No magnetometer on this board, so yaw still drifts slowly over long
// sessions (nothing to anchor absolute heading) — same caveat as before,
// just with usable roll/pitch now too.
#define MPU_ADDR 0x68
#define SDA_PIN 21
#define SCL_PIN 22

// LSB/unit at the power-on-default full-scale ranges (±250 deg/s,
// ±2g) — we don't touch GYRO_CONFIG/ACCEL_CONFIG, so these defaults apply.
#define GYRO_LSB_PER_DPS 131.0f
#define ACCEL_LSB_PER_G 16384.0f
#define DEG_TO_RAD_F 0.017453293f
#define G_TO_MS2 9.80665f

// Madgwick filter gain — higher beta trusts the accelerometer more
// (faster roll/pitch correction, noisier at rest); this is the standard
// starting value for a ~250dps-class MEMS gyro.
#define MADGWICK_BETA 0.1f

float q0 = 1.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;  // w,x,y,z
unsigned long last_us = 0;
unsigned long last_event_ms = 0;
#define STALL_TIMEOUT_MS 3000

bool mpuOk = false;
float gyroBiasX = 0.0f, gyroBiasY = 0.0f, gyroBiasZ = 0.0f;

// Raw MEMS gyros have a per-unit DC offset (often 1-2+ deg/s) that's
// otherwise silently integrated forever. The accel feedback in the
// Madgwick filter corrects roll/pitch drift over time regardless, but
// starting from a calibrated bias still makes convergence much faster
// and keeps yaw (which has no corrective reference at all) far more
// accurate. Average ~1s of samples right after init (board must be
// stationary at boot) and subtract from every later gyro reading.
void calibrateGyroBias() {
  const int N = 200;
  float sumX = 0, sumY = 0, sumZ = 0;
  int got = 0;
  for (int i = 0; i < N; i++) {
    int16_t raw[6];
    if (readMpu(raw)) {
      sumX += raw[3] / GYRO_LSB_PER_DPS * DEG_TO_RAD_F;
      sumY += raw[4] / GYRO_LSB_PER_DPS * DEG_TO_RAD_F;
      sumZ += raw[5] / GYRO_LSB_PER_DPS * DEG_TO_RAD_F;
      got++;
    }
    delay(5);
  }
  gyroBiasX = (got > 0) ? (sumX / got) : 0.0f;
  gyroBiasY = (got > 0) ? (sumY / got) : 0.0f;
  gyroBiasZ = (got > 0) ? (sumZ / got) : 0.0f;
}

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

bool initMpu() {
  i2cBusRecovery();
  Wire.begin();
  Wire.setClock(400000);

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x75);  // WHO_AM_I
  if (Wire.endTransmission(false) != 0) return false;
  Wire.requestFrom(MPU_ADDR, 1, true);
  if (Wire.available() < 1) return false;
  uint8_t whoami = Wire.read();
  // 0x71/0x73: MPU9250/9255, 0x70: MPU6500, 0x68: MPU6050 — accept any,
  // we never touch the magnetometer regardless of which is actually on
  // the board.
  if (whoami != 0x71 && whoami != 0x73 && whoami != 0x70 && whoami != 0x68) {
    return false;
  }

  // PWR_MGMT_1: wake from sleep, clock source = PLL/gyro.
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0x01);
  if (Wire.endTransmission(true) != 0) return false;
  delay(50);

  q0 = 1.0f; q1 = 0.0f; q2 = 0.0f; q3 = 0.0f;
  last_us = micros();
  return true;
}

bool readMpu(int16_t out[7]) {
  // ACCEL_XOUT_H(0x3B) .. GYRO_ZOUT_L(0x48), 14 bytes, skip TEMP (2 bytes
  // in the middle) by reading all 14 and discarding indices 3,4.
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(MPU_ADDR, 14, true) != 14) return false;

  uint8_t buf[14];
  for (int i = 0; i < 14; i++) buf[i] = Wire.read();

  out[0] = (int16_t)((buf[0] << 8) | buf[1]);    // ax
  out[1] = (int16_t)((buf[2] << 8) | buf[3]);    // ay
  out[2] = (int16_t)((buf[4] << 8) | buf[5]);    // az
  // buf[6],buf[7] = temp, skipped
  out[3] = (int16_t)((buf[8] << 8) | buf[9]);    // gx
  out[4] = (int16_t)((buf[10] << 8) | buf[11]);  // gy
  out[5] = (int16_t)((buf[12] << 8) | buf[13]);  // gz
  return true;
}

static float invSqrt(float x) {
  return 1.0f / sqrtf(x);
}

// Standard Madgwick 6DOF (gyro+accel, no magnetometer) AHRS update.
// gx/gy/gz in rad/s, ax/ay/az in any consistent unit (normalised
// internally), dt in seconds.
void madgwickUpdate(float gx, float gy, float gz, float ax, float ay, float az, float dt) {
  float recipNorm;
  float s0, s1, s2, s3;
  float qDot1, qDot2, qDot3, qDot4;

  qDot1 = 0.5f * (-q1 * gx - q2 * gy - q3 * gz);
  qDot2 = 0.5f * (q0 * gx + q2 * gz - q3 * gy);
  qDot3 = 0.5f * (q0 * gy - q1 * gz + q3 * gx);
  qDot4 = 0.5f * (q0 * gz + q1 * gy - q2 * gx);

  if (!((ax == 0.0f) && (ay == 0.0f) && (az == 0.0f))) {
    recipNorm = invSqrt(ax * ax + ay * ay + az * az);
    ax *= recipNorm;
    ay *= recipNorm;
    az *= recipNorm;

    float _2q0 = 2.0f * q0, _2q1 = 2.0f * q1, _2q2 = 2.0f * q2, _2q3 = 2.0f * q3;
    float _4q0 = 4.0f * q0, _4q1 = 4.0f * q1, _4q2 = 4.0f * q2;
    float _8q1 = 8.0f * q1, _8q2 = 8.0f * q2;
    float q0q0 = q0 * q0, q1q1 = q1 * q1, q2q2 = q2 * q2, q3q3 = q3 * q3;

    s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay;
    s1 = _4q1 * q3q3 - _2q3 * ax + 4.0f * q0q0 * q1 - _2q0 * ay - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az;
    s2 = 4.0f * q0q0 * q2 + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az;
    s3 = 4.0f * q1q1 * q3 - _2q1 * ax + 4.0f * q2q2 * q3 - _2q2 * ay;
    recipNorm = invSqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3);
    s0 *= recipNorm; s1 *= recipNorm; s2 *= recipNorm; s3 *= recipNorm;

    qDot1 -= MADGWICK_BETA * s0;
    qDot2 -= MADGWICK_BETA * s1;
    qDot3 -= MADGWICK_BETA * s2;
    qDot4 -= MADGWICK_BETA * s3;
  }

  q0 += qDot1 * dt;
  q1 += qDot2 * dt;
  q2 += qDot3 * dt;
  q3 += qDot4 * dt;

  recipNorm = invSqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
  q0 *= recipNorm; q1 *= recipNorm; q2 *= recipNorm; q3 *= recipNorm;
}

void setup() {
  Serial.begin(115200);
  delay(500);
  mpuOk = initMpu();
  if (mpuOk) calibrateGyroBias();
  last_event_ms = millis();
  Serial.println(mpuOk ? "OK: MPU init succeeded" : "ERR: MPU init failed");
}

void loop() {
  if (!mpuOk) {
    delay(500);
    mpuOk = initMpu();
    if (mpuOk) calibrateGyroBias();
    last_event_ms = millis();
    return;
  }

  int16_t raw[6];
  if (readMpu(raw)) {
    unsigned long now_us = micros();
    float dt = (now_us - last_us) / 1000000.0f;
    last_us = now_us;
    last_event_ms = millis();

    float ax = raw[0] / ACCEL_LSB_PER_G * G_TO_MS2;
    float ay = raw[1] / ACCEL_LSB_PER_G * G_TO_MS2;
    float az = raw[2] / ACCEL_LSB_PER_G * G_TO_MS2;
    float gx = raw[3] / GYRO_LSB_PER_DPS * DEG_TO_RAD_F - gyroBiasX;
    float gy = raw[4] / GYRO_LSB_PER_DPS * DEG_TO_RAD_F - gyroBiasY;
    float gz = raw[5] / GYRO_LSB_PER_DPS * DEG_TO_RAD_F - gyroBiasZ;

    if (dt > 0.0001f && dt < 1.0f) {
      madgwickUpdate(gx, gy, gz, ax, ay, az, dt);
    }

    Serial.print("IMU,");
    Serial.print(q0, 5); Serial.print(',');
    Serial.print(q1, 5); Serial.print(',');
    Serial.print(q2, 5); Serial.print(',');
    Serial.print(q3, 5); Serial.print(',');
    Serial.print(gx, 5); Serial.print(',');
    Serial.print(gy, 5); Serial.print(',');
    Serial.print(gz, 5); Serial.print(',');
    Serial.print(ax, 5); Serial.print(',');
    Serial.print(ay, 5); Serial.print(',');
    Serial.println(az, 5);
  }

  if (millis() - last_event_ms > STALL_TIMEOUT_MS) {
    Serial.println("WARN: no IMU data, recovering bus and reinitializing");
    mpuOk = initMpu();
    if (mpuOk) calibrateGyroBias();
    last_event_ms = millis();
  }
}
