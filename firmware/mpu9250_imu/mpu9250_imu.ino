#include <Wire.h>

// Swapped in to replace the BNO085 (see bno085_imu.ino's notes) after
// confirming the BNO08x's well-documented I2C clock-stretching SDA/SCL
// setup-time violation (Adafruit forums/GitHub issue #53) was causing the
// bus to lock up ~15-20s into streaming on this jumper-wire setup, with no
// pure-firmware fix available short of an extra pull-up resistor.
//
// ekf.yaml only fuses orientation.yaw + angular_velocity.z from imu/data
// (see its imu0_config — every other field is masked out), so a plain
// MPU6500/9250 (gyro+accel, no magnetometer/AK8963 needed) is sufficient:
// yaw is integrated from the gyro Z rate alone instead of a full 9-axis
// AHRS fusion. This will drift slowly over long sessions (no magnetometer
// to correct it) but is far simpler/more reliable, and fine for both the
// short in-place rotation calibration test and the after-turn-skew fix
// ekf.yaml was originally written for.
#define MPU_ADDR 0x68
#define SDA_PIN 21
#define SCL_PIN 22

// LSB/unit at the power-on-default full-scale ranges (±250 deg/s,
// ±2g) — we don't touch GYRO_CONFIG/ACCEL_CONFIG, so these defaults apply.
#define GYRO_LSB_PER_DPS 131.0f
#define ACCEL_LSB_PER_G 16384.0f
#define DEG_TO_RAD_F 0.017453293f
#define G_TO_MS2 9.80665f

float yaw = 0.0f;
unsigned long last_us = 0;
unsigned long last_event_ms = 0;
#define STALL_TIMEOUT_MS 3000

bool mpuOk = false;

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

  yaw = 0.0f;
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

void setup() {
  Serial.begin(115200);
  delay(500);
  mpuOk = initMpu();
  last_event_ms = millis();
  Serial.println(mpuOk ? "OK: MPU init succeeded" : "ERR: MPU init failed");
}

void loop() {
  if (!mpuOk) {
    delay(500);
    mpuOk = initMpu();
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
    float gx = raw[3] / GYRO_LSB_PER_DPS * DEG_TO_RAD_F;
    float gy = raw[4] / GYRO_LSB_PER_DPS * DEG_TO_RAD_F;
    float gz = raw[5] / GYRO_LSB_PER_DPS * DEG_TO_RAD_F;

    yaw += gz * dt;

    float half = yaw * 0.5f;
    float qw = cos(half);
    float qz = sin(half);

    Serial.print("IMU,");
    Serial.print(qw, 5); Serial.print(',');
    Serial.print(0.0, 5); Serial.print(',');
    Serial.print(0.0, 5); Serial.print(',');
    Serial.print(qz, 5); Serial.print(',');
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
    last_event_ms = millis();
  }
}
