#include <Wire.h>

#define SDA_PIN 21
#define SCL_PIN 22

void setup() {
  Serial.begin(115200);
  delay(200);
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);
}

void loop() {
  Serial.println("SCAN_START");
  int found = 0;
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    uint8_t err = Wire.endTransmission();
    if (err == 0) {
      Serial.print("FOUND 0x");
      Serial.println(addr, HEX);
      found++;
    }
  }
  Serial.print("SCAN_DONE, devices=");
  Serial.println(found);
  delay(2000);
}
