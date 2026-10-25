#include <Arduino_LED_Matrix.h>
#include <MsgPack.h>  // Explicit include lets App CLI discover RouterBridge's dependency.
#include <Arduino_RouterBridge.h>

// The matrix is a central 8 x 13 window of Willow's 105-qubit topology.
Arduino_LED_Matrix matrix;
uint16_t nodeRows[8] = {0};
uint16_t directRows[8] = {0};
uint16_t awareRows[8] = {0};
uint8_t frame[104] = {0};
int startPixel = -1;
int endPixel = -1;
int weakPixel = -1;
bool blocked = false;
bool bonusMode = false;
int depthA = 1;
int depthB = 1;

void setRouteRow(int row, int nodes, int direct, int aware) {
  if (row < 0 || row >= 8) return;
  nodeRows[row] = uint16_t(nodes) & 0x1fff;
  directRows[row] = uint16_t(direct) & 0x1fff;
  awareRows[row] = uint16_t(aware) & 0x1fff;
}

void setRouteMarkers(int start, int end, int weak, bool isBlocked) {
  startPixel = start;
  endPixel = end;
  weakPixel = weak;
  blocked = isBlocked;
}

void setBonusDepths(int a, int b) {
  depthA = max(1, a);
  depthB = max(1, b);
}

void setDisplayMode(bool showBonus) {
  bonusMode = showBonus;
}

void setup() {
  matrix.begin();
  matrix.setGrayscaleBits(3);
  matrix.clear();
  pinMode(LED4_R, OUTPUT);
  pinMode(LED4_G, OUTPUT);
  pinMode(LED4_B, OUTPUT);
  Bridge.begin();
  Bridge.provide_safe("set_route_row", setRouteRow);
  Bridge.provide_safe("set_route_markers", setRouteMarkers);
  Bridge.provide_safe("set_bonus_depths", setBonusDepths);
  Bridge.provide_safe("set_display_mode", setDisplayMode);
}

void loop() {
  // Blue = fewest hops, green = error-aware. The white dot sweeps across
  // each route, while the weak qubit blinks at full brightness.
  const uint32_t tick = millis() / 160;
  if (bonusMode) {
    // Upper two rows = candidate A; lower two rows = candidate B.
    // A shorter bar means a shallower circuit. The winning bar is brighter.
    const int longest = max(depthA, depthB);
    const int lengthA = max(1, (13 * depthA + longest - 1) / longest);
    const int lengthB = max(1, (13 * depthB + longest - 1) / longest);
    for (int y = 0; y < 8; ++y) {
      for (int x = 0; x < 13; ++x) {
        uint8_t level = 0;
        if ((y == 1 || y == 2) && x < lengthA) level = depthA <= depthB ? 7 : 3;
        if ((y == 5 || y == 6) && x < lengthB) level = depthB <= depthA ? 7 : 3;
        frame[y * 13 + x] = level;
      }
    }
    matrix.draw(frame);
    const bool tie = depthA == depthB;
    digitalWrite(LED4_R, tie ? LOW : HIGH);
    digitalWrite(LED4_G, tie || depthB < depthA ? LOW : HIGH);
    digitalWrite(LED4_B, tie || depthA < depthB ? LOW : HIGH);
    delay(70);
    return;
  }
  const bool aware = ((tick / 26) % 2) == 1;
  const int sweep = tick % 13;
  const uint16_t* route = aware ? awareRows : directRows;
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 13; ++x) {
      const int pixel = y * 13 + x;
      const uint16_t bit = uint16_t(1) << x;
      uint8_t level = (nodeRows[y] & bit) ? 1 : 0;
      if (route[y] & bit) level = (x == sweep) ? 7 : 5;
      if (pixel == weakPixel) level = blocked ? 0 : ((tick / 3) % 2 ? 7 : 2);
      if (pixel == startPixel || pixel == endPixel) level = 7;
      frame[pixel] = level;
    }
  }
  matrix.draw(frame);
  digitalWrite(LED4_R, HIGH);
  digitalWrite(LED4_G, aware ? LOW : HIGH);
  digitalWrite(LED4_B, aware ? HIGH : LOW);
  delay(70);
}
