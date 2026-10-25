# Willow Route Lab for Arduino UNO Q

A board-first exhibit for the UNO Q's built-in **8 × 13 LED matrix**. It compares a shortest-hop route with a route that avoids a weak qubit, then can show the two exact bonus-circuit depths as LED bars. The companion web page controls both views. No external display, LEDs, buttons, or Willow access are needed.

## Run on the UNO Q

**Arduino IDE and Arduino App Lab are separate programs.** Arduino IDE can upload the MCU sketch, but this exhibit also needs its Python route engine and WebUI brick. Use App Lab, or the board's preinstalled `arduino-app-cli` over USB/SSH, to run all three parts together.

1. Open **Arduino App Lab** and import/open this entire `arduino-uno-q-willow` folder as an App. Connect the UNO Q and press **Run**. App Lab should build `sketch/sketch.ino` for the board and start `python/main.py` on its Linux processor.
2. Watch the **built-in LED matrix**. It cycles through three routing scenarios without any browser. Blue status LED: fewest hops. Green status LED: error-aware. The weak qubit blinks; the route is brighter than the surrounding topology.
3. To control it, open the App Lab web view or visit `http://<UNO-Q-IP>:7000` from a device on the same network. Change the scenario, weak-link error, or block the weak qubit. The board and page update together.
4. Scroll to **The combined Dicke solution**. Choose a graded bonus `(N, n)` and press **Show depth duel on matrix**. Upper LED bars show candidate A; lower bars show B. The shorter bar glows brighter. The status LED is blue for A, green for B, and white for a tie. **Return matrix to routing** switches back. After about half a minute without interaction, the board resumes its routing cycle.

The matrix has 104 pixels and Willow has 105 qubits. The board shows rows 3–10 and columns 1–13 of the supplied topology; the companion page shows all 105 qubits. The demo routes are chosen to remain in that matrix window.

The app was also deployed successfully over USB to `/home/arduino/ArduinoApps/WillowRouteLab` using the ADB tool bundled with Arduino IDE and the UNO Q's onboard App CLI. When launching the CLI from an ADB shell, set `TMPDIR=/tmp`: the ADB shell otherwise sets a temporary path that does not exist on the UNO Q's Linux system. The required MCU libraries are pinned in `sketch/sketch.yaml` so its isolated build can find them. Reimport the whole folder in App Lab and press **Run** after changing the MCU sketch or Python app; a browser reload alone cannot update the board firmware.

### Reconnect after unplugging USB

The board is configured to run Willow Route Lab at startup. Reattach it by USB-C and run `./connect_usb.sh` from this folder; then open `http://127.0.0.1:7000/`. The helper finds Arduino IDE's bundled ADB, starts the app only if needed, and restores the USB port forward. The `127.0.0.1` address works only while the USB connection and forward are active.

For a cable-free exhibit, power the UNO Q separately and join it to the same Wi-Fi network as your phone or laptop using Arduino App Lab. Then open `http://<UNO-Q-LAN-IP>:7000/`. The board's Wi-Fi is **not configured on this setup yet**; a USB-powered board disconnected from the Mac will also need another power source. The public GitHub Pages website currently does not pair with the board or relay its controls.

## Preview without hardware

Run from this folder:

```sh
python3 preview.py
```

Then open `http://127.0.0.1:8765/?preview=1`. The page uses the same routing and bonus-selection data but does not control the UNO Q. If port 8765 is occupied, run `python3 preview.py --port 8767` and open the printed URL.

Run the route checks with:

```sh
python3 -m unittest discover -s tests -v
```

## What the numbers mean

The topology and couplers come from the supplied Willow layout. The edge errors are **deterministic synthetic teaching values**, and the weak-qubit slider overrides edges touching the marked qubit. The direct route minimizes hops. The error-aware route minimizes a sum of `-log(1-p)` costs, assuming each preterminal move uses a SWAP of three CZ gates and the final interaction uses one CZ. The displayed estimated survival is the product of the corresponding `(1-p)` terms. It does not include idling, decoherence, crosstalk, readout, or actual Willow calibration, so it is **not a hardware fidelity forecast**. This version uses classical Dijkstra routing, not AI.

The bonus depths in `assets/bonus-depths.json` come from all 56 graded pairs of the combined notebook. Both candidates passed exact ideal-state checks on the static Willow topology. The matrix only displays their depths; the UNO Q does not simulate the circuits or access quantum hardware. The main-task rule shown on the page is the notebook's selection rule: the alternative team construction runs first for `(7, 3)` and `(9, 4)`, with the portfolio default for other pairs and a validity fallback.

## Source and third-party code

- [Arduino UNO Q manual](https://docs.arduino.cc/tutorials/uno-q/user-manual/) for the matrix and hybrid processor architecture.
- [Arduino Color your LEDs example](https://github.com/arduino/app-bricks-examples/tree/main/inspirational/platform_unoq/color-your-leds) for App Lab's WebUI pattern.
- `assets/libs/arduino.js` is copied from Arduino's example and carries an MPL-2.0 header. `assets/libs/socket.io.min.js` is the bundled Socket.IO client and carries an MIT header. They are included so the board page can work offline.

This project is independent of the public website, so PSC findings can change that site without breaking the board demo.
