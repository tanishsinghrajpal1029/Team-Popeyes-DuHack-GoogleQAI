"""App Lab entry point: route on Linux, draw on the UNO Q microcontroller."""

import threading
import time

from arduino.app_utils import App, Bridge
from arduino.app_bricks.web_ui import WebUI

from bonus import select_depths
from routing import SCENARIOS, solve


ui = WebUI()
guard = threading.Lock()
state = {"scenario": "crossbar", "weak_error_percent": 12.0, "block_weak": False}
last_touch = time.monotonic()
last_cycle = last_touch


def publish():
    plan = solve(**state)
    matrix = plan["matrix"]
    try:
        for row in range(8):
            Bridge.call(
                "set_route_row", row, matrix["nodes"][row],
                matrix["direct"][row], matrix["aware"][row],
            )
        Bridge.call(
            "set_route_markers", matrix["start_pixel"],
            matrix["end_pixel"], matrix["weak_pixel"], plan["block_weak"],
        )
        Bridge.call("set_display_mode", False)
        plan["board_status"] = "Matrix updated"
    except Exception as exc:
        plan["board_status"] = f"Matrix update failed: {exc}"
    ui.send_message("route_result", plan)


def on_route_request(_client_id, message):
    global last_touch
    if not isinstance(message, dict):
        ui.send_message("route_error", "Request must be an object")
        return
    candidate = {
        "scenario": message.get("scenario", state["scenario"]),
        "weak_error_percent": message.get("weak_error_percent", state["weak_error_percent"]),
        "block_weak": message.get("block_weak", state["block_weak"]),
    }
    try:
        solve(**candidate)
        with guard:
            state.update(candidate)
            last_touch = time.monotonic()
            publish()
    except ValueError as exc:
        ui.send_message("route_error", str(exc))


def on_bonus_request(_client_id, message):
    global last_touch
    if not isinstance(message, dict):
        ui.send_message("bonus_error", "Request must be an object")
        return
    try:
        result = select_depths(message.get("N"), message.get("n"))
        with guard:
            Bridge.call("set_bonus_depths", result["candidate_a"], result["candidate_b"])
            Bridge.call("set_display_mode", True)
            last_touch = time.monotonic()
        result["board_status"] = "Bonus depth bars on matrix"
        ui.send_message("bonus_result", result)
    except ValueError as exc:
        ui.send_message("bonus_error", str(exc))
    except Exception as exc:
        ui.send_message("bonus_error", f"Matrix update failed: {exc}")


def loop():
    global last_cycle
    time.sleep(0.5)
    now = time.monotonic()
    # Without a browser, the board cycles three demos on its own.
    if now - last_touch > 25 and now - last_cycle > 12:
        with guard:
            keys = list(SCENARIOS)
            state["scenario"] = keys[(keys.index(state["scenario"]) + 1) % len(keys)]
            last_cycle = now
            publish()


ui.on_message("route_request", on_route_request)
ui.on_message("bonus_request", on_bonus_request)
publish()
App.run(user_loop=loop)
