"""
Ask the circuit.

A small terminal chat where Gemini answers questions about the Dicke state
circuit by calling the team's own simulator. Every number in an answer comes
from the simulator, never from the model.

Setup
    pip install -U google-genai
    set GEMINI_API_KEY in the same PowerShell window

Run
    python ask_the_circuit.py
    python ask_the_circuit.py --mock     plumbing test only, returns FAKE numbers
"""

import sys

from google import genai
from google.genai import errors, types

# Tried in order. If one is overloaded (503) or unavailable, the next is used.
MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.5-flash",
]

USE_MOCK = "--mock" in sys.argv


# ---------------------------------------------------------------------------
# PLUG IN THE REAL SIMULATOR HERE
# Replace the body so it calls the project code and returns a plain dict of
# numbers, for example fidelity, cnot_count and depth.
# ---------------------------------------------------------------------------
def _simulate(n_qubits: int, k: int, noise: float) -> dict:
    raise NotImplementedError("Connect _simulate to the project simulator")


def _mock_simulate(n_qubits: int, k: int, noise: float) -> dict:
    # FAKE values, only here to test the chat plumbing.
    return {
        "MOCK_DATA_DO_NOT_USE": True,
        "n_qubits": n_qubits,
        "k": k,
        "noise": noise,
        "fidelity": round(max(0.0, 1.0 - 10 * noise), 4),
        "cnot_count": 4 * n_qubits,
        "depth": 3 * n_qubits,
    }


def run_dicke(n_qubits: int, k: int, noise: float = 0.0) -> dict:
    """Runs the team's Dicke state preparation circuit in the simulator and
    returns measured metrics such as fidelity, CNOT count and depth.

    Args:
        n_qubits: Total number of qubits.
        k: Number of excitations, the Hamming weight of the Dicke state.
        noise: Per gate error probability, 0.0 for ideal, 0.01 means 1 percent.

    Returns:
        A dict of metrics from the simulator, or a dict with an error key.
    """
    fn = _mock_simulate if USE_MOCK else _simulate
    try:
        out = fn(int(n_qubits), int(k), float(noise))
    except Exception as e:
        out = {"error": str(e)}
    print(f"  [simulator] n={n_qubits} k={k} noise={noise} -> {out}")
    return out


SYSTEM_RULES = (
    "You answer questions about a Dicke state preparation circuit for a quantum "
    "computing hackathon. For any numeric claim about the circuit, call run_dicke "
    "and report only what it returns. Never estimate or recall numbers. If the tool "
    "returns an error, say so plainly. Keep answers short and direct. When a "
    "question compares settings, call the tool once per setting and give a compact "
    "comparison."
)


def make_chat(client, model):
    return client.chats.create(
        model=model,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_RULES,
            tools=[run_dicke],
        ),
    )


def main():
    # 30 second timeout so a stalled model fails fast instead of hanging.
    client = genai.Client(http_options=types.HttpOptions(timeout=30000))
    if USE_MOCK:
        print("MOCK MODE, all numbers are fake.")
    print("Ask the circuit. Type quit to exit.")

    idx = 0
    chat = make_chat(client, MODELS[idx])
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit", "exit", ""):
            break

        while True:
            try:
                reply = chat.send_message(q)
                print("\n" + (reply.text or "(no text returned)"))
                break
            except errors.APIError as e:
                print(f"  [{MODELS[idx]} failed with {e.code}]")
                idx += 1
                if idx >= len(MODELS):
                    print("All models failed, try again in a minute.")
                    idx = 0
                    chat = make_chat(client, MODELS[idx])
                    break
                print(f"  [switching to {MODELS[idx]}]")
                chat = make_chat(client, MODELS[idx])


if __name__ == "__main__":
    main()
