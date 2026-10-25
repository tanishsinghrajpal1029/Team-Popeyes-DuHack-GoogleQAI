"""
Ask the circuit.

A small terminal chat where Gemini answers questions about the team's Dicke
state circuit by calling the real circuit builder in main.py. Every number in
an answer comes from that code, never from the model.

Setup
    Save this file in the same folder as main.py.
    pip install -U google-genai
    Set GEMINI_API_KEY in the same PowerShell window.

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


def _simulate(n_qubits: int, k: int) -> dict:
    """Build the circuit with main.create_dicke_state_adaptive and measure it."""
    import cirq
    import numpy as np
    import main  # the team's code, must be in the same folder

    circuit, data_qubits = main.create_dicke_state_adaptive(n_qubits, k)

    out = {
        "n_qubits": n_qubits,
        "k": k,
        "depth_moments": len(circuit),
        "two_qubit_gates": sum(1 for op in circuit.all_operations() if len(op.qubits) == 2),
        "total_gates": sum(1 for _ in circuit.all_operations()),
        "qubits_used": len(circuit.all_qubits()),
    }

    # Ideal (noiseless) check that the output really is a Dicke state.
    # Uses probabilities only, so global and virtual Z phases do not matter.
    try:
        all_q = sorted(circuit.all_qubits())
        data = sorted(data_qubits)
        state = cirq.Simulator().simulate(circuit, qubit_order=all_q).final_state_vector
        probs = np.abs(state) ** 2
        probs = probs.reshape([2] * len(all_q))
        other_axes = tuple(i for i, q in enumerate(all_q) if q not in set(data))
        if other_axes:
            probs = probs.sum(axis=other_axes)
        # axes of probs now follow the order of data qubits within all_q
        n = len(data)
        flat = probs.reshape(-1)
        weights = np.array([bin(i).count("1") for i in range(2 ** n)])
        in_k = flat[weights == k]
        out["ideal_prob_in_weight_k"] = round(float(in_k.sum()), 6)
        if len(in_k) > 0 and in_k.sum() > 0:
            target = 1.0 / len(in_k)
            out["max_deviation_from_uniform"] = round(float(np.max(np.abs(in_k / in_k.sum() - target))), 6)
    except Exception as e:
        out["ideal_check_error"] = str(e)[:200]

    return out


def _mock_simulate(n_qubits: int, k: int) -> dict:
    # FAKE values, only here to test the chat plumbing.
    return {
        "MOCK_DATA_DO_NOT_USE": True,
        "n_qubits": n_qubits,
        "k": k,
        "depth_moments": 3 * n_qubits,
        "two_qubit_gates": 4 * n_qubits,
        "total_gates": 9 * n_qubits,
        "qubits_used": n_qubits,
        "ideal_prob_in_weight_k": 1.0,
        "max_deviation_from_uniform": 0.0,
    }


def run_dicke(n_qubits: int, k: int) -> dict:
    """Builds the team's Dicke state preparation circuit for n_qubits qubits and
    k excitations, and returns measured properties of the circuit.

    Returned keys are depth_moments (circuit depth), two_qubit_gates (count of
    CZ style entangling gates), total_gates, qubits_used, ideal_prob_in_weight_k
    (probability the noiseless circuit lands on states with exactly k ones, 1.0
    is perfect) and max_deviation_from_uniform (how far the weight k amplitudes
    are from equal, 0.0 is perfect). The tool has no noise model.

    Args:
        n_qubits: Total number of qubits in the Dicke state.
        k: Number of excitations, the Hamming weight of the Dicke state.

    Returns:
        A dict of metrics, or a dict with an error key.
    """
    fn = _mock_simulate if USE_MOCK else _simulate
    try:
        out = fn(int(n_qubits), int(k))
    except Exception as e:
        out = {"error": str(e)[:300]}
    print(f"  [simulator] n={n_qubits} k={k} -> {out}")
    return out


SYSTEM_RULES = (
    "You answer questions about a Dicke state preparation circuit for a quantum "
    "computing hackathon. For any numeric claim about the circuit, call run_dicke "
    "and report only what it returns. Never estimate or recall numbers. The tool "
    "has no noise model, so if asked about noise or hardware fidelity, say that "
    "it reports circuit structure and an ideal noiseless check only. If the tool "
    "returns an error, say so plainly. Keep answers short and direct. When a "
    "question compares settings, call the tool once per setting and give a "
    "compact comparison."
)


def make_chat(client, model):
    return client.chats.create(
        model=model,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_RULES,
            tools=[run_dicke],
        ),
    )


def main_loop():
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
    main_loop()
