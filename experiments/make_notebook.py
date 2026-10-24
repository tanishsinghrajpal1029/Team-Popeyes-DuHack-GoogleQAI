"""Build submission.ipynb: the challenge notebook with main.py filled into its two stub cells.

`create_dicke_state_adaptive` goes into the bonus stub cell; the rest of main.py (every helper
plus `create_dicke_state`) replaces the standard-track stub cell. A final cell checks a few
(N, n) cases with the notebook's own `test_circuit` and `Willow105.validate_circuit`.

    .venv/bin/python experiments/make_notebook.py            # write submission.ipynb
    .venv/bin/python experiments/make_notebook.py --run      # also execute it (no Jupyter needed)
"""

import argparse
import copy
import json
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "[standard_track] Google Quantum AI Challenge.ipynb"
OUT = ROOT / "submission.ipynb"

CHECK_CELL = '''\
# Sanity check on the fake calibration above: exact state and valid Willow routing.
import time

for N, n in [(2, 1), (4, 2), (5, 1), (6, 3), (8, 4), (10, 5), (10, 0), (7, 7)]:
    t0 = time.time()
    circuit, qubits = create_dicke_state(
        N, n, cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors
    )
    build_s = time.time() - t0
    cirq_google.Willow105.validate_circuit(circuit)
    test_circuit(circuit, n, qubits)
    n_cz = sum(1 for op in circuit.all_operations() if op.gate == cirq.CZ)
    print(f"D({N},{n}): {len(circuit)} moments, {n_cz} CZ, built in {build_s:.1f} s -- exact, valid")

for N, n in [(4, 2), (10, 5), (15, 7)]:
    circuit, qubits = create_dicke_state_adaptive(N, n)
    cirq_google.Willow105.validate_circuit(circuit)
    if N <= 10:
        test_circuit(circuit, n, qubits)
    print(f"adaptive D({N},{n}): {len(circuit)} moments")
'''


def split_main(src: str) -> tuple[str, str]:
    """Return (main.py without the adaptive function, the adaptive function)."""
    lines = src.splitlines(keepends=True)
    start = next(i for i, l in enumerate(lines) if l.startswith("def create_dicke_state_adaptive("))
    end = next(i for i in range(start + 1, len(lines)) if lines[i][:1] not in ("", " ", "\n", ")"))
    adaptive = "".join(lines[start:end]).rstrip() + "\n"
    rest = "".join(lines[:start] + lines[end:])
    return rest, adaptive


def code_cell(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


def build() -> dict:
    nb = json.loads(TEMPLATE.read_text())
    rest, adaptive = split_main((ROOT / "main.py").read_text())
    cells = nb["cells"]

    def find(prefix):
        return next(i for i, c in enumerate(cells)
                    if c["cell_type"] == "code" and "".join(c["source"]).startswith(prefix))

    cells[find("def create_dicke_state(")] = code_cell(rest)
    cells[find("def create_dicke_state_adaptive(")] = code_cell(adaptive)
    i = find("single_qubit_dropouts = pick_single_qubit_dropouts")
    cells.insert(i + 1, code_cell(CHECK_CELL))
    # drop the template's trailing empty cell
    if cells and cells[-1]["cell_type"] == "code" and not "".join(cells[-1]["source"]).strip():
        cells.pop()
    # The checks need the calibration and helper cells first, but the adaptive function is
    # defined further down; move the check cell to the end so a top-to-bottom run works.
    cells.append(cells.pop(i + 1))
    return nb


def run(nb: dict) -> None:
    """Execute the code cells in order in one namespace, as a top-to-bottom kernel run would."""
    ns: dict = {"__name__": "__main__"}
    for k, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        t0 = time.time()
        exec(compile("".join(c["source"]), f"<cell {k}>", "exec"), ns)
        print(f"[cell {k} ok, {time.time() - t0:.1f} s]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    nb = build()
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(nb['cells'])} cells)")
    if args.run:
        run(copy.deepcopy(nb))
