# Team Popeyes — Google Quantum AI Dicke-state challenge

The submission is the notebook `[standard_track] Google Quantum AI Challenge.ipynb`. It contains the full, line-by-line
documented solution (`create_dicke_state`), the bonus solution (`create_dicke_state_adaptive`), verification, a noisy
emulator with design experiments, and the AI-usage disclosure.

## Results at a glance

| Instance $(N,n)$ | CZs | Moments | Emulated expected score (ideal) |
| --- | --- | --- | --- |
| (6, 1) W state | 9 | 11 | 1.12 (1.20) |
| (6, 3) | 23 | 25 | 0.86 (1.05) |
| (8, 2) | 23 | 23 | 0.88 (1.04) |
| (8, 4) | 49 | 49 | 0.70 (1.01) |

* Every circuit is exact (checked for all $N\le10$ and all $n$ on two calibrations) and uses only CZs on calibrated couplers
  and `PhasedXZGate`s.
* W states use $2N-3$ CZs, which a numerical search shows is optimal for $N\le4$.
* Bonus (depth) score with the organisers' formula: 65.3, versus 39.7 for the plain hardware-track circuits.

## Layout

| Path | Purpose |
| --- | --- |
| `dicke_solution.py` | Single source of the solution code and its markdown documentation (cells marked with `# %%`) |
| `tools/build_notebook.py` | Rebuilds the notebook from `tools/challenge_template.ipynb` plus the documented sources |
| `tools/nb_*.py` | Notebook-only cells: verification, emulator and experiments, bonus scoring, AI disclosure |
| `tools/noise_sim.py`, `tools/experiments.py` | Full design experiments quoted in the notebook |
| `tools/fake_calibration.py` | The notebook's fake-calibration generators, usable outside Jupyter |
| `tools/verify.py` | Exhaustive exactness and hardware-rule check (`N <= 12`, all `n`) |

Rebuild and check:

```bash
.venv/bin/python tools/build_notebook.py
.venv/bin/python tools/verify.py 2
.venv/bin/python tools/experiments.py all
```
