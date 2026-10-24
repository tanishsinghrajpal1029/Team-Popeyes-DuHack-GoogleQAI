# Dicke State Preparation on Google Willow

Duke Quantum Hackathon (DuQuantum), Google Quantum AI standard track, October 2026

## The problem

The task is a function, `create_dicke_state`, that takes a number of qubits N, a number of excitations n, and the chip's calibration data, and returns a Cirq circuit that prepares the Dicke state on Google's 105 qubit Willow chip. The Dicke state is the equal superposition of every bitstring with exactly n ones. The organizers run each circuit on the real hardware and score the raw results, so the circuit has to be exact on paper and also hold up against hardware noise.

## Approach

The approach is to first make sure every case works, then make the circuits shorter, and then protect them from noise.

**Get something correct for every case first.** A case that raises `NotImplementedError` scores nothing, while even a weak circuit still scores. So the first priority was a construction that is exact for every N and n.

**Then cut the two things hardware punishes.** On Willow, every CZ gate adds error, and every extra moment gives waiting qubits more time to decay. Looking at the numbers, idle decay over a deep circuit costs roughly as much as the gates, so depth and CZ count are treated as equally important.

**Use the calibration data.** The calibration data shows which qubits and couplers are good and which are broken on the day, so the code should choose its own qubits instead of using a fixed layout.

## Completed so far

### 1. Exact baseline for every case (`baseline.py`)

The baseline starts from the Bärtschi and Eidenbenz (2019) construction, which builds the Dicke state by deciding qubits one at a time. When r excitations remain among j undecided qubits, the current qubit keeps an excitation with amplitude √(r/j). This is the same counting rule as picking a random subset, so the result is exactly the Dicke state.

The original version has gates between qubits that are far apart, which Willow does not allow. Each step is rewritten as a chain of exact gates on three neighbouring qubits, so the whole circuit runs along one path through the chip. The path is chosen from the calibration data to avoid broken qubits and couplers and to minimise CZ error, single qubit error and readout error. For n > N/2 the code builds the smaller case and flips every qubit, which costs no extra CZ gates.

### 2. Splitting tree for W states (`w_tree.py`)

For n = 1 the baseline passes the single excitation down the line one qubit at a time, which is slow. Instead, the excitation now spreads like a tree. One qubit shares with a neighbour, then those two share with two new neighbours, then four, and so on. Depth grows with log N instead of N. The code picks the root and the order of splits from the calibration data.

| N | Baseline moments | Tree moments |
|---|---|---|
| 8 | 29 | 11 |
| 12 | 45 | 15 |
| 100 | about 395 | 51 to 63 |

The CZ count stays at 2N − 3, so the gain comes entirely from the shorter circuit.

### 3. Dynamical decoupling (`decoupling.py`)

The notebook names dynamical decoupling as important. While a qubit waits, its phase drifts slowly, which damages the phase relationships the X basis score checks. Flipping a waiting qubit with X and Y pulses in the middle of its wait makes the drift cancel itself. The pulses multiply to the identity, so the ideal state is unchanged, and they go into empty slots of existing moments, so the circuit does not get any deeper.

### 4. Putting it together (`solution.py`)

`solution.py` is the main entry point. It uses the tree for n = 1 and n = N − 1, the baseline for everything else, and adds decoupling at the end.

## Checks

`test_solution.py` checks every case it builds.

* The state is exactly the Dicke state (fidelity 1 to within 10⁻¹³) for every N and n up to 13
* Every two qubit gate is a CZ on a calibrated pair, and every single qubit gate is a PhasedXZGate on a calibrated qubit
* Every circuit passes `cirq_google.Willow105.validate_circuit`
* Every circuit stays under 1,750 moments and builds well under the one minute limit

These were run on several fake calibrations from the notebook's own generators, including broken qubits and couplers.

## Findings

* Each step of the nearest neighbour Bärtschi and Eidenbenz construction needs at least 5 CZ gates. A numerical search showed that versions with 2, 3 or 4 CZ gates cannot be exact. This is why the baseline is expensive for larger n and why it needs to be restructured.
* A computer search found an exact circuit for D(4, 2) with 5 CZ gates on a square of qubits, compared with 7 for the analytic route, and showed 4 is not possible. Small cases have real room for improvement.
* The organizers' `test_circuit` uses `keep_indices = range(qubits)`, which fails because `qubits` is a set. The tests here use `range(len(qubits))`.

## Next

1. A two row split for larger n. Divide the excitations between two rows of qubits first, then finish both rows at the same time. This should cut both depth and CZ count substantially for the cases where the baseline is heaviest.
2. Shorter fixed circuits for small cases, found offline and placed on the chip at run time.
3. The bonus question, starting by reusing the main circuits.

## Running it

```bash
pip install --pre cirq cirq-google
python test_solution.py 8
```

The last line should read `all exact`.

## References

A. Bärtschi and S. Eidenbenz, *Deterministic Preparation of Dicke States*, FCT 2019, arXiv:1904.07358
