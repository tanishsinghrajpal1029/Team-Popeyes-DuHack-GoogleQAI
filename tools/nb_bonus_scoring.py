# pyright: reportUndefinedVariable=false
# Notebook fragment: executed inside the challenge notebook after the bonus solution.
# %% [markdown]
# The organisers' `score_solution` calls a `validate_circuit` helper that the notebook does not define. Our local stand-in
# checks the device rules with `cirq_google.Willow105.validate_circuit` and, for $N\le14$ (state-vector size $2^N$), the exact
# state with the notebook's `test_circuit`. Our bonus circuits contain no measurements, so the readout-diagonal rule for
# mid-circuit measurements is satisfied trivially.

# %%
def validate_circuit(circuit, num_excitations, qubits):
    """Local stand-in for the organisers' checker."""
    cirq_google.Willow105.validate_circuit(circuit)  # gate set and nearest-neighbour CZs on Willow105
    assert not any(cirq.is_measurement(op) for op in circuit.all_operations()), 'unexpected measurement'  # readout rule
    assert circuit.all_qubits() <= set(qubits), 'unexpected ancilla'  # state lives on the returned qubits
    if len(qubits) <= 14:  # state-vector check where affordable
        test_circuit(circuit, num_excitations, qubits)  # exact Dicke state


# %% [markdown]
# Running the organisers' scoring function (it loops over $2\le N\le20$, $1\le n\le\lfloor N/2\rfloor$, about a minute). For reference,
# compiling the hardware-track candidates on the ideal device without the bonus-specific tree and packing gives 39.7.

# %%
_t = _time.monotonic()  # timer
print(f'bonus score = {score_solution():.2f}  ({_time.monotonic() - _t:.0f} s)')  # organisers' metric
for _N in (8, 12, 16, 20):  # depth table for a few sizes
    print(_N, [len(create_dicke_state_adaptive(_N, _n)[0]) for _n in range(1, _N // 2 + 1)])  # moments per n
