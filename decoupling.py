"""Dynamical decoupling: protective pulses on qubits that sit idle.

A qubit that holds part of the state and has nothing to do for a while slowly
picks up random phase. Flipping it with X (and Y) pulses in the middle of the
idle stretch makes that phase cancel out. The pulses multiply to the identity,
so the ideal state is unchanged, and they go into empty slots of existing
moments, so the circuit does not get any deeper.
"""

import cirq

_X = cirq.PhasedXZGate(x_exponent=1, z_exponent=0, axis_phase_exponent=0)
_Y = cirq.PhasedXZGate(x_exponent=1, z_exponent=0, axis_phase_exponent=0.5)


def add_decoupling(circuit, min_gap=2):
    """Fill idle stretches of every qubit (after its first gate) with X X or X Y X Y.

    Idle stretches run until the end of the circuit, since the measurement comes
    right after.
    """
    moments = [list(m.operations) for m in circuit]
    busy = {}
    for t, mom in enumerate(moments):
        for op in mom:
            for q in op.qubits:
                busy.setdefault(q, []).append(t)
    T = len(moments)
    for q, times in busy.items():
        times = sorted(set(times))
        # gaps after each busy moment
        bounds = list(zip(times, times[1:] + [T]))
        for start, end in bounds:
            gap = end - start - 1          # free moments strictly between
            if gap < min_gap:
                continue
            seq = [_X, _Y, _X, _Y] if gap >= 4 else [_X, _X]
            L = len(seq)
            # evenly spaced pulse positions inside the free window
            slots = [start + 1 + int((i + 0.5) * gap / L) for i in range(L)]
            for s, g in zip(slots, seq):
                moments[s].append(g.on(q))
    return cirq.Circuit(cirq.Moment(m) for m in moments)
