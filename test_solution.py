import itertools, math, time, sys
import cirq, cirq_google, numpy as np
from solution import create_dicke_state

def create_dicke_statevector(num_qubits, num_excitations):
    psi = np.zeros(2**num_qubits)
    norm = 1 / np.sqrt(math.comb(num_qubits, num_excitations))
    for ex in itertools.combinations(range(num_qubits), num_excitations):
        psi[sum(2**q for q in ex)] = norm
    return psi

def fidelity(circuit, n, qubits):
    allq = circuit.all_qubits()
    anc = allq.difference(qubits)
    order = sorted(qubits) + sorted(anc)
    psi = cirq.Simulator(dtype=np.complex128).simulate(circuit, qubit_order=order).final_state_vector
    if anc:
        psi = cirq.sub_state_vector(psi, keep_indices=range(len(qubits)))
    return abs(psi @ create_dicke_statevector(len(qubits), n))**2

def pick_single_qubit_dropouts(rate, rng):
    allq = np.array(list(cirq_google.Willow105.metadata.qubit_set))
    return set(allq[rng.random(len(allq)) < rate])
def gen_q(median, spread, drop, unc, rng):
    allq = np.array(list(cirq_google.Willow105.metadata.qubit_set))
    w = allq[np.array([q not in drop for q in allq])]
    e = rng.lognormal(np.log(median), spread, len(w))
    return dict(zip(w, zip(e, unc*np.ones(len(w)))))
def gen_2q(rng, drop, median=2.5e-3, spread=0.5, dropout=0.01, unc=5e-4):
    pairs = np.array(list(cirq_google.Willow105.metadata.qubit_pairs))
    ok = np.array([all(q not in drop for q in p) for p in pairs])
    work = (rng.random(len(pairs)) > dropout) * ok
    wp = [tuple(p) for p in pairs[work]]
    e = rng.lognormal(np.log(median), spread, len(wp))
    return dict(zip(wp, zip(e, unc*np.ones(len(wp)))))
def fake(seed):
    rng = np.random.default_rng(seed)
    d = pick_single_qubit_dropouts(0.01, rng)
    return (gen_2q(rng, d), gen_q(4e-4, .5, d, 2e-4, rng),
            gen_q(5e-3, .5, d, 2e-4, rng), gen_q(1e-2, .5, d, 3e-4, rng))

def check_rules(circuit, cz, sq):
    pairs = {frozenset(p) for p in cz}
    for op in circuit.all_operations():
        if len(op.qubits) == 2:
            assert op.gate == cirq.CZ, op
            assert frozenset(op.qubits) in pairs, op
        else:
            assert isinstance(op.gate, cirq.PhasedXZGate), op
            assert op.qubits[0] in sq, op
    cirq_google.Willow105.validate_circuit(circuit)
    assert len(circuit) <= 1750

if __name__ == '__main__':
    maxN = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    errs = fake(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    print(f"{'N':>3}{'n':>3}{'fid':>14}{'CZ':>6}{'moments':>9}{'time':>7}")
    worst = 1
    for N in range(1, maxN + 1):
        for n in range(0, N + 1):
            t = time.time()
            c, qs = create_dicke_state(N, n, *errs)
            dt = time.time() - t
            check_rules(c, errs[0], errs[1])
            assert len(qs) == N
            f = fidelity(c, n, qs)
            worst = min(worst, f)
            ncz = sum(1 for op in c.all_operations() if len(op.qubits) == 2)
            if n <= N // 2 + 1:
                print(f"{N:>3}{n:>3}{f:>14.10f}{ncz:>6}{len(c):>9}{dt:>7.2f}")
            assert np.isclose(f, 1.0), (N, n, f)
    print("all exact, worst fidelity", worst)
