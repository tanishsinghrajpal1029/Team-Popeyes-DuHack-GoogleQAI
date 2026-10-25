<!-- Moved verbatim from the notebook cells of Bryan's solution (origin/bryan @ b94cb62,
dicke_solution.py), which main.py vendors. Function names refer to main.py, where
create_dicke_state(_adaptive) is renamed _bryan_create_dicke_state(_adaptive). -->

# DuQuantum solution: exact, hardware-aware Dicke-state preparation on Willow

This section contains our complete solution. Several *exact* constructions are generated,
placed on the calibrated device graph, compiled to the native gate set
(`CZ` + `PhasedXZGate`) with dynamical decoupling, and the candidate with the highest
*predicted* hardware success probability is returned.

**Notation.** $N$ = number of qubits, $n$ = number of excitations (the notebook's names),
$k=\min(n, N-n)$ = the excitation count we actually prepare (see the complement trick below).
$R_y(\theta)=e^{-i\theta Y/2}$, so $R_y(\theta)|0\rangle=\cos\frac\theta2|0\rangle+\sin\frac\theta2|1\rangle$.

**Pipeline.**

| Stage | What it does | Why |
| --- | --- | --- |
| 1. Exact building blocks | Splits, Givens rotations and "split-and-cyclic-shift" (SCS) blocks written directly in CZs | Every CZ costs ~0.25% error, so each block uses the fewest CZs we could prove or find |
| 2. Logical constructions | W-state tree ($k=1$), Bärtschi–Eidenbenz (BE) path and a two-arm "split" construction ($k\ge2$) | Different shapes trade CZ count against depth |
| 3. Placement | Search trees / paths / ladders on the calibrated coupler graph | Uses the best qubits, couplers and readout |
| 4. Compilation | Merge 1-qubit gates into `PhasedXZ`, ALAP layering into alternating 1q/CZ moments, XY dynamical decoupling | Native gate set, minimal idle exposure |
| 5. Selection | Predict $\log F$ (gates + readout + idling) for every candidate and keep the best | Calibration-aware choice |

**Complement trick.** $|D^{(N)}_n\rangle = X^{\otimes N}|D^{(N)}_{N-n}\rangle$ because flipping every bit maps
weight-$n$ strings bijectively onto weight-$(N-n)$ strings. We therefore always prepare
$k=\min(n,N-n)\le N/2$ excitations and, if $n>N/2$, append $X$ on every qubit. The extra $X$'s merge into
the final single-qubit layer, so the trick costs nothing.

## 1. Exact building blocks

We build circuits from a tiny internal instruction list: `('U', q, M)` applies the $2\times2$ unitary `M`
to qubit `q`, and `('CZ', a, b)` applies a CZ. Single-qubit matrices are merged later into one
`PhasedXZGate` between consecutive CZs, so they are free in CZ count; only CZs matter.

**(a) CNOT from one CZ.** $\mathrm{CNOT}_{c\to t}=H_t\,\mathrm{CZ}_{ct}\,H_t$ because $HZH=X$.

**(b) Controlled rotation onto a known $|0\rangle$: one CZ.** For any $\varphi$,
$R_y(\varphi)\,Z\,R_y(-\varphi)=R_y(2\varphi)Z$ (since $ZR_y(\varphi)Z=R_y(-\varphi)$), and $Z|0\rangle=|0\rangle$. Hence
$$R_y(\tfrac\theta2)_t\,\mathrm{CZ}_{ct}\,R_y(-\tfrac\theta2)_t:\quad |0\rangle_c|0\rangle_t\mapsto|0\rangle|0\rangle,\qquad |1\rangle_c|0\rangle_t\mapsto|1\rangle R_y(\theta)|0\rangle .$$
A general controlled-$R_y$ needs two CZs; one suffices whenever the target is guaranteed to be
$|0\rangle$ when the control is $|1\rangle$. We use this "known-target" trick everywhere.

**(c) Excitation transfer (W-state split): two CZs.** $|1\rangle_s|0\rangle_d\mapsto\sqrt{1-p}\,|10\rangle+\sqrt p\,|01\rangle$,
$|00\rangle\mapsto|00\rangle$: a known-target rotation of $d$ followed by $\mathrm{CNOT}_{d\to s}$ (which clears $s$ when $d$
took the excitation). Two CZs are necessary: with one CZ, $(A_1\otimes B_1)\,\mathrm{CZ}\,(A_0\otimes B_0)$ applied to $|x\rangle|0\rangle$
gives $\sum_y (A_0)_{yx}A_1|y\rangle\otimes B_1Z^y|b\rangle$; keeping $|00\rangle$ a product state forces $A_0$ to be diagonal, and then
the $|10\rangle$ input also stays a product state, so the entangled branch cannot be produced.

**(d) Givens rotation: two CZs.** $G(\theta)=\exp\!\big[\tfrac\theta2(|10\rangle\langle01|-|01\rangle\langle10|)\big]=\exp\!\big[i\tfrac\theta4(XY-YX)\big]$
fixes $|00\rangle,|11\rangle$ and rotates $|01\rangle\mapsto\cos\frac\theta2|01\rangle+\sin\frac\theta2|10\rangle$. Conjugating by $S$ on qubit $b$ maps
$XY-YX\to-(XX+YY)$, and $R_x(\frac\pi2)^{\otimes2}$ maps $YY\to ZZ$. Finally
$\mathrm{CNOT}_{a\to b}\,[R_x(\alpha)\otimes R_z(\beta)]\,\mathrm{CNOT}_{a\to b}=e^{-i\alpha XX/2}e^{-i\beta ZZ/2}$, so
$$G(\theta)=V^\dagger W^\dagger\,\mathrm{CNOT}_{ab}\,[R_x(\tfrac\theta2)\otimes R_z(\tfrac\theta2)]\,\mathrm{CNOT}_{ab}\,W V,\quad V=S_b,\ W=R_x(\tfrac\pi2)^{\otimes2}.$$

**(e) Moving-hole gate $G_3$: five CZs** (an exhaustive numerical search found no 3- or 4-CZ circuit on a line). On qubits
$(c,a,b)$ with $a$ in the middle and input domain $\{000,001,011,101,111\}$ it maps $001\mapsto\cos\frac\theta2|001\rangle+\sin\frac\theta2|010\rangle$
(partial move of a "hole" from $a$ to $b$), $101\mapsto|110\rangle$ (full move when $c=1$, i.e. the hole is "in transit"), and fixes
the rest. Construction: $\mathrm{CNOT}_{a\to b}\,M\,\mathrm{CNOT}_{a\to b}$ with
$M=R_y(\alpha)Z^{c}R_y(\beta)Z^{b}R_y(-\beta)Z^{c}R_y(-\alpha)$ acting on $a$, $\alpha=\frac{\theta+\pi}4$, $\beta=\frac{\theta-\pi}4$. After the first
CNOT, $b=1\Rightarrow a=0$, and pushing the $Z$'s through the rotations gives total $R_y$ angles $0$ ($c=b=0$), $0$ ($c=1,b=0$),
$\theta$ ($c=0,b=1$) and $\pi$ ($c=b=1$), exactly as required.

**(f) Merged top block: four CZs instead of 2 + 5 = 7.** Found by an exhaustive numerical search over CZ orderings
(only the ordering $ab,bc,ab,bc$ works) and then derived analytically. On a line $a-b-c$ with domain
$\{000,001,011\}$:
1. known-target rotation $b\to a$ with angle $\theta_2$ (only the $011$ input has $b=1$, and there $a=0$);
2. a Givens rotation $G(\theta_1)$ on $(b,c)$ whose middle $R_x(\theta_1/2)$ on $b$ is replaced by
   $R_x(\frac\pi2)\,\mathrm{CZ}_{ab}\,R_x(\frac{\theta_1-\pi}2)$. For $a=0$ nothing changes (the $R_x$ angles add to $\theta_1/2$), so
   $001\mapsto\cos\frac{\theta_1}2|001\rangle+\sin\frac{\theta_1}2|010\rangle$ and $011$ is fixed. For $a=1$ the middle becomes $R_x(\pi-\frac{\theta_1}2)Z$;
   moving the $Z$ out of the Givens frame shows that the $|11\rangle_{bc}$ input is rotated by an extra $\pi-\theta_1$ in the
   $\{|01\rangle,|10\rangle\}$ plane and lands exactly on $|10\rangle_{bc}$. So $|111\rangle\mapsto|110\rangle$ up to a phase $-i$;
3. an $S$ gate on $a$ removes that phase ($a=1$ only in that branch).

Every block is checked numerically against its defining isometry in the verification section.

## 2. Logical constructions

### 2.1 W states ($k=1$): excitation tree, $2N-3$ CZs, depth $O(\log N)$

Pick a spanning tree of $N$ qubits with root $r$, and write $s(v)$ for the size of the subtree under $v$. Starting from
$|1\rangle_r$, every node $v$ that currently "owns" the excitation for $m_v$ qubits (itself plus its not-yet-served child subtrees)
transfers it to child $c$ with probability $s(c)/m_v$. The probability that the excitation reaches a given qubit is the product
of conditional probabilities along its root path, which telescopes to $1/N$; all amplitudes stay real and positive, so the
result is exactly $|W_N\rangle$.

The *first* transfer from the root acts on the product state $|1\rangle_r|0\rangle_c$, so it needs only one CZ:
prepare $r$ in $\sqrt p|0\rangle+\sqrt{1-p}|1\rangle$, then apply $\mathrm{CNOT}_{r\to c}$ and $X_c$. Total: $1+2(N-2)=2N-3$ CZs. An exhaustive numerical
search confirmed that $W_3$ needs 3 and $W_4$ needs 5 CZs, so this count is optimal at those sizes.

Children are served in order of decreasing subtree height, so the longest chain starts first; the depth is roughly
$2\times$(tree height + branching), i.e. logarithmic in $N$ for a bushy tree on the 2D grid.

### 2.2 General $k$: Bärtschi–Eidenbenz (BE) on a path

BE prepares $|D^{(N)}_k\rangle=U_N|0^{N-k}1^k\rangle$ with $U_N=\prod_{m=N}^{2}\mathrm{SCS}_m$. Before step $m$ the first $m$ positions hold
$|0^{m-r}1^r\rangle$ (ones right-justified, $r$ = remaining excitations). Since
$$|D^{(m)}_r\rangle=\sqrt{\tfrac rm}\,|D^{(m-1)}_{r-1}\rangle|1\rangle+\sqrt{\tfrac{m-r}m}\,|D^{(m-1)}_r\rangle|0\rangle,$$
step $m$ must map $|0^{m-r}1^r\rangle\mapsto\sqrt{\tfrac rm}|0^{m-r}1^r\rangle+\sqrt{\tfrac{m-r}m}|0^{m-r-1}1^r0\rangle$: either position $m$ keeps its 1, or the
"hole" at the boundary $m-r$ travels to $m$ (a cyclic shift). Only the window $m-k,\dots,m$ is touched.

We implement $\mathrm{SCS}_m$ as a *local sweep* along the window, so it needs only nearest-neighbour CZs: the hole moves
one site at a time, partially at its original boundary and fully while "in transit". The first two sweep steps form the
4-CZ merged top block; the remaining steps are $G_3$ gates. Cost per step: 2 CZs ($k=1$), 4 CZs ($k=2$), $4+5(k-2)$ in general
(textbook BE uses one non-local three-qubit gate per extra branch, each several CNOTs, **plus** SWAP routing on a line).

*Caveat handled in code:* when $m\le k$ the window is the whole prefix and the all-ones input $|1^m\rangle$ can occur; the
merged block is not valid there (numerically it would need at least 6 CZs), so those steps use the plain sweep, which fixes $|1\dots1\rangle$.

### 2.3 General $k$: two-arm "split" construction (our main $k\ge2$ workhorse)

Split the qubits into a left arm $L$ ($a$ qubits) and a right arm $R$ ($b=N-a$ qubits). Then
$$|D^{(N)}_k\rangle=\sum_j\sqrt{P(j)}\;|D^{(a)}_{k-j}\rangle_L|D^{(b)}_j\rangle_R,\qquad P(j)=\frac{\binom bj\binom a{k-j}}{\binom Nk}\ \text{(hypergeometric)}.$$
Because the total weight $k$ is *fixed* at the start, the weight distribution is cheap to prepare in unary:
1. **Thermometer chain** on the first $k$ qubits $r_1..r_k$ of $R$: $r_1\leftarrow R_y$ with $P(j\ge1)$, then known-target rotations
   $r_{u-1}\to r_u$ with conditional probabilities $P(j\ge u)/P(j\ge u-1)$. Result: $\sum_j\sqrt{P(j)}|1^j0^{k-j}\rangle$ ($k-1$ CZs, exact).
2. **Rung copy.** The left arm's first $k$ qubits are laid out as a parallel row in *reverse* order, so that $l_{k+1-u}$ neighbours
   $r_u$ (a "ladder"). The left arm needs the thermometer $l_v=[k-j\ge v]=\lnot r_{k+1-v}$, so one parallel layer of CNOTs plus $X$ gates
   writes it ($k$ CZs, depth 1).
3. Each arm now holds a right-justified unary input, so we run the BE unitary on both arms **in parallel**.

Compared with one long BE path, the two arms run concurrently (about half the depth) and the construction also *saves CZs*,
because the expensive full-window sweeps happen on half as many positions. Examples: $(N,k)=(6,3)$: 23 vs 36 CZs; $(8,4)$: 49 vs 77.

## 3. Compilation: native gates, layering, dynamical decoupling

1. **Merge.** All single-qubit matrices on a qubit between two of its CZs are multiplied into one unitary and emitted
   as a single `PhasedXZGate` (every single-qubit unitary is a `PhasedXZ` up to global phase). This minimises the
   single-qubit gate count and error.
2. **Layer.** CZs form a DAG (two CZs depend on each other if they share a qubit). We compute the ASAP depth $D$ and then
   place every CZ **as late as possible** (ALAP) within $D$ layers. Moments alternate *1q layer / CZ layer*, which matches
   how Google calibrates parallel CZ moments. ALAP keeps each qubit in $|0\rangle$ for as long as possible, and $|0\rangle$ is
   immune to both $T_1$ decay and dephasing, so idle error only accumulates after a qubit's first gate.
3. **Dynamical decoupling (DD).** Once a qubit is active, every 1q slot in which it would otherwise idle receives an $X$ or $Y$
   pulse (alternating, XY4-style). The last slot of each idle window receives the inverse of the accumulated Pauli product,
   merged with the real gate that belongs there. Paulis multiply to a Pauli and the window contains no CZ on that qubit, so
   the logical unitary is **exactly** unchanged, while quasi-static $Z$ errors (detuning, spectator phases from neighbouring CZs)
   are echoed out. The challenge text flags DD as important; our emulator (later section) quantifies the trade-off, since each
   pulse costs about one single-qubit error.

## 4. Calibration model, success-probability estimate, placement search

**Pessimistic error rates.** Each calibration entry is a measured mean $\hat e$ with statistical uncertainty $\sigma$. Choosing the
minimum of many noisy estimates systematically selects under-estimates (the "winner's curse"), so we rank with
$e=\hat e+\beta\sigma$, $\beta=1$ (a one-standard-error penalty), following the plan's uncertainty-aware placement.

**Predicted log-fidelity** of a compiled candidate (higher is better):
$$\log F=\sum_{\mathrm{CZ}}\log(1-e_{\mathrm{CZ}})+\sum_{\mathrm{1q}}\log(1-e_{\mathrm{1q}})+\sum_{q}\log\!\big(1-[(1-p)e^{0\to1}_q+p\,e^{1\to0}_q]\big)-\sum_q\frac{t^{\mathrm{active}}_q}{T_{\mathrm{idle}}}.$$
The readout term uses the exact excitation probability $p=n/N$ of a Dicke qubit (asymmetric readout matters: $e^{1\to0}\approx2e^{0\to1}$).
$t^{\mathrm{active}}_q$ is the time from the qubit's first gate to measurement (25 ns per 1q moment, 42 ns per CZ moment), and
$T_{\mathrm{idle}}=60\,\mu$s is a nominal combined $T_1$/dephasing time for Willow-class qubits with DD (it only has to rank
candidates, not predict absolute numbers).

**Search spaces** (exhaustive over starting points, pruned by cheap proxies, then compiled and scored exactly):
* *Trees* ($k=1$): for every root and three depth penalties, a Prim-like greedy growth adds the cheapest frontier qubit
  (2 CZs on its edge + readout + single-qubit cost + depth penalty).
* *Paths* (BE): beam search over simple paths, with position-dependent weights equal to the actual number of CZs / 1q gates
  the construction places on each edge / qubit.
* *Ladders* (split): every straight $2\times k$ ladder on the grid, extended into two disjoint arms by beam search.

## 5. `_bryan_create_dicke_state`

Candidate families per $k=\min(n,N-n)$:

| $k$ | Families tried |
| --- | --- |
| 0 | No entangling gates: the $N$ qubits with the best relevant readout error, optionally flipped |
| 1 | Excitation trees (all roots × 3 depth penalties) |
| $\ge2$ | BE paths (top beams) **and** two-arm split ladders (top ladders × arm extensions) |

Every candidate is compiled and scored with the predicted log-fidelity; the best one with at most 1750 moments is returned.
A wall-clock budget (30 s, leaving a 2x margin under the 60 s limit for slower machines) stops the search early. Impossible requests
($n<0$, $n>N$, more qubits than usable ones) raise `NotImplementedError`.

## 6. Bonus: shallowest circuits (adaptive circuits allowed, no noise)

The bonus scores depth only (`len(circuit)`), on the ideal Willow105 connectivity. We considered the adaptive designs
in the plan (measurement-based fan-out and the "binary index → one-hot → $X$-basis measurement → $Z$ correction" W-state
scheme). For $N\le20$ they lose to our unitary constructions: a measurement moment plus a classically controlled
correction moment already cost 2 moments, the constant-depth fan-out gadget needs several more, and the binary→one-hot
decoder needs multi-controlled gates routed across the grid, whereas the excitation tree reaches $N$ qubits in
$\approx4\lceil\log_2N\rceil+1$ moments. Moreover mid-circuit measurements must cover whole readout diagonals
($\text{row}+\text{col}=$ const), which forces ancillas onto separate diagonals and lengthens routing. We therefore submit
**non-adaptive** circuits (explicitly allowed) and spend the effort on depth:

* **W states:** a *broadcast tree*. In round $t$ every informed qubit hands the excitation to one uninformed neighbour, so the
  informed set can double every two CZ layers; we build the tree by greedy bipartite matching round by round from every
  possible root and keep the shallowest compiled circuit.
* **$k\ge2$:** the two-arm split construction (both arms in parallel) and the BE path, compiled on the ideal device, keeping
  the shallower.
* **Packing:** for the bonus, single-qubit gates may share moments with CZs on other qubits (no hardware calibration constraint
  applies), so we re-insert every operation at the earliest possible moment and keep whichever version is shorter.
