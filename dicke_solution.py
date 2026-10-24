# %% [markdown]
# # DuQuantum solution: exact, hardware-aware Dicke-state preparation on Willow
#
# This section contains our complete solution. It follows the research plan in
# `DuQuantum_Research_and_Implementation_Plan.md`: several *exact* constructions are generated,
# placed on the calibrated device graph, compiled to the native gate set
# (`CZ` + `PhasedXZGate`) with dynamical decoupling, and the candidate with the highest
# *predicted* hardware success probability is returned.
#
# **Notation.** $N$ = number of qubits, $n$ = number of excitations (the notebook's names),
# $k=\min(n, N-n)$ = the excitation count we actually prepare (see the complement trick below).
# $R_y(\theta)=e^{-i\theta Y/2}$, so $R_y(\theta)|0\rangle=\cos\frac\theta2|0\rangle+\sin\frac\theta2|1\rangle$.
#
# **Pipeline.**
#
# | Stage | What it does | Why |
# | --- | --- | --- |
# | 1. Exact building blocks | Splits, Givens rotations and "split-and-cyclic-shift" (SCS) blocks written directly in CZs | Every CZ costs ~0.25% error, so each block uses the fewest CZs we could prove or find |
# | 2. Logical constructions | W-state tree ($k=1$), Bärtschi–Eidenbenz (BE) path and a two-arm "split" construction ($k\ge2$) | Different shapes trade CZ count against depth |
# | 3. Placement | Search trees / paths / ladders on the calibrated coupler graph | Uses the best qubits, couplers and readout |
# | 4. Compilation | Merge 1-qubit gates into `PhasedXZ`, ALAP layering into alternating 1q/CZ moments, XY dynamical decoupling | Native gate set, minimal idle exposure |
# | 5. Selection | Predict $\log F$ (gates + readout + idling) for every candidate and keep the best | Calibration-aware choice |
#
# **Complement trick.** $|D^{(N)}_n\rangle = X^{\otimes N}|D^{(N)}_{N-n}\rangle$ because flipping every bit maps
# weight-$n$ strings bijectively onto weight-$(N-n)$ strings. We therefore always prepare
# $k=\min(n,N-n)\le N/2$ excitations and, if $n>N/2$, append $X$ on every qubit. The extra $X$'s merge into
# the final single-qubit layer, so the trick costs nothing.

# %%
import heapq  # priority queues for Prim-style tree growth and beam search
import math  # exact binomials, logs, arcsines
import time  # wall-clock budget (the challenge allows one minute per call)

import cirq  # circuit objects and native gates
import cirq_google  # Willow105 device description (used by the bonus solution)
import numpy as np  # 2x2 matrix algebra for gate merging

# %% [markdown]
# ## 1. Exact building blocks
#
# We build circuits from a tiny internal instruction list: `('U', q, M)` applies the $2\times2$ unitary `M`
# to qubit `q`, and `('CZ', a, b)` applies a CZ. Single-qubit matrices are merged later into one
# `PhasedXZGate` between consecutive CZs, so they are free in CZ count; only CZs matter.
#
# **(a) CNOT from one CZ.** $\mathrm{CNOT}_{c\to t}=H_t\,\mathrm{CZ}_{ct}\,H_t$ because $HZH=X$.
#
# **(b) Controlled rotation onto a known $|0\rangle$: one CZ.** For any $\varphi$,
# $R_y(\varphi)\,Z\,R_y(-\varphi)=R_y(2\varphi)Z$ (since $ZR_y(\varphi)Z=R_y(-\varphi)$), and $Z|0\rangle=|0\rangle$. Hence
# $$R_y(\tfrac\theta2)_t\,\mathrm{CZ}_{ct}\,R_y(-\tfrac\theta2)_t:\quad |0\rangle_c|0\rangle_t\mapsto|0\rangle|0\rangle,\qquad |1\rangle_c|0\rangle_t\mapsto|1\rangle R_y(\theta)|0\rangle .$$
# A general controlled-$R_y$ needs two CZs; one suffices whenever the target is guaranteed to be
# $|0\rangle$ when the control is $|1\rangle$. We use this "known-target" trick everywhere.
#
# **(c) Excitation transfer (W-state split): two CZs.** $|1\rangle_s|0\rangle_d\mapsto\sqrt{1-p}\,|10\rangle+\sqrt p\,|01\rangle$,
# $|00\rangle\mapsto|00\rangle$: a known-target rotation of $d$ followed by $\mathrm{CNOT}_{d\to s}$ (which clears $s$ when $d$
# took the excitation). Two CZs are necessary: with one CZ, $(A_1\otimes B_1)\,\mathrm{CZ}\,(A_0\otimes B_0)$ applied to $|x\rangle|0\rangle$
# gives $\sum_y (A_0)_{yx}A_1|y\rangle\otimes B_1Z^y|b\rangle$; keeping $|00\rangle$ a product state forces $A_0$ to be diagonal, and then
# the $|10\rangle$ input also stays a product state, so the entangled branch cannot be produced.
#
# **(d) Givens rotation: two CZs.** $G(\theta)=\exp\!\big[\tfrac\theta2(|10\rangle\langle01|-|01\rangle\langle10|)\big]=\exp\!\big[i\tfrac\theta4(XY-YX)\big]$
# fixes $|00\rangle,|11\rangle$ and rotates $|01\rangle\mapsto\cos\frac\theta2|01\rangle+\sin\frac\theta2|10\rangle$. Conjugating by $S$ on qubit $b$ maps
# $XY-YX\to-(XX+YY)$, and $R_x(\frac\pi2)^{\otimes2}$ maps $YY\to ZZ$. Finally
# $\mathrm{CNOT}_{a\to b}\,[R_x(\alpha)\otimes R_z(\beta)]\,\mathrm{CNOT}_{a\to b}=e^{-i\alpha XX/2}e^{-i\beta ZZ/2}$, so
# $$G(\theta)=V^\dagger W^\dagger\,\mathrm{CNOT}_{ab}\,[R_x(\tfrac\theta2)\otimes R_z(\tfrac\theta2)]\,\mathrm{CNOT}_{ab}\,W V,\quad V=S_b,\ W=R_x(\tfrac\pi2)^{\otimes2}.$$
#
# **(e) Moving-hole gate $G_3$: five CZs** (an exhaustive numerical search found no 3- or 4-CZ circuit on a line). On qubits
# $(c,a,b)$ with $a$ in the middle and input domain $\{000,001,011,101,111\}$ it maps $001\mapsto\cos\frac\theta2|001\rangle+\sin\frac\theta2|010\rangle$
# (partial move of a "hole" from $a$ to $b$), $101\mapsto|110\rangle$ (full move when $c=1$, i.e. the hole is "in transit"), and fixes
# the rest. Construction: $\mathrm{CNOT}_{a\to b}\,M\,\mathrm{CNOT}_{a\to b}$ with
# $M=R_y(\alpha)Z^{c}R_y(\beta)Z^{b}R_y(-\beta)Z^{c}R_y(-\alpha)$ acting on $a$, $\alpha=\frac{\theta+\pi}4$, $\beta=\frac{\theta-\pi}4$. After the first
# CNOT, $b=1\Rightarrow a=0$, and pushing the $Z$'s through the rotations gives total $R_y$ angles $0$ ($c=b=0$), $0$ ($c=1,b=0$),
# $\theta$ ($c=0,b=1$) and $\pi$ ($c=b=1$), exactly as required.
#
# **(f) Merged top block: four CZs instead of 2 + 5 = 7.** Found by an exhaustive numerical search over CZ orderings
# (only the ordering $ab,bc,ab,bc$ works) and then derived analytically. On a line $a-b-c$ with domain
# $\{000,001,011\}$:
# 1. known-target rotation $b\to a$ with angle $\theta_2$ (only the $011$ input has $b=1$, and there $a=0$);
# 2. a Givens rotation $G(\theta_1)$ on $(b,c)$ whose middle $R_x(\theta_1/2)$ on $b$ is replaced by
#    $R_x(\frac\pi2)\,\mathrm{CZ}_{ab}\,R_x(\frac{\theta_1-\pi}2)$. For $a=0$ nothing changes (the $R_x$ angles add to $\theta_1/2$), so
#    $001\mapsto\cos\frac{\theta_1}2|001\rangle+\sin\frac{\theta_1}2|010\rangle$ and $011$ is fixed. For $a=1$ the middle becomes $R_x(\pi-\frac{\theta_1}2)Z$;
#    moving the $Z$ out of the Givens frame shows that the $|11\rangle_{bc}$ input is rotated by an extra $\pi-\theta_1$ in the
#    $\{|01\rangle,|10\rangle\}$ plane and lands exactly on $|10\rangle_{bc}$. So $|111\rangle\mapsto|110\rangle$ up to a phase $-i$;
# 3. an $S$ gate on $a$ removes that phase ($a=1$ only in that branch).
#
# Every block is checked numerically against its defining isometry in the verification section.

# %%
_I2 = np.eye(2, dtype=complex)  # identity
_X = np.array([[0, 1], [1, 0]], dtype=complex)  # Pauli X
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)  # Pauli Y
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)  # Hadamard
_S = np.diag([1, 1j]).astype(complex)  # phase gate S = sqrt(Z)


def _RY(theta):
    """R_y(theta) = exp(-i theta Y / 2): real rotation, |0> -> cos(theta/2)|0> + sin(theta/2)|1>."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)  # half-angle cosine and sine
    return np.array([[c, -s], [s, c]], dtype=complex)  # standard real rotation matrix


def _RX(theta):
    """R_x(theta) = exp(-i theta X / 2)."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)  # half-angle cosine and sine
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)  # rotation about the X axis


def _RZ(theta):
    """R_z(theta) = exp(-i theta Z / 2)."""
    return np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)])  # diagonal phase rotation


def _u(q, m):
    """Instruction: apply the 2x2 unitary m to qubit q."""
    return ('U', q, m)  # tuples are cheap; matrices are merged later


def _cz(a, b):
    """Instruction: CZ between qubits a and b (symmetric)."""
    return ('CZ', a, b)  # the only two-qubit gate we ever emit


def _cnot(control, target):
    """CNOT = H_t CZ H_t (one CZ, derivation (a))."""
    return [_u(target, _H), _cz(control, target), _u(target, _H)]  # HZH = X on the target


def _cry_on_zero(control, target, theta):
    """Controlled-R_y(theta), valid when target = |0> whenever control = |1> (one CZ, derivation (b))."""
    return [_u(target, _RY(-theta / 2)), _cz(control, target), _u(target, _RY(theta / 2))]  # R_y(t/2) CZ R_y(-t/2)


def _split_angle(p):
    """Angle theta with sin^2(theta/2) = p, i.e. R_y(theta)|0> has |1>-probability p."""
    return 2 * math.asin(math.sqrt(min(max(p, 0.0), 1.0)))  # clamping guards tiny float overshoots


def _transfer(src, dst, p):
    """|1>_src|0>_dst -> sqrt(1-p)|10> + sqrt(p)|01>, |00> -> |00> (two CZs, derivation (c))."""
    return _cry_on_zero(src, dst, _split_angle(p)) + _cnot(dst, src)  # rotate dst, then clear src if dst took it


def _givens(a, b, theta, middle_a=None):
    """Givens rotation G(theta) on (a, b) with two CZs (derivation (d)).

    `middle_a` optionally replaces the middle R_x(theta/2) on qubit a (used by the merged top block).
    """
    if middle_a is None:  # plain Givens rotation
        middle_a = [_u(a, _RX(theta / 2))]  # the XX-part angle
    ops = [_u(b, _S), _u(a, _RX(np.pi / 2)), _u(b, _RX(np.pi / 2))]  # W V: map XY-YX onto -(XX+ZZ)
    ops += _cnot(a, b)  # CNOT conjugation turns X_a into X_a X_b and Z_b into Z_a Z_b ...
    ops += middle_a + [_u(b, _RZ(theta / 2))]  # ... so these rotations become exp(-i t XX/2) exp(-i t ZZ/2)
    ops += _cnot(a, b)  # close the CNOT conjugation
    ops += [_u(a, _RX(-np.pi / 2)), _u(b, _RX(-np.pi / 2)), _u(b, _S.conj().T)]  # undo W and V
    return ops


def _g3(c, a, b, theta):
    """Moving-hole gate on (c, a, b): partial move a->b if c=0, full move if c=1 (five CZs, derivation (e))."""
    al = (theta + np.pi) / 4  # alpha in the derivation
    be = (theta - np.pi) / 4  # beta in the derivation
    ops = _cnot(a, b)  # afterwards b = 1 implies a = 0, enabling known-target rotations of a
    ops += [_u(a, _RY(-al)), _cz(c, a), _u(a, _RY(-be)), _cz(b, a), _u(a, _RY(be)), _cz(c, a), _u(a, _RY(al))]  # M
    ops += _cnot(a, b)  # map |a=1,b=1> back to |a=1,b=0>, completing the excitation move
    return ops


def _top_block(a, b, c, p_top, p_next):
    """Merged first two steps of an SCS sweep on a line a-b-c, four CZs (derivation (f)).

    Domain {000, 001, 011} (qubit order a, b, c). The excitation on c moves to b with probability p_next
    when the input is 001, and to a with probability p_top when the input is 011.
    """
    t2 = _split_angle(p_top)  # rotation for the 011 branch (landing site a)
    t1 = _split_angle(p_next)  # rotation for the 001 branch (landing site b)
    ops = _cry_on_zero(b, a, t2)  # step 1: only 011 has b = 1, and there a = 0
    middle = [_u(b, _RX((t1 - np.pi) / 2)), _cz(a, b), _u(b, _RX(np.pi / 2))]  # step 2: Z_b inserted iff a = 1
    ops += _givens(b, c, t1, middle_a=middle)  # Givens on (b, c) with the modified middle layer
    ops += [_u(a, _S)]  # step 3: cancel the -i phase acquired by the a = 1 branch
    return ops


# %% [markdown]
# ## 2. Logical constructions
#
# ### 2.1 W states ($k=1$): excitation tree, $2N-3$ CZs, depth $O(\log N)$
#
# Pick a spanning tree of $N$ qubits with root $r$, and write $s(v)$ for the size of the subtree under $v$. Starting from
# $|1\rangle_r$, every node $v$ that currently "owns" the excitation for $m_v$ qubits (itself plus its not-yet-served child subtrees)
# transfers it to child $c$ with probability $s(c)/m_v$. The probability that the excitation reaches a given qubit is the product
# of conditional probabilities along its root path, which telescopes to $1/N$; all amplitudes stay real and positive, so the
# result is exactly $|W_N\rangle$.
#
# The *first* transfer from the root acts on the product state $|1\rangle_r|0\rangle_c$, so it needs only one CZ:
# prepare $r$ in $\sqrt p|0\rangle+\sqrt{1-p}|1\rangle$, then apply $\mathrm{CNOT}_{r\to c}$ and $X_c$. Total: $1+2(N-2)=2N-3$ CZs. An exhaustive numerical
# search confirmed that $W_3$ needs 3 and $W_4$ needs 5 CZs, so this count is optimal at those sizes.
#
# Children are served in order of decreasing subtree height, so the longest chain starts first; the depth is roughly
# $2\times$(tree height + branching), i.e. logarithmic in $N$ for a bushy tree on the 2D grid.
#
# ### 2.2 General $k$: Bärtschi–Eidenbenz (BE) on a path
#
# BE prepares $|D^{(N)}_k\rangle=U_N|0^{N-k}1^k\rangle$ with $U_N=\prod_{m=N}^{2}\mathrm{SCS}_m$. Before step $m$ the first $m$ positions hold
# $|0^{m-r}1^r\rangle$ (ones right-justified, $r$ = remaining excitations). Since
# $$|D^{(m)}_r\rangle=\sqrt{\tfrac rm}\,|D^{(m-1)}_{r-1}\rangle|1\rangle+\sqrt{\tfrac{m-r}m}\,|D^{(m-1)}_r\rangle|0\rangle,$$
# step $m$ must map $|0^{m-r}1^r\rangle\mapsto\sqrt{\tfrac rm}|0^{m-r}1^r\rangle+\sqrt{\tfrac{m-r}m}|0^{m-r-1}1^r0\rangle$: either position $m$ keeps its 1, or the
# "hole" at the boundary $m-r$ travels to $m$ (a cyclic shift). Only the window $m-k,\dots,m$ is touched.
#
# We implement $\mathrm{SCS}_m$ as a *local sweep* along the window, so it needs only nearest-neighbour CZs: the hole moves
# one site at a time, partially at its original boundary and fully while "in transit". The first two sweep steps form the
# 4-CZ merged top block; the remaining steps are $G_3$ gates. Cost per step: 2 CZs ($k=1$), 4 CZs ($k=2$), $4+5(k-2)$ in general
# (textbook BE uses one non-local three-qubit gate per extra branch, each several CNOTs, **plus** SWAP routing on a line).
#
# *Caveat handled in code:* when $m\le k$ the window is the whole prefix and the all-ones input $|1^m\rangle$ can occur; the
# merged block is not valid there (numerically it would need at least 6 CZs), so those steps use the plain sweep, which fixes $|1\dots1\rangle$.
#
# ### 2.3 General $k$: two-arm "split" construction (our main $k\ge2$ workhorse)
#
# Split the qubits into a left arm $L$ ($a$ qubits) and a right arm $R$ ($b=N-a$ qubits). Then
# $$|D^{(N)}_k\rangle=\sum_j\sqrt{P(j)}\;|D^{(a)}_{k-j}\rangle_L|D^{(b)}_j\rangle_R,\qquad P(j)=\frac{\binom bj\binom a{k-j}}{\binom Nk}\ \text{(hypergeometric)}.$$
# Because the total weight $k$ is *fixed* at the start, the weight distribution is cheap to prepare in unary:
# 1. **Thermometer chain** on the first $k$ qubits $r_1..r_k$ of $R$: $r_1\leftarrow R_y$ with $P(j\ge1)$, then known-target rotations
#    $r_{u-1}\to r_u$ with conditional probabilities $P(j\ge u)/P(j\ge u-1)$. Result: $\sum_j\sqrt{P(j)}|1^j0^{k-j}\rangle$ ($k-1$ CZs, exact).
# 2. **Rung copy.** The left arm's first $k$ qubits are laid out as a parallel row in *reverse* order, so that $l_{k+1-u}$ neighbours
#    $r_u$ (a "ladder"). The left arm needs the thermometer $l_v=[k-j\ge v]=\lnot r_{k+1-v}$, so one parallel layer of CNOTs plus $X$ gates
#    writes it ($k$ CZs, depth 1).
# 3. Each arm now holds a right-justified unary input, so we run the BE unitary on both arms **in parallel**.
#
# Compared with one long BE path, the two arms run concurrently (about half the depth) and the construction also *saves CZs*,
# because the expensive full-window sweeps happen on half as many positions. Examples: $(N,k)=(6,3)$: 23 vs 36 CZs; $(8,4)$: 49 vs 77.

# %%
def _scs_ops(window, m, all_ones_possible):
    """SCS_m on window = [p_k, ..., p_1, p_0] (p_0 = position m) as a nearest-neighbour sweep."""
    k = len(window) - 1  # window size minus one = largest number of excitations handled
    if k == 1:  # a single partial move = Givens rotation (it also fixes |11>)
        return _givens(window[0], window[1], _split_angle((m - 1) / m))  # move the hole with probability (m-1)/m
    ops = []  # accumulated instructions
    i = k  # current hole index: branch r = i has its hole at p_i
    if not all_ones_possible:  # the merged top block needs p_k = 0 on every input
        ops += _top_block(window[0], window[1], window[2], (m - k) / m, (m - k + 1) / m)  # branches r = k and k-1
        i = k - 2  # the block already performed the top two sweep steps
    while i >= 1:  # remaining sweep steps, top to bottom
        theta = _split_angle((m - i) / m)  # branch r = i moves its hole with probability (m-i)/m
        if i == k:  # very first step of an unmerged sweep: no "in transit" control yet
            ops += _givens(window[0], window[1], theta)  # partial move p_k <-> p_{k-1}
        else:  # generic step: the control is the site above the hole position
            ops += _g3(window[k - i - 1], window[k - i], window[k - i + 1], theta)  # (c, a, b) = (p_{i+1}, p_i, p_{i-1})
        i -= 1  # move one site towards the head p_0
    return ops


def dicke_unitary_ops(path, k):
    """BE unitary on `path`: maps |0^{N-r} 1^r> (ones at the END of the list) to |D^(N)_r> for every r <= k."""
    N = len(path)  # number of positions
    k = min(k, N)  # cannot hold more excitations than qubits
    ops = []  # accumulated instructions
    for m in range(N, 1, -1):  # decide positions N, N-1, ..., 2 (position 1 is then determined)
        kk = min(k, m - 1)  # the window reaches back k sites, but not past position 1
        ops += _scs_ops(list(path[m - 1 - kk: m]), m, all_ones_possible=(k >= m))  # all-ones input reachable iff k >= m
    return ops


def dicke_path_ops(path, k):
    """|D^(N)_k> on a path: X on the last k sites, then the BE unitary."""
    return [_u(q, _X) for q in path[len(path) - k:]] + dicke_unitary_ops(path, k)  # |0^{N-k}1^k>, then U_N


def split_ops(left, right, k):
    """Two-arm construction (Section 2.3).

    left = [l_1, ..., l_a], right = [r_1, ..., r_b]; l_1 and r_1 are the arm heads.
    Ladder requirement: l_{k+1-u} must neighbour r_u for every rung that is actually used.
    """
    a, b = len(left), len(right)  # arm lengths
    N = a + b  # total number of qubits
    P = [math.comb(b, j) * math.comb(a, k - j) / math.comb(N, k) if (0 <= k - j <= a and j <= b) else 0.0
         for j in range(k + 1)]  # hypergeometric law of j = number of excitations in the right arm
    tail = [sum(P[u:]) for u in range(k + 2)]  # tail[u] = P(j >= u)
    ops = []  # accumulated instructions
    for u in range(1, min(k, b) + 1):  # thermometer chain r_1 ... r_k
        if tail[u] <= 1e-15:  # j >= u impossible: r_u stays |0>, and so do all later r's
            break  # nothing more to prepare
        if u == 1:  # first bit from scratch
            ops.append(_u(right[0], _RY(_split_angle(tail[1]))))  # P(r_1 = 1) = P(j >= 1)
        else:  # later bits conditioned on the previous one (known |0> target)
            ops += _cry_on_zero(right[u - 2], right[u - 1], _split_angle(tail[u] / tail[u - 1]))  # P(j>=u | j>=u-1)
    for u in range(1, k + 1):  # rung copies l_{k+1-u} = NOT r_u
        v = k + 1 - u  # left index paired with r_u
        if v > a:  # left arm too short: this left bit does not exist (then r_u is always 1)
            continue  # nothing to write
        if u > b or tail[u] <= 1e-15:  # r_u is always 0, so l_v is always 1
            ops.append(_u(left[v - 1], _X))  # deterministic bit, no CZ needed
        elif tail[u] < 1 - 1e-15:  # r_u genuinely random: copy its complement across the rung
            ops += _cnot(right[u - 1], left[v - 1]) + [_u(left[v - 1], _X)]  # l_v = 1 XOR r_u
        # otherwise r_u is always 1 and l_v correctly stays |0>
    ops += dicke_unitary_ops(list(reversed(right)), min(k, b))  # BE on the right arm (head r_1 = last element)
    ops += dicke_unitary_ops(list(reversed(left)), min(k, a))  # BE on the left arm; runs in parallel
    return ops


def _subtree_sizes(root, children):
    """Subtree sizes s(v) for a rooted tree given as {node: [children]}."""
    order = [root]  # BFS order
    for v in order:  # the list grows while we iterate
        order.extend(children.get(v, []))  # append children
    size = {}  # node -> subtree size
    for v in reversed(order):  # leaves first
        size[v] = 1 + sum(size[c] for c in children.get(v, []))  # itself plus its children's subtrees
    return size


def w_tree_ops(root, children):
    """|W_N> on a rooted tree with 2N-3 CZs (Section 2.1)."""
    size = _subtree_sizes(root, children)  # s(v)
    kids = children.get(root, [])  # root's children in serving order
    if not kids:  # single qubit: |W_1> = |1>
        return [_u(root, _X)]  # just flip it
    first = kids[0]  # first child, served with the 1-CZ product-state split
    p = size[first] / size[root]  # probability that the excitation goes to the first child's subtree
    ops = [_u(root, _RY(_split_angle(1 - p)))]  # root = sqrt(p)|0> + sqrt(1-p)|1>
    ops += _cnot(root, first) + [_u(first, _X)]  # -> sqrt(p)|01> + sqrt(1-p)|10>
    stack = [root]  # depth-first traversal keeps every qubit's instruction order valid
    while stack:  # visit every node once
        v = stack.pop()  # current owner of an excitation
        remaining = size[v] - (size[first] if v == root else 0)  # mass still owned by v
        for c in children.get(v, []):  # serve the children in order
            if v == root and c == first:  # already served by the 1-CZ split
                continue  # skip it
            ops += _transfer(v, c, size[c] / remaining)  # hand child c its share
            remaining -= size[c]  # v now owns less
        stack.extend(reversed(children.get(v, [])))  # children then process their own subtrees
    return ops


# %% [markdown]
# ## 3. Compilation: native gates, layering, dynamical decoupling
#
# 1. **Merge.** All single-qubit matrices on a qubit between two of its CZs are multiplied into one unitary and emitted
#    as a single `PhasedXZGate` (every single-qubit unitary is a `PhasedXZ` up to global phase). This minimises the
#    single-qubit gate count and error.
# 2. **Layer.** CZs form a DAG (two CZs depend on each other if they share a qubit). We compute the ASAP depth $D$ and then
#    place every CZ **as late as possible** (ALAP) within $D$ layers. Moments alternate *1q layer / CZ layer*, which matches
#    how Google calibrates parallel CZ moments. ALAP keeps each qubit in $|0\rangle$ for as long as possible, and $|0\rangle$ is
#    immune to both $T_1$ decay and dephasing, so idle error only accumulates after a qubit's first gate.
# 3. **Dynamical decoupling (DD).** Once a qubit is active, every 1q slot in which it would otherwise idle receives an $X$ or $Y$
#    pulse (alternating, XY4-style). The last slot of each idle window receives the inverse of the accumulated Pauli product,
#    merged with the real gate that belongs there. Paulis multiply to a Pauli and the window contains no CZ on that qubit, so
#    the logical unitary is **exactly** unchanged, while quasi-static $Z$ errors (detuning, spectator phases from neighbouring CZs)
#    are echoed out. The challenge text flags DD as important; our emulator (later section) quantifies the trade-off, since each
#    pulse costs about one single-qubit error.

# %%
def _is_identity(u, atol=1e-9):
    """True if u is the identity up to a global phase."""
    phase = u[0, 0] if abs(u[0, 0]) > 0.5 else 1.0  # an identity-like matrix has |u00| ~ 1
    return np.allclose(u / phase, _I2, atol=atol)  # compare after removing the phase


def _phxz(q, u):
    """Native gate for an arbitrary single-qubit unitary."""
    return cirq.PhasedXZGate.from_matrix(u).on(q)  # exact up to a global phase


def compile_layers(ops, dd=True):
    """Instruction list -> cirq.Circuit with alternating PhasedXZ / CZ moments (ALAP, optional DD)."""
    pending = {}  # qubit -> product of 1q matrices since its last CZ
    last = {}  # qubit -> index of its most recent CZ node
    nodes = []  # CZ nodes: qubits, pre-gates, DAG edges
    for kind, a, b in ops:  # walk the instructions in program order
        if kind == 'U':  # single-qubit matrix b on qubit a
            pending[a] = b @ pending.get(a, _I2)  # left-multiply: later gates act after earlier ones
            continue  # nothing else to do
        nodes.append({'qubits': (a, b),  # a CZ node
                      'pre': {a: pending.pop(a, None), b: pending.pop(b, None)},  # 1q gates right before it
                      'preds': {last[q] for q in (a, b) if q in last},  # CZs that must come first
                      'succs': set()})  # filled below
        last[a] = last[b] = len(nodes) - 1  # this CZ is now the latest on both qubits
    trailing = pending  # 1q gates after a qubit's last CZ (or the only gates of CZ-free qubits)
    for i, nd in enumerate(nodes):  # build successor lists
        for p in nd['preds']:  # every predecessor ...
            nodes[p]['succs'].add(i)  # ... points forward to i
    asap = [0] * len(nodes)  # earliest CZ layer (1-based)
    for i, nd in enumerate(nodes):  # program order is a topological order
        asap[i] = 1 + max((asap[p] for p in nd['preds']), default=0)  # one after the latest predecessor
    depth = max(asap, default=0)  # number of CZ layers
    layer = [0] * len(nodes)  # ALAP layer
    for i in range(len(nodes) - 1, -1, -1):  # reverse topological order
        layer[i] = min((layer[s] for s in nodes[i]['succs']), default=depth + 1) - 1  # one before the earliest successor
    n_mom = 2 * depth + 1  # moments: 1q, CZ, 1q, CZ, ..., 1q
    slots = [dict() for _ in range(n_mom)]  # 1q moments: qubit -> matrix
    cz_moments = [[] for _ in range(n_mom)]  # CZ moments
    q_layers = {}  # qubit -> list of (CZ layer, pre-gate)
    for i, nd in enumerate(nodes):  # place the CZs
        cz_moments[2 * layer[i] - 1].append(cirq.CZ(*nd['qubits']))  # CZ layer t lives in moment 2t-1
        for q in nd['qubits']:  # remember each qubit's timeline
            q_layers.setdefault(q, []).append((layer[i], nd['pre'][q]))  # its pre-gate goes right before this CZ
    for q in set(q_layers) | set(trailing):  # every qubit that appears
        seq = sorted(q_layers.get(q, []), key=lambda t: t[0])  # its CZs in time order
        if not seq:  # no CZ at all: put its only gate in the final 1q moment (stay |0> as long as possible)
            u = trailing.get(q)  # its merged gate
            if u is not None and not _is_identity(u):  # skip identities
                slots[n_mom - 1][q] = u  # last moment
            continue  # done with this qubit
        t0, u0 = seq[0]  # first CZ of this qubit
        if u0 is not None:  # gates before the first CZ (the qubit is |0> before them, so no DD there)
            slots[2 * t0 - 2][q] = u0  # 1q moment right before CZ layer t0
        windows = [(seq[j][0], seq[j + 1][0], seq[j + 1][1], 'mid') for j in range(len(seq) - 1)]  # between CZs
        windows.append((seq[-1][0], depth + 1, trailing.get(q), 'tail'))  # after the last CZ, until the end
        for t_a, t_b, u, kind in windows:  # each idle window of an active qubit
            moms = [2 * t for t in range(t_a, t_b)]  # its 1q moments
            u = _I2 if u is None else u  # net single-qubit operation required in this window
            if not dd or len(moms) < 2:  # no idle CZ layer inside (or DD disabled)
                if not _is_identity(u):  # only emit real gates
                    slots[moms[-1] if kind == 'mid' else moms[0]][q] = u  # just before next CZ / just after last CZ
                continue  # next window
            pulses = [(_X if i % 2 == 0 else _Y) for i in range(len(moms) - 1)]  # X, Y, X, Y, ...
            acc = _I2  # running product of the pulses
            for p in pulses:  # accumulate
                acc = p @ acc  # later pulses act after earlier ones
            pulses.append(acc.conj().T)  # the final slot undoes the product: window total = identity
            if kind == 'mid':  # the real gate must act right before the next CZ
                pulses[-1] = u @ pulses[-1]  # merge it into the final slot
            else:  # tail window: apply the real gate first, then decouple
                pulses[0] = pulses[0] @ u  # merge it into the first slot
            for mom, p in zip(moms, pulses):  # emit
                if not _is_identity(p):  # identity slots stay empty
                    slots[mom][q] = p  # one PhasedXZ per qubit per moment
    moments = []  # assemble the circuit
    for i in range(n_mom):  # in time order
        ops_i = [_phxz(q, u) for q, u in slots[i].items()] if i % 2 == 0 else cz_moments[i]  # 1q or CZ moment
        if ops_i:  # drop empty moments
            moments.append(cirq.Moment(ops_i))  # one homogeneous moment
    return cirq.Circuit(moments)  # final native circuit


# %% [markdown]
# ## 4. Calibration model, success-probability estimate, placement search
#
# **Pessimistic error rates.** Each calibration entry is a measured mean $\hat e$ with statistical uncertainty $\sigma$. Choosing the
# minimum of many noisy estimates systematically selects under-estimates (the "winner's curse"), so we rank with
# $e=\hat e+\beta\sigma$, $\beta=1$ (a one-standard-error penalty), following the plan's uncertainty-aware placement.
#
# **Predicted log-fidelity** of a compiled candidate (higher is better):
# $$\log F=\sum_{\mathrm{CZ}}\log(1-e_{\mathrm{CZ}})+\sum_{\mathrm{1q}}\log(1-e_{\mathrm{1q}})+\sum_{q}\log\!\big(1-[(1-p)e^{0\to1}_q+p\,e^{1\to0}_q]\big)-\sum_q\frac{t^{\mathrm{active}}_q}{T_{\mathrm{idle}}}.$$
# The readout term uses the exact excitation probability $p=n/N$ of a Dicke qubit (asymmetric readout matters: $e^{1\to0}\approx2e^{0\to1}$).
# $t^{\mathrm{active}}_q$ is the time from the qubit's first gate to measurement (25 ns per 1q moment, 42 ns per CZ moment), and
# $T_{\mathrm{idle}}=60\,\mu$s is a nominal combined $T_1$/dephasing time for Willow-class qubits with DD (it only has to rank
# candidates, not predict absolute numbers).
#
# **Search spaces** (exhaustive over starting points, pruned by cheap proxies, then compiled and scored exactly):
# * *Trees* ($k=1$): for every root and three depth penalties, a Prim-like greedy growth adds the cheapest frontier qubit
#   (2 CZs on its edge + readout + single-qubit cost + depth penalty).
# * *Paths* (BE): beam search over simple paths, with position-dependent weights equal to the actual number of CZs / 1q gates
#   the construction places on each edge / qubit.
# * *Ladders* (split): every straight $2\times k$ ladder on the grid, extended into two disjoint arms by beam search.

# %%
T_1Q = 25e-9  # duration of a single-qubit moment (s)
T_CZ = 42e-9  # duration of a CZ moment (s)
T_IDLE = 60e-6  # nominal effective coherence time used to price idling (s)


class Calibration:
    """Pessimistic error rates and the usable coupling graph built from the four calibration dictionaries."""

    def __init__(self, cz, sq, e0, e1, beta=1.0):
        def pess(v):  # mean + beta * sigma, clipped to a sane range
            mean, std = v  # calibration tuple
            return float(min(max(mean + beta * std, 1e-6), 0.5))  # keeps the logarithms finite

        self.sq = {q: pess(v) for q, v in sq.items()}  # usable qubits = those with single-qubit data
        med0 = float(np.median([v[0] for v in e0.values()])) if e0 else 0.01  # fallback 0->1 readout error
        med1 = float(np.median([v[0] for v in e1.values()])) if e1 else 0.02  # fallback 1->0 readout error
        self.e0 = {q: pess(e0[q]) if q in e0 else med0 for q in self.sq}  # 0 -> 1 readout error
        self.e1 = {q: pess(e1[q]) if q in e1 else med1 for q in self.sq}  # 1 -> 0 readout error
        self.cz = {}  # frozenset pair -> CZ error
        for pair, v in cz.items():  # only couplers whose two qubits are usable
            a, b = tuple(pair)  # unpack (tuple or frozenset)
            if a in self.sq and b in self.sq and a != b:  # both ends usable
                key = frozenset((a, b))  # orientation-free key
                self.cz[key] = min(self.cz.get(key, 1.0), pess(v))  # keep the better value if listed twice
        self.adj = {q: set() for q in self.sq}  # adjacency of the usable graph
        for key in self.cz:  # fill it
            a, b = tuple(key)  # endpoints
            self.adj[a].add(b)  # undirected ...
            self.adj[b].add(a)  # ... edge

    def edge_cost(self, a, b):
        """-log(1 - CZ error) for coupler (a, b)."""
        return -math.log(1 - self.cz[frozenset((a, b))])  # additive cost

    def ro_cost(self, q, p):
        """-log(1 - readout error) for a qubit that is |1> with probability p."""
        return -math.log(1 - ((1 - p) * self.e0[q] + p * self.e1[q]))  # mixture of both error types

    def sq_cost(self, q):
        """-log(1 - single-qubit error)."""
        return -math.log(1 - self.sq[q])  # additive cost


def estimate_log_fidelity(circuit, cal, data_qubits, p_exc):
    """Predicted log success probability of a compiled circuit (see the formula above)."""
    total = 0.0  # running log-fidelity
    first = {}  # qubit -> index of its first moment
    durations = []  # moment durations
    for i, moment in enumerate(circuit):  # scan the moments
        is_cz = any(len(op.qubits) == 2 for op in moment)  # moments are homogeneous: CZ or 1q
        durations.append(T_CZ if is_cz else T_1Q)  # its duration
        for op in moment:  # gate errors
            for q in op.qubits:  # remember when each qubit becomes active
                first.setdefault(q, i)  # first occurrence only
            if len(op.qubits) == 2:  # CZ
                total += math.log(1 - cal.cz[frozenset(op.qubits)])  # its error
            else:  # PhasedXZ
                total += math.log(1 - cal.sq[op.qubits[0]])  # its error
    cum = np.cumsum([0.0] + durations)  # cumulative time at each moment boundary
    for q, i in first.items():  # idling from the first gate until measurement
        total -= (cum[-1] - cum[i]) / T_IDLE  # linearised decay
    for q in data_qubits:  # readout
        total -= cal.ro_cost(q, p_exc)  # its error
    return total


def _tree_children(parent):
    """{child: parent} -> {node: [children sorted by decreasing subtree height]}."""
    children = {}  # node -> children
    for v, u in parent.items():  # invert the parent map
        children.setdefault(u, []).append(v)  # u is v's parent
    height = {}  # memoised subtree heights

    def h(v):  # height of v's subtree
        if v not in height:  # compute once
            height[v] = 1 + max((h(c) for c in children.get(v, [])), default=0)  # 1 + tallest child
        return height[v]  # cached value

    for v in list(children):  # order every child list
        children[v].sort(key=lambda c: (-h(c), str(c)))  # tallest first (critical path first), deterministic ties
    return children


def grow_tree(cal, root, N, p_exc, depth_weight):
    """Prim-like greedy N-node tree from `root`; returns {child: parent} or None."""
    in_tree = {root}  # nodes already in the tree
    parent = {}  # child -> parent
    depth = {root: 0}  # tree depth of each node
    heap = []  # frontier edges keyed by cost

    def push(u):  # add u's frontier edges
        for v in cal.adj[u]:  # neighbours
            if v not in in_tree:  # only new nodes
                c = (2 * cal.edge_cost(u, v) + cal.ro_cost(v, p_exc) + 4 * cal.sq_cost(v)
                     + depth_weight * (depth[u] + 1))  # 2 CZs + readout + ~4 1q gates + depth penalty
                heapq.heappush(heap, (c, str(v), v, u))  # str(v) breaks ties deterministically

    push(root)  # start from the root
    while len(in_tree) < N and heap:  # grow until N nodes
        _, _, v, u = heapq.heappop(heap)  # cheapest frontier edge
        if v in in_tree:  # stale entry
            continue  # skip
        in_tree.add(v)  # accept v
        parent[v] = u  # reached via u
        depth[v] = depth[u] + 1  # one level deeper
        push(v)  # extend the frontier
    return parent if len(in_tree) == N else None  # None if the component is too small


def beam_paths(cal, N, edge_w, node_w, p_exc, beam=200):
    """Beam search for low-cost simple paths of N qubits with position-dependent weights."""
    beams = [(node_w[0] * cal.sq_cost(q) + cal.ro_cost(q, p_exc), (q,)) for q in sorted(cal.sq, key=str)]  # all starts
    for pos in range(1, N):  # append one position at a time
        cand = {}  # dedupe partial paths by (qubit set, end)
        for cost, path in beams:  # expand every beam
            u = path[-1]  # current end
            for v in cal.adj[u]:  # possible next qubit
                if v in path:  # simple paths only
                    continue  # skip
                c = cost + edge_w[pos - 1] * cal.edge_cost(u, v) + node_w[pos] * cal.sq_cost(v) + cal.ro_cost(v, p_exc)  # new cost
                key = (frozenset(path) | {v}, v)  # equivalent partial paths
                if key not in cand or cand[key][0] > c:  # keep the best representative
                    cand[key] = (c, path + (v,))  # extended path
        beams = heapq.nsmallest(beam, cand.values(), key=lambda t: t[0])  # prune
        if not beams:  # dead end everywhere
            return []  # no path of this length
    return beams  # sorted best-first


def extend_path(cal, prefix, length, forbidden, p_exc, w_edge, w_node, beam=30):
    """Beam-search continuations of `prefix` to `length` qubits avoiding `forbidden`."""
    beams = [(0.0, tuple(prefix))]  # start from the prefix
    while len(beams[0][1]) < length:  # until long enough
        cand = {}  # dedupe
        for cost, path in beams:  # expand
            u = path[-1]  # current end
            for v in cal.adj[u]:  # neighbours
                if v in path or v in forbidden:  # keep the arms disjoint and simple
                    continue  # skip
                c = cost + w_edge * cal.edge_cost(u, v) + w_node * cal.sq_cost(v) + cal.ro_cost(v, p_exc)  # new cost
                key = (frozenset(path) | {v}, v)  # equivalence class
                if key not in cand or cand[key][0] > c:  # keep the best
                    cand[key] = (c, path + (v,))  # extended path
        beams = heapq.nsmallest(beam, cand.values(), key=lambda t: t[0])  # prune
        if not beams:  # cannot extend
            return []  # failure
    return beams  # best-first continuations


def ladder_candidates(cal, k, p_exc, w_edge, w_node):
    """All straight 2 x k ladders: right row r_1..r_k and a left row with l_{k+1-u} next to r_u."""
    out = []  # (cost, left_head, right_head)
    for q in sorted(cal.sq, key=str):  # r_1
        for d in ((0, 1), (0, -1), (1, 0), (-1, 0)):  # direction of the right row
            for s in ((d[1], d[0]), (-d[1], -d[0])):  # side on which the left row lies
                right = [cirq.GridQubit(q.row + u * d[0], q.col + u * d[1]) for u in range(k)]  # r_1..r_k
                left = [cirq.GridQubit(r.row + s[0], r.col + s[1]) for r in reversed(right)]  # l_1..l_k (reversed)
                if any(x not in cal.sq for x in right + left):  # all qubits must be usable
                    continue  # skip
                edges = (list(zip(right, right[1:])) + list(zip(left, left[1:]))
                         + [(right[u], left[k - 1 - u]) for u in range(k)])  # rows + rungs
                if any(frozenset(e) not in cal.cz for e in edges):  # all couplers must be usable
                    continue  # skip
                cost = (sum(w_edge * cal.edge_cost(*e) for e in edges)
                        + sum(w_node * cal.sq_cost(x) + cal.ro_cost(x, p_exc) for x in right + left))  # proxy cost
                out.append((cost, left, right))  # keep
    out.sort(key=lambda t: t[0])  # cheapest first
    return out


def path_profile(N, k):
    """CZ count per edge and gate count per position of the BE construction (weights for the path search)."""
    qs = cirq.LineQubit.range(N)  # abstract line
    circ = compile_layers(dicke_path_ops(qs, k), dd=False)  # compile once
    pos = {q: i for i, q in enumerate(qs)}  # qubit -> position index
    edge_w = [0] * max(N - 1, 1)  # CZs per edge
    node_w = [0] * N  # 1q gates per position
    for op in circ.all_operations():  # tally
        if len(op.qubits) == 2:  # CZ (always nearest-neighbour)
            edge_w[min(pos[q] for q in op.qubits)] += 1  # its edge index
        else:  # PhasedXZ
            node_w[pos[op.qubits[0]]] += 1  # its position
    return edge_w, node_w


# %% [markdown]
# ## 5. `create_dicke_state`
#
# Candidate families per $k=\min(n,N-n)$:
#
# | $k$ | Families tried |
# | --- | --- |
# | 0 | No entangling gates: the $N$ qubits with the best relevant readout error, optionally flipped |
# | 1 | Excitation trees (all roots × 3 depth penalties) |
# | $\ge2$ | BE paths (top beams) **and** two-arm split ladders (top ladders × arm extensions) |
#
# Every candidate is compiled and scored with the predicted log-fidelity; the best one with at most 1750 moments is returned.
# A wall-clock budget (30 s, leaving a 2x margin under the 60 s limit for slower machines) stops the search early. Impossible requests
# ($n<0$, $n>N$, more qubits than usable ones) raise `NotImplementedError`.

# %%
def _x_all(qubits):
    """Instructions flipping every qubit (complement trick)."""
    return [_u(q, _X) for q in qubits]  # merged into the final 1q layer by the compiler


def generate_candidates(N, n, cal, *, dd=True, deadline=None, max_trees=40, max_paths=10, max_ladders=12):
    """Yield exact candidates (circuit, qubit_list, kind) for |D^(N)_n> on the calibrated device."""
    deadline = deadline if deadline is not None else time.monotonic() + 30.0  # default budget
    k = min(n, N - n)  # prepare the lighter of the two complementary states
    flip = n > N - n  # append X everywhere afterwards if we prepared the complement
    p_exc = n / N  # excitation probability of each Dicke qubit (for readout pricing)
    if k == 0:  # computational basis state: no entanglement needed
        qs = sorted(cal.sq, key=lambda q: ((cal.e1[q] if flip else cal.e0[q]) + cal.sq[q], str(q)))[:N]  # best readout
        gate = cirq.PhasedXZGate(x_exponent=1 if flip else 0, z_exponent=0, axis_phase_exponent=0)  # X or identity
        yield cirq.Circuit(cirq.Moment(gate.on(q) for q in qs)), qs, 'trivial'  # explicit gate keeps qubits visible
        return  # only candidate
    if k == 1:  # W-state trees
        seen = set()  # avoid compiling duplicate trees
        proxies = []  # (proxy cost, root name, root, parent map)
        for w in (0.0, 0.01, 0.03):  # depth penalties: from pure quality to compact trees
            for r in sorted(cal.sq, key=str):  # every root
                parent = grow_tree(cal, r, N, p_exc, w)  # greedy tree
                if parent is None:  # component too small
                    continue  # skip
                key = (r, frozenset(parent.items()))  # identity of the tree
                if key in seen:  # duplicate
                    continue  # skip
                seen.add(key)  # remember it
                cost = (sum(2 * cal.edge_cost(v, u) for v, u in parent.items())
                        + sum(cal.ro_cost(q, p_exc) for q in [r, *parent]))  # cheap proxy: CZs + readout
                proxies.append((cost, str(r), r, parent))  # keep
        proxies.sort(key=lambda t: (t[0], t[1]))  # best proxies first
        for _, _, r, parent in proxies[:max_trees]:  # compile the most promising trees
            qs = [r, *parent]  # its qubits
            ops = w_tree_ops(r, _tree_children(parent)) + (_x_all(qs) if flip else [])  # exact construction
            yield compile_layers(ops, dd=dd), qs, 'tree'  # candidate
            if time.monotonic() > deadline:  # out of time
                return  # stop
        return  # trees beat paths for k = 1 (same CZ count, logarithmic instead of linear depth)
    edge_w, node_w = path_profile(N, k)  # per-position weights of the BE path
    for _, path in beam_paths(cal, N, edge_w, node_w, p_exc)[:max_paths]:  # best paths
        ops = dicke_path_ops(list(path), k) + (_x_all(path) if flip else [])  # exact construction
        yield compile_layers(ops, dd=dd), list(path), 'path'  # candidate
        if time.monotonic() > deadline:  # out of time
            return  # stop
    a, b = N // 2, N - N // 2  # balanced arms (b >= a >= k because k <= N/2)
    ew, nw = path_profile(b, k)  # typical arm weights
    w_edge = sum(ew) / max(len(ew), 1)  # average CZs per arm edge
    w_node = sum(nw) / max(len(nw), 1)  # average 1q gates per arm qubit
    for _, left0, right0 in ladder_candidates(cal, k, p_exc, w_edge, w_node)[:max_ladders]:  # best ladders
        used = set(left0) | set(right0)  # ladder qubits
        for _, right in extend_path(cal, right0, b, used, p_exc, w_edge, w_node)[:2]:  # right-arm options
            for _, left in extend_path(cal, left0, a, used | set(right), p_exc, w_edge, w_node)[:1]:  # best left arm
                qs = list(left) + list(right)  # all qubits
                ops = split_ops(list(left), list(right), k) + (_x_all(qs) if flip else [])  # exact construction
                yield compile_layers(ops, dd=dd), qs, 'split'  # candidate
        if time.monotonic() > deadline:  # out of time
            return  # stop


def create_dicke_state(
    num_qubits: int,
    num_excitations: int,
    cz_pauli_error_rate: dict[
        tuple[cirq.GridQubit, cirq.GridQubit], tuple[float, float]
    ],
    sq_pauli_error_rate: dict[cirq.GridQubit, tuple[float, float]],
    zero_errors: dict[cirq.GridQubit, tuple[float, float]],
    one_errors: dict[cirq.GridQubit, tuple[float, float]],
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """Create a Cirq circuit that produces a Dicke state.

    Args:
        num_qubits: The number, N, of qubits in the Dicke state.
        num_excitations: The number, n, of 1s in the state.
        cz_pauli_error_rate: Measured CZ Pauli error rates and their statistical uncertainties.
        sq_pauli_error_rate: Measured single-qubit Pauli error rates and their statistical uncertainties.
        zero_errors: Measured parallel 0 -> 1 readout error rates and their statistical uncertainties.
        one_errors: Measured 1 -> 0 readout error rates and their statistical uncertainties.

    Returns:
        A Cirq circuit to produce Dicke state and a set indicating which qubits the resulting state lives on.

    Raises:
        NotImplementedError: The requested (N, n) combination is not implemented.
    """
    start = time.monotonic()  # the challenge counts > 60 s as NotImplementedError
    N, n = int(num_qubits), int(num_excitations)  # accept numpy integers too
    if N < 1 or n < 0 or n > N:  # no such Dicke state
        raise NotImplementedError(f'invalid (N, n) = ({N}, {n})')  # explicit refusal
    cal = Calibration(cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors)  # pessimistic model
    if N > len(cal.sq):  # not enough working qubits
        raise NotImplementedError('not enough usable qubits')  # cannot embed
    best = None  # (predicted log-fidelity, circuit, qubits)
    for circ, qs, _ in generate_candidates(N, n, cal, deadline=start + 30.0):  # exact candidates
        if len(circ) > 1750:  # would not be run on the hardware
            continue  # discard
        score = estimate_log_fidelity(circ, cal, qs, n / N)  # predicted log success probability
        if best is None or score > best[0]:  # keep the best
            best = (score, circ, qs)  # new incumbent
    if best is None:  # nothing embeddable / everything too deep
        raise NotImplementedError('no embedding within the 1750-moment limit')  # explicit refusal
    return best[1], set(best[2])  # circuit and the qubits that hold the state


# %% [markdown]
# ## 6. Bonus: shallowest circuits (adaptive circuits allowed, no noise)
#
# The bonus scores depth only (`len(circuit)`), on the ideal Willow105 connectivity. We considered the adaptive designs
# in the plan (measurement-based fan-out and the "binary index → one-hot → $X$-basis measurement → $Z$ correction" W-state
# scheme). For $N\le20$ they lose to our unitary constructions: a measurement moment plus a classically controlled
# correction moment already cost 2 moments, the constant-depth fan-out gadget needs several more, and the binary→one-hot
# decoder needs multi-controlled gates routed across the grid, whereas the excitation tree reaches $N$ qubits in
# $\approx4\lceil\log_2N\rceil+1$ moments. Moreover mid-circuit measurements must cover whole readout diagonals
# ($\text{row}+\text{col}=$ const), which forces ancillas onto separate diagonals and lengthens routing. We therefore submit
# **non-adaptive** circuits (explicitly allowed) and spend the effort on depth:
#
# * **W states:** a *broadcast tree*. In round $t$ every informed qubit hands the excitation to one uninformed neighbour, so the
#   informed set can double every two CZ layers; we build the tree by greedy bipartite matching round by round from every
#   possible root and keep the shallowest compiled circuit.
# * **$k\ge2$:** the two-arm split construction (both arms in parallel) and the BE path, compiled on the ideal device, keeping
#   the shallower.
# * **Packing:** for the bonus, single-qubit gates may share moments with CZs on other qubits (no hardware calibration constraint
#   applies), so we re-insert every operation at the earliest possible moment and keep whichever version is shorter.

# %%
def _ideal_calibration():
    """Uniform 'calibration' covering every Willow105 qubit and coupler (the bonus is noise-free)."""
    device = cirq_google.Willow105  # the target device
    qubits = sorted(device.metadata.qubit_set)  # all 105 qubits
    cz = {tuple(sorted(p)): (1e-3, 0.0) for p in device.metadata.qubit_pairs}  # every coupler, equal cost
    flat = {q: (1e-3, 0.0) for q in qubits}  # equal single-qubit and readout costs
    return Calibration(cz, flat, flat, flat, beta=0.0)  # deterministic, uniform model


def broadcast_tree(cal, root, N):
    """Greedy round-by-round matching tree: every informed node recruits at most one new neighbour per round."""
    informed = [root]  # informed nodes in order of recruitment
    parent = {}  # child -> parent
    center = (np.mean([q.row for q in cal.sq]), np.mean([q.col for q in cal.sq]))  # prefer central growth
    while len(informed) < N:  # until N qubits hold the excitation
        taken = set()  # qubits recruited this round
        new = []  # (child, parent) pairs of this round
        for u in list(informed):  # each informed node may recruit once
            options = [v for v in cal.adj[u] if v not in parent and v != root and v not in taken]  # free neighbours
            if not options or len(informed) + len(new) >= N:  # nothing to do / enough nodes
                continue  # next node
            v = min(options, key=lambda q: (len([w for w in cal.adj[q] if w not in parent and w != root]) == 0,
                                            (q.row - center[0]) ** 2 + (q.col - center[1]) ** 2, str(q)))  # central, non-dead-end
            taken.add(v)  # reserve it
            new.append((v, u))  # recruit
        if not new:  # stuck (cannot happen on a connected device with N <= 105)
            return None  # give up on this root
        for v, u in new:  # commit the round
            parent[v] = u  # tree edge
            informed.append(v)  # v can recruit from the next round on
    return parent  # {child: parent}


def pack_earliest(circuit):
    """Re-insert every operation at the earliest moment allowed by its qubits (mixed 1q/CZ moments)."""
    return cirq.Circuit(circuit.all_operations(), strategy=cirq.InsertStrategy.EARLIEST)  # cirq does the packing


def create_dicke_state_adaptive(
    num_qubits: int, num_excitations: int
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """Create a Cirq circuit that produces a Dicke state, allowing for adaptive circuits.

    Args:
        num_qubits: The number, N, of qubits in the Dicke state.
        num_excitations: The number, n, of 1s in the state.

    Returns:
        A Cirq circuit to produce Dicke state and a set indicating which qubits the resulting state lives on.

    Raises:
        NotImplementedError: The requested (N, n) combination is not implemented.
    """
    N, n = int(num_qubits), int(num_excitations)  # accept numpy integers
    if N < 1 or n < 0 or n > N or N > 105:  # impossible request
        raise NotImplementedError(f'invalid (N, n) = ({N}, {n})')  # explicit refusal
    cal = _ideal_calibration()  # all qubits, uniform costs
    k = min(n, N - n)  # complement trick, as in the hardware solution
    flip = n > N - n  # whether to flip everything at the end
    cands = []  # (circuit, qubits)
    if k == 1:  # W states: broadcast trees from every root
        for r in sorted(cal.sq, key=str):  # every possible root
            parent = broadcast_tree(cal, r, N)  # shallow tree
            if parent is None:  # failed
                continue  # skip
            qs = [r, *parent]  # its qubits
            ops = w_tree_ops(r, _tree_children(parent)) + (_x_all(qs) if flip else [])  # exact construction
            cands.append((compile_layers(ops, dd=False), qs))  # no DD in a noise-free setting
    else:  # k = 0 or k >= 2: reuse the hardware candidate generator on the ideal device
        for circ, qs, _ in generate_candidates(N, n, cal, dd=False, max_trees=5, max_paths=2, max_ladders=3):  # few suffice
            cands.append((circ, qs))  # keep
    if not cands:  # nothing found
        raise NotImplementedError('no embedding found')  # explicit refusal
    best = None  # (depth, circuit, qubits)
    for circ, qs in cands:  # pick the shallowest, trying both layouts
        for c in (circ, pack_earliest(circ)):  # layered and packed versions
            if best is None or len(c) < best[0]:  # strictly shallower
                best = (len(c), c, qs)  # new incumbent
    return best[1], set(best[2])  # circuit and data qubits
