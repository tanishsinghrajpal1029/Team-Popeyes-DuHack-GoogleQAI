"""The judges' scoring functions (verbatim), plus fast equivalents and score statistics.

Fast path:
  * The Z term depends only on the counts of the K = C(N, n) valid strings.
  * The X term depends only on the Hamming-weight histogram of the X-basis shots.
    Averaging (-1)^{|x & S|} over all k-subsets S gives K_k(w(x)) / C(N, k), where
    K_k(w) = sum_j (-1)^j C(w, j) C(N - w, k - j) is a Krawtchouk polynomial.
That makes it cheap to draw thousands of simulated 10k-shot runs from an exact
distribution, which is what the heavy-tailed score needs (see CLAUDE.md).
"""

import functools
import itertools
import math

import numpy as np


# ---- verbatim from the notebook (cell 7) ----------------------------------------------


def check_z_bitstring_distribution(
    bitstrings: np.ndarray, num_excitations: int
) -> float:
    """Measure how close the observed Z-basis distribution of bitstrings is to
    the ideal one. The lower the score, the better.

    Args:
        bitstrings: An array of shape (n_shots, n_qubits) containing the n_shots Z-basis bitstrings.
        num_excitations: The weight, n, of the Dicke state.

    Returns:
        A measure of how close the distribution is to the ideal one.
    """
    # first, let's see how many bitstrings have one excitation
    n_shots, n_qubits = bitstrings.shape
    measured_excitations = np.sum(bitstrings, axis=1)
    fraction_in_subspace = np.mean(measured_excitations == num_excitations)

    # now, let's check how uniform the distribution is over states with n excitations
    chi2 = 0
    ideal_prob = 1 / math.comb(n_qubits, num_excitations)
    for indices in itertools.combinations(range(n_qubits), num_excitations):
        ideal_state = np.zeros(n_qubits)
        ideal_state[np.array(indices)] = 1
        match_count = np.sum(np.all(bitstrings == ideal_state, axis=1))
        expected_count = n_shots * fraction_in_subspace * ideal_prob
        chi2 += (match_count - expected_count) ** 2 / expected_count

    return ideal_prob * chi2 / fraction_in_subspace


def get_correlator_average(bitstrings: np.ndarray, correlator_size: int) -> float:
    """Get the average of all weight-correlator_size correlators in `bitstrings`.

    Args:
        bitstrings: The input bitstrings, shape (n_shots, n_qubits).
        correlator_size: Compute all correlators of this weight.

    Returns:
        The average of all weight-k correlators.
    """
    correlator_sum = 0.0
    n_shots, n_qubits = bitstrings.shape
    for indices in itertools.combinations(range(n_qubits), correlator_size):
        correlator_sum += 1 - 2 * np.mean(np.sum(bitstrings[:, indices], axis=1) % 2)
    return correlator_sum / math.comb(n_qubits, correlator_size)


def check_x_bitstring_distribution(
    bitstrings: np.ndarray, num_excitations: int, max_correlator_size: int
) -> float:
    """Check expected correlations in the X basis. The lower the score, the better.

    Args:
        bitstrings: An array of shape (n_shots, n_qubits) containing the n_shots X-basis bitstrings.
        max_correlator_size: The maximum weight of correlators to consider.
        num_excitations: The weight, n, of the Dicke state.

    Returns:
        A measure of how the correlations in the X-basis measurements compare to expectation.
    """
    comb = lambda n, k: math.comb(n, k) if (k >= 0 and n >= k) else 0
    n_shots, n_qubits = bitstrings.shape
    score = 0.0
    for correlator_size in range(1, max_correlator_size + 1):
        expected_correlator = (
            comb(correlator_size, correlator_size // 2)
            * comb(n_qubits - correlator_size, num_excitations - correlator_size // 2)
            / comb(n_qubits, num_excitations)
            if correlator_size % 2 == 0
            else 0.0
        )
        measured_correlator = get_correlator_average(bitstrings, correlator_size)
        score += (
            (expected_correlator - measured_correlator) ** 2 / expected_correlator**2
            if expected_correlator != 0
            else (expected_correlator - measured_correlator) ** 2
        )

    return score


def overall_score(
    z_bitstrings: np.ndarray,
    x_bitstrings: np.ndarray,
    num_excitations: int,
    max_correlator_size: int,
) -> float:
    """An overall score that checks how the Z and X basis bitstrings compare to their ideal
    distributions. The higher the score, the better.

    Args:
        z_bitstrings: The bitstrings measured in the Z basis
        x_bitstrings: The bitstrings measured in the X basis
        num_excitations: The weight, n, of the Dicke state.
        max_correlator_size: The maximum weight of correlators to consider

    Returns:
        A measure of how similar the bitstrings are to their ideal distributions.
    """
    return 1 / (
        check_z_bitstring_distribution(z_bitstrings, num_excitations)
        + check_x_bitstring_distribution(
            x_bitstrings, num_excitations, max_correlator_size
        )
    )


# ---- fast equivalents ------------------------------------------------------------------


def _comb(a: int, b: int) -> int:
    return math.comb(a, b) if 0 <= b <= a else 0


def expected_correlators(num_qubits: int, num_excitations: int, max_k: int) -> np.ndarray:
    """Ideal average weight-k X correlators, k = 1..max_k (same formula as the judges)."""
    N, n = num_qubits, num_excitations
    return np.array(
        [
            _comb(k, k // 2) * _comb(N - k, n - k // 2) / _comb(N, n) if k % 2 == 0 else 0.0
            for k in range(1, max_k + 1)
        ]
    )


def krawtchouk_matrix(num_qubits: int, max_k: int) -> np.ndarray:
    """A[w, k-1] = average over k-subsets S of (-1)^{|x & S|} for any x of weight w."""
    N = num_qubits
    A = np.zeros((N + 1, max_k))
    for w in range(N + 1):
        for k in range(1, max_k + 1):
            s = sum((-1) ** j * _comb(w, j) * _comb(N - w, k - j) for j in range(k + 1))
            A[w, k - 1] = s / _comb(N, k)
    return A


def x_term_from_weights(
    weight_probs: np.ndarray, num_excitations: int, max_k: int
) -> np.ndarray:
    """X term from (one or many) X-basis Hamming-weight distributions.

    weight_probs has shape (..., N+1) and holds empirical or exact frequencies.
    """
    N = weight_probs.shape[-1] - 1
    measured = weight_probs @ krawtchouk_matrix(N, max_k)
    ideal = expected_correlators(N, num_excitations, max_k)
    denom = np.where(ideal != 0, ideal**2, 1.0)
    return (((ideal - measured) ** 2) / denom).sum(axis=-1)


def z_term_from_counts(valid_counts: np.ndarray, n_shots: int) -> np.ndarray:
    """Z term from counts of the K valid strings, shape (..., K). Inf if no valid shots."""
    K = valid_counts.shape[-1]
    in_subspace = valid_counts.sum(axis=-1)
    f = in_subspace / n_shots
    expected = in_subspace / K
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = (((valid_counts - expected[..., None]) ** 2) / expected[..., None]).sum(axis=-1)
        z = chi2 / (K * f)
    return np.where(in_subspace > 0, z, np.inf)


def popcounts(num_qubits: int) -> np.ndarray:
    """Hamming weight of every index 0 .. 2^N - 1."""
    return np.bitwise_count(np.arange(2**num_qubits, dtype=np.uint64)).astype(np.int64)


def fast_overall_score(
    z_bitstrings: np.ndarray, x_bitstrings: np.ndarray, num_excitations: int, max_k: int
) -> float:
    """Same value as `overall_score`, without the C(N, k) loops."""
    n_shots, N = z_bitstrings.shape
    ints = z_bitstrings.astype(np.int64) @ (1 << np.arange(N, dtype=np.int64))
    valid = np.flatnonzero(popcounts(N) == num_excitations)
    counts = np.bincount(ints, minlength=2**N)[valid]
    weights = np.bincount(x_bitstrings.sum(axis=1).astype(int), minlength=N + 1) / n_shots
    z = z_term_from_counts(counts, n_shots)
    x = x_term_from_weights(weights, num_excitations, max_k)
    return float(1 / (z + x))


# ---- expected score (the actual objective) -----------------------------------------------------


def expected_score(K: int, f: float, lam: float, x: float) -> float:
    """E[1 / (z + x)] for one judged draw, integrating the chi^2 shot noise exactly.

    z = chi^2 / (K f), with chi^2 ~ noncentral chi^2(K - 1, lam) (lam = population
    imbalance, 0 for a perfect state). x is held at its mean; Jensen makes that a slight
    underestimate. This is what the judges' total is the sum of, in expectation. It is
    heavy-tailed for K <= 3, which is why resampled means are too noisy to use there.
    Substituting c = t^2 removes the c^(-1/2) singularity of the chi^2_1 density.
    """
    from scipy import integrate, stats

    if K <= 1:
        return 1.0 / x if x > 0 else math.inf
    dist = stats.ncx2(K - 1, lam) if lam > 1e-12 else stats.chi2(K - 1)
    a = K * f
    t_max = math.sqrt(dist.isf(1e-14))
    peak = math.sqrt(a * x)  # width of the 1/(c/a + x) spike near c = 0, in t
    val, _ = integrate.quad(lambda t: dist.pdf(t * t) * 2 * t / (t * t / a + x), 0, t_max,
                            points=[min(peak, t_max / 2)], limit=500)
    return val


# ---- the perfect-state ceiling ---------------------------------------------------------------


def ideal_distributions(num_qubits: int, num_excitations: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact Z- and X-basis distributions of a perfect Dicke state."""
    N = num_qubits
    psi = np.zeros(2**N)
    psi[popcounts(N) == num_excitations] = 1.0
    psi /= np.linalg.norm(psi)
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    t = psi.reshape((2,) * N)
    for j in range(N):
        t = np.moveaxis(np.tensordot(h, t, axes=([1], [j])), 0, j)
    return psi**2, t.reshape(-1) ** 2


@functools.lru_cache(maxsize=None)
def ceiling(num_qubits: int, num_excitations: int, max_k: int) -> dict:
    """Score statistics of a perfect state: what shot noise alone allows."""
    pz, px = ideal_distributions(num_qubits, num_excitations)
    return score_distribution(pz, px, num_qubits, num_excitations, max_k, repeats=20000, seed=12345)


# ---- score statistics from an exact output distribution ---------------------------------


def distribution_metrics(
    pz: np.ndarray, px: np.ndarray, num_qubits: int, num_excitations: int,
    max_k: int, n_shots: int = 10_000,
) -> dict:
    """Shot-noise-free diagnostics of exact Z/X output distributions over 2^N strings.

    f: probability of the right Hamming weight; delta_rms: rms relative deviation of the
    valid strings from their mean; lam: chi^2 noncentrality n_shots * f * mean(delta^2);
    x_inf: X term with infinitely many shots; corr_ratio: measured/ideal per even k.
    """
    N, n = num_qubits, num_excitations
    pc = popcounts(N)
    pv = pz[pc == n]
    f = float(pv.sum())
    delta = pv / pv.mean() - 1 if f > 0 else np.full_like(pv, np.nan)
    wx = np.bincount(pc, weights=px, minlength=N + 1)
    measured = wx @ krawtchouk_matrix(N, max_k)
    ideal = expected_correlators(N, n, max_k)
    nz = ideal != 0
    return dict(
        f=f,
        delta_rms=float(np.sqrt(np.mean(delta**2))),
        delta_max=float(np.max(np.abs(delta))),
        lam=float(n_shots * f * np.mean(delta**2)),
        x_inf=float(x_term_from_weights(wx, n, max_k)),
        corr_ratio={int(k): float(r) for k, r in zip(np.arange(1, max_k + 1)[nz], measured[nz] / ideal[nz])},
    )


def score_distribution(
    pz: np.ndarray, px: np.ndarray, num_qubits: int, num_excitations: int,
    max_k: int, repeats: int = 2000, n_shots: int = 10_000, seed: int = 0,
) -> dict:
    """Statistics of the judges' score over `repeats` simulated 10k-shot runs."""
    N, n = num_qubits, num_excitations
    rng = np.random.default_rng(seed)
    pc = popcounts(N)
    pv = pz[pc == n]
    cats = np.append(pv, max(0.0, 1.0 - pv.sum()))
    counts = rng.multinomial(n_shots, cats / cats.sum(), size=repeats)[:, :-1]
    z = z_term_from_counts(counts, n_shots)
    wx = np.bincount(pc, weights=px, minlength=N + 1)
    wcounts = rng.multinomial(n_shots, wx / wx.sum(), size=repeats) / n_shots
    x = x_term_from_weights(wcounts, n, max_k)
    s = 1 / (z + x)
    K = counts.shape[1]
    f = counts.sum(axis=1) / n_shots
    chi2 = z * K * f
    return dict(
        score_mean=float(s.mean()),  # unstable for K <= 3 (E[1/chi2_1] diverges): prefer median
        score_median=float(np.median(s)),
        score_p10=float(np.percentile(s, 10)),
        score_p90=float(np.percentile(s, 90)),
        loss_mean=float(np.mean(z + x)),  # well-behaved summary: mean of z + x
        z_mean=float(np.mean(z[np.isfinite(z)])) if np.isfinite(z).any() else np.inf,
        x_mean=float(x.mean()),
        chi2_ratio=float(np.mean(chi2[np.isfinite(chi2)]) / max(K - 1, 1)),  # 1 = shot noise only
    )
