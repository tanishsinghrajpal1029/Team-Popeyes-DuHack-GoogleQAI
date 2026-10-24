# AI prompt log

Every prompt given to an AI assistant during the hackathon, verbatim, as the sponsor
requires. The tool is Claude Code (model: Claude Opus 5.5). Answers to the assistant's
clarifying questions are logged too, since they steer the work.

## Session 1 — Sat Oct 24, 2026, ~12:20 PM

### Prompt 1 (plan mode)

> /plan understand the status, goals, and implementation plan for the challenge. revise it
> as needed and decide what to focus on in the immediate next steps.

Clarifying questions from the assistant, and Richard's answers:

- *How should the code be split between us?* → "AI implements all of it" (Richard makes
  the call at the checkpoints).
- *Any info on which (N, n) the judges run?* → "No info: cover everything".
- *Is anyone else working in parallel?* → "other teammates, but would be most efficient to
  develop sequentially. we can branch and merge latere. i've also uploaded a brief that a
  teammate made into this directory which may inform some of the constraints"

The resulting plan was approved by Richard. Its key outputs: the score-tail analysis (now in
CLAUDE.md, "How it is scored"), the revised priority order, and Steps 0–3.

### Checkpoint 1 (after baseline numbers)

The assistant presented the baseline table (W states near their ceiling except N=2;
general n far from it, with X term ≈ Z term at D⁸₄) and asked what should come next.
Richard's answer: "W tree, then general n (Recommended)".

### Prompt 2 (mid-task)

> make and switch to branch richard- dont push or commit anything onto main

(The assistant moved its two earlier local commits onto `richard` and reset local `main` to
`origin/main`. Nothing was pushed.)

### Prompt 3 (mid-task)

> what is the current overall score as the function of x and z? why is hardcoding necessary-
> this is dynamically applicable? did we build our own noise simulator?

### Prompt 4 (mid-task, pasted text, no other words)

A pasted analysis (not written by the assistant) proposing ceiling-normalized reporting
(median vs. K / median(χ²_{K−1})), using the median or mean loss z + x instead of the mean
score, explicit f and χ²/(K−1) columns, and a chain-vs-tree test with time-dependent noise.
The assistant adopted the reporting changes, ran the test, and corrected three factual
points: the tree uses 2 CZ per split (2N−3 total), our idle preset already models T1/Tφ,
and the tree was chosen for depth, not T1 balance. Full text of the paste:

> These aren’t six competing models, they’re two architectures across six test cases, and the raw scores aren’t comparable across cases because each (N, n) has its own ceiling. The useful comparison is each case against the best score a perfect state could get, which is set by shot noise: with K = C(N,n) valid strings, chi² for a perfect state is a χ² random variable with K−1 degrees of freedom, so the median achievable score is roughly K / median(χ²_{K−1}).
>
> case    K    ceiling (median)    yours (median)    fraction of ceiling
> W N=2 tree    2    4.4    4.37    ~99%
> W N=4 tree    4    1.69    1.60    ~95%
> W N=8 tree    8    1.26    1.15    ~91%
> D⁴₂ B-E    6    1.38    1.16    ~84%
> D⁶₃ B-E    20    1.09    0.53    ~49%
> D⁸₄ B-E    70    1.02    0.24    ~24%
>
> Two things fall out. First, the tree W-state circuit is essentially perfect in this noise model: its z values (0.50, 0.79, 0.96) are exactly (1 − 1/K)/f with f ≈ 1.0, 0.95, 0.91, meaning there’s no detectable population imbalance and the only loss is leakage from gate errors. Second, the B-E circuits fall off a cliff as n grows, and the x column tells you why: at D⁶₃ and D⁸₄ the coherence is mostly gone (x of 0.45 and 1.77 against a ceiling near 0), and f has dropped to around 0.6 and 0.4. That’s the O(nN) gate count biting, plus the three-qubit blocks for n ≥ 2 being expensive to decompose into CZs. It’s a leakage-and-coherence problem, not an angle or symmetry problem, which points at CZ count, layout search, and DD as the levers rather than the construction’s logic.
>
> Ignore the mean column. For small K the score is 1/(small noisy number), and for K=2 the expected value of 1/χ²₁ is literally infinite, so 62.7 is an artifact of a few lucky seeds. Use the median, or better, average the loss z + x across seeds, which is well-behaved. Worth remembering, though, that the judges take a single 10k-shot draw, so for small cases your realized score is a lottery around that median regardless of what you do.
>
> The tree architecture
>
> The W state is one excitation shared equally across N qubits, and the tree builds it by repeated amplitude splitting outward from a root.
>
> Pick a spanning tree of your N chosen qubits on the grid, ideally balanced so every leaf is a similar number of hops from the root. Put the root in |1⟩, everything else in |0⟩. Now process the tree top-down. Each node holds a pending “weight” equal to the number of qubits in its subtree, which is how many qubits will eventually share the amplitude it currently holds. When a node with subtree size s hands off to a child with subtree size c, it applies the split primitive: a controlled-RY on the child with sin²(θ/2) = c/s, followed by a CNOT from child back to parent. That moves exactly the fraction c/s of the parent’s probability onto the child, and the parent’s remaining weight drops to s − c. Repeat for each child; after the last one the parent is left with exactly 1/N of the total. Every node does this to its children, so when the leaves are reached each qubit holds 1/N, all with positive sign because RY rotations never introduce a minus.
>
> Cost: N − 1 splits total, each 3 CZs (two for the CRY, one for the CNOT), so 3(N−1) CZs. Splits in different branches touch disjoint qubits and run in the same moments, so depth is about 3 × (number of children along the longest root-to-leaf path), roughly 3·log₂N for a balanced tree versus 3(N−1) for a chain. That’s why N=8 still holds 91% of ceiling.
>
> The thing to be careful about when presenting it: this simulator applies a fixed error probability per gate and knows nothing about time, so it cannot show the T1-timing imbalance that made us prefer a balanced tree over a chain in the first place. The tree’s z-at-floor result is evidence the angles and gate sequence are exact, not evidence that the timing argument holds on hardware. If you want that evidence, add a crude time-dependent decay to the noise model and rerun chain versus tree for N=8; the chain should start showing systematic chi² excess.
>
> One more thing for the results table: add columns for f and for chi²/(K−1) separately. Right now you’re reading leakage versus imbalance off z by inference; having them explicit will make the slides much easier to defend.

### Prompt 5 (mid-task)

> what noise model are we using

### Prompt 6 (mid-task)

> explain the current approach in the meantime

### Prompt 7 (mid-task)

> what are the current running agents doing

### Prompt 8 (mid-task)

> do u need gpu? i can boot up a lambda ai cluster

(Answer: no; the workloads are tiny-statevector, latency-bound optimizations. More CPU
cores would help only for extra synthesis seeds.)

### Checkpoint: dynamical decoupling default

The assistant reported DD as neutral to slightly negative in simulation (short circuits;
the simulator lacks ZZ crosstalk) and asked for the default. Richard's answer:
"DD only on long idles (Recommended)".

### Prompt 9

> why are the 3 shells taking so long

### Prompt 10

> status?

### Prompt 11

> kill the ladder processes. eta on the 2nd job?

### Prompt 12

> is that most productive? check out the richard-vqe branch and see if its worth merging?

### Prompt 13

> that session is finished, you are good to commit it

(The assistant committed session 2's uncommitted work on `richard-vqe` in its worktree and
merged it into `richard`.)

## Session 2 — Sat Oct 24, 2026, ~2:10 PM (branch `richard-vqe`)

A second, parallel Claude Code session. Session 1 was still running jobs in the main
checkout on `richard`, so this session works in a separate git worktree
(`../Team-Popeyes-vqe`) on a new branch `richard-vqe` created from `richard`.

### Prompt 1

> see the current branch, and make a new richard-vqe branch. see if we can use VQE to be
> useful in this case? something about reducing circuit depth and increasing accuracy

### Prompt 2 (mid-task)

> do u want gpu? i can connect a lambda cloud cluster

(The assistant declined: every simulation here is at most 2^8 amplitudes, so the work is
CPU- and optimizer-bound, not FLOP-bound.)

### Prompt 14

> what is the new performance after the VQE-adjacent merge

### Prompt 15

> current eval score? 1/(x+z)? what dataset did we use for the testing? how does this
> compare to the actual testing on hardware? what are the still running jobs?

### Prompt 16 (relaying an organizer correction)

> did you correct: I wanted to correct two typos:
> In the bonus problem, the grading will be for 2 ≤ N ≤ 15, so score_solution should say
> `for num_qubits in range(2, 16)`.
> In `test_circuit`, `keep_indices = range(qubits)` should say
> `keep_indices = range(len(qubits))`

### Prompt 17

> the expected value of the remaining 2 shells?

### Prompt 18

> sure. can we commit and push to the branch now and add a md that explains this
> architecture and the performance metrics? are we using the proper performance metric?
> then lets tackle those

(The assistant pushed `richard`, added ARCHITECTURE.md, and replaced the resampled mean with
a semi-analytic expected score E[1/(z+x)] as the reported objective: median and % of
ceiling are diagnostics; the judges' total is a sum of single draws.)

(Following the metric change, the assistant added a build-time selector that picks the W2
qubit pair by predicted E[score]: +9% expected score, -5% median; kept for N = 2 only.)
