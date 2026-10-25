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

### Prompt 19 (mid-task, with a screenshot)

> [Image #2] was generated by another model. Does this model perform better or worse?

(The screenshot described another model's design, with tree W states, a "split"
Bärtschi–Eidenbenz, a 4-CZ block, ALAP, and uncertainty-aware placement, and gave scores for
(6,1) 1.12, (6,3) 0.86, (8,2) 0.88, (8,4) 0.70 against "best possible" K/(K−1). The assistant
compared on the same 1/E[z+x] metric: tie, ours +8%, theirs +7% (unverifiable), ours +11%.
Our CZ counts are 15 vs 23 and 27 vs 49 where theirs are given.)

(Bonus: the assistant added a non-adaptive `create_dicke_state_adaptive` that reuses the main
constructions on a uniform calibration, packed for depth: score 23.8 by the corrected
score_solution over 2 <= N <= 15.)

### Prompt 20

> how much improvement is captured by borrowing best of both worlds

(The assistant measured the borrowable pieces: the 7-CZ all-ones SCS block stays at 7 CZ
over 8 synthesis seeds; ALAP scheduling changes E[score] by -1.4..+0.1% under idle noise, so
it stays off; error+1σ placement is a no-op on uniform-uncertainty calibrations.)

### Prompt 21

> yes add the 1σ hedge and push

(Every placement decision now uses reported value + 1σ; 0 of 75 placements changed on the
fake calibrations, 256 tests pass.)

### Prompt 22

> what is the performance on absolute metrics currently

### Prompt 23

> performance compared to theoretical optimal so far?

### Prompt 24

> can you run and give results on these:
> D(5,2)
> D(6,3)
> D(8,4)
> D(10,5)

### Prompt 25 (pasted report from another session, plus a question)

> bryans branch's eval returned:
>
> Both background jobs are still running: N=9–10 plus the detuning-noise run, and the bonus
> scoring. I'll write up the full comparison when they finish. So far, on N ≤ 8 under
> richard's noise model the two solutions are nearly tied overall (Σ E[score] 100.6 for bryan
> vs 102.0 for richard). Richard wins clearly at k ≥ 3 for N ≤ 8, and bryan wins at k = 2 and
> for everything larger.
>
> is this true and how should we optimize if it is?

(The assistant re-ran both solutions through this repo's harness, in vs_bryan.csv: the claims
hold, with k = 2 going to bryan only for N ≥ 6. Bryan's bonus scores 35.3 vs our 23.8 under
the corrected range.)

Clarifying question from the assistant: how should the two solutions be combined? Richard's
answer: "Portfolio in main.py (Recommended)".

(The assistant vendored Bryan's dicke_solution.py into main.py with four renames, dispatched
k = 2 with N >= 6 and k >= 3 with N >= 9 to it, and made the bonus return the shallower circuit.
Portfolio: sum E[score] 102.0, sum of medians 27.9, bonus 37.2; hardening shows no over-moment
cases left.)

### Prompt 26

> merge to main

(The assistant merged `richard` into `main`. The only conflicts were add/add in requirements.txt
and .gitignore against teammates' origin/main commits; both pin the same cirq dev build, so they
were resolved as a union, keeping teammates' ipykernel. Tests passed, `main` was pushed, and
`richard` was fast-forwarded to it.)

### Prompt 27

> rerun the evals on the bryan + richard model for
> D(5,2)
> D(6,3)
> D(8,4)
> D(10,5)
> and give the Case, Our score, Perfect circuit (median), Ours as % of perfect

### Prompt 28 (plan mode)

> /plan our score seems to drop off heavily on big N. we should think of ways to address
> this issue while not sacrificing our current progress on small N. plan.

Clarifying questions from the assistant, and Richard's answers:

- *How far should "big N" go?* → "up to N = 15, we know the bonus is <= 15 bounded. so the
  regular may be bounded to like 10-12 ish? if targeting 15 is bad then we can reduce to
  around 12, then 10. depending on performance."
- *Where should re-synthesis run?* → "This machine, overnight (Recommended)"

### Prompt 29

> switching to an a100 with 30 vcpu to run on vm. make a handoff.md and include the future
> plan and next steps

(The assistant wrote HANDOFF.md: VM setup, current state, the big-N diagnosis, plan P0–P6,
commands, and gotchas.)

### Prompt 30

> can you run bechmarks comparing the performance of @tanish_submission.py and @main.py

(The assistant added `experiments/compare.py`, which scores any number of submission files
through `evaluate.evaluate_case` in parallel, and `--sub`/`--out` options on
`experiments/harden.py`. Results: `vs_tanish.csv`, `harden_tanish.csv`.)
### Prompt 31 (on the 30-vCPU VM)

> Look through @HANDOFF.md and complete the steps

(The assistant set up the VM venv, added checkpointing and per-job progress logs to
`experiments/synth.py` so long runs keep their best exact circuit if cut short, launched the
P2 synthesis runs for N = 9–10 and D8,3, and added the P0 golden test
`tests/test_golden_small_n.py` with `tests/golden_small_n.json` generated from `e028b05`.)

### Prompt 32 (mid-task, on the VM)

> new changes have come in from upstream, mainly that a new file tanish_submission.py shows
> slightly improved performance for higher n values. Pull upstream and validate if these
> results are correct, and if they are, consider them in plans.

(The assistant rebased onto `origin/richard`, reproduced `vs_tanish.csv`, re-ran the head-to-head
on fresh calibration seeds 3–5 (`vs_tanish_seeds345.csv`), added `experiments/mixed_moments.py`
to test how much of the gain comes from mixed CZ + 1q moments, and recorded the findings and
plan changes in HANDOFF.md §2b.)

### Prompt 33 (on the VM)

> combine the current best solution into an ipynb format for submission and commit it

(The assistant added `experiments/make_notebook.py`, which fills the challenge notebook's two
stub cells from `main.py` (the adaptive function into the bonus cell, everything else into the
standard-track cell) and appends a check cell running the notebook's `test_circuit` and
`Willow105.validate_circuit` on a few cases. Output: `submission.ipynb`; `--run` executes it
top to bottom.)

### Prompt(s) for commit `ce18281` ("optimization grind")

TODO (Richard): add the prompts from that session, if it used an AI assistant.

## Session 3 — Sun Oct 25, 2026, ~12:00 AM (on the VM)

### Prompt 34

> run a benchmarking comparison between the preliminary submission at @PreliminarySubmission.ipynb
> and most recent version @submission.ipynb

(The assistant extracted the code cells of both notebooks to scratch `.py` files, found the
preliminary one byte-identical to `tanish_submission.py`, ran `experiments/compare.py` on
N ≤ 10, seeds 0–2, idle noise (`prelim_vs_latest.csv`) and the notebook's `score_solution` for
the bonus. Result: Σ median 27.99 vs 27.53, Σ E 102.9 vs 90.8, bonus 37.15 vs 33.67.)

### Prompt 35

> how long until the main track comparison finishes?

### Prompt 36 (plan mode; model switched to Claude Fable 5.1)

> I want to do a merge and cleanup of the current codebase, mainly the richard branch and main
> branch. Create a plan on what needs to be kept, what can be removed, and what should be
> refactored.

Clarifying questions from the assistant, and Richard's answers:

- *What to do with `tanish_submission.py` and `PreliminarySubmission.ipynb`?* → "Delete both".
- *How aggressively to prune experiments/results?* → "Moderate (Recommended)".
- *HANDOFF.md?* → "Fold and delete (Recommended)".
- *The running synth.py job?* → "Stop it before cleanup (Recommended)".

Then: "Save the plan for now, I will use it in a bit".

### Prompt 37 (model: Claude Opus 5.5)

> follow the plan to merge branch richard into main and cleanup the codebase.

(The assistant stopped the synthesis job and committed its checkpoints. It then deleted the
two baseline files, HANDOFF.md, 7 unreferenced CSVs and the unused `_alap` option. It moved
the ~200 lines of notebook prose in the vendored Bryan code to `docs/bryan_constructions.md`
instead of deleting them, because they contain the derivations. HANDOFF's lasting content was
folded into ARCHITECTURE.md §6–7, and stale claims in README/ARCHITECTURE/CLAUDE.md were fixed.
`submission.ipynb` was regenerated, and it fast-forwarded `main` to `richard`.)

### Prompt 38

> export the 31-CZ D8,3 circuit and re-score it

(The assistant added `--nmax` to `synth.py --export` so the N = 9–10 results stay out of the
table, exported D8,3 (43 → 31 CZ), and re-scored it against the old circuit on seeds 0–5
(`d83_export.csv`). It also fixed `vqe.py`'s candidate cache, which was keyed by (N, k) only
and would have reused the old structure's angles.)

### Prompt 39

> how much longer do you estimate the run will take?

(Answer: ~40 min, probably ending at the 480-restart cap with few exact alternatives. Options:
measure only the deployed angles, or parallelize the restarts.)

### Prompt 40

> do option 1

(The assistant stopped the run and measured the deployed angles' coherent-error sensitivity
(`vqe_coherent_8_3_cz31.csv`). It came out close to the old pick (λ 107 vs 86 uniform, 86 vs 112
per-qubit), so the export was kept. It then regenerated the golden file and the notebook.)

### Prompt 41 (plan mode; model: Claude Fable 5.1)

> /plan I want to make a latex writeup documenting the important mathematical choices behind our
> project submission. Create a plan on the most important ideas to write about and how to describe
> them.

Clarifying questions from the assistant, and Richard's answers:

- *How long and for whom?* → "Appendix-style, ~3 pages".
- *Include Bryan's building-block derivations?* → "Brief summary only".
- *How to build the PDF (no LaTeX on the machine)?* → "Write .tex only, I compile in Overleaf
  (Recommended)".

Then: "Save the plan - I'll use it in a bit".

### Prompt 42 (model: Claude Opus 5.5)

> Execute from the plan file

(The assistant wrote `docs/writeup/main.tex` and `refs.bib` and copied two existing figures. The
document covers the score's heavy tail and the imbalance noncentrality, the complement and
permutation reductions, the W tree, whole-state synthesis and the robust angle pick, B-E with
synthesized SCS blocks, the placement cost read as log E[z], compilation, a summary of Bryan's
constructions, a results table and negative results. It checked the closed forms against
`scoring.expected_score`, checked the over-rotation versus phase-error orders on the deployed D4,2
circuit, and traced every number in the document to ARCHITECTURE.md, a summarized CSV, or that
check. No local LaTeX build was run.)

### Prompt 43 (model: Claude Opus 5.5)

> can you regenerate and benchmark the new submission.ipynb with its scoring functions for both
> the base problem and the bonus question?

(The assistant regenerated and executed the notebook with `make_notebook.py --run`. The output was
byte-identical to the committed file, and every sanity case passed. It then wrote
`experiments/bench_notebook.py`, which runs the notebook's own cells and scores with the
notebook's own functions. Standard track: 25 cases (N ≤ 10) × calibration seeds 0–2, idle noise,
1,000 sampled 10k-shot Z/X runs per case through the verbatim `overall_score`. All 75 builds were
exact and valid, and the verbatim and fast scorers agreed to 1e-14 on identical shots.
Σ median 27.64 and Σ E[score] 101.8 agree with ARCHITECTURE.md (27.9 / 102.0) within W₂
sampling noise. The judged total (one run per case, summed) has median 36.4, p10–p90 26.3–142,
and mean 100.8. Bonus: the notebook's `score_solution`, with a `validate_circuit` written from
the bonus text (the notebook never defines one), gave 37.1542 over 56 cases. All were exact and
routed, with no measurements, and the slowest build took 7.7 s. Results are in
`notebook_standard.csv`, `notebook_standard_totals.csv` and `notebook_bonus.csv`.)

### Prompt 44 (model: Claude Opus 5.5)

> can you rewrite main.tex so the math is understandable from a freshman undergraduate level and so
> that it uses @alan.sty for styling?

Follow-up: "retry now".

(The assistant rewrote `docs/writeup/main.tex` as a KOMA `scrartcl` document that loads
`\usepackage[noasy]{alan}`. It adds a crash course on qubits, gates and noise, and uses alan's
definition, fact, proposition, claim, example and remark boxes. Each step is explained from
first principles: the counting proofs of the ideal correlators and of the Krawtchouk shortcut,
why E[1/χ²_ν] diverges for ν ≤ 2, a worked W₄ tree example, and why over-rotation moves
populations at first order while phase errors move them only at second order. All numbers are
unchanged from the previous version. The local TeX Live lacks `derivative`, `yhmath` and
`thmtools`, so the assistant built them from CTAN in a scratch directory to check that the
document compiles: 14 pages, with no errors or warnings.)

### Prompt 45 (model: Claude Opus 5.5)

> further simplify @main.tex - we plan on including snippets of it as part of our final
> submission, so it should make sense to a new undergraduate reader and be able to be explained by
> a new undergraduate. There doesn't need to be too much specific detail, but rather we want to
> highlight 3-4 main points

(The assistant restructured `docs/writeup/main.tex` around four ideas, each in its own
self-contained section: (1) the score rewards small, balanced cases; (2) the W-state broadcast
tree; (3) computer search for exact circuits at N ≤ 8; (4) choosing qubits. It kept a one-page
background and a short "putting it together" section with a 6-row results table and lessons
learned. It removed the Krawtchouk shortcut, the closed forms for K = 2 and 3, the 1-CZ
impossibility argument, the coherent-error and minimax-regret analysis, the synthesized B-E
blocks, the winner's-curse correction and the negative-results list, and reduced the detail on
Bryan's construction to one sentence. The numbers it kept are unchanged from the previous
version. Two wordings were changed for accuracy: 1% imbalance gives λ ≈ 0.9, and the tree needs
"far fewer rounds than a line" rather than log N rounds. It compiled to 6 pages, down from 14,
with no warnings.)

### Prompt 46 (plan mode, model: Claude Opus 5.5)

> /plan I want to create manim animations / visualizations based on the docs in
> @docs/writeup/main.tex. Plan it

Answers to the assistant's clarifying questions: used in the 7-min live talk and as
repo/README extras; the core 5 scenes; classic 3b1b dark style.

(The assistant planned and then built `animations/`: `prep.py` runs the deployed code
(`main._best_w_tree`, `main.build`, `main._best_path`, `main._predict_tiny`) and
`experiments/synth.py` on calibration seed 0, and writes `animations/data/*.json`, with asserts
pinning the writeup's numbers (E[S] 51.4 / 33.3 / 1.3 for K = 2, 8.15 for K = 3, D⁴₂
synthesized at 5 CZ, W₁₀ tree 15 vs chain 37 moments). `scenes.py` has five scenes:
`DickeIntro`, `ScoreTail`, `WTree`, `Synthesis`, `Placement`. Each is split into beats, rendered
by `render.sh` into `figures/anim/` as 1080p60 MP4s, per-beat clips and README GIFs. On
seed 0 the W₂ pair scan picks the same pair as the generic tree layout, so the Placement
scene shows the scan's spread (best 79.6, median 41.2, worst 20.1 E[S]) instead of the
writeup's seed-averaged +9%.)

