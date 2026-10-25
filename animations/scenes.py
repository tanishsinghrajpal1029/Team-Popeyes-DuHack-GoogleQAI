"""Manim scenes for docs/writeup/main.tex (the four ideas) and docs/main.tex (the derivations).

  manim -ql animations/scenes.py DickeIntro            # quick preview
  animations/render.sh                                 # every scene, talk + README formats

The four-idea scenes: DickeIntro, ScoreTail, WTree, Synthesis, Placement. The docs/main.tex
scenes: RevisedScore, TransferBound, CoherentErrors, MergePass, BonusDepth.

Every number on screen comes from animations/data/*.json (written by animations/prep.py)
or, where noted, from the writeup's tables. Each scene is split into sections (beats);
render with --save_sections to get one clip per slide click.
"""

import math
import os
import sys

import numpy as np
from manim import (
    DOWN, LEFT, RIGHT, UP, WHITE, Arrow, Axes, Brace, Circle, Create, Cross, DashedLine,
    FadeIn, FadeOut, Flash, GrowFromEdge, Indicate, LaggedStart, Line, MathTex, Polygon,
    Rectangle, ReplacementTransform, Square, Succession, Swap, Text, Transform,
    ValueTracker, VGroup, Write, always_redraw, AnimationGroup, Circumscribe, Dot,
    SurroundingRectangle, interpolate_color, rate_functions,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    BAD, CZC, DIM, EXC, GOOD, Beats, Chip, bitstring, caption, err_color, load, node, share_bar, title,
)


# ---- Background ------------------------------------------------------------------------


class DickeIntro(Beats):
    def construct(self):
        self.next_section("strings")
        head = title("A Dicke state: every string with exactly n ones")
        strings = [format(i, "04b") for i in range(16)]
        grid = VGroup(*[bitstring(s, 30) for s in strings]).arrange_in_grid(4, 4, buff=(0.9, 0.45))
        note = caption("Four qubits give 16 possible bit strings").next_to(grid, DOWN, buff=0.6)
        self.play(Write(head))
        self.play(LaggedStart(*[FadeIn(m) for m in grid], lag_ratio=0.04), FadeIn(note))
        valid = [i for i, s in enumerate(strings) if s.count("1") == 2]
        self.play(*[grid[i].animate.set_opacity(0.15) for i in range(16) if i not in valid],
                  Transform(note, caption("Keep only the ones with exactly two 1s").move_to(note)))
        self.wait()

        self.next_section("amplitudes")
        chosen = VGroup(*[grid[i] for i in valid])
        self.play(FadeOut(VGroup(*[grid[i] for i in range(16) if i not in valid])), FadeOut(note),
                  chosen.animate.arrange(RIGHT, buff=0.7).shift(DOWN * 0.6))
        bars = VGroup(*[share_bar(1 / math.sqrt(6), max_height=2.4).next_to(s, UP, buff=0.2)
                        for s in chosen])
        amp = VGroup(*[MathTex(r"\tfrac{1}{\sqrt6}", font_size=30).next_to(b, UP, buff=0.1) for b in bars])
        eq = MathTex(r"\ket{D^4_2} = \tfrac{1}{\sqrt6}\big(\ket{0011}+\ket{0101}+\ket{0110}"
                     r"+\ket{1001}+\ket{1010}+\ket{1100}\big)", font_size=34).to_edge(DOWN, buff=0.9)
        k = MathTex(r"K = \binom{4}{2} = 6 \text{ valid strings, each measured with probability } 1/6",
                    font_size=30).next_to(eq, DOWN, buff=0.25)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.1), FadeIn(amp))
        self.play(Write(eq))
        self.play(FadeIn(k))
        self.wait()

        self.next_section("w_state")
        w_strings = ["0001", "0010", "0100", "1000"]
        w = VGroup(*[bitstring(s, 30) for s in w_strings]).arrange(RIGHT, buff=0.9).move_to(chosen)
        w_bars = VGroup(*[share_bar(0.5, max_height=2.4).next_to(s, UP, buff=0.2) for s in w])
        w_amp = VGroup(*[MathTex(r"\tfrac12", font_size=30).next_to(b, UP, buff=0.1) for b in w_bars])
        w_eq = MathTex(r"n = 1:\quad \ket{W_4} = \tfrac12\big(\ket{0001}+\ket{0010}+\ket{0100}+\ket{1000}\big)",
                       font_size=34).move_to(eq)
        self.play(ReplacementTransform(chosen, w), ReplacementTransform(bars, w_bars),
                  ReplacementTransform(amp, w_amp), ReplacementTransform(eq, w_eq), FadeOut(k))
        self.wait()

        self.next_section("symmetries")
        self.play(FadeOut(w_bars), FadeOut(w_amp), FadeOut(w_eq), w.animate.move_to(UP * 0.9))
        sym1 = caption("First symmetry: flipping every bit turns n ones into N − n", 28).to_edge(UP, buff=1.3)
        self.play(FadeIn(sym1))
        flipped = VGroup(*[bitstring("".join("1" if c == "0" else "0" for c in s), 30).move_to(m)
                           for s, m in zip(w_strings, w)])
        flip_eq = MathTex(r"X^{\otimes N}\ket{D^N_n} = \ket{D^N_{N-n}}", font_size=36).next_to(w, DOWN, buff=0.5)
        self.play(Transform(w, flipped), Write(flip_eq))
        self.wait(0.5)
        self.play(Transform(w, VGroup(*[bitstring(s, 30).move_to(m) for s, m in zip(w_strings, w)])))
        sym2 = caption("Second: swapping two qubits leaves the set of strings unchanged", 28).next_to(flip_eq, DOWN, buff=0.6)
        self.play(FadeIn(sym2))
        self.play(*[Swap(m[0], m[3]) for m in w], run_time=1.5)
        free = caption("So we only ever build n ≤ N/2, and the qubits can sit anywhere on the chip",
                       26, GOOD).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(free))
        self.wait(2)


# ---- Idea 1: the score ------------------------------------------------------------------


class ScoreTail(Beats):
    def construct(self):
        d = load("score.json")

        self.next_section("formula")
        head = title("Idea 1: small, balanced cases earn the most")
        S = MathTex(r"S = \frac{1}{z + x}", font_size=60).shift(UP * 1.2)
        zt = VGroup(MathTex(r"z = \frac{\chi^2}{K f}", font_size=40),
                    caption("Do all K valid strings show up equally often?", 22)).arrange(DOWN)
        xt = VGroup(MathTex(r"x", font_size=40), caption("Are the signs between them right?", 22)).arrange(DOWN)
        terms = VGroup(zt, xt).arrange(RIGHT, buff=1.5).next_to(S, DOWN, buff=0.8)
        self.play(Write(head))
        self.play(Write(S))
        self.play(FadeIn(terms))
        self.wait()

        self.next_section("lucky_draws")
        self.play(FadeOut(terms), S.animate.scale(0.7).to_corner(UP + RIGHT, buff=0.5))
        ax = Axes(x_range=[0, 3, 0.5], y_range=[0, 3, 1], x_length=8, y_length=4.2,
                  axis_config={"include_numbers": True, "font_size": 24}).shift(DOWN * 0.5)
        labels = ax.get_axis_labels(MathTex("z", font_size=32), Text("density", font_size=22))
        zs = np.array(d["z"])
        curves = {}
        for K, color in (("2", EXC), ("20", CZC)):
            ys = np.minimum(np.array(d["density"][K]), 3)
            curves[K] = ax.plot_line_graph(zs, ys, add_vertex_dots=False, line_color=color, stroke_width=4)
        tag2 = caption("K = 2", 24, EXC).next_to(ax.c2p(0.25, 2.6), RIGHT)
        tag20 = caption("K = 20", 24, CZC).next_to(ax.c2p(1.25, 1.75), UP)
        self.play(Create(ax), FadeIn(labels))
        self.play(Create(curves["20"]), FadeIn(tag20))
        self.play(Create(curves["2"]), FadeIn(tag2))

        cut = 0.05
        pts = [ax.c2p(0, 0)] + [ax.c2p(z, y) for z, y in zip(zs, np.minimum(d["density"]["2"], 3)) if z <= cut]
        pts += [ax.c2p(cut, 0)]
        shade = Polygon(*pts, stroke_width=0, fill_color=EXC, fill_opacity=0.8)
        p2, p20 = d["lucky"]["2"], d["lucky"]["20"]
        lucky = VGroup(
            caption("A lucky run lands below\nz = 0.05 and scores over 20", 26),
            caption(f"K = 2: {100 * p2:.0f}% of runs", 26, EXC),
            caption(f"K = 20: almost never ({p20:.0e})", 26, CZC),
        ).arrange(DOWN, aligned_edge=LEFT).to_edge(RIGHT, buff=0.4).shift(UP * 0.6)
        self.play(FadeIn(shade), Flash(ax.c2p(0.02, 2.8), color=EXC))
        self.play(FadeIn(lucky))
        self.wait()

        self.next_section("expected")
        self.play(FadeOut(VGroup(ax, labels, curves["2"], curves["20"], tag2, tag20, shade, lucky)))
        Ks, table = d["Ks"], d["expected"]
        lo, hi = math.log10(0.3), math.log10(100)
        bx = Axes(x_range=[0, len(Ks), 1], y_range=[lo, hi, 1], x_length=9, y_length=4.4,
                  axis_config={"include_ticks": False}).shift(DOWN * 0.4)
        yticks = VGroup(*[VGroup(Line(bx.c2p(0, math.log10(v)) + LEFT * 0.08, bx.c2p(0, math.log10(v))),
                                 Text(f"{v:g}", font_size=20).next_to(bx.c2p(0, math.log10(v)), LEFT, buff=0.15))
                          for v in (0.3, 1, 3, 10, 30, 100)])
        ylab = caption("average score E[S] (log scale)", 22).next_to(bx, UP, buff=0.2).align_to(bx, LEFT)
        xlab = VGroup(*[MathTex(f"K={K}", font_size=28).next_to(bx.c2p(i + 0.5, lo), DOWN) for i, K in enumerate(Ks)])
        delta = ValueTracker(0.0)

        def value(K):
            i = delta.get_value() / (d["deltas"][1] - d["deltas"][0])
            j = min(int(i), len(d["deltas"]) - 2)
            v = table[str(K)]
            return v[j] + (v[j + 1] - v[j]) * (i - j)

        def bars():
            g = VGroup()
            for i, K in enumerate(Ks):
                v = value(K)
                top = bx.c2p(i + 0.5, max(math.log10(v), lo + 1e-3))
                base = bx.c2p(i + 0.5, lo)
                color = interpolate_color(GOOD, BAD, min(delta.get_value() / 0.03, 1))
                bar = Rectangle(width=0.8, height=top[1] - base[1], stroke_width=0,
                                fill_color=color, fill_opacity=0.85)
                bar.move_to((top + base) / 2)
                g.add(VGroup(bar, Text(f"{v:.3g}", font_size=24).next_to(bar, UP, buff=0.1)))
            return g

        live = always_redraw(bars)
        note = caption("for a perfect state, with f = 0.9, x = 0.001 and 10,000 shots", 22, DIM).to_edge(DOWN, buff=0.3)
        bx.x_axis.set_opacity(0)  # it would sit at log10(1); draw a floor and a dashed S = 1 instead
        floor = Line(bx.c2p(0, lo), bx.c2p(len(Ks), lo), color=WHITE, stroke_width=2)
        one = DashedLine(bx.c2p(0, 0), bx.c2p(len(Ks), 0), color=DIM, stroke_width=2)
        self.play(Create(bx), Create(floor), Create(one), FadeIn(yticks), FadeIn(ylab), FadeIn(xlab))
        self.play(FadeIn(live), FadeIn(note))
        fifty = caption("One K = 2 case is worth\nabout fifty big ones", 28, EXC).to_edge(RIGHT, buff=0.5).shift(UP * 1.2)
        self.play(Circumscribe(live[0], color=EXC), FadeIn(fifty))
        self.wait()

        self.next_section("imbalance")
        readout = always_redraw(lambda: MathTex(
            rf"\text{{imbalance }}\delta = {100 * delta.get_value():.1f}\%", font_size=36,
        ).to_edge(RIGHT, buff=0.5).shift(UP * 1.2))
        self.play(FadeOut(fifty), FadeIn(readout))
        self.play(delta.animate.set_value(0.01), run_time=2.5, rate_func=rate_functions.linear)
        self.wait(1)
        self.play(delta.animate.set_value(0.03), run_time=3.5, rate_func=rate_functions.linear)
        self.wait()

        self.next_section("takeaway")
        take = caption("Balance first, then cut gates; tilting amplitudes to hide a bias isn't allowed.",
                       24, GOOD).to_edge(DOWN, buff=0.3)
        self.play(FadeOut(note), FadeIn(take))
        self.wait(2)


# ---- Idea 2: the W tree -----------------------------------------------------------------


def _fill(circle, share):
    """Probability held by a node shown as fill; sqrt so 1/N stays visible."""
    return circle.animate.set_fill(EXC, opacity=math.sqrt(share))


class WTree(Beats):
    def construct(self):
        d = load("wtree.json")

        self.next_section("toy")
        head = title("Idea 2: spread a W state along a tree, not a line")
        self.play(Write(head))
        # tree r -> a -> c, r -> b ; line r - a - c - b
        tpos = {"r": UP * 1.6, "a": LEFT * 1.2 + UP * 0.5, "b": RIGHT * 1.2 + UP * 0.5, "c": LEFT * 1.2 + DOWN * 0.6}
        lpos = {q: RIGHT * 1.3 * i + UP * 0.5 for i, q in enumerate("racb")}
        tree = {q: node(q).move_to(p + LEFT * 3.5) for q, p in tpos.items()}
        line = {q: node(q).move_to(p + RIGHT * 1.6) for q, p in lpos.items()}
        tedges = VGroup(*[Line(tree[a][0].get_center(), tree[b][0].get_center(), buff=0.3, color=DIM)
                          for a, b in ("ra", "rb", "ac")])
        ledges = VGroup(*[Line(line[a][0].get_center(), line[b][0].get_center(), buff=0.3, color=DIM)
                          for a, b in ("ra", "ac", "cb")])
        ttag = caption("tree", 28).next_to(tree["r"], UP, buff=0.4)
        ltag = caption("line", 28).match_y(ttag).match_x(VGroup(*line.values()))
        self.play(*[FadeIn(m) for m in (*tree.values(), *line.values(), tedges, ledges, ttag, ltag)])
        tr = caption("rounds: 0", 26).next_to(tree["c"], DOWN, buff=0.35)
        lr = caption("rounds: 0", 26).move_to(RIGHT * 3.55).match_y(tr)
        self.play(_fill(tree["r"][0], 1), _fill(line["r"][0], 1), FadeIn(tr), FadeIn(lr))
        share_note = caption("One excitation to share out: every qubit should end up holding 1/4", 24, DIM).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(share_note))

        def step(group, n, *anims):
            return AnimationGroup(*anims, Transform(group, caption(f"rounds: {n}", 26).move_to(group)))

        self.play(step(tr, 1, _fill(tree["r"][0], 0.5), _fill(tree["a"][0], 0.5), Indicate(tedges[0], color=EXC)),
                  step(lr, 1, _fill(line["r"][0], 0.25), _fill(line["a"][0], 0.75), Indicate(ledges[0], color=EXC)))
        self.play(step(tr, 2, *[_fill(tree[q][0], 0.25) for q in "rabc"],
                       Indicate(tedges[1], color=EXC), Indicate(tedges[2], color=EXC)),
                  step(lr, 2, _fill(line["a"][0], 0.25), _fill(line["c"][0], 0.5), Indicate(ledges[1], color=EXC)))
        self.play(Flash(tree["r"], color=GOOD), tr.animate.set_color(GOOD))
        self.play(step(lr, 3, _fill(line["c"][0], 0.25), _fill(line["b"][0], 0.25), Indicate(ledges[2], color=EXC)))
        self.wait()

        self.next_section("cost")
        cost = VGroup(
            MathTex(r"\ket{10}\mapsto\sqrt{1-p}\,\ket{10}+\sqrt{p}\,\ket{01}", font_size=34),
            caption("The first hand-off costs 1 CZ, every later one 2: 2N − 3 in total for both shapes", 24),
            caption("The difference: a tree does all of a round's hand-offs at once", 24, GOOD),
        ).arrange(DOWN, buff=0.25).to_edge(DOWN, buff=0.35)
        self.play(FadeOut(share_note), FadeIn(cost))
        self.wait(2)

        self.next_section("willow")
        self.play(*[FadeOut(m) for m in self.mobjects if m is not head])
        lay = load("layout.json")
        used = {tuple(q) for r in d["rounds"] for e in r for q in e[:2]} | {tuple(d["root"])}
        used |= {tuple(q) for q in d["chain"]}
        rows = [q[0] for q in used]
        cols = [q[1] for q in used]
        patch = [q for q in lay["qubits"]
                 if min(rows) - 1 <= q[0] <= max(rows) + 1 and min(cols) - 1 <= q[1] <= max(cols) + 1]
        left = Chip(lay, spacing=0.7, qubits=patch)
        right = Chip(lay, spacing=0.7, qubits=patch)
        VGroup(left, right).arrange(RIGHT, buff=1.2).shift(DOWN * 0.15)
        for chip in (left, right):
            for dot in chip.dots.values():
                dot.set_opacity(0.4)
        ltitle = caption(f"Tree on Willow: W₁₀ with {d['tree_cz']} CZ", 26).next_to(left, UP, buff=0.3)
        rtitle = caption(f"Line (our routed B-E baseline): {d['chain_cz']} CZ", 26).next_to(right, UP, buff=0.3)
        self.play(FadeIn(left), FadeIn(right), FadeIn(ltitle), FadeIn(rtitle))

        N = d["N"]
        tree_qs = [tuple(d["root"])] + [tuple(e[1]) for r in d["rounds"] for e in r]
        rings = {name: {q: node(radius=0.22).move_to(chip.dots[q])[0] for q in qs}
                 for name, chip, qs in (("tree", left, tree_qs), ("line", right, [tuple(q) for q in d["chain"]]))}
        self.play(*[FadeIn(c) for g in rings.values() for c in g.values()])

        # one circuit moment = SEC seconds, so the race shows the real depths (15 vs 37)
        SEC = 0.22
        held = {tuple(d["root"]): N}
        tree_steps = [_fill(rings["tree"][tuple(d["root"])], 1)]
        tree_anims = []
        per_round = d["tree_moments"] / len(d["rounds"]) * SEC
        for r in d["rounds"]:
            moves = []
            for p, c, s in r:
                p, c = tuple(p), tuple(c)
                held[p] -= s
                held[c] = s
                moves += [_fill(rings["tree"][p], held[p] / N), _fill(rings["tree"][c], s / N),
                          Indicate(left.line(p, c), color=EXC, scale_factor=1)]
            tree_anims.append(AnimationGroup(*moves, run_time=per_round))
        chain = [tuple(q) for q in d["chain"]]
        line_anims = []
        per_step = d["chain_moments"] / (N - 1) * SEC
        for i in range(N - 1):
            rest = (N - i - 1) / N
            line_anims.append(AnimationGroup(
                _fill(rings["line"][chain[i]], 1 / N), _fill(rings["line"][chain[i + 1]], rest),
                Indicate(right.line(chain[i], chain[i + 1]), color=EXC, scale_factor=1), run_time=per_step))
        clock = ValueTracker(0)
        tclock = always_redraw(lambda: caption(
            f"time steps: {min(int(clock.get_value()), d['tree_moments'])}", 26,
            GOOD if clock.get_value() >= d["tree_moments"] else WHITE).next_to(left, DOWN, buff=0.3))
        lclock = always_redraw(lambda: caption(
            f"time steps: {min(int(clock.get_value()), d['chain_moments'])}", 26).next_to(right, DOWN, buff=0.3))
        self.play(*tree_steps, _fill(rings["line"][chain[0]], 1), FadeIn(tclock), FadeIn(lclock))
        self.play(Succession(*tree_anims), Succession(*line_anims),
                  clock.animate(rate_func=rate_functions.linear, run_time=d["chain_moments"] * SEC)
                  .set_value(d["chain_moments"]))
        self.wait()

        self.next_section("result")
        res = VGroup(
            caption(f"With idle noise, the X term falls from {d['chain_x']:.4f} on the line "
                    f"to {d['tree_x']:.4f} on the tree", 24, GOOD),
            caption("(mean of 3 calibrations; each branch holds one excitation, "
                    "so decay can't unbalance it)", 20, DIM),
        ).arrange(DOWN, buff=0.15).to_edge(DOWN, buff=0.2)
        self.play(FadeOut(tclock), FadeOut(lclock), FadeIn(res))
        self.wait(2)


# ---- Idea 3: computer search ------------------------------------------------------------


class CircuitDiagram(VGroup):
    """Ry layers (boxes) between CZ layers (dot pairs) on N wires."""

    def __init__(self, N, cz_layers, width=7.0, **kw):
        super().__init__(**kw)
        cols = 2 * len(cz_layers) + 1
        dx = width / cols
        self.wires = VGroup(*[Line(LEFT * width / 2, RIGHT * width / 2, color=DIM, stroke_width=2)
                              .shift(DOWN * 0.8 * q) for q in range(N)])
        self.labels = VGroup(*[MathTex(f"q_{q}", font_size=30).next_to(w, LEFT) for q, w in enumerate(self.wires)])
        self.ry = VGroup()
        self.cz = []  # per layer: {edge: mobject}
        for col in range(cols):
            x = -width / 2 + dx * (col + 0.5)
            if col % 2 == 0:
                for q in range(N):
                    box = Square(0.42, color=WHITE, stroke_width=2, fill_color="#1a1a1a", fill_opacity=1)
                    box.move_to(RIGHT * x + DOWN * 0.8 * q)
                    self.ry.add(VGroup(box, MathTex(r"R_y", font_size=22).move_to(box)))
            else:
                layer = {}
                for k, (a, b) in enumerate(cz_layers[col // 2]):
                    xx = x + (k - (len(cz_layers[col // 2]) - 1) / 2) * 0.28
                    g = VGroup(Line(RIGHT * xx + DOWN * 0.8 * a, RIGHT * xx + DOWN * 0.8 * b, color=CZC, stroke_width=4),
                               Dot(RIGHT * xx + DOWN * 0.8 * a, color=CZC, radius=0.09),
                               Dot(RIGHT * xx + DOWN * 0.8 * b, color=CZC, radius=0.09))
                    layer[(a, b)] = g
                self.cz.append(layer)
        self.add(self.wires, self.labels, self.ry, *[g for layer in self.cz for g in layer.values()])


class Synthesis(Beats):
    def construct(self):
        d = load("synth.json")
        N = d["N"]

        self.next_section("template")
        head = title("Idea 3: let a computer search for exact circuits")
        self.play(Write(head))
        what = MathTex(r"\text{goal: }\ket{D^4_2}\qquad \text{building blocks: } R_y \text{ rotations and CZs between neighbours}",
                       font_size=32).next_to(head, DOWN, buff=0.35).to_edge(LEFT, buff=0.5)
        self.play(FadeIn(what))
        log = VGroup()
        circ = None
        for i, row in enumerate(d["depth_log"]):
            from_layers = [list(map(tuple, l)) for l in (d["start_layers"][:row["L"]])]
            new = CircuitDiagram(N, from_layers, width=1.9 * row["L"] + 1.0).shift(LEFT * 3 + DOWN * 0.8)
            ok = row["infidelity"] < 1e-13
            line = caption(f"{row['L']} CZ layer{'s' if row['L'] > 1 else ''}: "
                           + ("exact" if ok else f"still off by {row['infidelity']:.3f}"), 24, GOOD if ok else WHITE)
            log.add(line)
            log.arrange(DOWN, aligned_edge=LEFT).to_edge(RIGHT, buff=0.5).shift(UP * 0.3)
            if circ is None:
                self.play(FadeIn(new), FadeIn(line))
            else:
                self.play(ReplacementTransform(circ, new), FadeIn(line))
            circ = new
            self.wait(0.4)
        self.wait()

        self.next_section("descent")
        loss = d["loss"]
        ax = Axes(x_range=[0, len(loss), 5], y_range=[-16, 0, 4], x_length=4.2, y_length=3.2,
                  axis_config={"include_numbers": False}).to_edge(RIGHT, buff=0.6).shift(DOWN * 1.2)
        yl = VGroup(*[MathTex(f"10^{{{v}}}", font_size=22).next_to(ax.c2p(0, v), LEFT) for v in (0, -8, -16)])
        xl = caption("optimizer steps", 20).next_to(ax, DOWN, buff=0.15)
        tl = caption("distance from exact (1 − F)", 20).next_to(ax, UP, buff=0.15)
        graph = ax.plot_line_graph(list(range(len(loss))), [math.log10(v) for v in loss],
                                   add_vertex_dots=False, line_color=GOOD, stroke_width=4)
        tol = DashedLine(ax.c2p(0, -13), ax.c2p(len(loss), -13), color=DIM)
        toltag = caption("we call it exact below 10⁻¹³", 18, DIM).next_to(tol, UP, buff=0.05).align_to(tol, RIGHT)
        self.play(FadeOut(log), FadeIn(ax), FadeIn(yl), FadeIn(xl), FadeIn(tl), Create(tol), FadeIn(toltag))
        self.play(Create(graph), run_time=2.5)
        self.wait()

        self.next_section("prune")
        self.play(FadeOut(VGroup(ax, yl, xl, tl, graph, tol, toltag)))
        count = sum(map(len, d["start_layers"]))
        counter = caption(f"CZ: {count}", 34).move_to(RIGHT * 4.3 + UP * 0.5)
        rule = caption("Remove one CZ and re-fit.\nStill exact? Leave it out.\nOtherwise, put it back.", 22, DIM).next_to(counter, DOWN, buff=0.4)
        self.play(FadeIn(counter), FadeIn(rule))
        diagram = circ
        for st in d["steps"]:
            li, e = st["layer"], tuple(st["edge"])
            g = diagram.cz[li][e]
            self.play(g.animate.set_color(EXC), run_time=0.4)
            if st["accepted"]:
                count -= 1
                self.play(FadeOut(g, scale=1.6), Transform(counter, caption(f"CZ: {count}", 34, GOOD).move_to(counter)))
                del diagram.cz[li][e]
            else:
                x = Cross(g, stroke_color=BAD, stroke_width=5, scale_factor=0.6)
                why = caption(f"off by {st['infidelity']:.2f}", 20, BAD).next_to(g, UP, buff=0.1)
                self.play(FadeIn(x), FadeIn(why), run_time=0.4)
                self.play(FadeOut(x), FadeOut(why), g.animate.set_color(CZC), run_time=0.4)
        done = caption(f"{count} CZ, and still exact\nto machine precision", 26, GOOD).next_to(rule, DOWN, buff=0.4)
        self.play(FadeIn(done))
        self.wait()

        self.next_section("table")
        self.play(*[FadeOut(m) for m in self.mobjects if m is not head])
        t = d["table"]
        bx = Axes(x_range=[0, len(t["cases"]), 1], y_range=[0, 170, 40], x_length=10, y_length=4.4,
                  axis_config={"include_ticks": False}, y_axis_config={"include_numbers": True, "font_size": 22,
                                                                         "include_ticks": True}).shift(DOWN * 0.5)
        bars = VGroup()
        for i, (case, s, b) in enumerate(zip(t["cases"], t["search"], t["textbook"])):
            for off, v, color in ((-0.18, b, BAD), (0.18, s, GOOD)):
                top, base = bx.c2p(i + 0.5 + off, v), bx.c2p(i + 0.5 + off, 0)
                bar = Rectangle(width=0.34, height=top[1] - base[1], stroke_width=0, fill_color=color, fill_opacity=0.85)
                bar.move_to((top + base) / 2)
                bars.add(VGroup(bar, Text(str(v), font_size=20).next_to(bar, UP, buff=0.08)))
            bars.add(MathTex(case, font_size=30).next_to(bx.c2p(i + 0.5, 0), DOWN, buff=0.2))
        legend = VGroup(caption("textbook recipe (Bärtschi–Eidenbenz), routed", 22, BAD),
                        caption("found by computer search", 22, GOOD)).arrange(DOWN, aligned_edge=LEFT)
        legend.next_to(bx.c2p(0.2, 165), RIGHT).shift(DOWN * 0.2)
        ylab = caption("CZ gates used", 22).next_to(bx, UP, buff=0.1).align_to(bx, LEFT)
        self.play(Create(bx), FadeIn(ylab))
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.2) for b in bars], lag_ratio=0.05), FadeIn(legend))
        foot = caption("Past about 9 qubits, the search stops finding short circuits", 22, DIM).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(foot))
        self.wait(2)


# ---- Idea 4: placement ------------------------------------------------------------------


class Placement(Beats):
    def column(self, *mobs, buff=0.18):
        """Right-hand text column, top-aligned beside the chip."""
        g = VGroup(*mobs).arrange(DOWN, aligned_edge=LEFT, buff=buff)
        return g.move_to(RIGHT * 0.4 + UP * 2.4, aligned_edge=UP + LEFT)

    def construct(self):
        d = load("layout.json")

        self.next_section("chip")
        head = title("Idea 4: put the circuit on the best qubits")
        self.play(Write(head))
        chip = Chip(d, spacing=0.42)
        chip.move_to(LEFT * 3.4 + DOWN * 0.45)
        errs = [c[2] for c in d["couplers"] if c[2] is not None]
        lo, hi = min(errs), max(errs)
        self.play(FadeIn(chip))
        recolor = [chip.line(a, b).animate.set_color(err_color(e, lo, hi))
                   for a, b, e in d["couplers"] if e is not None]
        legend = self.column(
            caption("Willow as calibrated (seed 0)", 26),
            caption(f"best CZ error: {100 * lo:.2f}%", 22, GOOD),
            caption(f"typical: {100 * float(np.median(errs)):.2f}%", 22, EXC),
            caption(f"worst: {100 * hi:.2f}%", 22, BAD),
            caption("grey: not usable today", 22, DIM),
        )
        self.play(LaggedStart(*recolor, lag_ratio=0.005), FadeIn(legend))
        self.wait()

        self.next_section("beam")
        why = self.column(
            caption("Scoring a placement", 26),
            MathTex(r"f \approx \prod (1 - e)\;\Rightarrow\; -\log f \approx \sum e", font_size=32),
            caption("add up the error of every gate,\nwait and readout it touches", 22, DIM),
        )
        self.play(FadeOut(legend), FadeIn(why))
        ghosts, label = None, None
        for L, snap in enumerate(d["beam_snapshots"], start=1):
            g = VGroup()
            for path in snap:
                pts = [chip.dots[tuple(q)].get_center() for q in path]
                g.add(Dot(pts[0], radius=0.09, color=EXC) if len(pts) == 1 else
                      VGroup(*[Line(a, b, color=EXC, stroke_width=5, stroke_opacity=0.7)
                               for a, b in zip(pts, pts[1:])]))
            step = caption(f"beam search keeps the best\npaths of length {L}", 24, EXC)
            step.next_to(why, DOWN, buff=0.6).align_to(why, LEFT)
            if ghosts is None:
                self.play(FadeIn(g), FadeIn(step), run_time=0.7)
                label = step
            else:
                self.play(ReplacementTransform(ghosts, g), Transform(label, step), run_time=0.7)
            ghosts = g
        best = [chip.dots[tuple(q)].get_center() for q in d["best_path"]]
        path = VGroup(*[Line(a, b, color=GOOD, stroke_width=9) for a, b in zip(best, best[1:])],
                      *[Dot(p, radius=0.1, color=GOOD) for p in best])
        verdict = caption(f"best 6-qubit path costs {d['best_path_cost']:.4f};\n"
                          f"a typical kept path, {d['median_beam_cost']:.4f}", 24, GOOD)
        verdict.move_to(label, aligned_edge=UP + LEFT)
        self.play(FadeOut(ghosts), FadeIn(path), ReplacementTransform(label, verdict))
        self.wait()

        self.next_section("readout")
        intro = self.column(
            caption("Matching readout errors", 26),
            caption("each qubit misreads 1 as 0 at its own rate,\n"
                    "so |10⟩ and |01⟩ aren't read equally well", 20, DIM),
        )
        self.play(FadeOut(why), FadeOut(path), FadeOut(verdict), FadeIn(intro))
        panels, rings = VGroup(), VGroup()
        for key, name, color in (("matched", "matched pair", GOOD), ("leaky", "mismatched pair", BAD)):
            info = d[key]
            a, b = (tuple(q) for q in info["pair"])
            rings.add(*[SurroundingRectangle(chip.dots[q], color=color, buff=0.08) for q in (a, b)])
            p10, p01 = info["read"]
            delta = (p10 - p01) / (p10 + p01)
            bars = VGroup(*[share_bar((v - 0.95) / 0.05, width=0.5, max_height=1.5, color=color)
                            for v in (p10, p01)]).arrange(RIGHT, buff=0.35, aligned_edge=DOWN)
            names = VGroup(*[bitstring(s, 22).next_to(bar, DOWN, buff=0.1) for s, bar in zip(("10", "01"), bars)])
            vals = VGroup(*[Text(f"{v:.3f}", font_size=18).next_to(bar, UP, buff=0.08)
                            for v, bar in zip((p10, p01), bars)])
            panels.add(VGroup(caption(name, 22, color), VGroup(bars, names, vals),
                              MathTex(rf"\delta = {100 * abs(delta):.2f}\%", font_size=30, color=color)
                              ).arrange(DOWN, buff=0.2))
        panels.arrange(RIGHT, buff=1.0, aligned_edge=UP).next_to(intro, DOWN, buff=0.4).align_to(intro, LEFT)
        hint = caption("chance of reading correctly (axis starts at 0.95)", 18, DIM)
        hint.next_to(panels, DOWN, buff=0.2).align_to(intro, LEFT)
        for k in range(2):
            self.play(Create(rings[2 * k]), Create(rings[2 * k + 1]), FadeIn(panels[k]))
        self.play(FadeIn(hint))
        self.wait()

        self.next_section("w2_scan")
        scan = d["w2_scan"]
        es = [s[2] for s in scan]
        lo_e, hi_e, med = min(es), max(es), float(np.median(es))
        recolor = [chip.line(a, b).animate.set_color(err_color(1 / e, 1 / hi_e, 1 / lo_e))
                   for a, b, e in scan]
        tag = self.column(
            caption("For W₂ we simulate every pair", 26),
            caption("each coupler coloured by its expected score", 20, DIM),
            caption(f"worst pair   E[S] = {lo_e:.1f}", 24, BAD),
            caption(f"typical pair  E[S] = {med:.1f}", 24, EXC),
            caption(f"best pair    E[S] = {hi_e:.1f}", 24, GOOD),
            caption(f"{hi_e / med:.1f}× the typical pair,\njust by choosing well", 24, GOOD),
        )
        self.play(FadeOut(intro), FadeOut(panels), FadeOut(hint), FadeOut(rings))
        self.play(LaggedStart(*recolor, lag_ratio=0.005), FadeIn(tag))
        a, b = (tuple(q) for q in d["w2_picked"])
        pick = VGroup(Line(chip.dots[a].get_center(), chip.dots[b].get_center(), color=GOOD, stroke_width=10),
                      *[Dot(chip.dots[q].get_center(), radius=0.11, color=GOOD) for q in (a, b)])
        self.play(FadeIn(pick), Flash(pick.get_center(), color=GOOD, flash_radius=0.4), run_time=1.2)
        self.wait(2)


# ---- docs/main.tex: the organizers' revised ranking -------------------------------------


def _log_bars(ax, values, lo, color, size=24):
    """One bar per value on a log10 axis whose floor is `lo`, labelled with the value."""
    g = VGroup()
    for i, v in enumerate(values):
        top, base = ax.c2p(i + 0.5, math.log10(v)), ax.c2p(i + 0.5, lo)
        bar = Rectangle(width=0.8, height=top[1] - base[1], stroke_width=0, fill_color=color,
                        fill_opacity=0.85).move_to((top + base) / 2)
        g.add(VGroup(bar, Text(f"{v:.3g}", font_size=size).next_to(bar, UP, buff=0.1)))
    return g


def _log_axes(n, lo, hi, ticks):
    ax = Axes(x_range=[0, n, 1], y_range=[lo, hi, 1], x_length=9, y_length=4.2,
              axis_config={"include_ticks": False}).shift(DOWN * 0.5)
    ax.x_axis.set_opacity(0)  # it would sit at log10(1); draw a floor instead
    floor = Line(ax.c2p(0, lo), ax.c2p(n, lo), color=WHITE, stroke_width=2)
    yt = VGroup(*[VGroup(Line(ax.c2p(0, math.log10(v)) + LEFT * 0.08, ax.c2p(0, math.log10(v))),
                         Text(f"{v:g}", font_size=20).next_to(ax.c2p(0, math.log10(v)), LEFT, buff=0.15))
                  for v in ticks])
    return ax, VGroup(floor, yt)


class RevisedScore(Beats):
    def construct(self):
        d = load("revised.json")

        self.next_section("two_averages")
        head = title("The organizers changed how scores are averaged")
        self.play(Write(head))
        old = VGroup(caption("Before: the plain average score", 26),
                     MathTex(r"\mathbb{E}[S] = \mathbb{E}\Big[\frac{1}{z+x}\Big]", font_size=40),
                     MathTex(r"\mathbb{E}\big[1/\chi^2_\nu\big] = \infty \text{ for } \nu \le 2",
                             font_size=32, color=BAD)).arrange(DOWN, buff=0.35)
        new = VGroup(caption("Since Oct 25: average 1/S, then flip it", 26),
                     MathTex(r"\frac{1}{\mathbb{E}[1/S]} = \frac{1}{\mathbb{E}[z+x]}", font_size=40),
                     MathTex(r"\mathbb{E}[\chi^2] = K - 1 + \lambda \text{ is always finite}",
                             font_size=32, color=GOOD)).arrange(DOWN, buff=0.35)
        VGroup(old, new).arrange(RIGHT, buff=1.0).shift(DOWN * 0.3)
        self.play(FadeIn(old))
        self.play(FadeIn(new))
        self.wait()

        self.next_section("tail_gone")
        self.play(FadeOut(old), FadeOut(new))
        Ks = d["Ks"]
        lo, hi = math.log10(0.3), math.log10(100)
        ax, frame = _log_axes(len(Ks), lo, hi, (0.3, 1, 3, 10, 30, 100))
        xlab = VGroup(*[MathTex(f"K={K}", font_size=28).next_to(ax.c2p(i + 0.5, lo), DOWN) for i, K in enumerate(Ks)])
        before = _log_bars(ax, d["mean_S"], lo, EXC)
        after = _log_bars(ax, d["inv_mean"], lo, GOOD)
        tag = caption("average score E[S]", 28, EXC).move_to(ax.c2p(4.6, math.log10(60)))
        foot = caption(f"for a perfect state, with f = {d['f']}, x = {d['x']} and 10,000 shots", 22, DIM).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(ax), FadeIn(frame), FadeIn(xlab), FadeIn(foot))
        self.play(FadeIn(before), FadeIn(tag))
        self.wait(0.5)
        self.play(ReplacementTransform(before, after),
                  Transform(tag, caption("1 / E[z + x]", 28, GOOD).move_to(tag)), run_time=2)
        why = MathTex(r"\frac{1}{\mathbb{E}[z+x]} = \frac{Kf}{K-1+\lambda+Kfx}", font_size=34)
        why.move_to(ax.c2p(4.6, math.log10(14)))
        self.play(FadeIn(why))
        self.play(Transform(foot, caption("Lucky runs no longer count: a small case is now worth about f·K/(K−1), not 50",
                                          24, GOOD).move_to(foot)))
        self.wait()

        self.next_section("budget")
        self.play(FadeOut(VGroup(ax, frame, xlab, after, tag, why, foot)))
        # in this beat the colours separate the three terms of E[z + x]
        formula = MathTex(r"\mathbb{E}[z+x] =", r"\frac{K-1}{Kf}", "+", r"\frac{\lambda}{Kf}", "+", "x",
                          font_size=44).next_to(head, DOWN, buff=0.5)
        for i, c in ((1, BAD), (3, EXC), (5, CZC)):
            formula[i].set_color(c)
        formula.set_x(0)
        names = VGroup(caption("lost shots (f)", 22, BAD), caption("imbalance", 22, EXC),
                       caption("wrong signs", 22, CZC)).arrange(RIGHT, buff=0.8).next_to(formula, DOWN, buff=0.3)
        self.play(Write(formula), FadeIn(names))
        scale = 2.8
        rows = VGroup()
        for y, (name, b) in zip((0.0, -1.5), d["budgets"].items()):
            parts = [(b["errors"], BAD), (b["imbalance"], EXC), (b["x"], CZC)]
            segs = VGroup(*[Rectangle(width=max(v * scale, 1e-3), height=0.5, stroke_width=0,
                                      fill_color=c, fill_opacity=0.85) for v, c in parts]).arrange(RIGHT, buff=0)
            segs.move_to(LEFT * 3.2 + UP * (y - 0.4), aligned_edge=LEFT)
            total = sum(v for v, _ in parts)
            label = MathTex(rf"{name}\;(K={b['K']})", font_size=32).next_to(segs, LEFT, buff=0.3)
            score = MathTex(rf"\Rightarrow\ \tfrac{{1}}{{\mathbb{{E}}[z+x]}} = {1 / total:.2f}", font_size=30)
            score.next_to(segs, RIGHT, buff=0.3)
            nums = caption(" + ".join(f"{v:.3f}" for v, _ in parts), 20, DIM).next_to(segs, DOWN, buff=0.12)
            nums.align_to(segs, LEFT)
            rows.add(VGroup(label, segs, score, nums))
        for r in rows:
            self.play(FadeIn(r[0]), GrowFromEdge(r[1], LEFT), FadeIn(r[3]))
            self.play(FadeIn(r[2]))
        take = caption("For big K, imbalance barely matters; lost shots and wrong signs decide the score",
                       24, GOOD).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(take))
        self.wait()

        self.next_section("pairs")
        self.play(FadeOut(VGroup(formula, names, rows, take)))
        p = d["pairs"]
        bx = Axes(x_range=[0, 3, 1], y_range=[0, 0.9, 0.2], x_length=9, y_length=4,
                  axis_config={"include_ticks": False},
                  y_axis_config={"include_numbers": True, "font_size": 22, "include_ticks": True}).shift(DOWN * 0.4)
        series = (("before", "before the merge pass", DIM), ("deployed", "what we submitted", GOOD),
                  ("combined", "teammates' combined entry", BAD))
        bars = VGroup()
        for i, case in enumerate(p["cases"]):
            for j, (key, _, color) in enumerate(series):
                v = p[key][i][0]
                x = i + 0.5 + (j - 1) * 0.27
                top, base = bx.c2p(x, v), bx.c2p(x, 0)
                bar = Rectangle(width=0.25, height=top[1] - base[1], stroke_width=0, fill_color=color,
                                fill_opacity=0.85).move_to((top + base) / 2)
                bars.add(VGroup(bar, Text(f"{v:.2f}", font_size=16).next_to(bar, UP, buff=0.06)))
            bars.add(MathTex(case, font_size=30).next_to(bx.c2p(i + 0.5, 0), DOWN, buff=0.2))
        legend = VGroup(*[caption(name, 20, color) for _, name, color in series]).arrange(RIGHT, buff=0.5)
        legend.next_to(bx, UP, buff=0.15)
        ylab = caption("1/E[z+x] with gate noise, averaged over 3 calibrations", 20).next_to(legend, UP, buff=0.1).align_to(bx, LEFT)
        self.play(Create(bx), FadeIn(ylab), FadeIn(legend))
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.2) for b in bars], lag_ratio=0.05))
        foot = caption("Final judging is still one run per case on hardware, so we still aim for the best average",
                       22, DIM).to_edge(DOWN, buff=0.25)
        self.play(FadeIn(foot))
        self.wait(2)


# ---- docs/main.tex: one CZ cannot do a later transfer -----------------------------------


class TransferBound(Beats):
    def construct(self):
        d = load("transfer.json")

        self.next_section("task")
        head = title("Why each later W hand-off needs 2 CZ")
        self.play(Write(head))
        t1 = MathTex(r"\ket{00}", r"\mapsto", r"\ket{00}", font_size=44)
        t2 = MathTex(r"\ket{10}", r"\mapsto", r"\sqrt{1-p}\,\ket{10}+\sqrt{p}\,\ket{01}", font_size=44)
        VGroup(t1, t2).arrange(DOWN, aligned_edge=LEFT, buff=0.45).shift(UP * 1.0 + LEFT * 1.5)
        t1.shift(RIGHT * (t2[1].get_x() - t1[1].get_x()))
        tags = VGroup(caption("must stay unentangled", 24, GOOD).next_to(t1, RIGHT, buff=0.6),
                      caption("must become entangled", 24, EXC).next_to(t2, RIGHT, buff=0.5))
        tags[0].align_to(tags[1], LEFT)
        self.play(Write(t1), Write(t2))
        self.play(FadeIn(tags))
        why = VGroup(
            caption("After round one, qubit v only holds the excitation in some branches,", 24, DIM),
            caption("so a single gate has to get both inputs right at the same time", 24, DIM),
            caption("The first hand-off only ever sees |10⟩, so R_y and one CNOT (1 CZ) suffice", 24, GOOD),
        ).arrange(DOWN, buff=0.2).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(why[:2]))
        self.play(FadeIn(why[2]))
        self.wait()

        self.next_section("lemma")
        self.play(FadeOut(VGroup(t1, t2, tags, why)))
        eq = MathTex(r"\mathrm{CZ}\,(\alpha\ket0+\beta\ket1)\ket\phi", "=",
                     r"\alpha\ket0\ket\phi+\beta\ket1\,Z\ket\phi", font_size=38).next_to(head, DOWN, buff=0.45).set_x(0)
        rule = VGroup(caption("This stays unentangled exactly when αβ = 0 or Z|φ⟩ ∝ |φ⟩.", 22),
                      caption("In words: a CZ only entangles when neither qubit sits at |0⟩ or |1⟩", 22, EXC)
                      ).arrange(DOWN, buff=0.1).next_to(eq, DOWN, buff=0.25)
        self.play(Write(eq))
        self.play(FadeIn(rule))
        ta, tb = ValueTracker(0.0), ValueTracker(0.0)

        def dial(tr, center, name):
            def draw():
                a = tr.get_value()
                on_axis = abs(math.sin(2 * a)) < 0.03
                circ = Circle(radius=1.0, color=DIM, stroke_width=2).move_to(center)
                axes = VGroup(DashedLine(center + LEFT * 1.15, center + RIGHT * 1.15, color=DIM, stroke_width=1),
                              DashedLine(center + DOWN * 1.15, center + UP * 1.15, color=DIM, stroke_width=1))
                tip = center + np.array([math.cos(a), math.sin(a), 0])
                arrow = Arrow(center, tip, buff=0, color=GOOD if on_axis else EXC, stroke_width=5)
                return VGroup(circ, axes, arrow)
            g = always_redraw(draw)
            labels = VGroup(MathTex(r"\ket0", font_size=26).next_to(center + RIGHT * 1.15, RIGHT, buff=0.08),
                            MathTex(r"\ket1", font_size=26).next_to(center + UP * 1.15, UP, buff=0.05),
                            VGroup(*[caption(t, 20) for t in name.split("\n")]).arrange(DOWN, buff=0.05)
                            .next_to(center + DOWN * 1.15, DOWN, buff=0.1))
            return VGroup(g, labels)

        ca, cb = LEFT * 4.0 + DOWN * 1.1, LEFT * 0.6 + DOWN * 1.1
        da = dial(ta, ca, "qubit v\ncos a|0⟩ + sin a|1⟩")
        db = dial(tb, cb, "qubit c\ncos b|0⟩ + sin b|1⟩")
        readout = always_redraw(lambda: VGroup(
            caption("How entangled after the CZ?", 24),
            MathTex(r"\text{concurrence} = |\sin 2a\,\sin 2b|", font_size=32),
            MathTex(f"= {abs(math.sin(2 * ta.get_value()) * math.sin(2 * tb.get_value())):.2f}", font_size=40,
                    color=GOOD if abs(math.sin(2 * ta.get_value()) * math.sin(2 * tb.get_value())) < 0.01 else EXC),
        ).arrange(DOWN, buff=0.2).move_to(RIGHT * 4.1 + DOWN * 1.1))
        self.play(FadeIn(da), FadeIn(db), FadeIn(readout))
        self.play(tb.animate.set_value(math.pi / 4), run_time=1.5)  # v = |0>: nothing happens
        self.play(ta.animate.set_value(math.pi / 2), run_time=1.5)  # through superposition to |1>
        self.play(ta.animate.set_value(math.pi / 4), run_time=1.5)  # both in superposition
        self.wait(0.5)
        self.play(tb.animate.set_value(0.0), run_time=1.5)
        self.wait()

        self.next_section("proof")
        self.play(FadeOut(VGroup(eq, rule, da, db, readout)))
        wires = VGroup(*[Line(LEFT * 3.2, RIGHT * 3.2, color=DIM, stroke_width=2).shift(UP * (1.8 - 0.9 * i))
                         for i in range(2)])
        wl = VGroup(*[MathTex(q, font_size=30).next_to(w, LEFT) for q, w in zip("vc", wires)])

        def box(text, x, i, color=WHITE):
            sq = Square(0.55, color=color, stroke_width=2, fill_color="#1a1a1a", fill_opacity=1)
            sq.move_to(RIGHT * x + wires[i].get_center()[1] * UP)
            return VGroup(sq, MathTex(text, font_size=26, color=color).move_to(sq))

        pre = VGroup(box("A_0", -2, 0), box("B_0", -2, 1))
        cz = VGroup(Line(wires[0].get_center(), wires[1].get_center(), color=CZC, stroke_width=4),
                    Dot(wires[0].get_center(), color=CZC, radius=0.1), Dot(wires[1].get_center(), color=CZC, radius=0.1))
        post = VGroup(box("A_1", 2, 0), box("B_1", 2, 1))
        circuit = VGroup(wires, wl, pre, cz, post).shift(LEFT * 2.6)
        self.play(FadeIn(circuit))
        local = caption("single-qubit gates can't create\nor remove entanglement", 20, DIM).next_to(post, UP, buff=0.15)
        self.play(post.animate.set_opacity(0.3), FadeIn(local))
        before = VGroup(MathTex(r"\ket{00}\ \to\ A_0\ket0\otimes B_0\ket0", font_size=32),
                        MathTex(r"\ket{10}\ \to\ A_0\ket1\otimes B_0\ket0", font_size=32))
        before.arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(circuit, DOWN, buff=0.45).align_to(circuit, LEFT)
        btag = caption("the two inputs, just before the CZ", 22, DIM).next_to(before, RIGHT, buff=0.5)
        self.play(FadeIn(before), FadeIn(btag))
        cases = VGroup(
            VGroup(caption("case 1:", 24), MathTex(r"B_0\ket0\in\{\ket0,\ket1\}", font_size=30),
                   caption("then neither input gets entangled", 24)).arrange(RIGHT, buff=0.25),
            VGroup(caption("case 2:", 24), caption("otherwise, keeping |00⟩ unentangled needs", 24),
                   MathTex(r"A_0\ket0\in\{\ket0,\ket1\}", font_size=30)).arrange(RIGHT, buff=0.25),
            VGroup(caption("then", 24), MathTex(r"A_0\ket1\perp A_0\ket0", font_size=30),
                   caption("is a basis state too, so |10⟩ can't entangle", 24)).arrange(RIGHT, buff=0.25),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).next_to(before, DOWN, buff=0.4).align_to(before, LEFT)
        crosses = VGroup(*[caption("✗ dead end", 24, BAD).next_to(cases[i], RIGHT, buff=0.3) for i in (0, 2)])
        self.play(FadeIn(cases[0]))
        self.play(FadeIn(crosses[0]))
        self.play(FadeIn(cases[1]))
        self.play(FadeIn(cases[2]))
        self.play(FadeIn(crosses[1]))
        verdict = caption("Both cases fail, so one CZ can never do the hand-off for 0 < p < 1", 26, GOOD).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(verdict))
        self.wait()

        self.next_section("count")
        self.play(FadeOut(VGroup(circuit, local, before, btag, cases, crosses, verdict)))
        total = MathTex(r"\underbrace{1}_{\text{first}} + \underbrace{2\,(N-2)}_{\text{every later}} = 2N-3"
                        r"\ \text{CZ}", font_size=52).shift(UP * 0.6)
        opt = caption(f"A brute-force search confirms nothing does better for W₃ ({d['w_optimal']['3']} CZ) "
                      f"or W₄ ({d['w_optimal']['4']} CZ)", 24, GOOD).next_to(total, DOWN, buff=0.6)
        credit = caption("The argument is Bryan's; details in docs/bryan_constructions.md", 22, DIM).to_edge(DOWN, buff=0.4)
        self.play(Write(total))
        self.play(FadeIn(opt), FadeIn(credit))
        self.wait(2)


# ---- docs/main.tex: coherent errors ------------------------------------------------------


class CoherentErrors(Beats):
    def construct(self):
        d = load("coherent.json")

        self.next_section("geometry")
        head = title("Why over-rotation hurts more than phase errors")
        self.play(Write(head))
        eps = ValueTracker(0.0)
        U = 1.8  # screen length of the unperturbed amplitude psi_b = 1

        def plane(center, real_error):
            ax = VGroup(Line(center + LEFT * 0.5, center + RIGHT * 3.0, color=DIM, stroke_width=2),
                        Line(center + DOWN * 1.3, center + UP * 1.3, color=DIM, stroke_width=2))
            labels = VGroup(caption("Re", 18, DIM).next_to(ax[0], RIGHT, buff=0.1),
                            caption("Im", 18, DIM).next_to(ax[1], UP, buff=0.05))
            ref = DashedLine(center + RIGHT * U + DOWN * 1.2, center + RIGHT * U + UP * 1.2, color=DIM,
                             stroke_width=1) if real_error else Circle(radius=U, color=DIM, stroke_width=1).move_to(center)
            if not real_error:
                ref = ref.get_subcurve(0.85, 1.0).add(ref.copy().get_subcurve(0.0, 0.15))

            def draw():
                e = eps.get_value()
                tip = center + RIGHT * U
                step = (RIGHT if real_error else UP) * e * U
                g = VGroup(Arrow(center, tip, buff=0, color=WHITE, stroke_width=4))
                if e > 1e-3:
                    g.add(Arrow(tip, tip + step, buff=0, color=BAD, stroke_width=5, max_tip_length_to_length_ratio=0.4),
                          Arrow(center, tip + step, buff=0, color=EXC, stroke_width=5))
                return g
            return VGroup(ax, labels, ref, always_redraw(draw))

        cl, cr = LEFT * 5.6 + DOWN * 0.2, RIGHT * 1.2 + DOWN * 0.2
        left, right = plane(cl, True), plane(cr, False)
        ltitle = VGroup(caption("A slightly too-large rotation", 24),
                        caption("a real error, so it adds onto ψ", 20, DIM)).arrange(DOWN, buff=0.1)
        rtitle = VGroup(caption("A stray phase (from Z or CZ)", 24),
                        caption("an imaginary error, at right angles to ψ", 20, DIM)).arrange(DOWN, buff=0.1)
        ltitle.move_to(cl + RIGHT * 1.2 + UP * 2.0)
        rtitle.move_to(cr + RIGHT * 1.2 + UP * 2.0)
        lform = MathTex(r"(\psi_b+\varepsilon g)^2=\psi_b^2+", r"2\varepsilon\psi_b g", r"+\varepsilon^2g^2",
                        font_size=30).move_to(cl + RIGHT * 1.2 + DOWN * 1.9)
        lform[1].set_color(BAD)
        rform = MathTex(r"|\psi_b-i\varepsilon g|^2=\psi_b^2+", r"\varepsilon^2 g^2", font_size=30)
        rform.move_to(cr + RIGHT * 1.2 + DOWN * 1.9)
        rform[1].set_color(GOOD)
        lval = always_redraw(lambda: caption(f"the probability shifts by {2 * eps.get_value() + eps.get_value() ** 2:.3f}",
                                             22, BAD).next_to(lform, DOWN, buff=0.2))
        rval = always_redraw(lambda: caption(f"the probability shifts by {eps.get_value() ** 2:.3f}",
                                             22, GOOD).next_to(rform, DOWN, buff=0.2))
        etag = always_redraw(lambda: MathTex(rf"\varepsilon = {eps.get_value():.2f}", font_size=36)
                             .to_edge(DOWN, buff=0.25))
        self.play(FadeIn(left), FadeIn(right), FadeIn(ltitle), FadeIn(rtitle), FadeIn(etag))
        self.play(Write(lform), Write(rform), FadeIn(lval), FadeIn(rval))
        self.play(eps.animate.set_value(0.1), run_time=2, rate_func=rate_functions.linear)
        self.wait(0.5)
        self.play(eps.animate.set_value(0.35), run_time=2)
        self.wait()

        self.next_section("measured")
        self.play(*[FadeOut(m) for m in self.mobjects if m is not head])
        ax = Axes(x_range=[-3, -1, 1], y_range=[-6, 0, 1], x_length=6.5, y_length=4.3).shift(LEFT * 2.2 + DOWN * 0.5)
        ax.set_opacity(0)  # its axes would cross at 10^0; draw a bottom-left frame instead
        frame = VGroup(Line(ax.c2p(-3, -6), ax.c2p(-1, -6)), Line(ax.c2p(-3, -6), ax.c2p(-3, 0)),
                       *[Line(ax.c2p(v, -6), ax.c2p(v, -6) + UP * 0.1) for v in (-2, -1)],
                       *[Line(ax.c2p(-3, v), ax.c2p(-3, v) + RIGHT * 0.1) for v in (-4, -2, 0)])
        xl = VGroup(*[MathTex(f"10^{{{v}}}", font_size=24).next_to(ax.c2p(v, -6), DOWN) for v in (-3, -2, -1)])
        yl = VGroup(*[MathTex(f"10^{{{v}}}", font_size=24).next_to(ax.c2p(-3, v), LEFT) for v in (-6, -4, -2, 0)])
        xt = MathTex(r"\text{error size } \varepsilon", font_size=28).next_to(xl, DOWN, buff=0.1)
        yt = caption("How far the measured probabilities drift, on our deployed D⁴₂", 20).next_to(ax, UP, buff=0.2)
        self.play(Create(frame), FadeIn(xl), FadeIn(yl), FadeIn(xt), FadeIn(yt))
        es = [math.log10(e) for e in d["eps"]]
        notes = VGroup()
        for key, color, name in (("over", BAD, "over-rotation grows like ε"), ("z", GOOD, "a stray phase grows like ε²")):
            ys = [math.log10(v) for v in d["tv"][key]]
            g = ax.plot_line_graph(es, ys, line_color=color, stroke_width=4,
                                   vertex_dot_radius=0.06, vertex_dot_style={"color": color})
            ratio = d["tv"][key][4] / d["tv"][key][0]  # eps 1e-2 against 1e-3
            notes.add(VGroup(caption(name, 26, color),
                             caption(f"10× smaller ε, {ratio:.0f}× less drift", 22, color)).arrange(DOWN, aligned_edge=LEFT))
            notes.arrange(DOWN, aligned_edge=LEFT, buff=0.5).to_edge(RIGHT, buff=0.6).shift(UP * 0.4)
            self.play(Create(g), FadeIn(notes[-1]), run_time=1.5)
        self.wait()

        self.next_section("pick")
        self.play(*[FadeOut(m) for m in self.mobjects if m is not head])
        lead = caption("So when choosing between exact angle sets, we only guard against over-rotation", 24).next_to(head, DOWN, buff=0.5).set_x(0)
        loss = MathTex(r"L(c,\sigma)=\mathbb{E}[\mathrm{loss}]_c+\lambda_c\,\frac{(\sigma/1\%)^2}{Kf}", font_size=36)
        pick = MathTex(r"c^\star=\arg\min_c\,\max_\sigma\Big[L(c,\sigma)-\min_{c'} L(c',\sigma)\Big]", font_size=36)
        sig = caption("pick the set whose worst-case regret is smallest, for over-rotation σ from 0 to 1%", 22, DIM)
        VGroup(loss, pick, sig).arrange(DOWN, buff=0.35).next_to(lead, DOWN, buff=0.45)
        self.play(FadeIn(lead))
        self.play(Write(loss))
        self.play(Write(pick), FadeIn(sig))
        a, b = d["d52_lam"]
        bars = VGroup()
        for v, color, name in ((a, BAD, "first angles found"), (b, GOOD, "robust pick")):
            bar = Rectangle(width=v * 0.1, height=0.45, stroke_width=0, fill_color=color, fill_opacity=0.85)
            bars.add(VGroup(caption(name, 22, color), bar, Text(f"λ = {v}", font_size=22)).arrange(RIGHT, buff=0.3))
        bars.arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(sig, DOWN, buff=0.5)
        for row in bars:
            row[1].align_to(bars[0][1], LEFT)
            row[2].next_to(row[1], RIGHT, buff=0.3)
        btag = caption("Imbalance λ for D⁵₂ at 1% over-rotation (1.1–5.5× less sensitive across cases)", 22, GOOD)
        btag.next_to(bars, DOWN, buff=0.3)
        self.play(LaggedStart(*[GrowFromEdge(r[1], LEFT) for r in bars], lag_ratio=0.3),
                  FadeIn(VGroup(*[VGroup(r[0], r[2]) for r in bars])))
        self.play(FadeIn(btag))
        self.wait(2)


# ---- docs/main.tex: the single-qubit merge pass ------------------------------------------


class MergePass(Beats):
    DX = 0.088  # screen width of one moment

    def rows(self, timelines, kinds):
        """Per qubit, a dict moment -> mobject for the gate kinds in `kinds` ('c' or 's')."""
        out = []
        for i, row in enumerate(timelines):
            y = 1.3 - 0.75 * i
            marks = {}
            for t, ch in enumerate(row):
                if ch not in kinds:
                    continue
                x = -5.4 + self.DX * (t + 0.5)
                if ch == "c":
                    marks[t] = Square(0.2, stroke_width=0, fill_color=CZC, fill_opacity=1).move_to(RIGHT * x + UP * y)
                else:
                    marks[t] = Rectangle(width=0.05, height=0.4, stroke_width=0, fill_color=WHITE,
                                         fill_opacity=0.95).move_to(RIGHT * x + UP * y)
            out.append(marks)
        return out

    def construct(self):
        d = load("merge.json")
        st, tl = d["stats"], d["timelines"]
        n = len(tl["raw"][0])

        self.next_section("pulses")
        head = title("Merging single-qubit gates: one gate per run")
        self.play(Write(head))
        sub = caption("Bryan's D¹⁵₄, as we received it, puts a π pulse in every idle slot (CPMG)", 24, DIM).next_to(head, DOWN, buff=0.3).align_to(head, LEFT)
        wires = VGroup(*[Line(LEFT * 5.4 + UP * (1.3 - 0.75 * i), LEFT * 5.4 + RIGHT * self.DX * n + UP * (1.3 - 0.75 * i),
                              color=DIM, stroke_width=1.5) for i in range(len(d["qubits"]))])
        names = VGroup(*[Text(f"q{r},{c}", font_size=18).next_to(w, LEFT, buff=0.15) for (r, c), w in zip(d["qubits"], wires)])
        czs = self.rows(tl["raw"], "c")
        raw = self.rows(tl["raw"], "s")
        key = VGroup(Square(0.2, stroke_width=0, fill_color=CZC, fill_opacity=1), caption("CZ", 20),
                     Rectangle(width=0.05, height=0.4, stroke_width=0, fill_color=WHITE, fill_opacity=1),
                     caption("single-qubit gate", 20), caption(f"{n} time steps; 4 of the 15 qubits shown", 20, DIM)).arrange(RIGHT, buff=0.2)
        key.next_to(wires, DOWN, buff=0.35).align_to(wires, LEFT)
        counter = caption(f"single-qubit gates in total: {st['raw']['sq']}", 28).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(sub), Create(wires), FadeIn(names), FadeIn(key))
        self.play(FadeIn(VGroup(*[m for r in czs for m in r.values()])))
        self.play(LaggedStart(*[FadeIn(m) for r in raw for m in r.values()], lag_ratio=0.002), FadeIn(counter))
        self.wait()

        self.next_section("merge")
        merged = self.rows(tl["merged"], "s")
        moves, gone = [], []
        for q, row in enumerate(tl["raw"]):
            for t, mob in raw[q].items():
                # the run's product lands on the run's last gate: the next merged gate before a CZ
                j = next((u for u in range(t, n) if tl["merged"][q][u] == "s" or row[u] == "c"), None)
                if j is not None and tl["merged"][q][j] == "s":
                    moves.append(mob.animate.move_to(merged[q][j]))
                else:
                    gone.append(FadeOut(mob))
        rule = caption("Each run between two CZs becomes one PhasedXZ gate, or vanishes if it does nothing", 22, EXC)
        rule.move_to(sub, aligned_edge=LEFT)
        self.play(FadeOut(sub), FadeIn(rule))
        self.play(*moves, *gone, run_time=2)
        self.remove(*[m for r in raw for m in r.values()])
        self.add(*[m for r in merged for m in r.values()])
        self.play(Transform(counter, caption(f"single-qubit gates in total: {st['raw']['sq']} → {st['merged']['sq']}",
                                             28, GOOD).move_to(counter)))
        self.wait()

        self.next_section("budget")
        keep = VGroup(wires, names, key, *[m for r in czs for m in r.values()], *[m for r in merged for m in r.values()])
        self.play(FadeOut(keep), FadeOut(rule), FadeOut(counter))
        scale = 10
        rows = VGroup()
        for k, name in (("raw", "as received"), ("merged", "merged"), ("dd", "merged, plus sparse XY4")):
            segs = VGroup(Rectangle(width=st[k]["sq_err"] * scale, height=0.5, stroke_width=0, fill_color=WHITE, fill_opacity=0.85),
                          Rectangle(width=st[k]["cz_err"] * scale, height=0.5, stroke_width=0, fill_color=CZC, fill_opacity=0.85)
                          ).arrange(RIGHT, buff=0)
            nums = caption(f"{st[k]['sq_err']:.2f} + {st[k]['cz_err']:.2f}", 20, DIM)
            rows.add(VGroup(caption(name, 24), segs, nums))
        rows.arrange(DOWN, buff=0.5).shift(DOWN * 0.1)
        for r in rows:
            r[1].move_to(LEFT * 1.6 + UP * r.get_y(), aligned_edge=LEFT)
            r[0].next_to(r[1], LEFT, buff=0.3).align_to(LEFT * 1.9, RIGHT)
            r[2].next_to(r[1], RIGHT, buff=0.3)
        top = VGroup(caption("Total error added up:", 24), caption("single-qubit", 24),
                     caption("+", 24), caption("CZ", 24, CZC)).arrange(RIGHT, buff=0.15).next_to(head, DOWN, buff=0.5)
        self.play(FadeIn(top))
        for r in rows:
            self.play(FadeIn(r[0]), GrowFromEdge(r[1], LEFT), FadeIn(r[2]), run_time=0.8)
        why = caption("When there's nothing to refocus, each extra pulse just adds its own error", 24, GOOD)
        why.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(why))
        self.wait()

        self.next_section("exact")
        self.play(FadeOut(VGroup(rows, top, why)))
        prod = MathTex(r"g_m\cdots g_2\,g_1 = U_{\text{run}}", r"\;\Rightarrow\;\text{one PhasedXZ}", font_size=44)
        prod.next_to(head, DOWN, buff=0.7).set_x(0)
        facts = VGroup(
            caption("It's exact: a run of gates and its product are the same operation", 24),
            caption("Nothing moves earlier, so no qubit leaves |0⟩ any sooner", 24),
            caption(f"The circuit can only get shorter (here it stays at {st['merged']['moments']} steps)", 24),
            caption("Gates saved:  D¹⁵₄ 777 → 276,  "
                    + ",  ".join(f"{c} {a} → {b}" for c, (a, b) in (("D¹⁰₅", d["others"]["D^{10}_5"]),
                                                                   ("D⁹₄", d["others"]["D^9_4"]))), 24),
            caption("For N ≥ 9 the score rises 1–11% with gate noise and 0–21% with idle noise", 24, GOOD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(prod, DOWN, buff=0.6)
        self.play(Write(prod))
        self.play(LaggedStart(*[FadeIn(f) for f in facts], lag_ratio=0.5))
        self.wait()

        self.next_section("dd")
        self.play(FadeOut(VGroup(prod, facts)))
        self.play(FadeIn(keep))
        dd = self.rows(tl["dd"], "s")
        add = [FadeIn(m) for q, r in enumerate(dd) for t, m in r.items() if tl["merged"][q][t] != "s"]
        drop = [FadeOut(m) for q, r in enumerate(merged) for t, m in r.items() if tl["dd"][q][t] != "s"]
        echo = VGroup(caption("Then a few XY4 echoes go back in, only in long idle gaps after a qubit's first gate", 22, EXC),
                      MathTex(r"XYXY=(iZ)^2=-I", font_size=32)).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        echo.next_to(head, DOWN, buff=0.2).align_to(head, LEFT)
        self.play(FadeIn(echo))
        self.play(*add, *drop, run_time=1.5)
        foot = VGroup(caption(f"That brings the count from {st['merged']['sq']} to {st['dd']['sq']}. "
                              "Without DD our simulator scores 1–5% higher,", 20, DIM),
                      caption("but it can't model the slow drift DD cancels on real hardware, "
                              "so we keep the echoes", 20, DIM)).arrange(DOWN, buff=0.1).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(foot))
        self.wait(2)


# ---- docs/main.tex: the bonus track ------------------------------------------------------


class BonusDepth(Beats):
    def construct(self):
        d = load("bonus.json")

        self.next_section("score")
        head = title("Bonus track: only circuit depth counts")
        self.play(Write(head))
        S = MathTex(r"\sum_{2\le N\le15,\ 1\le n\le N/2}\frac{N\,n}{\text{moments}}", font_size=48).shift(UP * 1.2)
        note = caption("56 cases, all scored without noise", 24, DIM).next_to(S, DOWN, buff=0.3)
        first = VGroup(caption(f"our first entry: {d['first']}", 28, BAD),
                       caption(f"the leader: {d['leader']:.2f}", 28)).arrange(RIGHT, buff=0.8).next_to(note, DOWN, buff=0.6)
        exact = caption("All 56 of ours were exact, so the whole gap was depth", 24).next_to(first, DOWN, buff=0.3)
        uni = caption("We kept it measurement-free: the official validator rejects classically controlled gates",
                      22, DIM).to_edge(DOWN, buff=0.5)
        self.play(Write(S), FadeIn(note))
        self.play(FadeIn(first))
        self.play(FadeIn(exact), FadeIn(uni))
        self.wait()

        self.next_section("anatomy")
        self.play(FadeOut(VGroup(S, note, first, exact, uni)))
        tot, alone = d["d157"]["moments"], d["d157"]["arm_alone"]
        w = 11.0 / tot
        rest = Rectangle(width=(tot - alone) * w, height=0.7, stroke_width=0, fill_color=CZC, fill_opacity=0.85)
        arm = Rectangle(width=alone * w, height=0.7, stroke_width=0, fill_color=BAD, fill_opacity=0.85)
        bar = VGroup(rest, arm).arrange(RIGHT, buff=0).shift(UP * 0.6)
        lead = MathTex(r"D^{15}_7\text{ splits into arms of 7 and 8 qubits, each built by a B-E sweep}", font_size=34)
        lead.next_to(bar, UP, buff=0.8)
        br = Brace(arm, DOWN)
        btxt = caption(f"{alone} of the {tot} steps are the 8-qubit arm working alone", 24, BAD).next_to(br, DOWN)
        rtxt = caption("everything else", 20).next_to(rest, UP, buff=0.1)
        cost = caption("Each arm position costs about 18 steps at k ≥ 5 (10 at k = 3, 5 at k = 2)", 22, DIM).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(lead))
        self.play(GrowFromEdge(rest, LEFT), FadeIn(rtxt))
        self.play(GrowFromEdge(arm, LEFT), run_time=1.5)
        self.play(Create(br), FadeIn(btxt))
        self.play(FadeIn(cost))
        self.wait()

        self.next_section("isometry")
        self.play(FadeOut(VGroup(bar, lead, br, btxt, rtxt, cost)))
        m0, fixed = d["m0"], set(d["fixed_columns"])
        iso = MathTex(r"U_{m_0}:\ \ket{0^{m_0-r}1^r}\ \mapsto\ \ket{D^{m_0}_r},\qquad r\le\min(k,m_0)",
                      font_size=38).next_to(head, DOWN, buff=0.45).set_x(0)
        sub = caption("The last m₀ positions of a sweep only ever receive these inputs", 22, DIM).next_to(iso, DOWN, buff=0.2)
        dim = 2 ** m0
        cw = 9.0 / dim
        cols = VGroup(*[Rectangle(width=cw * 0.8, height=2.6, stroke_width=0,
                                  fill_color=GOOD if j in fixed else DIM, fill_opacity=1.0 if j in fixed else 0.25)
                        for j in range(dim)]).arrange(RIGHT, buff=cw * 0.2).shift(DOWN * 0.6)
        ctag = caption(f"For U₇ with r ≤ 2, only {len(fixed)} of the {dim} columns matter; the rest are free",
                       24, GOOD).next_to(cols, DOWN, buff=0.3)
        self.play(Write(iso), FadeIn(sub))
        self.play(FadeIn(cols))
        self.play(*[Indicate(cols[j], color=GOOD, scale_factor=1.5) for j in fixed], FadeIn(ctag))
        self.wait(0.5)
        loss = MathTex(r"1-\Big(\tfrac1B\sum_b\langle\mathrm{out}_b|\,U\,|\mathrm{in}_b\rangle\Big)^2", font_size=34)
        ltag = caption("is zero only if every column is right, all with one sign", 20)
        lg = VGroup(loss, ltag).arrange(RIGHT, buff=0.4).to_edge(DOWN, buff=0.35)
        self.play(FadeOut(ctag), FadeIn(lg))
        self.wait()

        self.next_section("results")
        self.play(FadeOut(VGroup(iso, sub, cols, lg)))
        blocks = VGroup()
        for b in d["blocks"]:
            sw = Rectangle(width=b["sweep"] * 0.05, height=0.4, stroke_width=0, fill_color=BAD, fill_opacity=0.85)
            bl = Rectangle(width=b["block"] * 0.05, height=0.4, stroke_width=0, fill_color=GOOD, fill_opacity=0.85)
            pair = VGroup(VGroup(sw, Text(f"sweep {b['sweep']}", font_size=20).next_to(sw, RIGHT, buff=0.15)),
                          VGroup(bl, Text(f"searched-for block {b['block']}", font_size=20).next_to(bl, RIGHT, buff=0.15)))
            pair.arrange(DOWN, aligned_edge=LEFT, buff=0.1)
            blocks.add(VGroup(MathTex(b["name"], font_size=30), pair).arrange(RIGHT, buff=0.3))
        blocks.arrange(DOWN, aligned_edge=LEFT, buff=0.35).to_edge(LEFT, buff=0.5).shift(UP * 0.9)
        for row in blocks:
            row[1].align_to(blocks[0][1], LEFT)
        btitle = caption("time steps", 22, DIM).next_to(blocks, UP, buff=0.2).align_to(blocks, LEFT)
        cases = VGroup(*[MathTex(rf"{c}:\ {a}\to{b}", font_size=32) for c, a, b in d["cases"]])
        cases.arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        ctitle = caption("time steps, first entry → now", 22, DIM)
        VGroup(ctitle, cases).arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_edge(RIGHT, buff=0.4).match_y(blocks)
        self.play(FadeIn(btitle), LaggedStart(*[FadeIn(r) for r in blocks], lag_ratio=0.4))
        self.play(FadeIn(ctitle), LaggedStart(*[FadeIn(c) for c in cases], lag_ratio=0.2))
        score = VGroup(MathTex(rf"{d['first']}\ \to\ ", font_size=44),
                       MathTex(rf"{d['now']}", font_size=52, color=GOOD)).arrange(RIGHT, buff=0.1)
        score.to_edge(DOWN, buff=1.0)
        ctx = VGroup(caption(f"Our teammates' entry scores {d['teammates']} on the same scorer; "
                             f"the first-round leader had {d['leader']:.2f}", 20, DIM),
                     caption(f"Still open: the {d['remaining']['cases']} cases with k ≥ 5, worth "
                             f"{d['remaining']['points']:.1f} of the {d['now']} points", 20, DIM)
                     ).arrange(DOWN, buff=0.08).next_to(score, DOWN, buff=0.2)
        self.play(FadeIn(score[0]))
        self.play(FadeIn(score[1], scale=1.3), FadeIn(ctx))
        self.wait(2)
