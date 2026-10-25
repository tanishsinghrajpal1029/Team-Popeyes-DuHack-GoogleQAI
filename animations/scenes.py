"""Manim scenes for the four ideas in docs/writeup/main.tex.

  manim -ql animations/scenes.py DickeIntro            # quick preview
  animations/render.sh                                 # every scene, talk + README formats

Every number on screen comes from animations/data/*.json (written by animations/prep.py)
or, where noted, from the writeup's tables. Each scene is split into sections (beats);
render with --save_sections to get one clip per slide click.
"""

import math
import os
import sys

import numpy as np
from manim import (
    DOWN, LEFT, RIGHT, UP, WHITE, Arrow, Axes, Create, Cross, DashedLine,
    FadeIn, FadeOut, Flash, GrowFromEdge, Indicate, LaggedStart, Line, MathTex, Polygon,
    Rectangle, ReplacementTransform, Scene, Square, Succession, Swap, Text, Transform,
    ValueTracker, VGroup, Wait, Write, always_redraw, AnimationGroup, Circumscribe, Dot,
    SurroundingRectangle, interpolate_color, rate_functions,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    BAD, CZC, DIM, EXC, GOOD, Chip, bitstring, caption, err_color, load, node, share_bar, title,
)


# ---- Background ------------------------------------------------------------------------


class DickeIntro(Scene):
    def construct(self):
        self.next_section("strings")
        head = title("A Dicke state: every string with exactly n ones")
        strings = [format(i, "04b") for i in range(16)]
        grid = VGroup(*[bitstring(s, 30) for s in strings]).arrange_in_grid(4, 4, buff=(0.9, 0.45))
        note = caption("N = 4 qubits: 16 bit strings").next_to(grid, DOWN, buff=0.6)
        self.play(Write(head))
        self.play(LaggedStart(*[FadeIn(m) for m in grid], lag_ratio=0.04), FadeIn(note))
        valid = [i for i, s in enumerate(strings) if s.count("1") == 2]
        self.play(*[grid[i].animate.set_opacity(0.15) for i in range(16) if i not in valid],
                  Transform(note, caption("keep the strings with n = 2 ones").move_to(note)))
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
        sym1 = caption("1. Flip every bit:  n ones become N − n ones", 28).to_edge(UP, buff=1.3)
        self.play(FadeIn(sym1))
        flipped = VGroup(*[bitstring("".join("1" if c == "0" else "0" for c in s), 30).move_to(m)
                           for s, m in zip(w_strings, w)])
        flip_eq = MathTex(r"X^{\otimes N}\ket{D^N_n} = \ket{D^N_{N-n}}", font_size=36).next_to(w, DOWN, buff=0.5)
        self.play(Transform(w, flipped), Write(flip_eq))
        self.wait(0.5)
        self.play(Transform(w, VGroup(*[bitstring(s, 30).move_to(m) for s, m in zip(w_strings, w)])))
        sym2 = caption("2. Swap two qubits:  the same set of strings", 28).next_to(flip_eq, DOWN, buff=0.6)
        self.play(FadeIn(sym2))
        self.play(*[Swap(m[0], m[3]) for m in w], run_time=1.5)
        free = caption("so we only build n ≤ N/2, and may place qubits anywhere on the chip",
                       26, GOOD).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(free))
        self.wait(2)


# ---- Idea 1: the score ------------------------------------------------------------------


class ScoreTail(Scene):
    def construct(self):
        d = load("score.json")

        self.next_section("formula")
        head = title("Idea 1: the score rewards small, balanced cases")
        S = MathTex(r"S = \frac{1}{z + x}", font_size=60).shift(UP * 1.2)
        zt = VGroup(MathTex(r"z = \frac{\chi^2}{K f}", font_size=40),
                    caption("are all K valid strings equally likely?", 24)).arrange(DOWN)
        xt = VGroup(MathTex(r"x", font_size=40), caption("are the signs right?", 24)).arrange(DOWN)
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
            caption("lucky draw: z < 0.05, so S > 20", 26),
            caption(f"K = 2:  {100 * p2:.0f}% of runs", 26, EXC),
            caption(f"K = 20:  {p20:.0e} of runs", 26, CZC),
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
        ylab = caption("expected score E[S]  (log scale)", 22).next_to(bx, UP, buff=0.2).align_to(bx, LEFT)
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
        note = caption("perfect state, f = 0.9, x = 0.001, 10,000 shots", 22, DIM).to_edge(DOWN, buff=0.3)
        bx.x_axis.set_opacity(0)  # it would sit at log10(1); draw a floor and a dashed S = 1 instead
        floor = Line(bx.c2p(0, lo), bx.c2p(len(Ks), lo), color=WHITE, stroke_width=2)
        one = DashedLine(bx.c2p(0, 0), bx.c2p(len(Ks), 0), color=DIM, stroke_width=2)
        self.play(Create(bx), Create(floor), Create(one), FadeIn(yticks), FadeIn(ylab), FadeIn(xlab))
        self.play(FadeIn(live), FadeIn(note))
        fifty = caption("one K = 2 case ≈ fifty large cases", 28, EXC).to_edge(RIGHT, buff=0.5).shift(UP * 1.2)
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
        take = caption("Balance first, then gate count. A known bias can't be fixed by tilting amplitudes.",
                       26, GOOD).to_edge(DOWN, buff=0.3)
        self.play(FadeOut(note), FadeIn(take))
        self.wait(2)


# ---- Idea 2: the W tree -----------------------------------------------------------------


def _fill(circle, share):
    """Probability held by a node shown as fill; sqrt so 1/N stays visible."""
    return circle.animate.set_fill(EXC, opacity=math.sqrt(share))


class WTree(Scene):
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
        share_note = caption("one excitation; finish with 1/4 on every qubit", 24, DIM).to_edge(DOWN, buff=0.4)
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
            caption("first transfer: 1 CZ,  every later one: 2 CZ  →  2N − 3 CZ either way", 24),
            caption("the tree runs a round's transfers at the same time", 24, GOOD),
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
        ltitle = caption(f"tree: W₁₀ on Willow, {d['tree_cz']} CZ", 26).next_to(left, UP, buff=0.3)
        rtitle = caption(f"line: W₁₀ on Willow, {d['chain_cz']} CZ", 26).next_to(right, UP, buff=0.3)
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
            caption(f"X term with idle noise:  line {d['chain_x']:.4f}  →  tree {d['tree_x']:.4f}", 26, GOOD),
            caption("one excitation in every branch: decay lowers all strings equally, no imbalance", 22, DIM),
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


class Synthesis(Scene):
    def construct(self):
        d = load("synth.json")
        N = d["N"]

        self.next_section("template")
        head = title("Idea 3: let a computer search for exact circuits")
        self.play(Write(head))
        what = MathTex(r"\text{target: }\ket{D^4_2}\qquad \text{template: } R_y \text{ layers} + \text{CZ between neighbours}",
                       font_size=32).next_to(head, DOWN, buff=0.35).to_edge(LEFT, buff=0.5)
        self.play(FadeIn(what))
        log = VGroup()
        circ = None
        for i, row in enumerate(d["depth_log"]):
            from_layers = [list(map(tuple, l)) for l in (d["start_layers"][:row["L"]])]
            new = CircuitDiagram(N, from_layers, width=1.9 * row["L"] + 1.0).shift(LEFT * 3 + DOWN * 0.8)
            ok = row["infidelity"] < 1e-13
            line = caption(f"{row['L']} CZ layer{'s' if row['L'] > 1 else ''}:  1 − F = "
                           + ("0  (exact)" if ok else f"{row['infidelity']:.3f}"), 24, GOOD if ok else WHITE)
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
        xl = caption("gradient steps", 20).next_to(ax, DOWN, buff=0.15)
        tl = caption("infidelity 1 − F", 20).next_to(ax, UP, buff=0.15)
        graph = ax.plot_line_graph(list(range(len(loss))), [math.log10(v) for v in loss],
                                   add_vertex_dots=False, line_color=GOOD, stroke_width=4)
        tol = DashedLine(ax.c2p(0, -13), ax.c2p(len(loss), -13), color=DIM)
        toltag = caption("exact: below 10⁻¹³", 18, DIM).next_to(tol, UP, buff=0.05).align_to(tol, RIGHT)
        self.play(FadeOut(log), FadeIn(ax), FadeIn(yl), FadeIn(xl), FadeIn(tl), Create(tol), FadeIn(toltag))
        self.play(Create(graph), run_time=2.5)
        self.wait()

        self.next_section("prune")
        self.play(FadeOut(VGroup(ax, yl, xl, tl, graph, tol, toltag)))
        count = sum(map(len, d["start_layers"]))
        counter = caption(f"CZ: {count}", 34).move_to(RIGHT * 4.3 + UP * 0.5)
        rule = caption("delete a CZ, re-optimize,\nkeep the deletion only\nif still exact", 22, DIM).next_to(counter, DOWN, buff=0.4)
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
                why = caption(f"1 − F = {st['infidelity']:.2f}", 20, BAD).next_to(g, UP, buff=0.1)
                self.play(FadeIn(x), FadeIn(why), run_time=0.4)
                self.play(FadeOut(x), FadeOut(why), g.animate.set_color(CZC), run_time=0.4)
        done = caption(f"{count} CZ, exact to\nmachine precision", 26, GOOD).next_to(rule, DOWN, buff=0.4)
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
        legend = VGroup(caption("textbook (Bärtschi–Eidenbenz, routed)", 22, BAD),
                        caption("computer search", 22, GOOD)).arrange(DOWN, aligned_edge=LEFT)
        legend.next_to(bx.c2p(0.2, 165), RIGHT).shift(DOWN * 0.2)
        ylab = caption("CZ count", 22).next_to(bx, UP, buff=0.1).align_to(bx, LEFT)
        self.play(Create(bx), FadeIn(ylab))
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.2) for b in bars], lag_ratio=0.05), FadeIn(legend))
        foot = caption("search stops scaling near N = 9", 22, DIM).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(foot))
        self.wait(2)


# ---- Idea 4: placement ------------------------------------------------------------------


class Placement(Scene):
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
            caption("Willow, calibration seed 0", 26),
            caption(f"CZ error   best {100 * lo:.2f}%", 22, GOOD),
            caption(f"median {100 * float(np.median(errs)):.2f}%", 22, EXC),
            caption(f"worst {100 * hi:.2f}%", 22, BAD),
            caption("grey: dropped out", 22, DIM),
        )
        self.play(LaggedStart(*recolor, lag_ratio=0.005), FadeIn(legend))
        self.wait()

        self.next_section("beam")
        why = self.column(
            caption("cost of a placement", 26),
            MathTex(r"f \approx \prod (1 - e)\;\Rightarrow\; -\log f \approx \sum e", font_size=32),
            caption("sum of the error of every gate,\nidle period and readout it uses", 22, DIM),
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
            step = caption(f"beam search: the best partial\npaths of length {L}", 24, EXC)
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
        verdict = caption(f"best 6-qubit path: cost {d['best_path_cost']:.4f}\n"
                          f"median of the kept beam: {d['median_beam_cost']:.4f}", 24, GOOD)
        verdict.move_to(label, aligned_edge=UP + LEFT)
        self.play(FadeOut(ghosts), FadeIn(path), ReplacementTransform(label, verdict))
        self.wait()

        self.next_section("readout")
        intro = self.column(
            caption("matching readout errors", 26),
            caption("each qubit misreads 1 as 0 at its own rate,\n"
                    "so |10⟩ and |01⟩ are read correctly unequally", 20, DIM),
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
        hint = caption("P(read correctly), axis from 0.95", 18, DIM)
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
            caption("W₂: simulate every pair exactly", 26),
            caption("couplers coloured by expected score", 20, DIM),
            caption(f"worst pair   E[S] = {lo_e:.1f}", 24, BAD),
            caption(f"median pair  E[S] = {med:.1f}", 24, EXC),
            caption(f"best pair    E[S] = {hi_e:.1f}", 24, GOOD),
            caption(f"{hi_e / med:.1f}× the median, from placement", 24, GOOD),
        )
        self.play(FadeOut(intro), FadeOut(panels), FadeOut(hint), FadeOut(rings))
        self.play(LaggedStart(*recolor, lag_ratio=0.005), FadeIn(tag))
        a, b = (tuple(q) for q in d["w2_picked"])
        pick = VGroup(Line(chip.dots[a].get_center(), chip.dots[b].get_center(), color=GOOD, stroke_width=10),
                      *[Dot(chip.dots[q].get_center(), radius=0.11, color=GOOD) for q in (a, b)])
        self.play(FadeIn(pick), Flash(pick.get_center(), color=GOOD, flash_radius=0.4), run_time=1.2)
        self.wait(2)
