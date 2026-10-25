"""Shared palette, TeX macros and drawing helpers for animations/scenes.py.

Each colour means one thing in every scene: an excitation (a "1") is YELLOW, a CZ / coupler
is BLUE, good / exact / best is GREEN, error / imbalance / the baseline is RED, inactive is
GREY. Background is manim's default black.
"""

import json
import math
import os

from manim import (
    BLUE, DOWN, GREEN, GREY_B, GREY_D, LEFT, RED, RIGHT, UL, UP, WHITE, YELLOW,
    Circle, Dot, Line, Rectangle, TexTemplate, Text, VGroup, config, interpolate_color,
)

HERE = os.path.dirname(os.path.abspath(__file__))

EXC, CZC, GOOD, BAD, DIM = YELLOW, BLUE, GREEN, RED, GREY_B
MONO = "DejaVu Sans Mono"

tex = TexTemplate()
tex.add_to_preamble(r"\newcommand{\ket}[1]{\lvert #1\rangle}")
config.tex_template = tex


def load(name: str) -> dict:
    with open(os.path.join(HERE, "data", name)) as fh:
        return json.load(fh)


def title(text: str) -> Text:
    return Text(text, font_size=34, weight="BOLD").to_corner(UL, buff=0.4)


def caption(text: str, size: int = 24, color=WHITE) -> Text:
    return Text(text, font_size=size, color=color)


def bitstring(bits: str, size: int = 32) -> Text:
    """A bit string with its 1s in the excitation colour."""
    t = Text(bits, font=MONO, font_size=size)
    for ch, glyph in zip(bits, t):
        glyph.set_color(EXC if ch == "1" else WHITE)
    return t


def share_bar(height: float, width: float = 0.35, max_height: float = 1.6, color=EXC) -> VGroup:
    """A vertical bar of `height` (0..1) inside an outline of `max_height`, bottom-aligned."""
    frame = Rectangle(width=width, height=max_height, stroke_color=GREY_D, stroke_width=2)
    fill = Rectangle(width=width, height=max(height * max_height, 1e-3), stroke_width=0,
                     fill_color=color, fill_opacity=0.9)
    fill.align_to(frame, DOWN)
    return VGroup(frame, fill)


def err_color(err: float, lo: float, hi: float):
    """Green (low error) -> yellow -> red (high error) on a log scale."""
    t = (math.log(err) - math.log(lo)) / (math.log(hi) - math.log(lo))
    t = min(max(t, 0.0), 1.0)
    return interpolate_color(GOOD, EXC, 2 * t) if t < 0.5 else interpolate_color(EXC, BAD, 2 * t - 1)


class Chip(VGroup):
    """Willow qubits as dots and couplers as lines, laid out on their grid coordinates."""

    def __init__(self, layout: dict, spacing: float = 0.55, qubits=None, **kw):
        super().__init__(**kw)
        keep = None if qubits is None else {tuple(q) for q in qubits}
        self.spacing = spacing
        self.dots, self.lines = {}, {}
        usable = {tuple(q) for q in layout["usable"]}
        for a, b, err in layout["couplers"]:
            a, b = tuple(a), tuple(b)
            if keep is not None and not (a in keep and b in keep):
                continue
            line = Line(self.pos(a), self.pos(b), stroke_width=3,
                        color=GREY_D if err is None else DIM, stroke_opacity=0.35 if err is None else 0.8)
            self.lines[frozenset((a, b))] = line
        for q in layout["qubits"]:
            q = tuple(q)
            if keep is not None and q not in keep:
                continue
            dot = Dot(self.pos(q), radius=0.07, color=WHITE if q in usable else GREY_D)
            self.dots[q] = dot
        self.add(*self.lines.values(), *self.dots.values())

    def pos(self, q):
        r, c = q
        return RIGHT * c * self.spacing + DOWN * r * self.spacing

    def line(self, a, b):
        return self.lines[frozenset((tuple(a), tuple(b)))]


def node(label: str = "", radius: float = 0.3, color=WHITE) -> VGroup:
    c = Circle(radius=radius, color=color, stroke_width=3)
    t = Text(label, font_size=24).move_to(c) if label else VGroup()
    return VGroup(c, t)
