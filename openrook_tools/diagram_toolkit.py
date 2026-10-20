"""Small, deterministic SVG primitives for the documentation diagrams.

Paint and fonts come from :mod:`openrook_tools.brand`. The canvas is transparent;
only nodes and label masks have page fills. Connectors are orthogonal elbows
or true circular arcs, and role treatments carry meaning without a rainbow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, cos, pi, sin, sqrt
from xml.sax.saxutils import escape

from openrook_tools.brand import DIAGRAM_SERIF, LIGHT, MONO, SANS, Theme


def _rounded_path(points: list[tuple[float, float]]) -> str:
    """Orthogonal route with eight-pixel corner radii (short segments clamp)."""
    if len(points) < 2:
        raise ValueError("A connector needs at least two points")
    for a, b in zip(points, points[1:]):
        if a[0] != b[0] and a[1] != b[1]:
            raise ValueError("Diagram connectors must be orthogonal")
    d = f"M{points[0][0]:.3f},{points[0][1]:.3f}"
    for a, b, c in zip(points, points[1:], points[2:]):
        incoming = abs(b[0] - a[0]) + abs(b[1] - a[1])
        outgoing = abs(c[0] - b[0]) + abs(c[1] - b[1])
        radius = min(8, incoming / 2, outgoing / 2)
        if not incoming or not outgoing:
            continue
        p = tuple(b[i] + (a[i] - b[i]) * radius / incoming for i in (0, 1))
        q = tuple(b[i] + (c[i] - b[i]) * radius / outgoing for i in (0, 1))
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        d += f" L{p[0]:.3f},{p[1]:.3f} A{radius},{radius} 0 0 {int(cross > 0)} {q[0]:.3f},{q[1]:.3f}"
    return d + f" L{points[-1][0]:.3f},{points[-1][1]:.3f}"


def loop_geometry(cx: float, cy: float, radius: float, count: int, w: int, h: int):
    """Grid-snapped stations and exact entry/exit angles on one shared circle."""
    if not 5 <= count <= 8:
        raise ValueError("A loop needs five to eight stations")
    boxes, angles = [], []
    for k in range(count):
        theta = -pi / 2 + k * 2 * pi / count
        x = round((cx + radius * cos(theta) - w / 2) / 4) * 4
        y = round((cy + radius * sin(theta) - h / 2) / 4) * 4
        boxes.append((x, y, w, h))
        hits = []
        for edge in (x, x + w):
            delta = radius**2 - (edge - cx)**2
            if delta >= 0:
                for sy in (-1, 1):
                    yy = cy + sy * sqrt(delta)
                    if y <= yy <= y + h:
                        hits.append(atan2(yy - cy, edge - cx))
        for edge in (y, y + h):
            delta = radius**2 - (edge - cy)**2
            if delta >= 0:
                for sx in (-1, 1):
                    xx = cx + sx * sqrt(delta)
                    if x <= xx <= x + w:
                        hits.append(atan2(edge - cy, xx - cx))
        relative = sorted((a - theta + pi) % (2 * pi) - pi for a in hits)
        if len(relative) != 2:
            raise ValueError("Each station must interrupt the ring at two edges")
        angles.append((theta + relative[0], theta + relative[1]))
    return boxes, angles


@dataclass
class Canvas:
    width: int
    height: int
    title: str
    subtitle: str
    theme: Theme = LIGHT
    slug: str = "diagram"
    kind: str = "process"
    pieces: list[str] = field(default_factory=list)

    def add(self, svg: str) -> None:
        self.pieces.append(svg)

    def text(self, x, y, value, *, size=18, weight=400, fill=None, anchor="start", font=SANS):
        self.add(
            f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill or self.theme.ink}">'
            f'{escape(str(value))}</text>'
        )

    def rect(self, x, y, w, h, *, fill="none", stroke="none", radius=6, dashed=False, sw=1.5, extra=""):
        dash = ' stroke-dasharray="6 5"' if dashed else ""
        self.add(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash}{extra}/>'
        )

    def line(self, x1, y1, x2, y2, *, color=None, sw=1):
        if x1 != x2 and y1 != y2:
            raise ValueError("Lines must be orthogonal")
        end = f"H{x2}" if y1 == y2 else f"V{y2}"
        self.add(f'<path d="M{x1},{y1} {end}" fill="none" '
                 f'stroke="{color or self.theme.border}" stroke-width="{sw}"/>')

    def region(self, x, y, w, h, label, *, focal=False, dashed=True):
        self.rect(x, y, w, h, stroke=self.theme.amber_ink if focal else self.theme.border,
                  radius=8, dashed=dashed, extra=' data-role="boundary"')
        self.text(x + 20, y + 30, label, size=16, weight=600,
                  fill=self.theme.amber_ink if focal else self.theme.muted)

    def node(self, x, y, w, h, title, sub="", *, role="step", focal=False, number=None, mono=False, size=21):
        t = self.theme
        self.add(f'<g data-node="{escape(title)}" data-role="{role}" data-focal="{str(focal).lower()}">')
        self.rect(x, y, w, h, fill=t.page, radius=6)
        if focal:
            self.rect(x, y, w, h, fill=t.amber, radius=6, extra=' fill-opacity="0.10"')
        if role == "input":
            self.line(x, y + h, x + w, y + h, color=t.muted)
        else:
            self.rect(x, y, w, h, stroke=t.amber_ink if focal else t.border,
                      dashed=role in ("optional", "external"), radius=6)
        if role == "store":
            self.line(x + 12, y + 8, x + w - 12, y + 8, color=t.border)
        tx = x + 20
        if number:
            self.text(tx, y + 22, number, size=12, weight=600, fill=t.muted)
            ty = y + 48
        else:
            ty = y + h / 2 - (4 if sub else -7)
        self.text(tx, ty, title, size=size, weight=600,
                  fill=t.amber_ink if focal else t.ink, font=MONO if mono else SANS)
        if sub:
            self.text(tx, ty + 25, sub, size=16, fill=t.muted)
        self.add('</g>')

    def path(self, d, *, dashed=False, arrow=True, accent=False, extra=""):
        color = self.theme.amber_ink if accent else self.theme.muted
        marker = f' marker-end="url(#{self.slug}-{self.theme.name}-arrow{"-accent" if accent else ""})"' if arrow else ""
        dash = ' stroke-dasharray="5 5"' if dashed else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.6"{dash}{marker}{extra}/>')

    def connector(self, points, *, dashed=False, accent=False, arrow=True):
        self.path(_rounded_path(points), dashed=dashed, accent=accent, arrow=arrow,
                  extra=' data-connector="orthogonal"')

    def label(self, x, y, value, *, size=12, anchor="middle"):
        """Opaque arrow-label mask; y is the baseline, caller reserves an 8px gap."""
        w = len(value) * size * .65 + 12
        left = x - w / 2 if anchor == "middle" else x - 6
        self.rect(left, y - size - 3, w, size + 8, fill=self.theme.page, radius=2,
                  extra=' data-role="label-mask"')
        self.text(x, y, value, size=size, fill=self.theme.muted, weight=600, anchor=anchor)

    def footer(self, text, *, second=""):
        y = self.height - (76 if second else 52)
        self.line(48, y - 20, self.width - 48, y - 20)
        self.text(48, y + 8, text, size=16, fill=self.theme.muted)
        if second:
            self.text(48, y + 34, second, size=16, fill=self.theme.muted)

    def to_svg(self) -> str:
        slug = f"{self.slug}-{self.theme.name}"
        defs = ''.join(
            f'<marker id="{slug}-arrow{suffix}" markerWidth="8" markerHeight="8" refX="8" refY="4" orient="auto" markerUnits="userSpaceOnUse">'
            f'<path d="M0,0 L8,4 L0,8 Z" fill="{color}"/></marker>'
            for suffix, color in (("", self.theme.muted), ("-accent", self.theme.amber_ink))
        )
        heading = Canvas(self.width, self.height, "", "", self.theme)
        heading.text(48, 36, f"OPENROOK  /  {self.kind.upper()}", size=13, weight=600, fill=self.theme.muted)
        heading.text(48, 84, self.title, size=38, font=DIAGRAM_SERIF)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}" role="img" aria-labelledby="{slug}-title {slug}-desc">\n'
            f'<title id="{slug}-title">{escape(self.title)}</title>\n'
            f'<desc id="{slug}-desc">{escape(self.subtitle)}</desc>\n'
            f'<defs>{defs}</defs>\n' + '\n'.join(heading.pieces + self.pieces) + '\n</svg>\n'
        )
