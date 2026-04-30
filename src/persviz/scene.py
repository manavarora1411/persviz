"""
Manim scene: signed-EDT superlevel filtration, production quality v3+.

Layout (16:9, 1920 x 1080):
  +----------------+-----------------+
  |  porous slice  |   EDT field +   |
  |  + superlevel  |   colorbar      |
  +----------------+-----------------+
  |  3D heightmap  |  persistence    |
  |  + sweep plane |  diagram        |
  +----------------+-----------------+

References
----------
- Edelsbrunner, Letscher, Zomorodian (2002). Topological persistence and
  simplification. DCG 28, 511-533.
- Robins, Wood, Sheppard (2011). IEEE PAMI 33(8), 1646-1658.
"""
from __future__ import annotations

import os

import numpy as np
from manim import (
    DEGREES,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    WHITE,
    DashedLine,
    DecimalNumber,
    Dot,
    ImageMobject,
    Line,
    MathTex,
    Polygon,
    Scene,
    ValueTracker,
    VGroup,
    always_redraw,
    config,
)
from manim.utils.rate_functions import linear

from persviz.data import load_image_or_npy, synthetic_phantom
from persviz.edt import signed_edt
from persviz.pairing import paired_features

config.background_color = "#0F1318"
config.frame_width = 16
config.frame_height = 9

GOLD = "#F4D35E"
GOLD_DIM = "#8A7A3D"
PORE_RGB = (200, 184, 154)
SOLID_RGB = (42, 49, 64)
SUPERLEVEL_RGB = (244, 211, 94)
WHITE_DIM = "#B8C2CC"
WHITE_FAINT = "#3A4250"
H0_COLOR = "#5DADE2"
H1_COLOR = "#E97A6F"


def _rasterize_field(field):
    f = field.astype(float)
    f = (f - f.min()) / (f.max() - f.min() + 1e-12)
    lo = np.array([31, 37, 48], dtype=float)
    mid = np.array([78, 60, 90], dtype=float)
    hi = np.array([244, 211, 94], dtype=float)
    rgb = np.where(
        f[..., None] < 0.5,
        lo + (mid - lo) * (f[..., None] * 2),
        mid + (hi - mid) * ((f[..., None] - 0.5) * 2),
    )
    return rgb.astype(np.uint8)


def _rasterize_binary_with_superlevel(binary, phi, t):
    h, w = binary.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[binary == 1] = SOLID_RGB
    img[binary == 0] = PORE_RGB
    sup = (phi >= t) & (binary == 0)
    img[sup] = SUPERLEVEL_RGB
    return img


def _nice_ticks(lo, hi, n=4):
    span = hi - lo
    raw_step = span / n
    mag = 10 ** np.floor(np.log10(raw_step))
    norm = raw_step / mag
    if norm < 1.5:
        step = 1 * mag
    elif norm < 3:
        step = 2 * mag
    elif norm < 7:
        step = 5 * mag
    else:
        step = 10 * mag
    start = np.ceil(lo / step) * step
    ticks = []
    v = start
    while v <= hi + 1e-9:
        ticks.append(round(float(v), 6))
        v += step
    return ticks


def _color_for_height(z_norm):
    z_norm = max(0.0, min(1.0, z_norm))
    if z_norm < 0.5:
        a = z_norm * 2
        c = np.array([31, 37, 48]) + a * (np.array([92, 74, 110]) - np.array([31, 37, 48]))
    else:
        a = (z_norm - 0.5) * 2
        c = np.array([92, 74, 110]) + a * (np.array([244, 211, 94]) - np.array([92, 74, 110]))
    return "#{:02x}{:02x}{:02x}".format(*c.astype(int))


class FiltrationScene(Scene):
    PHANTOM_SIZE = 128
    SEED = 0
    PERSISTENCE_THRESHOLD = 1.0
    SWEEP_DURATION = 22.0
    HOLD_AT_END = 2.5
    SURFACE_DOWNSAMPLE = 4

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.npy_path = os.environ.get("PERSVIZ_INPUT")

    def construct(self):
        if self.npy_path:
            binary = load_image_or_npy(self.npy_path, target_size=self.PHANTOM_SIZE)
        else:
            binary = synthetic_phantom(size=self.PHANTOM_SIZE, seed=self.SEED)

        phi = signed_edt(binary)
        feats_all = paired_features(phi)
        feats = [
            f for f in feats_all
            if not np.isfinite(f.death) or f.persistence > self.PERSISTENCE_THRESHOLD
        ]

        t_max = float(phi.max()) + 0.5
        t_min = float(phi.min()) - 0.5
        t_tracker = ValueTracker(t_max)

        title = MathTex(r"X_t = \{\,x : \phi(x) \geq t\,\}", color=WHITE).scale(0.7)
        title.to_edge(UP, buff=0.2)
        self.add(title)

        t_label = VGroup(
            MathTex("t", "=", color=WHITE_DIM).scale(0.65),
            DecimalNumber(t_max, num_decimal_places=2, color=GOLD, font_size=38),
        ).arrange(RIGHT, buff=0.14)
        t_label.next_to(title, DOWN, buff=0.12)
        t_label[1].add_updater(lambda m: m.set_value(t_tracker.get_value()))
        self.add(t_label)

        panel_w = 6.4
        panel_h = 3.1
        tl_center = np.array([-3.95, 1.7, 0])
        tr_center = np.array([3.95, 1.7, 0])
        bl_center = np.array([-3.95, -2.4, 0])
        br_center = np.array([3.95, -2.4, 0])

        self._build_slice_panel(binary, phi, tl_center, panel_h, t_tracker, feats)
        self._build_edt_panel(phi, tr_center, panel_h, t_tracker)
        self._build_3d_panel(phi, bl_center, panel_w, panel_h, t_tracker)
        diag_axes = self._build_diagram_panel(feats, br_center, panel_w, panel_h, t_tracker)

        self._setup_pairing_traces(feats, t_tracker, diag_axes)

        self.wait(0.6)
        self.play(
            t_tracker.animate.set_value(t_min),
            run_time=self.SWEEP_DURATION,
            rate_func=linear,
        )
        self.wait(self.HOLD_AT_END)

    def _build_slice_panel(self, binary, phi, center, panel_h, t_tracker, feats):
        H, W = phi.shape
        label = MathTex(r"\text{slice} + X_t", color=WHITE_DIM).scale(0.46)
        label.move_to(center + UP * (panel_h / 2 + 0.22))
        self.add(label)

        slice_img = ImageMobject(_rasterize_binary_with_superlevel(binary, phi, float(phi.max()) + 1))
        slice_img.set_resampling_algorithm(0)
        slice_img.height = panel_h
        slice_img.move_to(center)

        def update_slice(img):
            new_arr = _rasterize_binary_with_superlevel(binary, phi, t_tracker.get_value())
            img.pixel_array = np.dstack([
                new_arr,
                np.full(new_arr.shape[:2] + (1,), 255, dtype=np.uint8),
            ])

        slice_img.add_updater(update_slice)
        self.add(slice_img)

        pixel_size = panel_h / H

        def cell_to_point(row, col):
            x = center[0] - panel_h / 2 + (col + 0.5) * pixel_size
            y = center[1] + panel_h / 2 - (row + 0.5) * pixel_size
            return np.array([x, y, 0])

        self._slice_cell_to_point = cell_to_point

        birth_pulses = VGroup()
        for f in feats:
            pos = cell_to_point(*f.birth_cell)
            dot = Dot(point=pos, radius=0.12, color=GOLD,
                      stroke_width=1.6, stroke_color=WHITE)
            dot.feature = f
            dot._base_radius = 0.12
            dot.set_opacity(0)
            birth_pulses.add(dot)

        def update_pulses(group):
            t = t_tracker.get_value()
            for dot in group:
                f = dot.feature
                window = max(1.2, 0.08 * abs(f.birth) + 1.0)
                if t <= f.birth - window or t > f.birth:
                    dot.set_opacity(0)
                else:
                    progress = (f.birth - t) / window
                    fade = 1.0 - progress
                    scale_factor = 1.0 + 1.2 * np.exp(-progress * 3.0)
                    target_radius = dot._base_radius * scale_factor
                    current_radius = dot.width / 2
                    if current_radius > 1e-6:
                        dot.scale(target_radius / current_radius)
                    dot.set_opacity(fade)

        birth_pulses.add_updater(update_pulses)
        self.add(birth_pulses)

    def _build_edt_panel(self, phi, center, panel_h, t_tracker):
        label = MathTex(r"\phi:\ \text{signed EDT}", color=WHITE_DIM).scale(0.46)
        label.move_to(center + UP * (panel_h / 2 + 0.22))
        self.add(label)

        cbar_w = 0.18
        cbar_gap = 0.18
        img_h = panel_h
        img_w = panel_h
        img_center = center + LEFT * (cbar_w / 2 + cbar_gap / 2)

        edt_img = ImageMobject(_rasterize_field(phi))
        edt_img.set_resampling_algorithm(0)
        edt_img.height = img_h
        edt_img.move_to(img_center)
        self.add(edt_img)

        phi_min, phi_max = float(phi.min()), float(phi.max())
        n_strips = 64
        strip_h = img_h / n_strips
        cbar_x = img_center[0] + img_w / 2 + cbar_gap

        cbar_strips = VGroup()
        for k in range(n_strips):
            frac = (n_strips - 1 - k) / (n_strips - 1)
            color = _color_for_height(frac)
            y = img_center[1] + img_h / 2 - (k + 0.5) * strip_h
            strip = Polygon(
                np.array([cbar_x, y - strip_h / 2, 0]),
                np.array([cbar_x + cbar_w, y - strip_h / 2, 0]),
                np.array([cbar_x + cbar_w, y + strip_h / 2, 0]),
                np.array([cbar_x, y + strip_h / 2, 0]),
                fill_color=color, fill_opacity=1.0, stroke_width=0,
            )
            cbar_strips.add(strip)
        self.add(cbar_strips)

        cbar_outline = Polygon(
            np.array([cbar_x, img_center[1] - img_h / 2, 0]),
            np.array([cbar_x + cbar_w, img_center[1] - img_h / 2, 0]),
            np.array([cbar_x + cbar_w, img_center[1] + img_h / 2, 0]),
            np.array([cbar_x, img_center[1] + img_h / 2, 0]),
            fill_opacity=0, stroke_color=WHITE_DIM, stroke_width=1.0,
        )
        self.add(cbar_outline)

        for v in [phi_min, 0.0, phi_max]:
            if v < phi_min - 1e-6 or v > phi_max + 1e-6:
                continue
            frac = (v - phi_min) / (phi_max - phi_min + 1e-9)
            y = img_center[1] - img_h / 2 + frac * img_h
            tick = Line(
                np.array([cbar_x + cbar_w, y, 0]),
                np.array([cbar_x + cbar_w + 0.08, y, 0]),
                color=WHITE_DIM, stroke_width=1.2,
            )
            self.add(tick)
            num = DecimalNumber(v, num_decimal_places=1, font_size=18, color=WHITE_DIM)
            num.next_to(tick, RIGHT, buff=0.06)
            self.add(num)

        def make_t_indicator():
            t = t_tracker.get_value()
            t_clipped = max(min(t, phi_max), phi_min)
            frac = (t_clipped - phi_min) / (phi_max - phi_min + 1e-9)
            y = img_center[1] - img_h / 2 + frac * img_h
            arrow_w = 0.16
            tip = np.array([cbar_x - 0.02, y, 0])
            return Polygon(
                tip,
                tip + np.array([-arrow_w, arrow_w * 0.6, 0]),
                tip + np.array([-arrow_w, -arrow_w * 0.6, 0]),
                fill_color=GOLD, fill_opacity=1.0, stroke_width=0,
            )

        t_indicator = always_redraw(make_t_indicator)
        self.add(t_indicator)

    def _build_3d_panel(self, phi, center, panel_w, panel_h, t_tracker):
        ds = self.SURFACE_DOWNSAMPLE
        phi_ds = phi[::ds, ::ds]
        H, W = phi_ds.shape

        x_extent = panel_w * 0.55
        y_extent = panel_h * 0.5
        z_extent = panel_h * 0.55

        phi_min, phi_max = float(phi.min()), float(phi.max())
        z_scale = z_extent / (phi_max - phi_min + 1e-9)

        cos_a = np.cos(28 * DEGREES)
        sin_a = np.sin(28 * DEGREES)
        depth_squash = 0.45

        bottom_anchor_y = center[1] - panel_h * 0.42

        def project(i, j, z_world):
            u = (j / (W - 1) - 0.5) * x_extent
            v = (i / (H - 1) - 0.5) * y_extent * depth_squash
            sx = u + v * cos_a * 0.6
            sy = -v * sin_a + z_world
            return np.array([center[0] + sx, bottom_anchor_y + sy, 0])

        quad_indices = []
        for i in range(H - 1):
            for j in range(W - 1):
                z00 = (phi_ds[i, j] - phi_min) * z_scale
                z01 = (phi_ds[i, j + 1] - phi_min) * z_scale
                z10 = (phi_ds[i + 1, j] - phi_min) * z_scale
                z11 = (phi_ds[i + 1, j + 1] - phi_min) * z_scale
                avg_z = (z00 + z01 + z10 + z11) / 4.0
                p00 = project(i, j, z00)
                p01 = project(i, j + 1, z01)
                p11 = project(i + 1, j + 1, z11)
                p10 = project(i + 1, j, z10)
                paint_key = -i + 0.0001 * j
                quad_indices.append((paint_key, avg_z, p00, p01, p11, p10))

        quad_indices.sort(key=lambda q: q[0])

        quads = VGroup()
        for _, avg_z, p00, p01, p11, p10 in quad_indices:
            z_norm = avg_z / z_extent
            fill = _color_for_height(z_norm)
            quad = Polygon(
                p00, p01, p11, p10,
                fill_color=fill, fill_opacity=0.95,
                stroke_width=0.4, stroke_color=WHITE_DIM, stroke_opacity=0.25,
            )
            quads.add(quad)
        self.add(quads)

        label = MathTex(r"\phi\ \text{as height},\ \text{plane } z=t",
                        color=WHITE_DIM).scale(0.46)
        label.move_to(center + UP * (panel_h / 2 + 0.22))
        self.add(label)

        def make_plane():
            t = t_tracker.get_value()
            t_clipped = max(min(t, phi_max), phi_min)
            z_world = (t_clipped - phi_min) * z_scale
            corners = [
                project(0, 0, z_world),
                project(0, W - 1, z_world),
                project(H - 1, W - 1, z_world),
                project(H - 1, 0, z_world),
            ]
            return Polygon(
                *corners,
                fill_color=GOLD, fill_opacity=0.28,
                stroke_color=GOLD, stroke_width=2.2,
            )

        plane = always_redraw(make_plane)
        self.add(plane)

    def _build_diagram_panel(self, feats, center, panel_w, panel_h, t_tracker):
        label = MathTex(r"\text{persistence diagram}", color=WHITE_DIM).scale(0.46)
        label.move_to(center + UP * (panel_h / 2 + 0.22))
        self.add(label)

        finite_births = [f.birth for f in feats]
        finite_deaths = [f.death for f in feats if np.isfinite(f.death)]
        d_lo = min(finite_deaths) if finite_deaths else 0.0
        d_hi = max(finite_births) if finite_births else 1.0
        pad = 0.1 * (d_hi - d_lo + 1e-6)
        diag_lo, diag_hi = d_lo - pad, d_hi + pad
        ticks = _nice_ticks(diag_lo, diag_hi, n=4)

        diag_w = panel_w * 0.66
        diag_h = panel_h * 0.78
        frame_origin = center + DOWN * 0.05 + np.array([-diag_w / 2, -diag_h / 2, 0])

        def c2p(x, y):
            fx = (x - diag_lo) / (diag_hi - diag_lo)
            fy = (y - diag_lo) / (diag_hi - diag_lo)
            return frame_origin + np.array([fx * diag_w, fy * diag_h, 0])

        if diag_lo < 0 < diag_hi:
            gridx = Line(
                c2p(0, diag_lo), c2p(0, diag_hi),
                color=WHITE_FAINT, stroke_width=1.0,
            )
            gridy = Line(
                c2p(diag_lo, 0), c2p(diag_hi, 0),
                color=WHITE_FAINT, stroke_width=1.0,
            )
            self.add(gridx, gridy)

        x_axis = Line(c2p(diag_lo, diag_lo), c2p(diag_hi, diag_lo),
                      color=WHITE_DIM, stroke_width=1.5)
        y_axis = Line(c2p(diag_lo, diag_lo), c2p(diag_lo, diag_hi),
                      color=WHITE_DIM, stroke_width=1.5)
        self.add(x_axis, y_axis)

        tick_len = 0.08
        for v in ticks:
            if v < diag_lo - 1e-6 or v > diag_hi + 1e-6:
                continue
            xtick = Line(c2p(v, diag_lo), c2p(v, diag_lo) + DOWN * tick_len,
                         color=WHITE_DIM, stroke_width=1.2)
            self.add(xtick)
            xnum = DecimalNumber(v, num_decimal_places=0 if v == int(v) else 1,
                                 font_size=18, color=WHITE_DIM)
            xnum.next_to(c2p(v, diag_lo) + DOWN * tick_len, DOWN, buff=0.06)
            self.add(xnum)

            ytick = Line(c2p(diag_lo, v), c2p(diag_lo, v) + LEFT * tick_len,
                         color=WHITE_DIM, stroke_width=1.2)
            self.add(ytick)
            ynum = DecimalNumber(v, num_decimal_places=0 if v == int(v) else 1,
                                 font_size=18, color=WHITE_DIM)
            ynum.next_to(c2p(diag_lo, v) + LEFT * tick_len, LEFT, buff=0.06)
            self.add(ynum)

        diag_diag = Line(
            c2p(diag_lo, diag_lo), c2p(diag_hi, diag_hi),
            stroke_opacity=0.3, stroke_width=1.2, color=WHITE_DIM,
        )
        self.add(diag_diag)

        xlabel = MathTex(r"\text{birth}", color=WHITE_DIM).scale(0.42)
        xlabel.next_to(Line(c2p(diag_lo, diag_lo), c2p(diag_hi, diag_lo)),
                       DOWN, buff=0.42)
        ylabel = MathTex(r"\text{death}", color=WHITE_DIM).scale(0.42)
        ylabel.rotate(90 * DEGREES).next_to(
            Line(c2p(diag_lo, diag_lo), c2p(diag_lo, diag_hi)), LEFT, buff=0.5
        )
        self.add(xlabel, ylabel)

        class _DiagFrame:
            pass
        diag_axes = _DiagFrame()
        diag_axes.c2p = lambda x, y: c2p(x, y)

        diag_dots = VGroup()
        for f in feats:
            if not np.isfinite(f.death):
                continue
            color = H0_COLOR if f.dim == 0 else H1_COLOR
            d = Dot(
                point=c2p(f.birth, f.birth),
                radius=0.065, color=color,
                stroke_width=1.0, stroke_color=WHITE,
            )
            d.feature = f
            d.set_opacity(0)
            diag_dots.add(d)

        def update_dots(group):
            t = t_tracker.get_value()
            for d in group:
                f = d.feature
                if t > f.birth:
                    d.set_opacity(0)
                else:
                    d.set_opacity(1.0)
                    y = max(t, f.death)
                    d.move_to(c2p(f.birth, y))

        diag_dots.add_updater(update_dots)
        self.add(diag_dots)
        self._diag_dots = diag_dots

        threshold_line = always_redraw(
            lambda: DashedLine(
                c2p(diag_lo, max(min(t_tracker.get_value(), diag_hi), diag_lo)),
                c2p(diag_hi, max(min(t_tracker.get_value(), diag_hi), diag_lo)),
                color=GOLD_DIM, stroke_width=1.2, dash_length=0.06,
            )
        )
        self.add(threshold_line)

        legend = VGroup(
            VGroup(Dot(color=H0_COLOR, radius=0.08),
                   MathTex(r"H_0", color=WHITE_DIM).scale(0.5)).arrange(RIGHT, buff=0.14),
            VGroup(Dot(color=H1_COLOR, radius=0.08),
                   MathTex(r"H_1", color=WHITE_DIM).scale(0.5)).arrange(RIGHT, buff=0.14),
        ).arrange(RIGHT, buff=0.5)
        legend.move_to(center + DOWN * (panel_h / 2 + 0.18) + RIGHT * (panel_w * 0.32))
        self.add(legend)

        return diag_axes

    def _setup_pairing_traces(self, feats, t_tracker, diag_axes):
        birth_traces = VGroup()
        for f in feats:
            if not np.isfinite(f.death):
                continue
            color = H0_COLOR if f.dim == 0 else H1_COLOR
            slice_pt = self._slice_cell_to_point(*f.birth_cell)
            init_pt = diag_axes.c2p(f.birth, f.birth)
            line = Line(slice_pt, init_pt, color=color, stroke_width=1.8)
            line.set_stroke(opacity=0)
            line.feature = f
            line._slice_pt = slice_pt
            birth_traces.add(line)

        def update_birth_traces(group):
            t = t_tracker.get_value()
            for line in group:
                f = line.feature
                window = max(0.8, 0.05 * abs(f.birth) + 0.8)
                if t > f.birth or t < f.birth - window:
                    line.set_stroke(opacity=0)
                else:
                    progress = (f.birth - t) / window
                    fade = 1.0 - progress
                    y = max(t, f.death)
                    end_pt = diag_axes.c2p(f.birth, y)
                    line.put_start_and_end_on(line._slice_pt, end_pt)
                    line.set_stroke(opacity=fade * 0.7)

        birth_traces.add_updater(update_birth_traces)
        self.add(birth_traces)

        death_traces = VGroup()
        for f in feats:
            if not np.isfinite(f.death) or f.death_cell is None:
                continue
            color = H0_COLOR if f.dim == 0 else H1_COLOR
            start = self._slice_cell_to_point(*f.death_cell)
            end = diag_axes.c2p(f.birth, f.death)
            line = Line(start, end, color=color, stroke_width=1.8)
            line.set_stroke(opacity=0)
            line.feature = f
            death_traces.add(line)

        def update_death_traces(group):
            t = t_tracker.get_value()
            for line in group:
                f = line.feature
                window = max(0.8, 0.05 * abs(f.death) + 0.8)
                if t > f.death or t < f.death - window:
                    line.set_stroke(opacity=0)
                else:
                    progress = (f.death - t) / window
                    fade = 1.0 - progress
                    line.set_stroke(opacity=fade * 0.7)

        death_traces.add_updater(update_death_traces)
        self.add(death_traces)
