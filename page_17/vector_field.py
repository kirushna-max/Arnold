#!/usr/bin/env python3
"""
Animated vector field F(x,y) = (1, sin x) — vertically-translation invariant.
  · Camera pans diagonally through world-space
  · Left-click  → draw integral curve + display its ODE
  · Right-click → clear all curves
"""

import subprocess
subprocess.run(["uv", "pip", "install", "numpy", "matplotlib"], check=True)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.colors as mcolors

# ── config ──────────────────────────────────────────────────────────────────
NX, NY = 25, 16          # arrow grid density
VIEW_W = 4 * np.pi       # visible x-span
VIEW_H = 8.0             # visible y-span
PAN_X  = 0.6             # world-units / second  (diagonal pan)
PAN_Y  = 0.28
FPS    = 25

COLORS = ["#00ffff", "#ff6f61", "#b5ea8c", "#f0a500",
          "#c77dff", "#ff9ff3", "#feca57", "#48dbfb"]

# ── figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor("#0d0d0d")
ax.set_facecolor("#0d0d0d")
ax.tick_params(colors="white")
ax.set_xlabel("x", color="white", fontsize=11)
ax.set_ylabel("y", color="white", fontsize=11)
for sp in ax.spines.values():
    sp.set_edgecolor("#333")

norm = mcolors.Normalize(vmin=-np.pi / 4, vmax=np.pi / 4)
cmap = plt.cm.plasma

cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, pad=0.02)
cb.set_label("slope angle  θ = arctan(sin x)  [rad]", color="white", fontsize=10)
cb.ax.yaxis.set_tick_params(color="white")
plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
cb.outline.set_edgecolor("#444")

ax.set_title(
    "F(x, y) = (1,  sin x)    ·    left-click → integral curve    ·    right-click → clear",
    color="white", fontsize=12)

eq_box = ax.text(
    0.02, 0.04, "", transform=ax.transAxes,
    color="white", fontsize=10, family="monospace", va="bottom",
    bbox=dict(boxstyle="round,pad=0.5", fc="#111", ec="#555", alpha=0.88),
    zorder=10)

# ── runtime state ────────────────────────────────────────────────────────────
curve_store = []    # list of (C, color, Line2D, PathCollection)
col_idx     = [0]
t_now       = [0.0]
quiv_h      = [None]

def view_bounds(t):
    cx, cy = PAN_X * t, PAN_Y * t
    return cx - VIEW_W / 2, cx + VIEW_W / 2, cy - VIEW_H / 2, cy + VIEW_H / 2

def refresh_eq():
    if not curve_store:
        eq_box.set_text("")
        eq_box.get_bbox_patch().set_edgecolor("#555")
        return
    rows = []
    for i, (C, col, *_) in enumerate(curve_store):
        sign = "+" if C >= 0 else "−"
        rows.append(f"[{i+1}]  dy/dx = sin(x)   ⟹   y = −cos(x) {sign} {abs(C):.2f}")
    eq_box.set_text("\n".join(rows))
    eq_box.get_bbox_patch().set_edgecolor(curve_store[-1][1])

# ── animation frame ──────────────────────────────────────────────────────────
def update(frame):
    t = frame / FPS
    t_now[0] = t
    xmin, xmax, ymin, ymax = view_bounds(t)

    xv = np.linspace(xmin, xmax, NX)
    yv = np.linspace(ymin, ymax, NY)
    X, Y = np.meshgrid(xv, yv)
    V    = np.sin(X)
    mag  = np.sqrt(1 + V ** 2)
    slope = np.arctan2(V, np.ones_like(V))

    if quiv_h[0] is not None:
        quiv_h[0].remove()
    quiv_h[0] = ax.quiver(
        X, Y, 1 / mag, V / mag, slope,
        cmap=cmap, norm=norm, scale=28,
        width=0.004, headwidth=4, headlength=5, alpha=0.9, zorder=2)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Integral curves follow world coordinates through the panning window
    xc = np.linspace(xmin, xmax, 1000)
    for C, _, line, _ in curve_store:
        line.set_xdata(xc)
        line.set_ydata(-np.cos(xc) + C)

# ── mouse handler ────────────────────────────────────────────────────────────
def on_click(event):
    if event.inaxes is not ax:
        return

    if event.button == 3:          # right-click → clear
        for _, _, line, sc in curve_store:
            line.remove()
            sc.remove()
        curve_store.clear()
        col_idx[0] = 0
        refresh_eq()
        fig.canvas.draw_idle()
        return

    if event.button != 1:
        return

    x0, y0 = event.xdata, event.ydata
    # Integral curve of  dy/dx = sin(x)  through (x0, y0)
    #   ∫ sin(x) dx = −cos(x) + C   →   C = y0 + cos(x0)
    C = y0 + np.cos(x0)

    color = COLORS[col_idx[0] % len(COLORS)]
    col_idx[0] += 1

    xmin, xmax, ymin, ymax = view_bounds(t_now[0])
    xc = np.linspace(xmin, xmax, 1000)
    line, = ax.plot(xc, -np.cos(xc) + C, color=color, lw=2.2, alpha=0.9, zorder=5)
    sc    = ax.scatter([x0], [y0], c=color, s=60, zorder=6)

    curve_store.append((C, color, line, sc))
    refresh_eq()
    fig.canvas.draw_idle()

fig.canvas.mpl_connect("button_press_event", on_click)

ani = anim.FuncAnimation(fig, update, interval=1000 // FPS,
                         blit=False, cache_frame_data=False)

plt.tight_layout()
plt.show()
