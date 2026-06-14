#!/usr/bin/env python3
"""
F(t, y) = (1, sinh y) — horizontally-translation invariant.

  Single unstable equilibrium at y = 0.
  Slope = cosh(y) ≥ 1, increases symmetrically and without bound on both sides.
  Solutions blow up to ±∞ in FINITE TIME — there is no attractor.

Integral curve through (t₀, y₀):
  dy/dt = sinh(y)   ⟹   y(t) = 2·arctanh( K·e^{t−t₀} )
  K = tanh(y₀/2) ∈ (−1, 1)
  Blowup at t_blow = t₀ − ln|K|   (curve ceases to exist beyond this)

  · Camera pans horizontally
  · Left-click  → integral curve + ODE info
  · Right-click → clear
"""

import subprocess
subprocess.run(["uv", "pip", "install", "numpy", "matplotlib"], check=True)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.colors as mcolors

# ── config ───────────────────────────────────────────────────────────────────
NX, NY = 26, 20
VIEW_W = 8.0
VIEW_H = 6.0
PAN_X  = 0.5          # world-units / second, horizontal only
FPS    = 25

COLORS = ["#00ffff", "#ff6f61", "#b5ea8c", "#f0a500",
          "#c77dff", "#ff9ff3", "#feca57", "#48dbfb"]

# ── figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor("#0d0d0d")
ax.set_facecolor("#0d0d0d")
ax.tick_params(colors="white")
ax.set_xlabel("t", color="white", fontsize=11)
ax.set_ylabel("y", color="white", fontsize=11)
for sp in ax.spines.values():
    sp.set_edgecolor("#333")

# slope angle = arctan(sinh y) ∈ (−π/2, π/2) — use full seismic range
norm = mcolors.Normalize(vmin=-np.pi / 2, vmax=np.pi / 2)
cmap = plt.cm.seismic   # blue (steep down) → white (flat) → red (steep up)

cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, pad=0.02)
cb.set_label("slope angle  θ = arctan(sinh y)  [rad]", color="white", fontsize=10)
cb.ax.yaxis.set_tick_params(color="white")
plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
cb.outline.set_edgecolor("#444")

ax.set_title(
    "F(t, y) = (1,  sinh y)    ·    slope = cosh(y), grows without bound    ·    "
    "solutions blow up in finite time    ·    left-click → curve    ·    right-click → clear",
    color="white", fontsize=10.5)

# Equilibrium guide line at y = 0
ax.axhline(0, color="#ff5555", lw=1.3, ls="--", alpha=0.5, zorder=1)
ax.text(0.01, 0.505, "← unstable  y = 0", transform=ax.transAxes,
        color="#ff7777", fontsize=9, va="center", alpha=0.7)

eq_box = ax.text(
    0.02, 0.04, "", transform=ax.transAxes,
    color="white", fontsize=9.5, family="monospace", va="bottom",
    bbox=dict(boxstyle="round,pad=0.5", fc="#111", ec="#555", alpha=0.88),
    zorder=10)

# ── state ────────────────────────────────────────────────────────────────────
# curve_store: (K, t0, color, Line2D, scatter, vline)
curve_store = []
col_idx     = [0]
t_now       = [0.0]
quiv_h      = [None]

def view_bounds(t):
    ct = PAN_X * t
    return ct - VIEW_W / 2, ct + VIEW_W / 2, -VIEW_H / 2, VIEW_H / 2

def curve_data(K, t0, tmin, tmax):
    """y = 2·arctanh(K·e^{t−t₀}), NaN where |K·eᵗ| ≥ 1 (past blowup)."""
    tc  = np.linspace(tmin, tmax, 1500)
    arg = K * np.exp(np.clip(tc - t0, -30, 30))
    yc  = np.where(np.abs(arg) < 0.99999, 2 * np.arctanh(np.clip(arg, -0.99999, 0.99999)), np.nan)
    return tc, yc

def refresh_eq():
    if not curve_store:
        eq_box.set_text("")
        eq_box.get_bbox_patch().set_edgecolor("#555")
        return
    rows = ["dy/dt = sinh(y)   ⟹   y = 2·arctanh( K · e^{t−t₀} )"]
    for i, (K, t0, col, *_) in enumerate(curve_store):
        if abs(K) > 1e-10:
            t_blow = t0 - np.log(abs(K))
            rows.append(f"  [{i+1}]  K = {K:+.3f}   t₀ = {t0:.2f}   t_blow = {t_blow:.2f}")
        else:
            rows.append(f"  [{i+1}]  K = 0   (rests at equilibrium y = 0)")
    eq_box.set_text("\n".join(rows))
    eq_box.get_bbox_patch().set_edgecolor(curve_store[-1][2])

# ── animation ────────────────────────────────────────────────────────────────
def update(frame):
    t = frame / FPS
    t_now[0] = t
    tmin, tmax, ymin, ymax = view_bounds(t)

    tv = np.linspace(tmin, tmax, NX)
    yv = np.linspace(ymin, ymax, NY)
    T, Y  = np.meshgrid(tv, yv)
    V     = np.sinh(Y)
    mag   = np.sqrt(1 + V ** 2)
    slope = np.arctan2(V, np.ones_like(V))

    if quiv_h[0] is not None:
        quiv_h[0].remove()
    quiv_h[0] = ax.quiver(
        T, Y, 1 / mag, V / mag, slope,
        cmap=cmap, norm=norm, scale=28,
        width=0.004, headwidth=4, headlength=5, alpha=0.9, zorder=2)

    ax.set_xlim(tmin, tmax)
    ax.set_ylim(ymin, ymax)

    for K, t0, _, line, _, _ in curve_store:
        tc, yc = curve_data(K, t0, tmin, tmax)
        line.set_xdata(tc)
        line.set_ydata(yc)

# ── click handler ─────────────────────────────────────────────────────────────
def on_click(event):
    if event.inaxes is not ax:
        return

    if event.button == 3:           # right-click → clear
        for _, _, _, line, sc, vline in curve_store:
            line.remove()
            sc.remove()
            vline.remove()
        curve_store.clear()
        col_idx[0] = 0
        refresh_eq()
        fig.canvas.draw_idle()
        return

    if event.button != 1:
        return

    t0, y0 = event.xdata, event.ydata
    K = np.tanh(y0 / 2)             # ∈ (−1, 1) always

    color = COLORS[col_idx[0] % len(COLORS)]
    col_idx[0] += 1

    tmin, tmax, ymin, ymax = view_bounds(t_now[0])
    tc, yc = curve_data(K, t0, tmin, tmax)

    line, = ax.plot(tc, yc, color=color, lw=2.2, alpha=0.9, zorder=5)
    sc    = ax.scatter([t0], [y0], c=color, s=60, zorder=6)

    # Dotted vertical line at the blowup time
    if abs(K) > 1e-10:
        t_blow = t0 - np.log(abs(K))
        vline, = ax.plot([t_blow, t_blow], [-50, 50],
                         color=color, lw=1.0, ls=":", alpha=0.5, zorder=4)
    else:
        vline, = ax.plot([], [], color=color)  # no blowup at equilibrium

    curve_store.append((K, t0, color, line, sc, vline))
    refresh_eq()
    fig.canvas.draw_idle()

fig.canvas.mpl_connect("button_press_event", on_click)

ani = anim.FuncAnimation(fig, update, interval=1000 // FPS,
                         blit=False, cache_frame_data=False)

plt.tight_layout()
plt.show()
