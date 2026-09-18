"""
Builds the two MIP figures for the GRL revision:

    fig5.pdf                  main text  -- thermodynamic buffering, one panel
    figS_mip_diagnostics.pdf  SI         -- the five supporting diagnostics

WHY THIS FILE EXISTS
--------------------
The package's README describes the Fig 5 split ("panel (a) alone is the
main-text Figure 5; panels (b)-(f) moved to a standalone SI figure"), but no
script in the package actually performs it: unified_buffering_analysis.py
still only emits the combined 6-panel figure, at 33 x 5.5 in. The two
delivered PDFs were produced by hand outside the repo, so they cannot be
regenerated and they drifted from the code. This script closes that gap --
it reuses unified_buffering_analysis.py's loaders and aggregation verbatim
and does nothing but lay out and annotate the results.

Run it from a directory containing the same inputs that
unified_buffering_analysis.py needs (RCEMIP/, the four .pkl files, the two
.npz files).
"""
import os
import runpy

import matplotlib
matplotlib.use("Agg")           # never open a window
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

ISOTHERM = 228.0

# ---------------------------------------------------------------------------
# Reuse the existing pipeline unchanged.
# ---------------------------------------------------------------------------
_u = runpy.run_path("unified_buffering_analysis.py")
iso_data = _u["iso_data"]
aggregate_isotherm = _u["aggregate_isotherm"]
VARS_ALL = _u["VARS_ALL"]

keys, agg = aggregate_isotherm(ISOTHERM)
sources = np.array([k[1] for k in keys])
names = np.array([k[0] for k in keys])

N_LOADED = len(iso_data)
N_USED = len(keys)
_dropped = [k for k in iso_data if k not in keys]

# Same type scale as analysis_minimal.py, and each figure is authored at the
# width it will be printed at, so a tick label is the same size in Fig. 5 as
# it is in Fig. 1. See the note above LINEWIDTH_IN in analysis_minimal.py.
LINEWIDTH_IN = 5.50              # see the note above LINEWIDTH_IN in analysis_minimal.py
W_FULL = 0.95 * LINEWIDTH_IN     # 5.225 in -- Fig. 5, at width=0.95\linewidth
W_SI   = LINEWIDTH_IN            # 5.50 in -- SI figure, at width=\textwidth
                                 # (assumes agutexSI2019 shares the main
                                 # class's text width; check if it does not)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
})

# Named text roles, shared by BOTH figures this script writes, so the
# main-text panel and its SI companion read as one pair. Previously the
# annotations here ran 8.5 / 7.5 / 7.2 / 6.5 pt with no rule behind which
# was which.
FS_LEGEND = 8.0    # every legend (set via rcParams above)
FS_ANNOT  = 7.5    # statistics boxes and caption notes
FS_POINT  = 6.5    # per-point model labels

# ---------------------------------------------------------------------------
# Styling.
#
# The delivered figures used blue / red / green filled circles for the three
# multi-model groups. Red-vs-green is the single most common confusion pair
# (~8% of men), and in these panels red and green are exactly the two groups
# that overlap most in the crowded center of the cloud. Swapped for a
# blue / orange / purple set that survives both deuteranopia and protanopia,
# and given each group a distinct MARKER SHAPE as well, so the encoding is
# never carried by hue alone.
# ---------------------------------------------------------------------------
STYLE = {
    "RCE_small":    dict(marker="o", color="#4C72B0", s=42, label="RCEMIP, small domain"),
    "RCE_large":    dict(marker="s", color="#DD8452", s=42, label="RCEMIP, large domain"),
    "AMIP_CMIP":    dict(marker="^", color="#8172B3", s=48, label="CMIP6 AMIP-future4K"),
    "AMIP_SCREAM":  dict(marker="*", color="gold",    s=320, label="SCREAM AMIP (Cess1)"),
    "SCREAM_cess2": dict(marker="*", color="k",       s=320, label="SCREAM (Cess2, this study)"),
}
ORDER = ["RCE_small", "RCE_large", "AMIP_CMIP", "AMIP_SCREAM", "SCREAM_cess2"]


def draw_groups(ax, xp, yp, legend_handles=None):
    for tag in ORDER:
        sel = sources == tag
        if not np.any(sel):
            continue
        st = dict(STYLE[tag])
        lab = st.pop("label")
        h = ax.scatter(xp[sel], yp[sel], edgecolor="k", linewidth=0.5,
                       zorder=3 if tag.startswith("RCE") or tag == "AMIP_CMIP" else 4,
                       label=lab, **st)
        if legend_handles is not None:
            legend_handles.append(h)


def fit(xp, yp):
    ok = np.isfinite(xp) & np.isfinite(yp)
    return linregress(xp[ok], yp[ok])


# ===========================================================================
# FIGURE 5 (main text): thermodynamic buffering
# ===========================================================================
x = agg["S"] * 100.0
y = agg["tau"] * 100.0
r = fit(x, y)

fig, ax = plt.subplots(figsize=(W_FULL, 5.1), constrained_layout=True)

lo = min(np.nanmin(x), np.nanmin(y)) - 1.0
hi = max(np.nanmax(x), np.nanmax(y)) + 1.0
xx = np.array([lo, hi])

ax.plot(xx, xx, color="0.45", lw=1.2, ls="--", zorder=1, label="1:1")
ax.plot(xx, r.slope * xx + r.intercept, color="k", lw=1.6, zorder=2,
        label="Least-squares fit")

# The scientific content of this panel is the OFFSET between the fit and 1:1
# (tau_sub responds ~1:1 with S_inv, displaced by the Delta p term), so shade
# it rather than leaving the reader to measure the gap by eye.
ax.fill_between(xx, xx, r.slope * xx + r.intercept, color="k", alpha=0.06, zorder=0)

draw_groups(ax, x, y)

# ---------------------------------------------------------------------------
# Model names: the delivered version printed all 82 at fontsize 6 with no
# collision handling, which produced an unreadable black mass across the
# center of the plot -- the single worst legibility problem in the figure
# set. Only the extremes carry information a reader can act on, so label
# those and let the SI table carry the full list.
# ---------------------------------------------------------------------------
_d = np.hypot(x - np.nanmean(x), y - np.nanmean(y))
_cand = [i for i in np.argsort(_d)[::-1]
         if np.isfinite(x[i]) and np.isfinite(y[i])]
# Greedy thinning: take the most extreme points, but skip any that sit within
# MIN_SEP of one already labeled, so the labels themselves cannot pile up
# (three of the six most-extreme models are neighbours in the lower-left tail).
MIN_SEP, MAX_LABELS = 2.5, 5
_label_idx = []
for i in _cand:
    if len(_label_idx) >= MAX_LABELS:
        break
    if all(np.hypot(x[i] - x[j], y[i] - y[j]) > MIN_SEP for j in _label_idx):
        _label_idx.append(i)

# Placement is solved in DATA coordinates against the two reference lines.
# Both lines run at ~45 degrees straight through the middle of the plot, so a
# label nudged by a fixed dx/dy lands on one of them about as often as not.
# For each label we walk candidate offsets -- perpendicular to the trend
# first, both directions, at increasing distance -- and keep the first whose
# text rectangle sits inside the axes and is crossed by neither line.
#
# Working in data coordinates rather than display coordinates is deliberate:
# only the text's SIZE is read from the renderer (which is reliable), while
# its position and the containment test come from the axis limits we set
# ourselves. Display-space bounding boxes are invalidated by every relayout,
# which silently let a label escape into the title.
_SQ = 1.0 / np.sqrt(2.0)
_CANDIDATES = []
for _d in (2.0, 3.0, 4.2, 5.6, 7.0):
    _CANDIDATES += [(-_SQ * _d, _SQ * _d), (_SQ * _d, -_SQ * _d)]   # up-left, down-right
for _d in (3.0, 4.5, 6.0):
    _CANDIDATES += [(-_d, 0.0), (_d, 0.0)]                          # horizontal fallback

_LINES = [(1.0, 0.0), (r.slope, r.intercept)]      # (m, c) for y = m*x + c


def _crossed(rect):
    """Is rect (X0, Y0, X1, Y1) cut by either reference line?"""
    X0, Y0, X1, Y1 = rect
    for m, c in _LINES:
        vals = [Y - (m * X + c) for X in (X0, X1) for Y in (Y0, Y1)]
        if min(vals) <= 0.0 <= max(vals):
            return True
    return False


fig.canvas.draw()
_rend = fig.canvas.get_renderer()
# data units per display pixel, from the transform's scale (position-independent)
_o = ax.transData.transform((0.0, 0.0))
_ux = ax.transData.transform((1.0, 0.0))[0] - _o[0]
_uy = ax.transData.transform((0.0, 1.0))[1] - _o[1]
_MARGIN = 0.45          # data units of clearance demanded on every side

for i in _label_idx:
    nm = names[i].split(".")[0]
    _probe = ax.text(0, 0, nm, fontsize=FS_POINT)
    _bb = _probe.get_window_extent(_rend)
    _w, _h = _bb.width / _ux, _bb.height / _uy
    _probe.remove()

    _chosen = None
    for _dx, _dy in _CANDIDATES:
        _px, _py = x[i] + _dx, y[i] + _dy
        X0, X1 = (_px, _px + _w) if _dx > 0 else ((_px - _w, _px) if _dx < 0
                                                  else (_px - _w / 2, _px + _w / 2))
        Y0, Y1 = (_py, _py + _h) if _dy > 0 else ((_py - _h, _py) if _dy < 0
                                                  else (_py - _h / 2, _py + _h / 2))
        rect = (X0 - _MARGIN, Y0 - _MARGIN, X1 + _MARGIN, Y1 + _MARGIN)
        if not (lo <= rect[0] and rect[2] <= hi and lo <= rect[1] and rect[3] <= hi):
            continue                                    # never leave the frame
        if _chosen is None:
            _chosen = (_dx, _dy)                        # best in-frame option so far
        if not _crossed(rect):
            _chosen = (_dx, _dy)
            break
    _dx, _dy = _chosen if _chosen else (0.7, -1.5)
    ax.annotate(nm, (x[i], y[i]), xytext=(x[i] + _dx, y[i] + _dy),
                fontsize=FS_POINT, color="0.3",
                ha="left" if _dx > 0 else ("right" if _dx < 0 else "center"),
                va="bottom" if _dy > 0 else ("top" if _dy < 0 else "center"),
                arrowprops=dict(arrowstyle="-", color="0.6", lw=0.5,
                                shrinkA=0, shrinkB=3))

ax.text(0.035, 0.760, f"labeled: {len(_label_idx)} most extreme models\n"
                      "(full list in Supporting Information)",
        transform=ax.transAxes, fontsize=FS_POINT, color="0.45", va="top")

# ---------------------------------------------------------------------------
# N. The delivered figure said "N = 94 models" in the title and "N=82" in the
# in-panel annotation, with nothing explaining the difference. 94 is the
# number of models LOADED; 12 of them have no finite value for one or more of
# the six aggregated variables at this isotherm, so the regression -- and
# therefore slope=1.04 and R^2=0.98 -- is over 82. Report the number the
# statistics actually use, and say where the other 12 went.
# ---------------------------------------------------------------------------
ax.text(0.035, 0.965,
        f"slope = {r.slope:.2f} $\\pm$ {r.stderr:.2f}   "
        f"($R^2$ = {r.rvalue**2:.2f})\n"
        f"offset from 1:1 = {r.intercept:+.2f} % K$^{{-1}}$\n"
        f"$N$ = {N_USED} models",
        transform=ax.transAxes, va="top", ha="left", fontsize=FS_ANNOT,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.75", lw=0.6))

ax.set_xlabel(r"$\Delta \ln S_\mathrm{inv} / \Delta T_s$  [% K$^{-1}$]")
ax.set_ylabel(r"$\Delta \ln \tau_\mathrm{sub}^{-1} / \Delta T_s$  [% K$^{-1}$]")
ax.set_title(f"Thermodynamic buffering at the anvil peak "
             f"($T$ = {ISOTHERM:.0f} K)")
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_aspect("equal")           # a 1:1 line should look like 45 degrees
ax.grid(True, alpha=0.3)
# labelspacing carries the star markers (s=320), which are tall enough to
# crowd the rows either side at the default spacing.
ax.legend(frameon=False, loc="lower right", handletextpad=0.65,
          borderpad=0.5, labelspacing=0.85, handlelength=1.9)

fig.savefig("fig5.pdf", dpi=300)   # no bbox_inches: figsize IS the printed size
plt.close(fig)

# ===========================================================================
# FIGURE S1 (SI): the five supporting diagnostics
# ===========================================================================
dln_deltap = (agg["S"] - agg["tau"]) * 100.0

# Panel (a), "Cloud buffering", dropped per request: it tests a different
# question (does the robust tau_sub^-1 buffering propagate to a robust
# cross-model prediction of dlnC/dTs?) than the other four panels (which are
# all about the tau_sub^-1 <-> lapse-rate closure chain), and its R^2=0.06
# was already flagged as "no explanatory power". The remaining four panels
# are the self-contained closure-test set.

PANELS = [
    ("p", "S", r"$\Delta \ln p$", r"$\Delta \ln S_\mathrm{inv}$",
     r"$S_\mathrm{inv}$ vs. pressure"),
    ("gamma", "tau", r"$\Delta \ln \Gamma$", r"$\Delta \ln \tau_\mathrm{sub}^{-1}$",
     r"$\tau_\mathrm{sub}^{-1}$ vs. lapse rate"),
    ("gamma", "S", r"$\Delta \ln \Gamma$", r"$\Delta \ln S_\mathrm{inv}$",
     r"$S_\mathrm{inv}$ vs. lapse rate"),
    (None, None, r"$\Delta \ln S_\mathrm{inv}$", r"$\Delta \ln \Delta p$ (exact)",
     "Offset universality"),
]

# 4 panels -> a plain 2 x 2 grid, one shared legend below via fig.legend()
# rather than the donor 3x2's spare sixth cell (there isn't a spare cell to
# spend now).
fig, axs = plt.subplots(2, 2, figsize=(W_SI, 6.2), constrained_layout=True)
axs = axs.ravel()
handles = []

for i, (xk, yk, xlab, ylab, title) in enumerate(PANELS):
    ax = axs[i]
    if xk is None:                      # the Delta p offset panel
        xp, yp = agg["S"] * 100.0, dln_deltap
    else:
        xp, yp = agg[xk] * 100.0, agg[yk] * 100.0

    draw_groups(ax, xp, yp, legend_handles=handles if i == 0 else None)
    rr = fit(xp, yp)
    _lo = min(np.nanmin(xp), np.nanmin(yp)) - 0.8
    _hi = max(np.nanmax(xp), np.nanmax(yp)) + 0.8
    _xx = np.array([_lo, _hi])

    if xk is not None:
        ax.plot(_xx, _xx, color="0.55", lw=1.0, ls="--", zorder=1)

    _weak = rr.rvalue ** 2 < 0.25
    ax.plot(_xx, rr.slope * _xx + rr.intercept,
            color="0.6" if _weak else "k", lw=1.0 if _weak else 1.6,
            ls=":" if _weak else "-", zorder=2)

    _note = (f"slope = {rr.slope:.2f}, $R^2$ = {rr.rvalue**2:.2f}"
             + ("\n(no explanatory power)" if _weak else ""))
    ax.text(0.04, 0.96, _note, transform=ax.transAxes, va="top", fontsize=FS_ANNOT,
            color="0.35" if _weak else "k",
            bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.85))

    if xk is None:
        ax.axhline(np.nanmean(yp), color="k", ls="--", lw=1.0, zorder=2)
        ax.text(0.96, 0.06, ("mean = %.2f % s" % (np.nanmean(yp), "K$^{-1}$"))
                             .replace("-", "\u2212"),
                transform=ax.transAxes, ha="right", fontsize=FS_ANNOT,
                bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.85))
        ax.set_xlim(np.nanmin(xp) - 0.8, np.nanmax(xp) + 0.8)
    else:
        ax.set_xlim(_lo, _hi)
        ax.set_ylim(_lo, _hi)

    ax.set_xlabel(xlab + r"$/\Delta T_s$  [% K$^{-1}$]")
    ax.set_ylabel(ylab + r"$/\Delta T_s$  [% K$^{-1}$]")
    ax.set_title(f"({chr(97+i)}) {title}")     # relettered (a)-(d) now that
                                                # the dropped panel no longer
                                                # leaves a gap in the sequence
    ax.grid(True, alpha=0.3)

# Shared legend + annotation, below the 2x2 grid rather than in a spare cell.
leg = fig.legend(handles=handles, loc="lower center",
                  bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False,
                  handletextpad=0.7, columnspacing=1.4)
fig.text(0.5, -0.145,
         f"All panels: {ISOTHERM:.0f} K isotherm, $N$ = {N_USED} models. "
         f"Dashed grey line is 1:1.",
         ha="center", va="top", fontsize=FS_ANNOT, color="0.35")

fig.suptitle("RCEMIP + CMIP6 AMIP + SCREAM: supporting diagnostics\n"
             "(lapse-rate closure test)", fontsize=11)
fig.savefig("figS_mip_diagnostics_no_panel_a.pdf", dpi=300,
            bbox_inches="tight")
plt.close(fig)

print("\nwrote figS_mip_diagnostics_no_panel_a.pdf (4 panels, panel 'a' Cloud "
      "buffering dropped)")
