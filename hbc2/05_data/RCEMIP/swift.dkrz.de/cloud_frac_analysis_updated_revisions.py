import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import linregress

# =====================================================
# USER SETTINGS
# =====================================================

BASE_DIR = "RCE_large"   # "RCE_small" or "RCE_large"
CF_MODE = "cfv0"         # "cfv0" or "cfv1-cfv2"

SAVE_FIGS = True
OUT_PREFIX = f"RCEMIP_{BASE_DIR}_{CF_MODE}"

T_l = np.arange(295.0, 197.0, -1.0)

AVG_TMIN = 200.0
AVG_TMAX = 240.0

# RCEMIP's 295/300/305K triplet steps in 5K increments -- dln() below divides
# by this so every reported number is a true %/K, directly comparable to the
# SCREAM (4K perturbation) and AMIP-future4K results elsewhere in the paper.
# It's Friday, but this constant is not a fun one to get wrong, so it gets
# its own line rather than a silently-buried magic number.
DELTA_SST = 5.0

Rd = 287.0
cp = 1004.0
g = 9.81
p0 = 100000.0
kappa = Rd / cp
gam_d = g / cp

# =====================================================
# MODE SETTINGS
# =====================================================

domain = "small" if "small" in BASE_DIR else "large"

if CF_MODE == "cfv0":
    CF_VAR = "cfv0_avg"
    suffix295 = f"_RCE_{domain}295_cfv0-profiles.nc"
    suffix300 = f"_RCE_{domain}300_cfv0-profiles.nc"
    suffix305 = f"_RCE_{domain}305_cfv0-profiles.nc"
else:
    CF_VAR = "cfv1_avg"
    suffix295 = f"_RCE_{domain}295_cfv1-cfv2-profiles.nc"
    suffix300 = f"_RCE_{domain}300_cfv1-cfv2-profiles.nc"
    suffix305 = f"_RCE_{domain}305_cfv1-cfv2-profiles.nc"

pattern300 = f"*{suffix300}"

# =====================================================
# HELPERS
# =====================================================

def dln(x1, x0):
    """Fractional change per Kelvin of SST warming (already divided by
    DELTA_SST), so every dln(...) call downstream returns true %/K units."""
    denom = 0.5 * (x1 + x0)
    out = np.full_like(denom, np.nan, dtype=float)
    valid = np.isfinite(x0) & np.isfinite(x1) & (denom != 0)
    out[valid] = (x1[valid] - x0[valid]) / denom[valid] / DELTA_SST
    return out

def get_1d_var(ds, names):
    for name in names:
        if name in ds.variables:
            arr = np.asarray(ds[name]).squeeze()
            if arr.ndim == 1:
                return arr
    raise KeyError(f"Could not find any of {names}")

def get_pressure_pa(ds):
    p_raw = get_1d_var(ds, ["pa_avg", "pa"])
    if np.nanmax(p_raw) < 2000:
        return p_raw * 100.0
    return p_raw

def load_profile(path):
    ds = xr.open_dataset(path)

    T = get_1d_var(ds, ["ta_avg", "ta"])
    p = get_pressure_pa(ds)
    cf = get_1d_var(ds, [CF_VAR, "cfv0_avg", "cfv1_avg", "cfv2_avg"])

    theta = T * (p0 / p) ** kappa
    S = -np.gradient(theta, p, edge_order=2)
    rho = p / (Rd * T)

    ds.close()
    return T.astype(float), p.astype(float), cf.astype(float), S.astype(float), rho.astype(float)

def interp_branch(T, X):
    imin = np.argmin(T)

    T_cut = np.asarray(T[:imin+1], float)
    X_cut = np.asarray(X[:imin+1], float)

    valid = np.isfinite(T_cut) & np.isfinite(X_cut)
    T_cut = T_cut[valid]
    X_cut = X_cut[valid]

    if len(T_cut) < 2:
        return np.full_like(T_l, np.nan, dtype=float), imin

    order = np.argsort(T_cut)
    T_use = T_cut[order]
    X_use = X_cut[order]

    T_unique, idx = np.unique(T_use, return_index=True)
    X_unique = X_use[idx]

    if len(T_unique) < 2:
        return np.full_like(T_l, np.nan, dtype=float), imin

    f = interp1d(T_unique, X_unique, bounds_error=False, fill_value=np.nan)
    X_l = f(T_l)
    X_l[(T_l < T_unique.min()) | (T_l > T_unique.max())] = np.nan

    return X_l, imin

def derive(T, p, cf, S, rho):
    # =====================================================
    # Mirrors the SCREAM script's structure: interpolate raw fields onto the
    # T_l grid first, then derive gam/invS/dp/tau from the interpolated
    # quantities. Uses the native-grid static stability S = -d(theta)/dp
    # (as originally computed in load_profile), interpolated to T_l, in a
    # SCREAM-style "Gamma_d * (1 - correction)" formula.
    #
    # IMPORTANT: SCREAM's own gam = (1-S_l*ro)*gam_d formula assumes S_l is
    # a DRY-STATIC-ENERGY pressure derivative (units m^3/kg) -- that's what
    # SCREAM outputs natively. Our S here is a POTENTIAL-TEMPERATURE
    # pressure derivative (units K/Pa) -- a different physical quantity,
    # related by dtheta/dz = (theta/T)*(Gamma_d-Gamma). Using S directly in
    # SCREAM's unmodified formula reproduces the earlier units-mismatch bug
    # (off by a factor of order cp). The corrected, theta-consistent version
    # of the same "Gamma_d * (1 - ...)" structure is:
    #     Gamma = Gamma_d * (1 - S*rho*cp*(T/theta))
    # which is what's implemented below.
    # =====================================================
    cf_l, imin = interp_branch(T, cf)
    p_l, _ = interp_branch(T, p)
    S_l, _ = interp_branch(T, S)   # -d(theta)/dp, interpolated onto T_l

    ro_l = p_l / (Rd * T_l)                 # ideal gas law, evaluated exactly on the T_l grid
    theta_l = T_l * (p0 / p_l) ** kappa      # potential temperature, evaluated exactly on the T_l grid

    gamma = gam_d * (1.0 - S_l * ro_l * cp * (T_l / theta_l))

    gamma[~np.isfinite(gamma)] = np.nan
    gamma[gamma <= 0] = np.nan

    denom = (1.0 / gamma) - (1.0 / gam_d)
    invS = np.full_like(T_l, np.nan, dtype=float)
    good = np.isfinite(denom) & (np.abs(denom) > 1e-8)
    invS[good] = 1.0 / denom[good]

    dZ = np.full_like(T_l, np.nan, dtype=float)
    good_g = np.isfinite(gamma) & (gamma > 0)
    dZ[good_g] = 1.0 / gamma[good_g]

    dp = dZ * g * ro_l

    tau = np.full_like(T_l, np.nan, dtype=float)
    good_tau = np.isfinite(invS) & np.isfinite(dp) & (dp != 0)
    tau[good_tau] = invS[good_tau] / dp[good_tau]

    return {
        "cf_l": cf_l,
        "p_l": p_l,
        "S_l": S_l,
        "rho_l": ro_l,
        "gamma_l": gamma,
        "invS_l": invS,
        "dp_l": dp,
        "tau_l": tau,
        "imin": imin,
    }

def fit_stats(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 3:
        return np.nan, np.nan, np.nan
    slope, intercept, r_value, _, _ = linregress(x[valid], y[valid])
    return slope, intercept, r_value**2

def setup_lims(ax, x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    lo = min(np.nanmin(x[valid]), np.nanmin(y[valid]))
    hi = max(np.nanmax(x[valid]), np.nanmax(y[valid]))
    pad = 0.05 * (hi - lo) if hi > lo else 0.1
    lims = [lo - pad, hi + pad]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(lims, lims, "k--", label="1:1")
    return lims

# =====================================================
# FIND TRIPLETS
# =====================================================

files300 = sorted(glob.glob(os.path.join(BASE_DIR, pattern300)))
triplets = []

for f300 in files300:
    model = os.path.basename(f300).replace(suffix300, "")
    f295 = os.path.join(BASE_DIR, model + suffix295)
    f305 = os.path.join(BASE_DIR, model + suffix305)

    if os.path.exists(f295) and os.path.exists(f305):
        triplets.append((model, f295, f300, f305))

print(f"Found {len(triplets)} matched 295/300/305 triplets.")

# =====================================================
# FILTER MODELS
# =====================================================

exclude_exact = {
    "CNRM-CM6",
    "UKMO-GA7.1",
    "WRF-CRM",
}
exclude_exact = {
}

def keep_model(model):
    if model in exclude_exact:
        return False
    if "GCM" in model:
        return False
    if model.startswith("DALES"):
        return False
    return True

kept = []
removed = []

for triplet in triplets:
    model = triplet[0]
    if keep_model(model):
        kept.append(triplet)
    else:
        removed.append(model)

triplets = kept

print("\nRemoved models:")
for m in removed:
    print(" ", m)

print(f"\nRemaining after filtering: {len(triplets)} models")
for model, _, _, _ in triplets:
    print(" ", model)

# =====================================================
# DERIVE PROFILES
# =====================================================

profiles = []

for model, f295, f300, f305 in triplets:
    try:
        T295, p295, cf295, S295, rho295 = load_profile(f295)
        T300, p300, cf300, S300, rho300 = load_profile(f300)
        T305, p305, cf305, S305, rho305 = load_profile(f305)

        d295 = derive(T295, p295, cf295, S295, rho295)
        d300 = derive(T300, p300, cf300, S300, rho300)
        d305 = derive(T305, p305, cf305, S305, rho305)

        profiles.append((model, d295, d300, d305))

        print(
            f"{model:30s} "
            f"imin295={d295['imin']:3d} "
            f"imin300={d300['imin']:3d} "
            f"imin305={d305['imin']:3d} "
            f"valid_tau295={np.sum(np.isfinite(d295['tau_l'])):3d} "
            f"valid_tau300={np.sum(np.isfinite(d300['tau_l'])):3d} "
            f"valid_tau305={np.sum(np.isfinite(d305['tau_l'])):3d}"
        )

    except Exception as e:
        print(f"Skipping {model}: {e}")

print(f"\nUsable models: {len(profiles)}")

# =====================================================
# VERTICAL PROFILE PLOTS, 4 MODELS PER FIGURE
# =====================================================

batch_size = 4
n_models = len(profiles)

for start in range(0, n_models, batch_size):
    batch = profiles[start:start + batch_size]
    n_batch = len(batch)

    fig, axs = plt.subplots(
        n_batch, 5,
        figsize=(15, 2.8 * n_batch),
        sharey=True,
        constrained_layout=True
    )

    if n_batch == 1:
        axs = np.array([axs])

    for i, (model, d295, d300, d305) in enumerate(batch):

        for d, color, ls, label in [
            (d295, "tab:green", ":", "295 K"),
            (d300, "tab:blue", "-", "300 K"),
            (d305, "tab:red", "--", "305 K"),
        ]:
            axs[i, 0].plot(d["cf_l"], T_l, lw=1.8, color=color, ls=ls, label=label if i == 0 else None)
            axs[i, 1].plot(d["S_l"], T_l, lw=1.8, color=color, ls=ls)
            axs[i, 2].plot(d["invS_l"], T_l, lw=1.8, color=color, ls=ls)
            axs[i, 3].plot(d["dp_l"], T_l, lw=1.8, color=color, ls=ls)
            axs[i, 4].plot(d["tau_l"], T_l, lw=1.8, color=color, ls=ls)

        axs[i, 0].set_title(model, fontsize=10)
        axs[i, 0].set_ylabel("Temperature [K]")

        for j in range(5):
            axs[i, j].grid(True, alpha=0.3)
            axs[i, j].set_ylim(260, 198)

    axs[-1, 0].set_xlabel("Cloud fraction")
    axs[-1, 1].set_xlabel(r"$-d\theta/dp$ [K Pa$^{-1}$]")
    axs[-1, 2].set_xlabel(r"$S_{\rm inv}$")
    axs[-1, 3].set_xlabel(r"$\Delta p$")
    axs[-1, 4].set_xlabel(r"$\tau_{i,\rm sub}$")

    axs[0, 0].legend(frameon=False, loc="lower left")
    fig.suptitle(f"RCEMIP {domain}, {CF_MODE}: models {start+1}-{start+n_batch}", fontsize=14)

    if SAVE_FIGS:
        fig.savefig(
            f"{OUT_PREFIX}_profiles_{start+1:02d}_{start+n_batch:02d}.pdf",
            bbox_inches="tight",
            dpi=300
        )

    plt.show()

# =====================================================
# COMPUTE ISOTHERM-WISE dln FOR BOTH INCREMENTS, THEN AVERAGE
# (all now in true %/K units, per-Kelvin, via the updated dln() above)
# =====================================================

# =====================================================
# COMPUTE ISOTHERM-WISE dln FOR BOTH INCREMENTS, PER MODEL
# (all now in true %/K units, per-Kelvin, via the updated dln() above)
#
# Computed ONCE per model as a full T_l-indexed profile, then reused for
# either a range-average (like before) or single-isotherm slices -- so we
# can compare "average over 200-240K" against "exactly at 228K" (the
# manuscript's anvil-peak reference) and a few other isotherms, without
# recomputing anything from scratch each time.
# =====================================================

iso_data = {}  # model -> dict of T_l-indexed dln arrays

for model, d295, d300, d305 in profiles:
    dcf_300_295 = dln(d300["cf_l"], d295["cf_l"])
    dcf_305_300 = dln(d305["cf_l"], d300["cf_l"])
    dS_300_295 = dln(d300["invS_l"], d295["invS_l"])
    dS_305_300 = dln(d305["invS_l"], d300["invS_l"])
    dtau_300_295 = dln(d300["tau_l"], d295["tau_l"])
    dtau_305_300 = dln(d305["tau_l"], d300["tau_l"])
    dp_300_295 = dln(d300["p_l"], d295["p_l"])
    dp_305_300 = dln(d305["p_l"], d300["p_l"])
    dgamma_300_295 = dln(d300["gamma_l"], d295["gamma_l"])
    dgamma_305_300 = dln(d305["gamma_l"], d300["gamma_l"])

    iso_data[model] = {
        "cf": np.nanmean(np.stack([dcf_300_295, dcf_305_300]), axis=0),
        "S": np.nanmean(np.stack([dS_300_295, dS_305_300]), axis=0),
        "tau": np.nanmean(np.stack([dtau_300_295, dtau_305_300]), axis=0),
        "p": np.nanmean(np.stack([dp_300_295, dp_305_300]), axis=0),
        "gamma": np.nanmean(np.stack([dgamma_300_295, dgamma_305_300]), axis=0),
    }

VARS = ["S", "tau", "cf", "p", "gamma"]

def aggregate_range(Tmin, Tmax):
    """Average each model's per-isotherm dln arrays over [Tmin, Tmax]."""
    mask = (T_l <= Tmax) & (T_l >= Tmin)
    models_used, out = [], {v: [] for v in VARS}
    for model, d in iso_data.items():
        valid = mask.copy()
        for v in VARS:
            valid &= np.isfinite(d[v])
        if np.sum(valid) == 0:
            continue
        models_used.append(model)
        for v in VARS:
            out[v].append(np.nanmean(d[v][valid]))
    models_used = np.array(models_used)
    return models_used, {v: np.array(out[v]) for v in VARS}

def aggregate_isotherm(T_target):
    """Extract each model's dln value at the single T_l index nearest T_target."""
    idx = int(np.argmin(np.abs(T_l - T_target)))
    models_used, out = [], {v: [] for v in VARS}
    for model, d in iso_data.items():
        if not all(np.isfinite(d[v][idx]) for v in VARS):
            continue
        models_used.append(model)
        for v in VARS:
            out[v].append(d[v][idx])
    models_used = np.array(models_used)
    return models_used, {v: np.array(out[v]) for v in VARS}, T_l[idx]

def make_four_panel(models_used, agg, title_suffix, filename_suffix):
    """Standard 4-panel scatter (a-d), reused for every T-range/isotherm choice."""
    fig, axs = plt.subplots(1, 4, figsize=(22, 5.5), constrained_layout=True)
    panel_specs = [
        (agg["S"], agg["tau"], r"$\Delta \ln S_{\rm inv}$ [%K$^{-1}$]",
         r"$\Delta \ln \tau_{i,\rm sub}$ [%K$^{-1}$]", "(a) Thermodynamic buffering"),
        (agg["tau"], agg["cf"], r"$\Delta \ln \tau_{i,\rm sub}$ [%K$^{-1}$]",
         r"$\Delta \ln \mathrm{CF}$ [%K$^{-1}$]", "(b) Cloud buffering"),
        (agg["p"], agg["S"], r"$\Delta \ln p$ [%K$^{-1}$]",
         r"$\Delta \ln S_{\rm inv}$ [%K$^{-1}$]", "(c) Does $S_{\\rm inv}$ track $p$?"),
        (agg["gamma"], agg["tau"], r"$\Delta \ln \Gamma$ [%K$^{-1}$]",
         r"$\Delta \ln \tau_{i,\rm sub}$ [%K$^{-1}$]", "(d) Does $\\tau_{i,\\rm sub}$ track $\\Gamma$?"),
    ]
    results = {}
    for ax, (x, y, xlabel, ylabel, title) in zip(axs, panel_specs):
        xp, yp = x * 100, y * 100
        ax.scatter(xp, yp, s=80, edgecolor="k")
        lims = setup_lims(ax, xp, yp)
        slope, intercept, r2 = fit_stats(xp, yp)
        results[title] = (slope, r2)
        if np.isfinite(slope):
            xx = np.linspace(lims[0], lims[1], 200)
            ax.plot(xx, slope * xx + intercept, color="red", label=fr"Fit ($R^2$={r2:.2f})")
        for xi, yi, name in zip(xp, yp, models_used):
            if np.isfinite(xi) and np.isfinite(yi):
                ax.text(xi, yi, name, fontsize=7, ha="left", va="bottom")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        ax.text(0.05, 0.92, f"Slope = {slope:.2f}\n$R^2$ = {r2:.2f}\nN = {len(models_used)}",
                transform=ax.transAxes, va="top")

    fig.suptitle(f"RCEMIP {domain}, {CF_MODE}: {title_suffix}", fontsize=13)
    if SAVE_FIGS:
        fig.savefig(f"{OUT_PREFIX}_four_panel_{filename_suffix}.pdf", bbox_inches="tight", dpi=300)
    plt.show()
    return results

# =====================================================
# RUN BOTH: the original range-average, AND single-isotherm slices
# =====================================================

# (1) Range average, as before (200-240K)
models_range, agg_range = aggregate_range(AVG_TMIN, AVG_TMAX)
print(f"\n=== Range average {AVG_TMIN:.0f}-{AVG_TMAX:.0f}K (N={len(models_range)}) ===")
res_range = make_four_panel(models_range, agg_range,
                             f"mean %/K over {AVG_TMAX:.0f}-{AVG_TMIN:.0f} K (range average)",
                             "range_avg")
for title, (slope, r2) in res_range.items():
    print(f"  {title}: slope={slope:.2f}  R2={r2:.2f}")

# (2) Single isotherms: 228K (the manuscript's anvil-peak reference) plus a
# spread (220, 230, 240, 250) to see whether these relationships hold at one
# level, or are an artifact of averaging across a range.
isotherm_targets = [220, 228, 230, 240, 250]
isotherm_results = {}  # T -> {panel_title: (slope, r2)}

for T_target in isotherm_targets:
    models_iso, agg_iso, T_actual = aggregate_isotherm(T_target)
    print(f"\n=== Isotherm T={T_actual:.0f}K (N={len(models_iso)}) ===")
    res_iso = make_four_panel(models_iso, agg_iso,
                               f"exactly at T={T_actual:.0f}K (single isotherm)",
                               f"isotherm_{int(T_actual)}K")
    isotherm_results[T_actual] = res_iso
    for title, (slope, r2) in res_iso.items():
        print(f"  {title}: slope={slope:.2f}  R2={r2:.2f}")

# =====================================================
# SENSITIVITY SUMMARY: does slope/R^2 change across isotherms?
# One compact figure instead of squinting at 5 separate 4-panel figures.
# =====================================================

panel_titles = list(next(iter(isotherm_results.values())).keys())
Ts_sorted = sorted(isotherm_results.keys())

fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
for title in panel_titles:
    slopes = [isotherm_results[T][title][0] for T in Ts_sorted]
    r2s = [isotherm_results[T][title][1] for T in Ts_sorted]
    axs[0].plot(Ts_sorted, slopes, marker="o", label=title)
    axs[1].plot(Ts_sorted, r2s, marker="o", label=title)

axs[0].set_xlabel("Isotherm T [K]")
axs[0].set_ylabel("Slope")
axs[0].set_title("Does the slope change with height?")
axs[0].invert_xaxis()
axs[0].grid(True, alpha=0.3)
axs[0].legend(fontsize=7, frameon=False)

axs[1].set_xlabel("Isotherm T [K]")
axs[1].set_ylabel(r"$R^2$")
axs[1].set_title("Does the fit strength change with height?")
axs[1].invert_xaxis()
axs[1].set_ylim(0, 1)
axs[1].grid(True, alpha=0.3)
axs[1].legend(fontsize=7, frameon=False)

fig.suptitle(f"RCEMIP {domain}, {CF_MODE}: sensitivity of the four relationships to isotherm choice", fontsize=13)
if SAVE_FIGS:
    fig.savefig(f"{OUT_PREFIX}_isotherm_sensitivity.pdf", bbox_inches="tight", dpi=300)
plt.show()
