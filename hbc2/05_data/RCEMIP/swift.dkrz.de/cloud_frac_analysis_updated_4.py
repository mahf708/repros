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

BASE_DIR = "RCE_small"   # "RCE_small" or "RCE_large"
CF_MODE = "cfv0"         # "cfv0" or "cfv1-cfv2"

SAVE_FIGS = True
OUT_PREFIX = f"RCEMIP_{BASE_DIR}_{CF_MODE}"

T_l = np.arange(295.0, 197.0, -1.0)

AVG_TMIN = 200.0
AVG_TMAX = 240.0

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
    denom = 0.5 * (x1 + x0)
    out = np.full_like(denom, np.nan, dtype=float)
    valid = np.isfinite(x0) & np.isfinite(x1) & (denom != 0)
    out[valid] = (x1[valid] - x0[valid]) / denom[valid]
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
    cf_l, imin = interp_branch(T, cf)
    p_l, _ = interp_branch(T, p)
    S_l, _ = interp_branch(T, S)
    rho_l, _ = interp_branch(T, rho)

    gamma = (1.0 - S_l * rho_l) * gam_d
    gamma[~np.isfinite(gamma)] = np.nan
    gamma[gamma <= 0] = np.nan

    denom = (1.0 / gamma) - (1.0 / gam_d)
    invS = np.full_like(T_l, np.nan, dtype=float)
    good = np.isfinite(denom) & (np.abs(denom) > 1e-8)
    invS[good] = 1.0 / denom[good]

    dZ = np.full_like(T_l, np.nan, dtype=float)
    good_g = np.isfinite(gamma) & (gamma > 0)
    dZ[good_g] = 1.0 / gamma[good_g]

    dp = dZ * g * rho_l

    tau = np.full_like(T_l, np.nan, dtype=float)
    good_tau = np.isfinite(invS) & np.isfinite(dp) & (dp != 0)
    tau[good_tau] = invS[good_tau] / dp[good_tau]

    return {
        "cf_l": cf_l,
        "p_l": p_l,
        "S_l": S_l,
        "rho_l": rho_l,
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
# =====================================================

avg_mask = (T_l <= AVG_TMAX) & (T_l >= AVG_TMIN)

models_used = []
dln_CF_avg = []
dln_S_avg = []
dln_tau_avg = []

for model, d295, d300, d305 in profiles:

    dcf_300_295 = dln(d300["cf_l"], d295["cf_l"])
    dcf_305_300 = dln(d305["cf_l"], d300["cf_l"])

    dS_300_295 = dln(d300["invS_l"], d295["invS_l"])
    dS_305_300 = dln(d305["invS_l"], d300["invS_l"])

    dtau_300_295 = dln(d300["tau_l"], d295["tau_l"])
    dtau_305_300 = dln(d305["tau_l"], d300["tau_l"])

    dcf_iso = np.nanmean(np.stack([dcf_300_295, dcf_305_300]), axis=0)
    dS_iso = np.nanmean(np.stack([dS_300_295, dS_305_300]), axis=0)
    dtau_iso = np.nanmean(np.stack([dtau_300_295, dtau_305_300]), axis=0)

    valid = (
        np.isfinite(dcf_iso)
        & np.isfinite(dS_iso)
        & np.isfinite(dtau_iso)
        & avg_mask
    )

    if np.sum(valid) == 0:
        print(f"Skipping {model}: no valid dln points in {AVG_TMAX:.0f}-{AVG_TMIN:.0f} K")
        continue

    models_used.append(model)
    dln_CF_avg.append(np.nanmean(dcf_iso[valid]))
    dln_S_avg.append(np.nanmean(dS_iso[valid]))
    dln_tau_avg.append(np.nanmean(dtau_iso[valid]))

models_used = np.array(models_used)
dln_CF_avg = np.array(dln_CF_avg)
dln_S_avg = np.array(dln_S_avg)
dln_tau_avg = np.array(dln_tau_avg)

print("\nLayer-mean dln values:")
for m, s, t, c in zip(models_used, dln_S_avg, dln_tau_avg, dln_CF_avg):
    print(f"{m:25s} dlnS={s: .3f} dlnTau={t: .3f} dlnCF={c: .3f}")

# =====================================================
# TWO-STEP BUFFERING SCATTER WITH MODEL LABELS
# =====================================================

fig, axs = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)

# -------------------------
# Panel a: tau_i_sub vs S_inv
# -------------------------

x1 = dln_S_avg
y1 = dln_tau_avg

axs[0].scatter(x1, y1, s=80, edgecolor="k")
lims1 = setup_lims(axs[0], x1, y1)

slope1, intercept1, r2_1 = fit_stats(x1, y1)
if np.isfinite(slope1):
    xx = np.linspace(lims1[0], lims1[1], 200)
    axs[0].plot(xx, slope1 * xx + intercept1, color="red", label=fr"Fit ($R^2$={r2_1:.2f})")

for xi, yi, name in zip(x1, y1, models_used):
    if np.isfinite(xi) and np.isfinite(yi):
        axs[0].text(xi, yi, name, fontsize=7, ha="left", va="bottom")

axs[0].set_xlabel(r"$\overline{\Delta \ln S_{\rm inv}}$")
axs[0].set_ylabel(r"$\overline{\Delta \ln \tau_{i,\rm sub}}$")
axs[0].set_title("(a) Thermodynamic buffering")
axs[0].grid(True, alpha=0.3)
axs[0].legend(frameon=False)

axs[0].text(
    0.05, 0.92,
    f"Slope = {slope1:.2f}\n$R^2$ = {r2_1:.2f}\nN = {len(models_used)}",
    transform=axs[0].transAxes,
    va="top"
)

# -------------------------
# Panel b: CF vs tau_i_sub
# -------------------------

x2 = dln_tau_avg
y2 = dln_CF_avg

axs[1].scatter(x2, y2, s=80, edgecolor="k")
lims2 = setup_lims(axs[1], x2, y2)

slope2, intercept2, r2_2 = fit_stats(x2, y2)
if np.isfinite(slope2):
    xx = np.linspace(lims2[0], lims2[1], 200)
    axs[1].plot(xx, slope2 * xx + intercept2, color="red", label=fr"Fit ($R^2$={r2_2:.2f})")

for xi, yi, name in zip(x2, y2, models_used):
    if np.isfinite(xi) and np.isfinite(yi):
        axs[1].text(xi, yi, name, fontsize=7, ha="left", va="bottom")

axs[1].set_xlabel(r"$\overline{\Delta \ln \tau_{i,\rm sub}}$")
axs[1].set_ylabel(r"$\overline{\Delta \ln \mathrm{CF}}$")
axs[1].set_title("(b) Cloud buffering")
axs[1].grid(True, alpha=0.3)
axs[1].legend(frameon=False)

axs[1].text(
    0.05, 0.92,
    f"Slope = {slope2:.2f}\n$R^2$ = {r2_2:.2f}\nN = {len(models_used)}",
    transform=axs[1].transAxes,
    va="top"
)

fig.suptitle(
    f"RCEMIP {domain}, {CF_MODE}: mean dln over {AVG_TMAX:.0f}-{AVG_TMIN:.0f} K",
    fontsize=14
)

if SAVE_FIGS:
    fig.savefig(f"{OUT_PREFIX}_two_step_buffering.pdf", bbox_inches="tight", dpi=300)

plt.show()
