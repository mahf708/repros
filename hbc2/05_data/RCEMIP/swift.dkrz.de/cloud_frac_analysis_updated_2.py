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

BASE_DIR = "RCE_large"   # change to "RCE_large" if needed
CF_MODE = "cfv0"         # "cfv0" or "cfv1-cfv2"

T_l = np.arange(295.0, 197.0, -1.0)   # 295, 294, ..., 198 K

AVG_TMIN = 200.0
AVG_TMAX = 240.0

# constants
Rd = 287.0
cp = 1004.0
g = 9.81
p0 = 100000.0
kappa = Rd / cp
gam_d = g / cp

# =====================================================
# CF MODE SETTINGS
# =====================================================

if BASE_DIR.endswith("small"):
    domain = "small"
elif BASE_DIR.endswith("large"):
    domain = "large"
else:
    raise ValueError("BASE_DIR should end with 'small' or 'large'.")

if CF_MODE == "cfv0":
    CF_VAR = "cfv0_avg"
    pattern300 = f"*_RCE_{domain}300_cfv0-profiles.nc"
    suffix300 = f"_RCE_{domain}300_cfv0-profiles.nc"
    suffix305 = f"_RCE_{domain}305_cfv0-profiles.nc"
else:
    CF_VAR = "cfv1_avg"
    pattern300 = f"*_RCE_{domain}300_cfv1-cfv2-profiles.nc"
    suffix300 = f"_RCE_{domain}300_cfv1-cfv2-profiles.nc"
    suffix305 = f"_RCE_{domain}305_cfv1-cfv2-profiles.nc"

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
                return arr, name
    raise KeyError(f"Could not find any of {names} in dataset")

def get_pressure_pa(ds):
    p_raw, pname = get_1d_var(ds, ["pa_avg", "pa"])
    units = ds[pname].attrs.get("units", "").lower()

    if "hpa" in units or "mb" in units or (units == "" and np.nanmax(p_raw) < 2000):
        return p_raw * 100.0
    return p_raw

def load_profile(path, cf_var=CF_VAR):
    ds = xr.open_dataset(path)

    T, _ = get_1d_var(ds, ["ta_avg", "ta"])
    p = get_pressure_pa(ds)
    cf, _ = get_1d_var(ds, [cf_var, "cfv0_avg", "cfv1_avg", "cfv2_avg", "cfv0", "cfv1", "cfv2"])

    theta = T * (p0 / p) ** kappa
    S = -np.gradient(theta, p, edge_order=2)
    rho = p / (Rd * T)

    ds.close()
    return T.astype(float), p.astype(float), cf.astype(float), S.astype(float), rho.astype(float)

def interp_monotonic_branch(T, X, T_target):
    imin = np.argmin(T)

    T_cut = np.asarray(T[:imin+1], float)
    X_cut = np.asarray(X[:imin+1], float)

    valid = np.isfinite(T_cut) & np.isfinite(X_cut)
    T_cut = T_cut[valid]
    X_cut = X_cut[valid]

    if len(T_cut) < 2:
        return np.full_like(T_target, np.nan, dtype=float), imin

    order = np.argsort(T_cut)
    T_use = T_cut[order]
    X_use = X_cut[order]

    T_unique, idx = np.unique(T_use, return_index=True)
    X_unique = X_use[idx]

    if len(T_unique) < 2:
        return np.full_like(T_target, np.nan, dtype=float), imin

    f = interp1d(
        T_unique,
        X_unique,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan
    )

    X_l = f(T_target)
    X_l[(T_target < T_unique.min()) | (T_target > T_unique.max())] = np.nan
    return X_l, imin

def derive_Tcoord(T, p, cf, S, rho, T_target):
    cf_l, imin = interp_monotonic_branch(T, cf,  T_target)
    p_l, _     = interp_monotonic_branch(T, p,   T_target)
    S_l, _     = interp_monotonic_branch(T, S,   T_target)
    rho_l, _   = interp_monotonic_branch(T, rho, T_target)

    gamma_l = (1.0 - S_l * rho_l) * gam_d
    gamma_l[~np.isfinite(gamma_l)] = np.nan
    gamma_l[gamma_l <= 0] = np.nan

    denom = (1.0 / gamma_l) - (1.0 / gam_d)
    invS_l = np.full_like(T_target, np.nan, dtype=float)
    good = np.isfinite(denom) & (np.abs(denom) > 1e-8)
    invS_l[good] = 1.0 / denom[good]

    dZ_l = np.full_like(T_target, np.nan, dtype=float)
    good_g = np.isfinite(gamma_l) & (gamma_l > 0)
    dZ_l[good_g] = 1.0 / gamma_l[good_g]

    dp_l = dZ_l * g * rho_l

    tau_i_sub_l = np.full_like(T_target, np.nan, dtype=float)
    good_tau = np.isfinite(invS_l) & np.isfinite(dp_l) & (dp_l != 0)
    tau_i_sub_l[good_tau] = invS_l[good_tau] / dp_l[good_tau]

    return {
        "cf_l": cf_l,
        "p_l": p_l,
        "S_l": S_l,
        "rho_l": rho_l,
        "gamma_l": gamma_l,
        "invS_l": invS_l,
        "dp_l": dp_l,
        "tau_i_sub_l": tau_i_sub_l,
        "imin": imin,
    }

def fit_stats(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 3:
        return np.nan, np.nan, np.nan, np.nan
    slope, intercept, r_value, p_value, std_err = linregress(x[valid], y[valid])
    return slope, intercept, r_value**2, p_value

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
# FIND MATCHED PAIRS
# =====================================================

files300 = sorted(glob.glob(os.path.join(BASE_DIR, pattern300)))

pairs = []
for f300 in files300:
    stem300 = os.path.basename(f300)
    model = stem300.replace(suffix300, "")
    f305 = os.path.join(BASE_DIR, f"{model}{suffix305}")
    if os.path.exists(f305):
        pairs.append((model, f300, f305))

print(f"Found {len(pairs)} matched 300K/305K pairs.")

# =====================================================
# FILTER OUT PROBLEMATIC MODELS
# =====================================================

exclude_exact = {
    "CNRM-CM6",
    "FV3",
    "IPSL-CM6",
    "UCLA-CRM",
    "UKMO-GA7.1"}
#    "WRF-CRM",
#}
#exclude_exact = {"ICON-LEM-CRM","ICON-NWP-CRM","MPAS","FV3"}
#exclude_exact = {} 
def keep_model(model):
    if model in exclude_exact:
        return False
    if "GCM" in model:
        return False
    if model.startswith("DALES"):
        return False
    return True

pairs = [pair for pair in pairs if keep_model(pair[0])]

print(f"Remaining after filtering: {len(pairs)} models")
for model, _, _ in pairs:
    print(" ", model)

# =====================================================
# DERIVE T-COORD PROFILES
# =====================================================

profiles = []

for model, f300, f305 in pairs:
    try:
        T300, p300, cf300, S300, rho300 = load_profile(f300, cf_var=CF_VAR)
        T305, p305, cf305, S305, rho305 = load_profile(f305, cf_var=CF_VAR)

        tc300 = derive_Tcoord(T300, p300, cf300, S300, rho300, T_l)
        tc305 = derive_Tcoord(T305, p305, cf305, S305, rho305, T_l)

        profiles.append((model, tc300, tc305))

        print(
            f"{model:30s} "
            f"imin300={tc300['imin']:3d} imin305={tc305['imin']:3d} "
            f"valid tau300={np.sum(np.isfinite(tc300['tau_i_sub_l'])):3d} "
            f"valid tau305={np.sum(np.isfinite(tc305['tau_i_sub_l'])):3d}"
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

    for i, (model, d300, d305) in enumerate(batch):
        axs[i, 0].plot(d300["cf_l"], T_l, lw=1.8, color="tab:blue", label="300 K")
        axs[i, 0].plot(d305["cf_l"], T_l, lw=1.8, color="tab:red", ls="--", label="305 K")
        axs[i, 0].set_title(model, fontsize=10)
        axs[i, 0].set_ylabel("Temperature [K]")

        axs[i, 1].plot(d300["S_l"], T_l, lw=1.8, color="tab:blue")
        axs[i, 1].plot(d305["S_l"], T_l, lw=1.8, color="tab:red", ls="--")

        axs[i, 2].plot(d300["invS_l"], T_l, lw=1.8, color="tab:blue")
        axs[i, 2].plot(d305["invS_l"], T_l, lw=1.8, color="tab:red", ls="--")

        axs[i, 3].plot(d300["dp_l"], T_l, lw=1.8, color="tab:blue")
        axs[i, 3].plot(d305["dp_l"], T_l, lw=1.8, color="tab:red", ls="--")

        axs[i, 4].plot(d300["tau_i_sub_l"], T_l, lw=1.8, color="tab:blue")
        axs[i, 4].plot(d305["tau_i_sub_l"], T_l, lw=1.8, color="tab:red", ls="--")

        for j in range(5):
            axs[i, j].grid(True, alpha=0.3)
            axs[i, j].set_ylim(260, 198)

    axs[-1, 0].set_xlabel("Cloud fraction")
    axs[-1, 1].set_xlabel(r"$-d\theta/dp$ [K Pa$^{-1}$]")
    axs[-1, 2].set_xlabel(r"$S_{\rm inv}$")
    axs[-1, 3].set_xlabel(r"$\Delta p$")
    axs[-1, 4].set_xlabel(r"$\tau_{i,\rm sub}$")

    axs[0, 0].legend(frameon=False, loc="lower left")
    fig.suptitle(f"RCEMIP {domain}, {CF_MODE}: models {start+1}–{start+n_batch}", fontsize=14)

    plt.show()

# =====================================================
# COMPUTE ISOTHERM-WISE dln, THEN AVERAGE 240-200 K
# =====================================================

avg_mask = (T_l <= AVG_TMAX) & (T_l >= AVG_TMIN)

models_used = []
dln_CF_avg = []
dln_S_avg = []
dln_tau_avg = []

for model, d300, d305 in profiles:
    dln_cf_iso = dln(d305["cf_l"], d300["cf_l"])
    dln_S_iso = dln(d305["invS_l"], d300["invS_l"])
    dln_tau_iso = dln(d305["tau_i_sub_l"], d300["tau_i_sub_l"])

    valid = (
        np.isfinite(dln_cf_iso)
        & np.isfinite(dln_S_iso)
        & np.isfinite(dln_tau_iso)
        & avg_mask
    )

    if np.sum(valid) == 0:
        print(f"Skipping {model}: no valid points in {AVG_TMAX:.0f}-{AVG_TMIN:.0f} K")
        continue

    models_used.append(model)
    dln_CF_avg.append(np.nanmean(dln_cf_iso[valid]))
    dln_S_avg.append(np.nanmean(dln_S_iso[valid]))
    dln_tau_avg.append(np.nanmean(dln_tau_iso[valid]))

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

slope1, intercept1, r2_1, pval1 = fit_stats(x1, y1)
if np.isfinite(slope1):
    xx = np.linspace(lims1[0], lims1[1], 200)
    axs[0].plot(xx, slope1 * xx + intercept1, color="red", label=fr"Fit ($R^2$={r2_1:.2f})")

for xi, yi, name in zip(x1, y1, models_used):
    if np.isfinite(xi) and np.isfinite(yi):
        axs[0].text(xi, yi, name, fontsize=7, ha="left", va="bottom")

axs[0].set_xlabel(r"$\overline{\Delta \ln S_{\rm inv}}$")
axs[0].set_ylabel(r"$\overline{\Delta \ln \tau_{i,\rm sub}}$")
axs[0].set_title(r"(a) Thermodynamic buffering")
axs[0].grid(True, alpha=0.3)
axs[0].legend(frameon=False, loc="best")
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

slope2, intercept2, r2_2, pval2 = fit_stats(x2, y2)
if np.isfinite(slope2):
    xx = np.linspace(lims2[0], lims2[1], 200)
    axs[1].plot(xx, slope2 * xx + intercept2, color="red", label=fr"Fit ($R^2$={r2_2:.2f})")

for xi, yi, name in zip(x2, y2, models_used):
    if np.isfinite(xi) and np.isfinite(yi):
        axs[1].text(xi, yi, name, fontsize=7, ha="left", va="bottom")

axs[1].set_xlabel(r"$\overline{\Delta \ln \tau_{i,\rm sub}}$")
axs[1].set_ylabel(r"$\overline{\Delta \ln \mathrm{CF}}$")
axs[1].set_title(r"(b) Cloud buffering")
axs[1].grid(True, alpha=0.3)
axs[1].legend(frameon=False, loc="best")
axs[1].text(
    0.05, 0.92,
    f"Slope = {slope2:.2f}\n$R^2$ = {r2_2:.2f}\nN = {len(models_used)}",
    transform=axs[1].transAxes,
    va="top"
)

fig.suptitle(f"RCEMIP {domain}, {CF_MODE}: mean dln over {AVG_TMAX:.0f}-{AVG_TMIN:.0f} K", fontsize=14)

plt.show()
