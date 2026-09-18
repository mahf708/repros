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

BASE_DIR = "RCE_small"   # change to "RCE_large" if needed
CF_MODE = "cfv0"         # "cfv0" or "cfv1-cfv2"

# Common temperature coordinate
T_l = np.arange(295.0, 197.0, -1.0)   # 295, 294, ..., 198 K

# Layer over which to average dln values
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

if CF_MODE == "cfv0":
    CF_VAR = "cfv0_avg"
    pattern300 = "*_RCE_small300_cfv0-profiles.nc" if BASE_DIR.endswith("small") else "*_RCE_large300_cfv0-profiles.nc"
    suffix300 = "_RCE_small300_cfv0-profiles.nc" if BASE_DIR.endswith("small") else "_RCE_large300_cfv0-profiles.nc"
    suffix305 = "_RCE_small305_cfv0-profiles.nc" if BASE_DIR.endswith("small") else "_RCE_large305_cfv0-profiles.nc"
else:
    CF_VAR = "cfv1_avg"
    pattern300 = "*_RCE_small300_cfv1-cfv2-profiles.nc" if BASE_DIR.endswith("small") else "*_RCE_large300_cfv1-cfv2-profiles.nc"
    suffix300 = "_RCE_small300_cfv1-cfv2-profiles.nc" if BASE_DIR.endswith("small") else "_RCE_large300_cfv1-cfv2-profiles.nc"
    suffix305 = "_RCE_small305_cfv1-cfv2-profiles.nc" if BASE_DIR.endswith("small") else "_RCE_large305_cfv1-cfv2-profiles.nc"

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
    S = -np.gradient(theta, p, edge_order=2)   # positive stability [K Pa^-1]
    rho = p / (Rd * T)

    ds.close()
    return T.astype(float), p.astype(float), cf.astype(float), S.astype(float), rho.astype(float)

def interp_monotonic_branch(T, X, T_target):
    """
    Use only the first monotonic cooling branch:
    from index 0 to first Tmin.
    """
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

# =====================================================
# FIND MATCHED 300/305 FILE PAIRS
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

exclude_exact = {
    "CNRM-CM6",
    "UKMO-GA7.1",
    "WRF-CRM",
    "FV3"  
}
#exclude_exact = {}

def keep_model(model):
    if model in exclude_exact:
        return False
    #if "GCM" in model:
    #    return False
    if model.startswith("DALES"):
        return False
    return True

pairs = [pair for pair in pairs if keep_model(pair[0])]

print("Keeping models:")
for model, _, _ in pairs:
    print(" ", model)
# =====================================================
# FILTER OUT PROBLEMATIC MODELS
# =====================================================


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

    except Exception as e:
        print(f"Skipping {model}: {e}")

print(f"\nUsable models: {len(profiles)}")

# =====================================================
# COMPUTE dln ON EACH ISOTHERM, THEN AVERAGE 240–200 K
# =====================================================

avg_mask = (T_l <= AVG_TMAX) & (T_l >= AVG_TMIN)

dln_CF_avg = []
dln_tau_avg = []
models_used = []

for model, d300, d305 in profiles:
    dln_cf_iso = dln(d305["cf_l"], d300["cf_l"])
    dln_tau_iso = dln(d305["tau_i_sub_l"], d300["tau_i_sub_l"])

    valid = np.isfinite(dln_cf_iso) & np.isfinite(dln_tau_iso) & avg_mask
    if np.sum(valid) == 0:
        print(f"Skipping {model}: no valid points in 240–200 K")
        continue

    dln_CF_avg.append(np.nanmean(dln_cf_iso[valid]))
    dln_tau_avg.append(np.nanmean(dln_tau_iso[valid]))
    models_used.append(model)

dln_CF_avg = np.array(dln_CF_avg)
dln_tau_avg = np.array(dln_tau_avg)

print(f"\nModels used in scatter: {len(models_used)}")
for m, xval, yval in zip(models_used, dln_tau_avg, dln_CF_avg):
    print(f"{m:25s}  dln_tau={xval: .3f}  dln_CF={yval: .3f}")

# =====================================================
# REGRESSION
# =====================================================

mask = np.isfinite(dln_CF_avg) & np.isfinite(dln_tau_avg)
x = dln_tau_avg[mask]
y = dln_CF_avg[mask]
models_plot = np.array(models_used)[mask]

slope, intercept, r_value, p_value, std_err = linregress(x, y)
r2 = r_value**2

print(f"\nSlope = {slope:.2f}")
print(f"R^2   = {r2:.2f}")
print(f"N     = {len(x)}")

# =====================================================
# PLOT
# =====================================================

fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

ax.scatter(x, y, s=90, edgecolor="k")

lims = [min(np.min(x), np.min(y)), max(np.max(x), np.max(y))]
pad = 0.05 * (lims[1] - lims[0]) if lims[1] > lims[0] else 0.1
lims = [lims[0] - pad, lims[1] + pad]

ax.plot(lims, lims, "k--", label="1:1")

x_fit = np.linspace(lims[0], lims[1], 200)
y_fit = slope * x_fit + intercept
ax.plot(x_fit, y_fit, color="red", label=fr"Fit ($R^2$={r2:.2f})")

ax.set_xlim(lims)
ax.set_ylim(lims)

ax.set_xlabel(r"$\overline{\Delta \ln \tau_{i,\rm sub}}$")
ax.set_ylabel(r"$\overline{\Delta \ln \mathrm{CF}}$")
ax.set_title("RCEMIP models: mean dln over 240–200 K")

ax.text(
    0.05, 0.92,
    f"Slope = {slope:.2f}\n$R^2$ = {r2:.2f}\nN = {len(x)}",
    transform=ax.transAxes,
    va="top"
)

ax.grid(True, alpha=0.3)
ax.legend(frameon=False)

plt.show()
