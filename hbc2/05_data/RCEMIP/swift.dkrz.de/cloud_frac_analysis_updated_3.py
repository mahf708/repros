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

BASE_DIR = "RCE_large"   # or "RCE_large"
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
# CF MODE SETTINGS
# =====================================================

domain = "small" if "small" in BASE_DIR else "large"

if CF_MODE == "cfv0":
    CF_VAR = "cfv0_avg"
    suffix295 = f"_RCE_{domain}295_cfv0-profiles.nc"
    suffix300 = f"_RCE_{domain}300_cfv0-profiles.nc"
    suffix305 = f"_RCE_{domain}305_cfv0-profiles.nc"
    pattern300 = f"*{suffix300}"
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
    out = np.full_like(denom, np.nan)
    valid = np.isfinite(x0) & np.isfinite(x1) & (denom != 0)
    out[valid] = (x1[valid] - x0[valid]) / denom[valid]
    return out

def get_1d_var(ds, names):
    for name in names:
        if name in ds.variables:
            arr = np.asarray(ds[name]).squeeze()
            if arr.ndim == 1:
                return arr
    raise KeyError

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

    theta = T * (p0 / p)**kappa
    S = -np.gradient(theta, p)
    rho = p / (Rd * T)

    ds.close()
    return T, p, cf, S, rho

def interp_branch(T, X):
    imin = np.argmin(T)
    T_cut = T[:imin+1]
    X_cut = X[:imin+1]

    order = np.argsort(T_cut)
    T_use = T_cut[order]
    X_use = X_cut[order]

    T_unique, idx = np.unique(T_use, return_index=True)
    X_unique = X_use[idx]

    f = interp1d(T_unique, X_unique, bounds_error=False, fill_value=np.nan)
    X_l = f(T_l)
    X_l[(T_l < T_unique.min()) | (T_l > T_unique.max())] = np.nan

    return X_l

def derive(T, p, cf, S, rho):
    cf_l = interp_branch(T, cf)
    p_l = interp_branch(T, p)
    S_l = interp_branch(T, S)
    rho_l = interp_branch(T, rho)

    gamma = (1 - S_l * rho_l) * gam_d
    gamma[gamma <= 0] = np.nan

    denom = (1/gamma) - (1/gam_d)
    invS = np.full_like(denom, np.nan)
    good = np.isfinite(denom)
    invS[good] = 1 / denom[good]

    dZ = 1/gamma
    dp = dZ * g * rho_l

    tau = np.full_like(dp, np.nan)
    good = np.isfinite(invS) & np.isfinite(dp) & (dp != 0)
    tau[good] = invS[good] / dp[good]

    return dict(cf_l=cf_l, invS_l=invS, tau_l=tau)

# =====================================================
# FIND TRIPLETS
# =====================================================

files300 = glob.glob(os.path.join(BASE_DIR, pattern300))
triplets = []

for f300 in files300:
    model = os.path.basename(f300).replace(suffix300, "")
    f295 = os.path.join(BASE_DIR, model + suffix295)
    f305 = os.path.join(BASE_DIR, model + suffix305)

    if os.path.exists(f295) and os.path.exists(f305):
        triplets.append((model, f295, f300, f305))

print("Triplets:", len(triplets))

# =====================================================
# FILTER MODELS
# =====================================================

def keep(model):
    if "GCM" in model: return False
    if model.startswith("DALES"): return False
    if model in ["CNRM-CM6", "UKMO-GA7.1", "WRF-CRM"]: return False
    return True

triplets = [t for t in triplets if keep(t[0])]

# =====================================================
# DERIVE PROFILES
# =====================================================

profiles = []

for model, f295, f300, f305 in triplets:
    T295, p295, cf295, S295, rho295 = load_profile(f295)
    T300, p300, cf300, S300, rho300 = load_profile(f300)
    T305, p305, cf305, S305, rho305 = load_profile(f305)

    d295 = derive(T295, p295, cf295, S295, rho295)
    d300 = derive(T300, p300, cf300, S300, rho300)
    d305 = derive(T305, p305, cf305, S305, rho305)

    profiles.append((model, d295, d300, d305))

# =====================================================
# TWO-STEP SCATTER
# =====================================================

mask = (T_l <= AVG_TMAX) & (T_l >= AVG_TMIN)

X = []
Y = []
models = []

for model, d295, d300, d305 in profiles:
    dcf1 = dln(d300["cf_l"], d295["cf_l"])
    dcf2 = dln(d305["cf_l"], d300["cf_l"])

    dtau1 = dln(d300["tau_l"], d295["tau_l"])
    dtau2 = dln(d305["tau_l"], d300["tau_l"])

    dS1 = dln(d300["invS_l"], d295["invS_l"])
    dS2 = dln(d305["invS_l"], d300["invS_l"])

    dcf = np.nanmean([dcf1, dcf2], axis=0)
    dtau = np.nanmean([dtau1, dtau2], axis=0)
    dS = np.nanmean([dS1, dS2], axis=0)

    valid = np.isfinite(dcf) & np.isfinite(dtau) & mask

    if np.sum(valid) == 0:
        continue

    X.append(np.nanmean(dtau[valid]))
    Y.append(np.nanmean(dcf[valid]))
    models.append(model)

X = np.array(X)
Y = np.array(Y)

# =====================================================
# PLOT
# =====================================================

fig, ax = plt.subplots(figsize=(6,6))

ax.scatter(X, Y, s=80)

for xi, yi, name in zip(X, Y, models):
    ax.text(xi, yi, name, fontsize=8)

lims = [min(X.min(), Y.min()), max(X.max(), Y.max())]
ax.plot(lims, lims, 'k--')

slope, intercept, r, _, _ = linregress(X, Y)
ax.plot(lims, slope*np.array(lims)+intercept, 'r')

ax.set_xlabel("dln tau_i_sub")
ax.set_ylabel("dln CF")
ax.set_title("RCEMIP Scatter")

if SAVE_FIGS:
    fig.savefig(f"{OUT_PREFIX}_scatter.pdf", dpi=300)

plt.show()
