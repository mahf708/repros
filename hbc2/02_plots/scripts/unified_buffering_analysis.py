import os
import glob
import pickle
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import linregress, spearmanr

# =====================================================
# USER SETTINGS
# =====================================================

# --- RCEMIP ---
RCEMIP_PATH_PREFIX = "RCEMIP/swift.dkrz.de"  # RCE_small/RCE_large subdirectories are assumed to live here
RCEMIP_BASE_DIRS = [f"{RCEMIP_PATH_PREFIX}/RCE_small", f"{RCEMIP_PATH_PREFIX}/RCE_large"]
CF_MODE = "cfv0"

# --- AMIP (CMIP6 multi-model + SCREAM) ---
# Assumed to be in the current working directory alongside this script.
CMIP_CTRL_PATH = "CMIP_amip_19832008_dataset_RRTMG_pressure_Tcoord_1d_v3.pkl"
CMIP_P4K_PATH = "CMIP_amip4k_19832008_dataset_RRTMG_pressure_Tcoord_1d_v3_CNRM.pkl"  # confirmed: full 15-model ensemble despite the filename
SCREAM_CTRL_PATH = "SCREAM_ne1024_amip_dataset_RRTMG_pressure_Tcoord_1d_v3.pkl"
SCREAM_P4K_PATH = "SCREAM_ne1024_amip4k_dataset_RRTMG_pressure_Tcoord_1d_v3.pkl"

RCEMIP_DELTA_SST = 5.0  # RCEMIP's 295/300/305K triplet steps
AMIP_DELTA_SST = 4.0    # AMIP-future4K perturbation -- verify against your colleague's actual protocol

SAVE_FIGS = True
OUT_PREFIX = "Unified_buffering"

AVG_TMIN, AVG_TMAX = 200.0, 240.0
ISOTHERM_TARGETS = [220, 228, 230, 240, 250]

# Per-domain exclusions for RCEMIP -- defaults to the same list for both small/large;
# override independently if needed.
_DEFAULT_EXCLUDE = {"CNRM-CM6", "UKMO-GA7.1", "WRF-CRM"}
_DEFAULT_EXCLUDE = {}
EXCLUDE_BY_DOMAIN = {"small": set(_DEFAULT_EXCLUDE), "large": set(_DEFAULT_EXCLUDE)}

# Color/marker style per source, used in every plot so all four sources are
# visually distinguishable at a glance.
DOMAIN_STYLE = {
    "RCE_small": dict(marker="o", color="tab:blue", edgecolor="k", s=80, zorder=2),
    "RCE_large": dict(marker="o", color="tab:red", edgecolor="k", s=80, zorder=2),
    "AMIP_CMIP": dict(marker="o", color="tab:green", edgecolor="k", s=80, zorder=2),
    "AMIP_SCREAM": dict(marker="*", color="gold", edgecolor="k", s=350, zorder=3),        # "cess1" -- the earlier SCREAM experiment in the AMIP pickle
    "SCREAM_cess2": dict(marker="*", color="black", edgecolor="white", s=350, zorder=4),  # THIS paper's actual Cess-Potter simulations
}

# --- SCREAM cess2 (this paper's own data): already-cached .npz files from
# the earlier analysis_minimal.py run (run_new=True pass), so this does NOT
# need NERSC/lustre netCDF access at all -- just read access to these two
# files. Update PLUS4K path if the actual companion filename differs.
SCREAM_CESS2_CTRL_NPZ = "control_vars_minimal_full_record.npz"
SCREAM_CESS2_P4K_NPZ = "plus4k_vars_minimal_full_record.npz"
SCREAM_CESS2_DELTA_SST = 4.0  # matches this paper's Cess-Potter +4K perturbation
SCREAM_CESS2_T_l = np.linspace(197, 297, 100)  # NOTE: yet another distinct grid from
                                                # both RCEMIP (295->197) and AMIP (190-270)
                                                # -- handled generically via the per-entry
                                                # T_l design already in place.

Rd = 287.0
cp = 1004.0
g = 9.81
p0 = 100000.0
kappa = Rd / cp
gam_d = g / cp

VARS = ["S", "tau", "cf", "p", "gamma"]
VARS_ALL = VARS + ["S_baseline"]

# =====================================================
# SHARED HELPERS
# =====================================================

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

def make_dln(delta_sst):
    """Returns a dln() function with the given SST-step normalization baked
    in -- RCEMIP and AMIP use different perturbation sizes (5K vs 4K), so
    each source needs its own version rather than one shared constant."""
    def dln(x1, x0):
        denom = 0.5 * (x1 + x0)
        out = np.full_like(denom, np.nan, dtype=float)
        valid = np.isfinite(x0) & np.isfinite(x1) & (denom != 0)
        out[valid] = (x1[valid] - x0[valid]) / denom[valid] / delta_sst
        return out
    return dln

dln_rcemip = make_dln(RCEMIP_DELTA_SST)
dln_amip = make_dln(AMIP_DELTA_SST)

# =====================================================
# RCEMIP LOADING (unchanged physics from the combined-domain script, just
# folded into the unified iso_data structure below)
# =====================================================

RCEMIP_T_l = np.arange(295.0, 197.0, -1.0)  # RCEMIP's own grid -- NOTE: different
                                            # from AMIP's grid (190-270K, increasing).
                                            # Each iso_data entry stores its own T_l
                                            # array so aggregation works generically
                                            # across both sources without forcing a
                                            # false common grid.

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

def load_rcemip_profile(path, cf_var):
    ds = xr.open_dataset(path)
    T = get_1d_var(ds, ["ta_avg", "ta"])
    p = get_pressure_pa(ds)
    cf = get_1d_var(ds, [cf_var, "cfv0_avg", "cfv1_avg", "cfv2_avg"])
    theta = T * (p0 / p) ** kappa
    S = -np.gradient(theta, p, edge_order=2)
    rho = p / (Rd * T)
    ds.close()
    return T.astype(float), p.astype(float), cf.astype(float), S.astype(float), rho.astype(float)

def rcemip_interp_branch(T, X):
    T = np.asarray(T, float)
    X = np.asarray(X, float)
    # Most RCEMIP files store the profile surface-first (T decreasing toward
    # the end), so the "monotonic branch" up to the global temperature
    # minimum runs from the start of the array. A few models (GEOS-GCM,
    # FV3) store it top-of-atmosphere-first instead -- the opposite
    # convention -- which this function silently mishandled: argmin(T)
    # would land near the START of the array, collapsing T_cut to at most a
    # couple of points regardless of how good the rest of the profile is.
    # Detect the direction from the finite endpoints and flip onto the
    # surface-first convention before doing anything else, so the rest of
    # this function's logic (unchanged below) applies correctly either way.
    finite = np.isfinite(T)
    if finite.sum() >= 2:
        first_T = T[finite][0]
        last_T = T[finite][-1]
        if first_T < last_T:
            T = T[::-1]
            X = X[::-1]
    imin = np.argmin(T)
    T_cut = np.asarray(T[:imin+1], float)
    X_cut = np.asarray(X[:imin+1], float)
    valid = np.isfinite(T_cut) & np.isfinite(X_cut)
    T_cut, X_cut = T_cut[valid], X_cut[valid]
    if len(T_cut) < 2:
        return np.full_like(RCEMIP_T_l, np.nan, dtype=float), imin
    order = np.argsort(T_cut)
    T_use, X_use = T_cut[order], X_cut[order]
    T_unique, idx = np.unique(T_use, return_index=True)
    X_unique = X_use[idx]
    if len(T_unique) < 2:
        return np.full_like(RCEMIP_T_l, np.nan, dtype=float), imin
    f = interp1d(T_unique, X_unique, bounds_error=False, fill_value=np.nan)
    X_l = f(RCEMIP_T_l)
    X_l[(RCEMIP_T_l < T_unique.min()) | (RCEMIP_T_l > T_unique.max())] = np.nan
    return X_l, imin

def derive_rcemip(T, p, cf, S, rho):
    cf_l, imin = rcemip_interp_branch(T, cf)
    p_l, _ = rcemip_interp_branch(T, p)
    S_l, _ = rcemip_interp_branch(T, S)

    ro_l = p_l / (Rd * RCEMIP_T_l)
    theta_l = RCEMIP_T_l * (p0 / p_l) ** kappa

    gamma = gam_d * (1.0 - S_l * ro_l * cp * (RCEMIP_T_l / theta_l))
    gamma[~np.isfinite(gamma)] = np.nan
    gamma[gamma <= 0] = np.nan

    denom = (1.0 / gamma) - (1.0 / gam_d)
    invS = np.full_like(RCEMIP_T_l, np.nan, dtype=float)
    good = np.isfinite(denom) & (np.abs(denom) > 1e-8)
    invS[good] = 1.0 / denom[good]

    dZ = np.full_like(RCEMIP_T_l, np.nan, dtype=float)
    good_g = np.isfinite(gamma) & (gamma > 0)
    dZ[good_g] = 1.0 / gamma[good_g]
    dp = dZ * g * ro_l

    tau = np.full_like(RCEMIP_T_l, np.nan, dtype=float)
    good_tau = np.isfinite(invS) & np.isfinite(dp) & (dp != 0)
    tau[good_tau] = invS[good_tau] / dp[good_tau]

    return {"cf_l": cf_l, "p_l": p_l, "invS_l": invS, "gamma_l": gamma, "dp_l": dp, "tau_l": tau}

def load_rcemip_domain(base_dir, cf_mode):
    domain = "small" if "small" in base_dir else "large"
    if cf_mode == "cfv0":
        cf_var = "cfv0_avg"
        suffix295, suffix300, suffix305 = (f"_RCE_{domain}295_cfv0-profiles.nc",
                                            f"_RCE_{domain}300_cfv0-profiles.nc",
                                            f"_RCE_{domain}305_cfv0-profiles.nc")
    else:
        cf_var = "cfv1_avg"
        suffix295, suffix300, suffix305 = (f"_RCE_{domain}295_cfv1-cfv2-profiles.nc",
                                            f"_RCE_{domain}300_cfv1-cfv2-profiles.nc",
                                            f"_RCE_{domain}305_cfv1-cfv2-profiles.nc")

    pattern = f"*{suffix300}"
    abs_base_dir = os.path.abspath(base_dir)
    print(f"[RCEMIP-{domain}] Looking in: {abs_base_dir}")
    print(f"[RCEMIP-{domain}] Directory exists: {os.path.isdir(base_dir)}")
    print(f"[RCEMIP-{domain}] Glob pattern: {pattern}")

    files300 = sorted(glob.glob(os.path.join(base_dir, pattern)))

    if len(files300) == 0 and os.path.isdir(base_dir):
        sample = sorted(os.listdir(base_dir))[:10]
        print(f"[RCEMIP-{domain}] WARNING: directory exists but pattern matched 0 files. "
              f"First 10 items actually in this directory: {sample}")
    elif not os.path.isdir(base_dir):
        print(f"[RCEMIP-{domain}] WARNING: this directory does not exist at all -- "
              f"check RCEMIP_PATH_PREFIX and that the script is being run from the "
              f"expected working directory (relative paths are resolved from wherever "
              f"you launch the script, not from the script's own location).")

    triplets = []
    for f300 in files300:
        model = os.path.basename(f300).replace(suffix300, "")
        f295 = os.path.join(base_dir, model + suffix295)
        f305 = os.path.join(base_dir, model + suffix305)
        if os.path.exists(f295) and os.path.exists(f305):
            triplets.append((model, f295, f300, f305))

    print(f"[RCEMIP-{domain}] Found {len(triplets)} matched 295/300/305 triplets.")

    exclude = EXCLUDE_BY_DOMAIN.get(domain, set())

    def keep_model(model):
        if model in exclude:
            return False
        if "GCM" in model:
            return True
        if model.startswith("DALES"):
            return True
        return True

    kept, removed = [], []
    for triplet in triplets:
        if keep_model(triplet[0]):
            kept.append(triplet)
        else:
            removed.append(triplet[0])
    triplets = kept
    print(f"[RCEMIP-{domain}] Removed: {removed if removed else '(none)'}; remaining: {len(triplets)} models")

    domain_iso = {}
    for model, f295, f300, f305 in triplets:
        try:
            T295, p295, cf295, S295, rho295 = load_rcemip_profile(f295, cf_var)
            T300, p300, cf300, S300, rho300 = load_rcemip_profile(f300, cf_var)
            T305, p305, cf305, S305, rho305 = load_rcemip_profile(f305, cf_var)

            d295 = derive_rcemip(T295, p295, cf295, S295, rho295)
            d300 = derive_rcemip(T300, p300, cf300, S300, rho300)
            d305 = derive_rcemip(T305, p305, cf305, S305, rho305)

            dcf = np.nanmean(np.stack([dln_rcemip(d300["cf_l"], d295["cf_l"]), dln_rcemip(d305["cf_l"], d300["cf_l"])]), axis=0)
            dS = np.nanmean(np.stack([dln_rcemip(d300["invS_l"], d295["invS_l"]), dln_rcemip(d305["invS_l"], d300["invS_l"])]), axis=0)
            dtau = np.nanmean(np.stack([dln_rcemip(d300["tau_l"], d295["tau_l"]), dln_rcemip(d305["tau_l"], d300["tau_l"])]), axis=0)
            dp_ = np.nanmean(np.stack([dln_rcemip(d300["p_l"], d295["p_l"]), dln_rcemip(d305["p_l"], d300["p_l"])]), axis=0)
            dgamma = np.nanmean(np.stack([dln_rcemip(d300["gamma_l"], d295["gamma_l"]), dln_rcemip(d305["gamma_l"], d300["gamma_l"])]), axis=0)

            domain_iso[model] = {
                "cf": dcf, "S": dS, "tau": dtau, "p": dp_, "gamma": dgamma,
                "S_baseline": d300["invS_l"], "T_l": RCEMIP_T_l,
            }
        except Exception as e:
            print(f"[RCEMIP-{domain}] Skipping {model}: {e}")

    print(f"[RCEMIP-{domain}] Usable models: {len(domain_iso)}")
    return domain, domain_iso

# =====================================================
# AMIP LOADING (CMIP6 multi-model + SCREAM single-model)
# =====================================================

def load_pickle_dataset(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def amip_extract(var_dict, T_grid, model=None):
    da = var_dict[model] if model is not None else var_dict
    da = da.reindex(temp=T_grid)
    return da.values.astype(float)

def derive_amip(data, T_grid, model=None):
    cf = amip_extract(data["cl"], T_grid, model)
    Sinv = amip_extract(data["G"], T_grid, model)
    dp = amip_extract(data["dp"], T_grid, model)
    gamma = amip_extract(data["gamma"], T_grid, model)
    rho = amip_extract(data["rho"], T_grid, model)
    p = rho * Rd * T_grid
    tau = np.full_like(T_grid, np.nan, dtype=float)
    good = np.isfinite(Sinv) & np.isfinite(dp) & (dp != 0)
    tau[good] = Sinv[good] / dp[good]
    return {"cf_l": cf, "invS_l": Sinv, "gamma_l": gamma, "dp_l": dp, "p_l": p, "tau_l": tau}

def load_amip():
    print("[AMIP] Loading CMIP6 control/+4K...")
    cmip_ctrl = load_pickle_dataset(CMIP_CTRL_PATH)
    cmip_p4k = load_pickle_dataset(CMIP_P4K_PATH)
    print("[AMIP] Loading SCREAM control/+4K...")
    scream_ctrl = load_pickle_dataset(SCREAM_CTRL_PATH)
    scream_p4k = load_pickle_dataset(SCREAM_P4K_PATH)

    cmip_models = sorted(set(cmip_ctrl["G"].keys()) & set(cmip_p4k["G"].keys()))
    print(f"[AMIP] CMIP models (common to both states): {len(cmip_models)}")

    T_l_amip = cmip_ctrl["G"][cmip_models[0]]["temp"].values.astype(float)
    T_l_scream = scream_ctrl["G"]["temp"].values.astype(float)
    assert np.array_equal(T_l_amip, T_l_scream), "SCREAM and CMIP temperature grids differ"

    amip_iso = {}
    for model in cmip_models:
        d_ctrl = derive_amip(cmip_ctrl, T_l_amip, model)
        d_p4k = derive_amip(cmip_p4k, T_l_amip, model)
        amip_iso[(model, "AMIP_CMIP")] = {
            "S": dln_amip(d_p4k["invS_l"], d_ctrl["invS_l"]),
            "tau": dln_amip(d_p4k["tau_l"], d_ctrl["tau_l"]),
            "cf": dln_amip(d_p4k["cf_l"], d_ctrl["cf_l"]),
            "p": dln_amip(d_p4k["p_l"], d_ctrl["p_l"]),
            "gamma": dln_amip(d_p4k["gamma_l"], d_ctrl["gamma_l"]),
            "S_baseline": d_ctrl["invS_l"],
            "T_l": T_l_amip,
        }

    d_ctrl_scream = derive_amip(scream_ctrl, T_l_amip)
    d_p4k_scream = derive_amip(scream_p4k, T_l_amip)
    amip_iso[("SCREAM", "AMIP_SCREAM")] = {
        "S": dln_amip(d_p4k_scream["invS_l"], d_ctrl_scream["invS_l"]),
        "tau": dln_amip(d_p4k_scream["tau_l"], d_ctrl_scream["tau_l"]),
        "cf": dln_amip(d_p4k_scream["cf_l"], d_ctrl_scream["cf_l"]),
        "p": dln_amip(d_p4k_scream["p_l"], d_ctrl_scream["p_l"]),
        "gamma": dln_amip(d_p4k_scream["gamma_l"], d_ctrl_scream["gamma_l"]),
        "S_baseline": d_ctrl_scream["invS_l"],
        "T_l": T_l_amip,
    }
    return amip_iso

# =====================================================
# SCREAM cess2 LOADING
# =====================================================

def load_scream_cess2():
    print(f"[SCREAM cess2] Loading cached arrays from {SCREAM_CESS2_CTRL_NPZ} / {SCREAM_CESS2_P4K_NPZ}...")
    _ctrl = np.load(SCREAM_CESS2_CTRL_NPZ)
    _p4k = np.load(SCREAM_CESS2_P4K_NPZ)

    T, S, omega_dn, omega_up, CF = _ctrl['T'], _ctrl['S'], _ctrl['omega_dn'], _ctrl['omega_up'], _ctrl['CF']
    T_p4k, S_p4k, omega_dn_p4k, omega_up_p4k, CF_p4k = (
        _p4k['T'], _p4k['S'], _p4k['omega_dn'], _p4k['omega_up'], _p4k['CF'])
    p_mean = _ctrl['p_mean']

    def interpolate_to_T(var, var_p4k, ind=35, ind_p4k=27):
        x_ctrl, x_p4k = T[ind:], T_p4k[ind_p4k:]
        f_ctrl = interp1d(x_ctrl, var[ind:], kind='linear')
        f_p4k = interp1d(x_p4k, var_p4k[ind_p4k:], kind='linear')
        T_l_ctrl = np.clip(SCREAM_CESS2_T_l, x_ctrl.min(), x_ctrl.max())
        T_l_p4k = np.clip(SCREAM_CESS2_T_l, x_p4k.min(), x_p4k.max())
        return f_ctrl(T_l_ctrl), f_p4k(T_l_p4k)

    p_l, p_l_p4k = interpolate_to_T(p_mean, p_mean)
    S_l, S_l_p4k = interpolate_to_T(-S, -S_p4k)
    CF_l, CF_l_p4k = interpolate_to_T(CF, CF_p4k)
    omega_up_l, omega_up_l_p4k = interpolate_to_T(omega_up, omega_up_p4k)
    omega_dn_l, omega_dn_l_p4k = interpolate_to_T(omega_dn, omega_dn_p4k)

    ro = p_l / (Rd * SCREAM_CESS2_T_l)
    ro_p4k = p_l_p4k / (Rd * SCREAM_CESS2_T_l)

    gamma = (1 - S_l * ro) * gam_d
    gamma_p4k = (1 - S_l_p4k * ro_p4k) * gam_d

    denom = (1.0 / gamma) - (1.0 / gam_d)
    denom_p4k = (1.0 / gamma_p4k) - (1.0 / gam_d)
    invS = 1.0 / denom
    invS_p4k = 1.0 / denom_p4k

    dZ = 1 / gamma
    dZ_p4k = 1 / gamma_p4k
    dp = dZ * g * ro
    dp_p4k = dZ_p4k * g * ro_p4k

    tau = invS / dp
    tau_p4k = invS_p4k / dp_p4k

    dln = make_dln(SCREAM_CESS2_DELTA_SST)
    entry = {
        "S": dln(invS_p4k, invS),
        "tau": dln(tau_p4k, tau),
        "cf": dln(CF_l_p4k, CF_l),
        "p": dln(p_l_p4k, p_l),
        "gamma": dln(gamma_p4k, gamma),
        "S_baseline": invS,
        "T_l": SCREAM_CESS2_T_l,
    }
    print(f"[SCREAM cess2] Sanity check -- gamma range: {np.nanmin(gamma):.5f} to {np.nanmax(gamma):.5f} "
          f"(dry adiabatic gam_d={gam_d:.5f}); CF range: {np.nanmin(CF_l):.4f} to {np.nanmax(CF_l):.4f}")
    return entry

# =====================================================
# BUILD THE UNIFIED iso_data
# =====================================================

iso_data = {}

for base_dir in RCEMIP_BASE_DIRS:
    domain, domain_iso = load_rcemip_domain(base_dir, CF_MODE)
    source_tag = "RCE_small" if domain == "small" else "RCE_large"
    for model, d in domain_iso.items():
        iso_data[(model, source_tag)] = d

amip_iso = load_amip()
iso_data.update(amip_iso)

scream_cess2_entry = load_scream_cess2()
iso_data[("SCREAM_cess2", "SCREAM_cess2")] = scream_cess2_entry

print(f"\nTotal combined points across all sources: {len(iso_data)}")
for tag in DOMAIN_STYLE:
    n = sum(1 for k in iso_data if k[1] == tag)
    print(f"  {tag}: {n} points")

# =====================================================
# GENERIC AGGREGATION
# =====================================================

def aggregate_range(Tmin, Tmax):
    keys_used, out = [], {v: [] for v in VARS_ALL}
    for key, d in iso_data.items():
        T_l_this = d["T_l"]
        mask = (T_l_this >= Tmin) & (T_l_this <= Tmax)
        valid = mask.copy()
        for v in VARS_ALL:
            valid &= np.isfinite(d[v])
        if np.sum(valid) == 0:
            continue
        keys_used.append(key)
        for v in VARS_ALL:
            out[v].append(np.nanmean(d[v][valid]))
    return keys_used, {v: np.array(out[v]) for v in VARS_ALL}

def aggregate_isotherm(T_target):
    keys_used, out = [], {v: [] for v in VARS_ALL}
    for key, d in iso_data.items():
        T_l_this = d["T_l"]
        idx = int(np.argmin(np.abs(T_l_this - T_target)))
        if not all(np.isfinite(d[v][idx]) for v in VARS_ALL):
            continue
        keys_used.append(key)
        for v in VARS_ALL:
            out[v].append(d[v][idx])
    return keys_used, {v: np.array(out[v]) for v in VARS_ALL}

# =====================================================
# FIVE-PANEL PLOT
# =====================================================

def make_five_panel(keys_used, agg, title_suffix, filename_suffix):
    sources = np.array([k[1] for k in keys_used])
    names = np.array([k[0] for k in keys_used])
    dln_deltap = agg["S"] - agg["tau"]

    for tag in DOMAIN_STYLE:
        sel = sources == tag
        if np.sum(sel) > 0:
            print(f"  [{tag}] dln(Delta p): mean={np.nanmean(dln_deltap[sel])*100:.2f}%/K  "
                  f"std={np.nanstd(dln_deltap[sel])*100:.2f}%/K  N={np.sum(sel)}")

    fig, axs = plt.subplots(1, 6, figsize=(33, 5.5), constrained_layout=True)
    panel_specs = [
        ("S", "tau", r"$\Delta \ln S_{\rm inv}$ [%K$^{-1}$]", r"$\Delta \ln \tau_{i,\rm sub}$ [%K$^{-1}$]",
         "(a) Thermodynamic buffering"),
        ("tau", "cf", r"$\Delta \ln \tau_{i,\rm sub}$ [%K$^{-1}$]", r"$\Delta \ln \mathrm{CF}$ [%K$^{-1}$]",
         "(b) Cloud buffering"),
        ("p", "S", r"$\Delta \ln p$ [%K$^{-1}$]", r"$\Delta \ln S_{\rm inv}$ [%K$^{-1}$]",
         "(c) Does $S_{\\rm inv}$ track $p$?"),
        ("gamma", "tau", r"$\Delta \ln \Gamma$ [%K$^{-1}$]", r"$\Delta \ln \tau_{i,\rm sub}$ [%K$^{-1}$]",
         "(d) Does $\\tau_{i,\\rm sub}$ track $\\Gamma$?"),
        ("gamma", "S", r"$\Delta \ln \Gamma$ [%K$^{-1}$]", r"$\Delta \ln S_{\rm inv}$ [%K$^{-1}$]",
         "(f) Does $S_{\\rm inv}$ track $\\Gamma$ directly?"),
    ]
    results = {}
    for ax, (xkey, ykey, xlabel, ylabel, title) in zip([axs[0], axs[1], axs[2], axs[3], axs[5]], panel_specs):
        xp_all, yp_all = agg[xkey] * 100, agg[ykey] * 100
        for tag in DOMAIN_STYLE:
            sel = sources == tag
            if np.sum(sel) == 0:
                continue
            ax.scatter(xp_all[sel], yp_all[sel], label=tag, **DOMAIN_STYLE[tag])
        lims = setup_lims(ax, xp_all, yp_all)
        slope, intercept, r2 = fit_stats(xp_all, yp_all)
        results[title] = (slope, r2)
        if np.isfinite(slope):
            xx = np.linspace(lims[0], lims[1], 200)
            ax.plot(xx, slope * xx + intercept, color="k", lw=1.5, label=fr"All-source fit ($R^2$={r2:.2f})")
        for xi, yi, name in zip(xp_all, yp_all, names):
            if np.isfinite(xi) and np.isfinite(yi):
                ax.text(xi, yi, name.split(".")[0], fontsize=6, ha="left", va="bottom")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=7)
        ax.text(0.05, 0.92, f"Slope={slope:.2f}, $R^2$={r2:.2f}, N={len(keys_used)}",
                transform=ax.transAxes, va="top", fontsize=8)

    ax = axs[4]
    xp_all, yp_all = agg["S"] * 100, dln_deltap * 100
    for tag in DOMAIN_STYLE:
        sel = sources == tag
        if np.sum(sel) == 0:
            continue
        ax.scatter(xp_all[sel], yp_all[sel], label=tag, **DOMAIN_STYLE[tag])
    slope_e, intercept_e, r2_e = fit_stats(xp_all, yp_all)
    results["(e) Is the offset universal?"] = (slope_e, r2_e)
    ax.axhline(0, color="gray", lw=1, ls=":")
    ax.axhline(np.nanmean(yp_all), color="k", ls="--", lw=1, label=f"mean = {np.nanmean(yp_all):.2f}%/K")
    xlims = [np.nanmin(xp_all) - 0.5, np.nanmax(xp_all) + 0.5]
    if np.isfinite(slope_e):
        xx = np.linspace(xlims[0], xlims[1], 200)
        ax.plot(xx, slope_e * xx + intercept_e, color="k", lw=1.5, label=fr"Fit ($R^2$={r2_e:.2f})")
    ax.set_xlim(xlims)
    for xi, yi, name in zip(xp_all, yp_all, names):
        if np.isfinite(xi) and np.isfinite(yi):
            ax.text(xi, yi, name.split(".")[0], fontsize=6, ha="left", va="bottom")
    ax.set_xlabel(r"$\Delta \ln S_{\rm inv}$ [%K$^{-1}$]")
    ax.set_ylabel(r"$\Delta \ln \Delta p$ (exact) [%K$^{-1}$]")
    ax.set_title("(e) Is the offset universal?")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=7)
    ax.text(0.05, 0.92, f"Slope={slope_e:.2f}, $R^2$={r2_e:.2f}, N={len(keys_used)}",
            transform=ax.transAxes, va="top", fontsize=8)

    fig.suptitle(f"RCEMIP + AMIP + SCREAM: {title_suffix}", fontsize=13)
    if SAVE_FIGS:
        fig.savefig(f"{OUT_PREFIX}_five_panel_{filename_suffix}.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)

    for title, (slope, r2) in results.items():
        print(f"    {title}: slope={slope:.2f} R2={r2:.2f}")
    return results

# =====================================================
# EMERGENT-CONSTRAINT PLOT
# =====================================================

def make_emergent_constraint_plot(keys_used, agg, title_suffix, filename_suffix):
    sources = np.array([k[1] for k in keys_used])
    names = np.array([k[0] for k in keys_used])

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    x = agg["S_baseline"]
    y = agg["S"] * 100

    for tag in DOMAIN_STYLE:
        sel = sources == tag
        if np.sum(sel) == 0:
            continue
        ax.scatter(x[sel], y[sel], label=tag, **DOMAIN_STYLE[tag])

    slope, intercept, r2 = fit_stats(x, y)
    valid = np.isfinite(x) & np.isfinite(y)

    slope_drop, intercept_drop, r2_drop, dropped_key = np.nan, np.nan, np.nan, None
    if np.sum(valid) >= 4 and np.isfinite(slope):
        resid = np.abs(y[valid] - (slope * x[valid] + intercept))
        keys_valid = [k for k, v in zip(keys_used, valid) if v]
        drop_idx = np.argmax(resid)
        dropped_key = keys_valid[drop_idx]
        keep_mask = np.ones(np.sum(valid), dtype=bool)
        keep_mask[drop_idx] = False
        x_v, y_v = x[valid], y[valid]
        slope_drop, intercept_drop, r2_drop = fit_stats(x_v[keep_mask], y_v[keep_mask])

    rho, _ = spearmanr(x[valid], y[valid]) if np.sum(valid) >= 3 else (np.nan, np.nan)

    if np.sum(valid) >= 3:
        xx = np.linspace(np.nanmin(x[valid]), np.nanmax(x[valid]), 200)
        ax.plot(xx, slope * xx + intercept, color="k", lw=1.5, label=fr"All-source fit ($R^2$={r2:.2f})")
        if np.isfinite(slope_drop):
            ax.plot(xx, slope_drop * xx + intercept_drop, color="orange", ls="--",
                    label=fr"Drop {dropped_key[0].split('.')[0] if dropped_key else '?'} ($R^2$={r2_drop:.2f})")

    for xi, yi, name in zip(x, y, names):
        if np.isfinite(xi) and np.isfinite(yi):
            ax.text(xi, yi, name.split(".")[0], fontsize=6, ha="left", va="bottom")

    ax.set_xlabel(r"Baseline (control) $S_{\rm inv}$")
    ax.set_ylabel(r"$\Delta \ln S_{\rm inv}$ [%K$^{-1}$]")
    ax.set_title(f"Emergent constraint? Baseline $S_{{\\rm inv}}$ vs. its own sensitivity\n{title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=7)
    ax.text(0.05, 0.05,
            f"All: slope={slope:.2f}, $R^2$={r2:.2f}\n"
            f"Drop largest outlier: slope={slope_drop:.2f}, $R^2$={r2_drop:.2f}\n"
            f"Spearman $\\rho$={rho:.2f}\nN = {len(keys_used)}",
            transform=ax.transAxes, va="bottom", fontsize=8)

    if SAVE_FIGS:
        fig.savefig(f"{OUT_PREFIX}_emergent_constraint_{filename_suffix}.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  [emergent constraint] all: slope={slope:.3f} R2={r2:.2f}  |  "
          f"drop {dropped_key}: slope={slope_drop:.3f} R2={r2_drop:.2f}  |  Spearman rho={rho:.2f}")

# =====================================================
# RUN
# =====================================================

keys_range, agg_range = aggregate_range(AVG_TMIN, AVG_TMAX)
print(f"\n=== Range average {AVG_TMIN:.0f}-{AVG_TMAX:.0f}K (N={len(keys_range)}) ===")
make_five_panel(keys_range, agg_range, f"mean %/K over {AVG_TMAX:.0f}-{AVG_TMIN:.0f} K (range average)", "range_avg")
make_emergent_constraint_plot(keys_range, agg_range, f"mean %/K over {AVG_TMAX:.0f}-{AVG_TMIN:.0f} K", "range_avg")

isotherm_results = {}
for T_target in ISOTHERM_TARGETS:
    keys_iso, agg_iso = aggregate_isotherm(T_target)
    print(f"\n=== Isotherm T={T_target:.0f}K (N={len(keys_iso)}) ===")
    res_iso = make_five_panel(keys_iso, agg_iso, f"exactly at T={T_target:.0f}K", f"isotherm_{int(T_target)}K")
    isotherm_results[T_target] = res_iso
    make_emergent_constraint_plot(keys_iso, agg_iso, f"exactly at T={T_target:.0f}K", f"isotherm_{int(T_target)}K")

panel_titles = list(next(iter(isotherm_results.values())).keys())
Ts_sorted = sorted(isotherm_results.keys())

fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
for title in panel_titles:
    slopes = [isotherm_results[T][title][0] for T in Ts_sorted]
    r2s = [isotherm_results[T][title][1] for T in Ts_sorted]
    axs[0].plot(Ts_sorted, slopes, marker="o", label=title)
    axs[1].plot(Ts_sorted, r2s, marker="o", label=title)

axs[0].set_xlabel("Isotherm T [K]"); axs[0].set_ylabel("Slope")
axs[0].set_title("Does the slope change with height?")
axs[0].invert_xaxis(); axs[0].grid(True, alpha=0.3); axs[0].legend(fontsize=7, frameon=False)

axs[1].set_xlabel("Isotherm T [K]"); axs[1].set_ylabel(r"$R^2$")
axs[1].set_title("Does the fit strength change with height?")
axs[1].invert_xaxis(); axs[1].set_ylim(0, 1); axs[1].grid(True, alpha=0.3); axs[1].legend(fontsize=7, frameon=False)

fig.suptitle("RCEMIP + AMIP + SCREAM: sensitivity of the four relationships to isotherm choice", fontsize=13)
if SAVE_FIGS:
    fig.savefig(f"{OUT_PREFIX}_isotherm_sensitivity.pdf", bbox_inches="tight", dpi=300)
plt.close(fig)
