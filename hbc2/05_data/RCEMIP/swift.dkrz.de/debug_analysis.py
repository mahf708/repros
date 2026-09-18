import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# -------------------------
# files
# -------------------------
f300 = "RCE_small/MESONH-VER_RCE_small300_cfv1-cfv2-profiles.nc"
f305 = "RCE_small/MESONH-VER_RCE_small305_cfv1-cfv2-profiles.nc"

CF_VAR = "cfv1_avg"

# common temperature grid: 295, 294, ..., 198 K
T_l = np.arange(295.0, 197.0, -1.0)

# constants
Rd = 287.0
cp = 1004.0
g = 9.81
p0 = 100000.0
kappa = Rd / cp
gam_d = g / cp

# -------------------------
# helpers
# -------------------------
def get_1d_var(ds, names):
    for name in names:
        if name in ds.variables:
            arr = np.asarray(ds[name]).squeeze()
            if arr.ndim == 1:
                return arr, name
    raise KeyError(f"Could not find any of {names}")

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
    cf, _ = get_1d_var(ds, [cf_var, "cfv1_avg", "cfv2_avg", "cfv1", "cfv2"])

    theta = T * (p0 / p) ** kappa
    S = -np.gradient(theta, p, edge_order=2)   # positive stability
    rho = p / (Rd * T)

    ds.close()
    return T.astype(float), p.astype(float), cf.astype(float), S.astype(float), rho.astype(float)

def interp_monotonic_branch(T, X, T_target):
    """
    Use only the first monotonic cooling branch: from index 0 to the first Tmin.
    """
    # first minimum temperature
    imin = np.argmin(T)

    T_cut = T[:imin+1]
    X_cut = X[:imin+1]

    valid = np.isfinite(T_cut) & np.isfinite(X_cut)
    T_cut = T_cut[valid]
    X_cut = X_cut[valid]

    if len(T_cut) < 2:
        return np.full_like(T_target, np.nan, dtype=float), imin

    # T_cut is decreasing; interp1d wants increasing x
    order = np.argsort(T_cut)
    T_use = T_cut[order]
    X_use = X_cut[order]

    # remove duplicates in T
    T_unique, idx = np.unique(T_use, return_index=True)
    X_unique = X_use[idx]

    f = interp1d(
        T_unique,
        X_unique,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan
    )
    X_l = f(T_target)

    # mask outside valid range
    X_l[(T_target < T_unique.min()) | (T_target > T_unique.max())] = np.nan
    return X_l, imin

def derive_Tcoord(T, p, cf, S, rho, T_target):
    cf_l, imin = interp_monotonic_branch(T, cf, T_target)
    p_l, _     = interp_monotonic_branch(T, p, T_target)
    S_l, _     = interp_monotonic_branch(T, S, T_target)
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
    dZ_l[good_g] = 1.0 / gamma_l[good_g]   # 1 K spacing

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

# -------------------------
# load files
# -------------------------
T300, p300, cf300, S300, rho300 = load_profile(f300)
T305, p305, cf305, S305, rho305 = load_profile(f305)

tc300 = derive_Tcoord(T300, p300, cf300, S300, rho300, T_l)
tc305 = derive_Tcoord(T305, p305, cf305, S305, rho305, T_l)

print("300 K first Tmin index:", tc300["imin"], " Tmin =", T300[tc300["imin"]])
print("305 K first Tmin index:", tc305["imin"], " Tmin =", T305[tc305["imin"]])

# -------------------------
# plot
# -------------------------
fig, axs = plt.subplots(1, 5, figsize=(15, 5), sharey=True, constrained_layout=True)

axs[0].plot(tc300["cf_l"], T_l, lw=2, label="300 K")
axs[0].plot(tc305["cf_l"], T_l, lw=2, ls="--", label="305 K")
axs[0].set_xlabel("Cloud fraction")
axs[0].set_ylabel("Temperature [K]")
axs[0].set_title("(a) Cloud fraction")
axs[0].legend(frameon=False, loc="lower left")

axs[1].plot(tc300["S_l"], T_l, lw=2)
axs[1].plot(tc305["S_l"], T_l, lw=2, ls="--")
axs[1].set_xlabel(r"$-d\theta/dp$ [K Pa$^{-1}$]")
axs[1].set_title("(b) Stability")

axs[2].plot(tc300["invS_l"], T_l, lw=2)
axs[2].plot(tc305["invS_l"], T_l, lw=2, ls="--")
axs[2].set_xlabel(r"$S_{\rm inv}$")
axs[2].set_title(r"(c) $S_{\rm inv}$")

axs[3].plot(tc300["dp_l"], T_l, lw=2)
axs[3].plot(tc305["dp_l"], T_l, lw=2, ls="--")
axs[3].set_xlabel(r"$\Delta p$")
axs[3].set_title(r"(d) $\Delta p$")

axs[4].plot(tc300["tau_i_sub_l"], T_l, lw=2)
axs[4].plot(tc305["tau_i_sub_l"], T_l, lw=2, ls="--")
axs[4].set_xlabel(r"$\tau_{i,\rm sub}$")
axs[4].set_title(r"(e) $\tau_{i,\rm sub}$")

for ax in axs:
    ax.grid(True, alpha=0.3)
    ax.set_ylim(260, 198)

plt.show()
