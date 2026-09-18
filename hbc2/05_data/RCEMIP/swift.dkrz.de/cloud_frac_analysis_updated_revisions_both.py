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

BASE_DIRS = ["RCE_small", "RCE_large"]   # both domains combined into one analysis
CF_MODE = "cfv0"                          # "cfv0" or "cfv1-cfv2"

# =====================================================
# WILLIAMS & JEEVANJEE (2025) REPLICATION SETTINGS
#
# Per their Methods (Section 2.1): "We used all cloud-resolving models for
# which the required data was available. The two exceptions to this are the
# UKMO model and the ICON model. The UKMO model submitted multiple variants
# and so we only take a single variant (UKMOi-vn11.1-CASIM)... We also
# discard the two ICON cloud-resolving models... The eight models we used
# are: dam, UKMOi-vn11.1-CASIM, WRF_COL_CRM, SAM_CRM, SCALE, MESONH,
# UCLA-CRM and CM1." They used RCE_small ONLY (RCE_large is never mentioned)
# and computed fractional changes directly between the 295K and 305K
# simulations (a single 10K step), not by averaging two 5K sub-steps as our
# default pipeline does.
#
# NOTE: the exact string matching below is a best-effort guess at how their
# eight model names map onto the filenames in this dataset -- verify this
# printed list actually finds 8 matches before trusting the replication run;
# if a name doesn't match, adjust WJ_MODEL_WHITELIST directly.
# =====================================================

RESTRICT_TO_WJ_MODELS = True   # set True to filter down to their 8-model list
WJ_SMALL_DOMAIN_ONLY = True     # they used RCE_small only, never RCE_large
WJ_MODEL_WHITELIST = {
    "DAM", "UKMO-CASIM", "WRF-COL-CRM", "SAM-CRM", "SCALE", "MESONH", "UCLA-CRM", "CM1",
}

USE_DIRECT_295_305 = True  # set True to use a single 295->305 (10K) comparison,
                            # matching their methodology, instead of averaging
                            # the two 5K sub-steps (295->300, 300->305)

SAVE_FIGS = True
_mode_tag = "WJreplication" if RESTRICT_TO_WJ_MODELS else "fullensemble"
OUT_PREFIX = f"RCEMIP_combined_{CF_MODE}_{_mode_tag}"

T_l = np.arange(295.0, 197.0, -1.0)

AVG_TMIN = 200.0
AVG_TMAX = 240.0

DELTA_SST = 5.0  # RCEMIP's 295/300/305K triplet steps -- dln() divides by this so every
                 # reported number is a true %/K.

Rd = 287.0
cp = 1004.0
g = 9.81
p0 = 100000.0
kappa = Rd / cp
gam_d = g / cp

# Per-domain exclusions -- defaults to the same list for both, since it's not clear
# whether dropping the exclusion for RCE_large (as in the version you just showed me)
# was intentional or a leftover from testing. Override EXCLUDE_BY_DOMAIN["large"]
# directly if you want it to differ from "small".
_DEFAULT_EXCLUDE = {"CNRM-CM6", "UKMO-GA7.1", "WRF-CRM"}
EXCLUDE_BY_DOMAIN = {
    "small": set(_DEFAULT_EXCLUDE),
    "large": set(_DEFAULT_EXCLUDE),
}

# Color/marker style per domain, used in every scatter plot below so you can see at a
# glance whether small- and large-domain results from the same model track each other.
DOMAIN_STYLE = {
    "small": dict(marker="o", color="tab:blue", edgecolor="k"),
    "large": dict(marker="o", color="tab:red", edgecolor="k"),
}

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

def load_profile(path, cf_var):
    ds = xr.open_dataset(path)

    T = get_1d_var(ds, ["ta_avg", "ta"])
    p = get_pressure_pa(ds)
    cf = get_1d_var(ds, [cf_var, "cfv0_avg", "cfv1_avg", "cfv2_avg"])

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
    # Static stability S = -d(theta)/dp on the native grid (from load_profile),
    # interpolated to T_l, combined with the theta-consistent SCREAM-style
    # formula Gamma = Gamma_d*(1 - S*rho*cp*(T/theta)) -- see prior discussion
    # for why the extra cp*(T/theta) factor is required when S is
    # potential-temperature-based rather than SCREAM's native
    # dry-static-energy-based diagnostic.
    cf_l, imin = interp_branch(T, cf)
    p_l, _ = interp_branch(T, p)
    S_l, _ = interp_branch(T, S)

    ro_l = p_l / (Rd * T_l)
    theta_l = T_l * (p0 / p_l) ** kappa

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
        "cf_l": cf_l, "p_l": p_l, "S_l": S_l, "rho_l": ro_l,
        "gamma_l": gamma, "invS_l": invS, "dp_l": dp, "tau_l": tau, "imin": imin,
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
# LOAD ONE DOMAIN (small or large) -- returns iso_data keyed by plain model name
# =====================================================

def load_domain(base_dir, cf_mode):
    domain = "small" if "small" in base_dir else "large"

    if cf_mode == "cfv0":
        cf_var = "cfv0_avg"
        suffix295 = f"_RCE_{domain}295_cfv0-profiles.nc"
        suffix300 = f"_RCE_{domain}300_cfv0-profiles.nc"
        suffix305 = f"_RCE_{domain}305_cfv0-profiles.nc"
    else:
        cf_var = "cfv1_avg"
        suffix295 = f"_RCE_{domain}295_cfv1-cfv2-profiles.nc"
        suffix300 = f"_RCE_{domain}300_cfv1-cfv2-profiles.nc"
        suffix305 = f"_RCE_{domain}305_cfv1-cfv2-profiles.nc"

    pattern300 = f"*{suffix300}"
    files300 = sorted(glob.glob(os.path.join(base_dir, pattern300)))
    triplets = []
    for f300 in files300:
        model = os.path.basename(f300).replace(suffix300, "")
        f295 = os.path.join(base_dir, model + suffix295)
        f305 = os.path.join(base_dir, model + suffix305)
        if os.path.exists(f295) and os.path.exists(f305):
            triplets.append((model, f295, f300, f305))

    print(f"[{domain}] Found {len(triplets)} matched 295/300/305 triplets.")

    exclude = EXCLUDE_BY_DOMAIN.get(domain, set())

    def keep_model(model):
        if model in exclude:
            return False
        if "GCM" in model:
            return False
        if model.startswith("DALES"):
            return False
        return True

    kept, removed = [], []
    for triplet in triplets:
        if keep_model(triplet[0]):
            kept.append(triplet)
        else:
            removed.append(triplet[0])
    triplets = kept

    print(f"[{domain}] Removed models: {removed if removed else '(none)'}")
    print(f"[{domain}] Remaining after filtering: {len(triplets)} models")

    domain_iso_data = {}
    domain_profiles = {}

    for model, f295, f300, f305 in triplets:
        try:
            T295, p295, cf295, S295, rho295 = load_profile(f295, cf_var)
            T300, p300, cf300, S300, rho300 = load_profile(f300, cf_var)
            T305, p305, cf305, S305, rho305 = load_profile(f305, cf_var)

            d295 = derive(T295, p295, cf295, S295, rho295)
            d300 = derive(T300, p300, cf300, S300, rho300)
            d305 = derive(T305, p305, cf305, S305, rho305)

            domain_profiles[model] = (d295, d300, d305)

            dcf_300_295 = dln(d300["cf_l"], d295["cf_l"]); dcf_305_300 = dln(d305["cf_l"], d300["cf_l"])
            dS_300_295 = dln(d300["invS_l"], d295["invS_l"]); dS_305_300 = dln(d305["invS_l"], d300["invS_l"])
            dtau_300_295 = dln(d300["tau_l"], d295["tau_l"]); dtau_305_300 = dln(d305["tau_l"], d300["tau_l"])
            dp_300_295 = dln(d300["p_l"], d295["p_l"]); dp_305_300 = dln(d305["p_l"], d300["p_l"])
            dgamma_300_295 = dln(d300["gamma_l"], d295["gamma_l"]); dgamma_305_300 = dln(d305["gamma_l"], d300["gamma_l"])

            if USE_DIRECT_295_305:
                # Matches W&J's methodology exactly: a single fractional change
                # computed directly between the 295K and 305K simulations (a 10K
                # step), rather than averaging two separate 5K sub-step estimates.
                # Note dln() itself still divides by DELTA_SST=5.0 -- for a true
                # per-Kelvin rate over the full 10K span this needs its OWN
                # 10K-normalized version, computed here directly rather than
                # reusing dln() (which assumes a 5K step).
                def dln10(x1, x0):
                    denom = 0.5 * (x1 + x0)
                    out = np.full_like(denom, np.nan, dtype=float)
                    valid = np.isfinite(x0) & np.isfinite(x1) & (denom != 0)
                    out[valid] = (x1[valid] - x0[valid]) / denom[valid] / 10.0
                    return out
                domain_iso_data[model] = {
                    "cf": dln10(d305["cf_l"], d295["cf_l"]),
                    "S": dln10(d305["invS_l"], d295["invS_l"]),
                    "tau": dln10(d305["tau_l"], d295["tau_l"]),
                    "p": dln10(d305["p_l"], d295["p_l"]),
                    "gamma": dln10(d305["gamma_l"], d295["gamma_l"]),
                    "S_baseline": d300["invS_l"],
                }
            else:
                domain_iso_data[model] = {
                    "cf": np.nanmean(np.stack([dcf_300_295, dcf_305_300]), axis=0),
                    "S": np.nanmean(np.stack([dS_300_295, dS_305_300]), axis=0),
                    "tau": np.nanmean(np.stack([dtau_300_295, dtau_305_300]), axis=0),
                    "p": np.nanmean(np.stack([dp_300_295, dp_305_300]), axis=0),
                    "gamma": np.nanmean(np.stack([dgamma_300_295, dgamma_305_300]), axis=0),
                    "S_baseline": d300["invS_l"],
                }
        except Exception as e:
            print(f"[{domain}] Skipping {model}: {e}")

    print(f"[{domain}] Usable models: {len(domain_iso_data)}")
    return domain, domain_profiles, domain_iso_data

# =====================================================
# LOAD BOTH DOMAINS, MERGE
# =====================================================

# iso_data keyed by (model, domain) tuples so the same model appearing in both
# domains is kept as two distinct points rather than overwriting/colliding.
iso_data = {}
profiles_by_domain = {}

_base_dirs_to_use = ["RCE_small"] if (RESTRICT_TO_WJ_MODELS and WJ_SMALL_DOMAIN_ONLY) else BASE_DIRS

for base_dir in _base_dirs_to_use:
    domain, domain_profiles, domain_iso_data = load_domain(base_dir, CF_MODE)
    profiles_by_domain[domain] = domain_profiles
    for model, d in domain_iso_data.items():
        if RESTRICT_TO_WJ_MODELS and model.upper() not in WJ_MODEL_WHITELIST:
            continue
        iso_data[(model, domain)] = d

if RESTRICT_TO_WJ_MODELS:
    found = sorted(set(k[0].upper() for k in iso_data.keys()))
    missing = sorted(WJ_MODEL_WHITELIST - set(found))
    print(f"\n[W&J replication] Matched {len(found)}/8 whitelisted models: {found}")
    if missing:
        print(f"[W&J replication] WARNING -- not found in dataset, check naming: {missing}")

print(f"\nTotal combined (model, domain) points: {len(iso_data)}")

VARS = ["S", "tau", "cf", "p", "gamma", "S_baseline"]

# =====================================================
# VERTICAL PROFILE PLOTS, 4 MODELS PER FIGURE, PER DOMAIN
# =====================================================

for domain, domain_profiles in profiles_by_domain.items():
    models = list(domain_profiles.keys())
    batch_size = 4
    for start in range(0, len(models), batch_size):
        batch = models[start:start + batch_size]
        n_batch = len(batch)

        fig, axs = plt.subplots(n_batch, 5, figsize=(15, 2.8 * n_batch), sharey=True, constrained_layout=True)
        if n_batch == 1:
            axs = np.array([axs])

        for i, model in enumerate(batch):
            d295, d300, d305 = domain_profiles[model]
            for d, color, ls, label in [(d295, "tab:green", ":", "295 K"), (d300, "tab:blue", "-", "300 K"), (d305, "tab:red", "--", "305 K")]:
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
            fig.savefig(f"RCEMIP_{domain}_{CF_MODE}_profiles_{start+1:02d}_{start+n_batch:02d}.pdf", bbox_inches="tight", dpi=300)
        plt.show()

# =====================================================
# AGGREGATION (range or single isotherm), OVER THE COMBINED (model, domain) SET
# =====================================================

def aggregate_range(Tmin, Tmax):
    mask = (T_l <= Tmax) & (T_l >= Tmin)
    keys_used, out = [], {v: [] for v in VARS}
    for key, d in iso_data.items():
        valid = mask.copy()
        for v in VARS:
            valid &= np.isfinite(d[v])
        if np.sum(valid) == 0:
            continue
        keys_used.append(key)
        for v in VARS:
            out[v].append(np.nanmean(d[v][valid]))
    return keys_used, {v: np.array(out[v]) for v in VARS}

def aggregate_isotherm(T_target):
    idx = int(np.argmin(np.abs(T_l - T_target)))
    keys_used, out = [], {v: [] for v in VARS}
    for key, d in iso_data.items():
        if not all(np.isfinite(d[v][idx]) for v in VARS):
            continue
        keys_used.append(key)
        for v in VARS:
            out[v].append(d[v][idx])
    return keys_used, {v: np.array(out[v]) for v in VARS}, T_l[idx]

def make_four_panel(keys_used, agg, title_suffix, filename_suffix):
    """Standard 4-panel scatter (a-d). Points are colored/shaped by domain
    (DOMAIN_STYLE) so small- vs large-domain results are visually distinguishable,
    but the fit (slope/R^2) is computed over the COMBINED set of both domains."""

    # =====================================================
    # EXACT dln(Delta p) PER POINT -- not a proxy, the literal identity residual
    # dln(tau_sub) = dln(S_inv) - dln(Delta p), so dln(Delta p) = dln(S_inv) - dln(tau_sub)
    # exactly, for every single model/domain point. Tests whether the buffering
    # is better understood as a near-universal ADDITIVE offset (small spread in
    # dlnDeltap relative to dlnS_inv's own spread) rather than a model-varying
    # multiplicative amplification -- i.e. whether panel (a)'s slope~1 reflects
    # dlnDeltap acting like a roughly constant intercept across the ensemble.
    # =====================================================
    dln_deltap_exact = agg["S"] - agg["tau"]
    print(f"  [exact dln(Delta p) check] mean={np.nanmean(dln_deltap_exact)*100:.2f}%/K  "
          f"std={np.nanstd(dln_deltap_exact)*100:.2f}%/K  "
          f"(for reference, dln(S_inv) itself spans mean={np.nanmean(agg['S'])*100:.2f}%/K, "
          f"std={np.nanstd(agg['S'])*100:.2f}%/K across the ensemble)")

    fig, axs = plt.subplots(1, 5, figsize=(27, 5.5), constrained_layout=True)
    panel_specs = [
        (agg["S"], agg["tau"], r"$\Delta \ln S_{\rm inv}$ [%K$^{-1}$]",
         r"$\Delta \ln \tau_{i,\rm sub}$ [%K$^{-1}$]", "(a) Thermodynamic buffering"),
        (agg["tau"], agg["cf"], r"$\Delta \ln \tau_{i,\rm sub}$ [%K$^{-1}$]",
         r"$\Delta \ln \mathrm{CF}$ [%K$^{-1}$]", "(b) Cloud buffering"),
        (agg["p"], agg["S"], r"$\Delta \ln p$ [%K$^{-1}$]",
         r"$\Delta \ln S_{\rm inv}$ [%K$^{-1}$]", "(c) Does $S_{\\rm inv}$ track $p$?"),
        (agg["gamma"], agg["tau"], r"$\Delta \ln \Gamma$ [%K$^{-1}$]",
         r"$\Delta \ln \tau_{i,\rm sub}$ [%K$^{-1}$]", "(d) Does $\\tau_{i,\\rm sub}$ track $\\Gamma$?"),
        (agg["S"], dln_deltap_exact, r"$\Delta \ln S_{\rm inv}$ [%K$^{-1}$]",
         r"$\Delta \ln \Delta p$ (exact) [%K$^{-1}$]", "(e) Is the offset universal?"),
    ]
    results = {}
    domains_present = sorted(set(k[1] for k in keys_used))

    for i, (ax, (x, y, xlabel, ylabel, title)) in enumerate(zip(axs, panel_specs)):
        xp, yp = x * 100, y * 100
        for dom in domains_present:
            sel = np.array([k[1] == dom for k in keys_used])
            style = DOMAIN_STYLE.get(dom, dict(marker="o", edgecolor="k"))
            ax.scatter(xp[sel], yp[sel], s=80, label=dom, **style)

        is_offset_panel = title.startswith("(e)")
        slope, intercept, r2 = fit_stats(xp, yp)
        results[title] = (slope, r2)

        if is_offset_panel:
            # Not a 1:1 test -- just autoscale, with y=0 and the mean-offset level
            # as reference lines, since the question here is "how flat/constant
            # is this", not "does it match a 1:1 relationship".
            ax.axhline(0, color="gray", lw=1, ls=":")
            mean_offset = np.nanmean(yp)
            ax.axhline(mean_offset, color="k", ls="--", lw=1,
                       label=f"mean = {mean_offset:.2f}%/K")
            xlims = [np.nanmin(xp) - 0.5, np.nanmax(xp) + 0.5]
            if np.isfinite(slope):
                xx = np.linspace(xlims[0], xlims[1], 200)
                ax.plot(xx, slope * xx + intercept, color="red", label=fr"Fit ($R^2$={r2:.2f})")
            ax.set_xlim(xlims)
        else:
            lims = setup_lims(ax, xp, yp)
            if np.isfinite(slope):
                xx = np.linspace(lims[0], lims[1], 200)
                ax.plot(xx, slope * xx + intercept, color="red", label=fr"Fit ($R^2$={r2:.2f})")

        for xi, yi, key in zip(xp, yp, keys_used):
            if np.isfinite(xi) and np.isfinite(yi):
                ax.text(xi, yi, key[0], fontsize=6, ha="left", va="bottom")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
        ax.text(0.05, 0.92, f"Slope = {slope:.2f}\n$R^2$ = {r2:.2f}\nN = {len(keys_used)}",
                transform=ax.transAxes, va="top")

    fig.suptitle(f"RCEMIP small+large combined, {CF_MODE}: {title_suffix}", fontsize=13)
    if SAVE_FIGS:
        fig.savefig(f"{OUT_PREFIX}_four_panel_{filename_suffix}.pdf", bbox_inches="tight", dpi=300)
    plt.show()
    return results

def make_emergent_constraint_plot(keys_used, agg, title_suffix, filename_suffix):
    """Emergent-constraint-style check: does each model's BASELINE (300K,
    un-warmed) S_inv predict its own SENSITIVITY (dlnS_inv/dTs)? A real
    across-model relationship here would mean present-day S_inv observations
    could help constrain the sensitivity itself -- worth checking directly
    since it's a different question from anything in the four/five-panel
    figure (baseline state vs. warming response, not response vs. response).

    Includes a basic outlier-robustness check: a single extreme point can
    dominate a least-squares slope/R^2 in a modest ensemble (N~40), so this
    also reports (a) the fit after dropping the single largest-residual
    point, and (b) the Spearman rank correlation, which is far less sensitive
    to one extreme value than the Pearson R^2 the main fit uses."""
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

    x = agg["S_baseline"]          # raw S_inv at 300K, not a fractional change
    y = agg["S"] * 100             # dlnS_inv/dTs, in %/K

    domains_present = sorted(set(k[1] for k in keys_used))
    for dom in domains_present:
        sel = np.array([k[1] == dom for k in keys_used])
        style = DOMAIN_STYLE.get(dom, dict(marker="o", edgecolor="k"))
        ax.scatter(x[sel], y[sel], s=80, label=dom, **style)

    slope, intercept, r2 = fit_stats(x, y)
    valid = np.isfinite(x) & np.isfinite(y)

    # --- Outlier-robustness: drop the single largest-residual point, refit ---
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

    # --- Spearman rank correlation: robust to a single extreme value by construction ---
    from scipy.stats import spearmanr
    if np.sum(valid) >= 3:
        rho, _ = spearmanr(x[valid], y[valid])
    else:
        rho = np.nan

    if np.sum(valid) >= 3:
        xx = np.linspace(np.nanmin(x[valid]), np.nanmax(x[valid]), 200)
        ax.plot(xx, slope * xx + intercept, color="red", label=fr"Fit, all points ($R^2$={r2:.2f})")
        if np.isfinite(slope_drop):
            ax.plot(xx, slope_drop * xx + intercept_drop,
                    color="orange", ls="--",
                    label=fr"Fit, drop {dropped_key[0] if dropped_key else '?'} ($R^2$={r2_drop:.2f})")

    for xi, yi, key in zip(x, y, keys_used):
        if np.isfinite(xi) and np.isfinite(yi):
            ax.text(xi, yi, key[0], fontsize=6, ha="left", va="bottom")

    ax.set_xlabel(r"Baseline (300K) $S_{\rm inv}$")
    ax.set_ylabel(r"$\Delta \ln S_{\rm inv}$ [%K$^{-1}$]")
    ax.set_title(f"Emergent constraint? Baseline $S_{{\\rm inv}}$ vs. its own sensitivity\n{title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=7)
    ax.text(0.05, 0.05,
            f"All points: slope={slope:.2f}, $R^2$={r2:.2f}\n"
            f"Drop largest outlier: slope={slope_drop:.2f}, $R^2$={r2_drop:.2f}\n"
            f"Spearman $\\rho$={rho:.2f}\nN = {len(keys_used)}",
            transform=ax.transAxes, va="bottom", fontsize=8)

    if SAVE_FIGS:
        fig.savefig(f"{OUT_PREFIX}_emergent_constraint_{filename_suffix}.pdf", bbox_inches="tight", dpi=300)
    plt.show()
    print(f"  [emergent constraint check] all points: slope={slope:.3f} R2={r2:.2f}  |  "
          f"drop largest outlier ({dropped_key[0] if dropped_key else 'n/a'}): "
          f"slope={slope_drop:.3f} R2={r2_drop:.2f}  |  Spearman rho={rho:.2f}")
    return slope, r2

# =====================================================
# RUN: range-average, then single isotherms, then sensitivity summary
# =====================================================

keys_range, agg_range = aggregate_range(AVG_TMIN, AVG_TMAX)
print(f"\n=== Range average {AVG_TMIN:.0f}-{AVG_TMAX:.0f}K (N={len(keys_range)}, both domains) ===")
res_range = make_four_panel(keys_range, agg_range,
                             f"mean %/K over {AVG_TMAX:.0f}-{AVG_TMIN:.0f} K (range average)",
                             "range_avg")
for title, (slope, r2) in res_range.items():
    print(f"  {title}: slope={slope:.2f}  R2={r2:.2f}")

make_emergent_constraint_plot(keys_range, agg_range,
                               f"mean %/K over {AVG_TMAX:.0f}-{AVG_TMIN:.0f} K (range average)",
                               "range_avg")

isotherm_targets = [220, 228, 230, 240, 250]
isotherm_results = {}

for T_target in isotherm_targets:
    keys_iso, agg_iso, T_actual = aggregate_isotherm(T_target)
    print(f"\n=== Isotherm T={T_actual:.0f}K (N={len(keys_iso)}, both domains) ===")
    res_iso = make_four_panel(keys_iso, agg_iso,
                               f"exactly at T={T_actual:.0f}K (single isotherm)",
                               f"isotherm_{int(T_actual)}K")
    isotherm_results[T_actual] = res_iso
    for title, (slope, r2) in res_iso.items():
        print(f"  {title}: slope={slope:.2f}  R2={r2:.2f}")

    make_emergent_constraint_plot(keys_iso, agg_iso,
                                   f"exactly at T={T_actual:.0f}K (single isotherm)",
                                   f"isotherm_{int(T_actual)}K")

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

fig.suptitle(f"RCEMIP small+large combined, {CF_MODE}: sensitivity of the four relationships to isotherm choice", fontsize=13)
if SAVE_FIGS:
    fig.savefig(f"{OUT_PREFIX}_isotherm_sensitivity.pdf", bbox_inches="tight", dpi=300)
plt.show()
