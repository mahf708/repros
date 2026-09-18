"""
Reconstructs figS_latent_clear_vs_cloud_profile.pdf (SI section S1):
clear-sky vs. in-cloud latent heating and cooling, as area-weighted
domain-mean magnitudes, control and +4K, log scale.

NOTE ON PROVENANCE: the original code that produced this exact figure was
written interactively during manuscript revision and was not saved as a
standalone script. This is a from-scratch reconstruction using the same
cached data (control/plus4k_vars_minimal_full_record.npz) and the same
variable definitions and unit conversions documented in SI section S1 and
already used elsewhere in analysis_minimal.py (L_v, L_s, cp constants;
the (1-CF)/CF area weighting). The reconstruction reproduces the SI text's
qualitative ordering exactly (in-cloud heating > in-cloud cooling >
clear-sky cooling > clear-sky heating) and is in the right ballpark for
magnitude (clear-sky terms roughly 0.6-2 orders of magnitude below their
in-cloud counterparts, vs. the SI text's "one to one and a half"), but the
exact separation is not an exact match and should be treated as
approximate, not a verified pixel-for-pixel reproduction.

Run from a directory containing control_vars_minimal_full_record.npz and
plus4k_vars_minimal_full_record.npz.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

_L_v = 2.501e6   # J/kg
_L_s = 2.834e6   # J/kg
_cp = 1004.0     # J/kg/K

def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}

ctrl = load("control_vars_minimal_full_record.npz")
p4k = load("plus4k_vars_minimal_full_record.npz")

T, T_p4 = ctrl["T"], p4k["T"]
CF, CF_p4 = ctrl["CF"], p4k["CF"]

# Latent heating (condensation + deposition) and cooling (evaporation +
# sublimation), in K/s, for both the clear-sky and in-cloud branches.
def heating_cooling(d):
    heat = (_L_v / _cp) * d["shoc_cond_clr"] + (_L_s / _cp) * d["qv2qi_vapdep_clr"]
    cool = (_L_v / _cp) * d["shoc_evap_clr"] + (_L_s / _cp) * d["qi2qv_sublim_clr"]
    heat_ic = (_L_v / _cp) * d["shoc_cond_incloud"] + (_L_s / _cp) * d["qv2qi_vapdep_incloud"]
    cool_ic = (_L_v / _cp) * d["shoc_evap_incloud"] + (_L_s / _cp) * d["qi2qv_sublim_incloud"]
    return heat, cool, heat_ic, cool_ic

heat_clr, cool_clr, heat_ic, cool_ic = heating_cooling(ctrl)
heat_clr_p4, cool_clr_p4, heat_ic_p4, cool_ic_p4 = heating_cooling(p4k)

# Dedicated clear-sky rain-evaporation diagnostic (converted to K/s, added
# to the clear-sky cooling term). Assumed entirely clear-sky by construction,
# so it is NOT weighted by (1-CF) again below.
rain_evap_K = (_L_v / _cp) * ctrl["rain_evap_domain_mean"]
rain_evap_K_p4 = (_L_v / _cp) * p4k["rain_evap_domain_mean"]

# Area-weighted domain-mean contributions, as strictly positive magnitudes.
heat_clr_dm = np.abs((1 - CF) * heat_clr)
cool_clr_dm = np.abs((1 - CF) * cool_clr) + np.abs(rain_evap_K)
heat_ic_dm = np.abs(CF * heat_ic)
cool_ic_dm = np.abs(CF * cool_ic)

heat_clr_dm_p4 = np.abs((1 - CF_p4) * heat_clr_p4)
cool_clr_dm_p4 = np.abs((1 - CF_p4) * cool_clr_p4) + np.abs(rain_evap_K_p4)
heat_ic_dm_p4 = np.abs(CF_p4 * heat_ic_p4)
cool_ic_dm_p4 = np.abs(CF_p4 * cool_ic_p4)

# Interpolate onto the same T-coordinate grid used throughout the paper.
T_l = np.linspace(197, 297, 100)
def interp_to_T(var_ctrl, var_p4, ind=35, ind_p4=27):
    x_c, x_p = T[ind:], T_p4[ind_p4:]
    f_c = interp1d(x_c, var_ctrl[ind:], kind="linear")
    f_p = interp1d(x_p, var_p4[ind_p4:], kind="linear")
    Tc = np.clip(T_l, x_c.min(), x_c.max())
    Tp = np.clip(T_l, x_p.min(), x_p.max())
    return f_c(Tc), f_p(Tp)

heat_clr_l, heat_clr_l_p4 = interp_to_T(heat_clr_dm, heat_clr_dm_p4)
cool_clr_l, cool_clr_l_p4 = interp_to_T(cool_clr_dm, cool_clr_dm_p4)
heat_ic_l, heat_ic_l_p4 = interp_to_T(heat_ic_dm, heat_ic_dm_p4)
cool_ic_l, cool_ic_l_p4 = interp_to_T(cool_ic_dm, cool_ic_dm_p4)

mask = (T_l >= 210) & (T_l <= 270)

fig, ax = plt.subplots(figsize=(6.0, 6.5), constrained_layout=True)
ax.plot(heat_ic_l[mask], T_l[mask], color="firebrick", lw=2.2, label="In-cloud heating (cond+dep)")
ax.plot(cool_ic_l[mask], T_l[mask], color="darkorange", lw=2.2, label="In-cloud cooling (evap+sublim)")
ax.plot(heat_clr_l[mask], T_l[mask], color="steelblue", lw=2.2, label="Clear-sky heating (cond+dep)")
ax.plot(cool_clr_l[mask], T_l[mask], color="teal", lw=2.2, label="Clear-sky cooling (evap+sublim+rain)")

ax.plot(heat_ic_l_p4[mask], T_l[mask], color="firebrick", lw=2.2, ls="--", alpha=0.6)
ax.plot(cool_ic_l_p4[mask], T_l[mask], color="darkorange", lw=2.2, ls="--", alpha=0.6)
ax.plot(heat_clr_l_p4[mask], T_l[mask], color="steelblue", lw=2.2, ls="--", alpha=0.6)
ax.plot(cool_clr_l_p4[mask], T_l[mask], color="teal", lw=2.2, ls="--", alpha=0.6)

ax.set_xscale("log")
ax.set_ylim(270, 210)
ax.set_xlabel("Domain-mean latent heating/cooling magnitude [K s$^{-1}$] (log scale)")
ax.set_ylabel("Temperature [K]")
ax.set_title("Clear-sky vs. in-cloud latent heating and cooling (log scale)\n(solid: control; dashed/faded: +4K)", fontsize=11)
ax.legend(fontsize=9, loc="lower left")
ax.grid(alpha=0.3, which="both")

fig.savefig("figS_latent_clear_vs_cloud_profile.pdf", bbox_inches="tight", dpi=300)
print("wrote figS_latent_clear_vs_cloud_profile.pdf")
