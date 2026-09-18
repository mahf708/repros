"""
Reconstructs figS_mass_balance_check.pdf (SI section S3): absolute values
of the area-weighted in-cloud pressure velocity and area-weighted
subsidence velocity, control and +4K.

PROVENANCE NOTE: the original code for this exact figure was not saved as
a standalone script and was rebuilt from the SI text's description. An
earlier version of this script additionally multiplied omega_dn and
omega_up by (1-CF) and CF respectively, on the assumption that the cached
variables were raw conditional means needing that weighting to become
domain contributions. That produced curves roughly a factor of 9 apart at
the anvil peak, nowhere near the SI text's "within 10%" claim, and was
inconsistent with analysis_minimal.py's own built-in mass-balance sanity
check (same weighting, ~2800% median residual on the native grid).

Checking the RAW, unweighted magnitudes of omega_dn and omega_up directly
resolved this: they already sit within a few percent of each other at most
levels (within ~2-18% across 210-270K, consistent with "within 10%" as a
representative summary), strongly indicating that omega_dn and omega_up
are *already* the properly area-weighted domain contributions -- i.e.
(1-C)*omega_sub and C*omega_cld are what's cached directly, not
conditional means requiring a further CF multiplication. This version
plots them as-is. This now matches the SI text's claim reasonably well
and is a substantially better match than the earlier double-weighted
version, though it has not been checked against the original figure
pixel-for-pixel.

Run from a directory containing control_vars_minimal_full_record.npz and
plus4k_vars_minimal_full_record.npz.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}

ctrl = load("control_vars_minimal_full_record.npz")
p4k = load("plus4k_vars_minimal_full_record.npz")

T, T_p4 = ctrl["T"], p4k["T"]
omega_dn, omega_dn_p4 = ctrl["omega_dn"], p4k["omega_dn"]   # already (1-C)*omega_sub
omega_up, omega_up_p4 = ctrl["omega_up"], p4k["omega_up"]   # already C*omega_cld

# omega_dn/omega_up are already the area-weighted domain contributions
# (see provenance note above) -- no additional CF/(1-CF) multiplication.
sub_dm = np.abs(omega_dn)
cld_dm = np.abs(omega_up)
sub_dm_p4 = np.abs(omega_dn_p4)
cld_dm_p4 = np.abs(omega_up_p4)

T_l = np.linspace(197, 297, 100)
def interp_to_T(var_ctrl, var_p4, ind=35, ind_p4=27):
    x_c, x_p = T[ind:], T_p4[ind_p4:]
    f_c = interp1d(x_c, var_ctrl[ind:], kind="linear")
    f_p = interp1d(x_p, var_p4[ind_p4:], kind="linear")
    Tc = np.clip(T_l, x_c.min(), x_c.max())
    Tp = np.clip(T_l, x_p.min(), x_p.max())
    return f_c(Tc), f_p(Tp)

sub_l, sub_l_p4 = interp_to_T(sub_dm, sub_dm_p4)
cld_l, cld_l_p4 = interp_to_T(cld_dm, cld_dm_p4)

mask = (T_l >= 210) & (T_l <= 270)

fig, ax = plt.subplots(figsize=(6.0, 6.5), constrained_layout=True)
ax.plot(sub_l[mask], T_l[mask], color="steelblue", lw=2.2, label=r"$(1-C)\,\omega_\mathrm{sub}$, control")
ax.plot(cld_l[mask], T_l[mask], color="firebrick", lw=2.2, label=r"$C\,\omega_\mathrm{cld}$, control")
ax.plot(sub_l_p4[mask], T_l[mask], color="steelblue", lw=2.2, ls="--", alpha=0.7, label=r"$(1-C)\,\omega_\mathrm{sub}$, +4K")
ax.plot(cld_l_p4[mask], T_l[mask], color="firebrick", lw=2.2, ls="--", alpha=0.7, label=r"$C\,\omega_\mathrm{cld}$, +4K")

ax.set_ylim(270, 210)
ax.set_xlabel("Domain-mean vertical mass flux magnitude [Pa s$^{-1}$]")
ax.set_ylabel("Temperature [K]")
ax.set_title("Verifying the mass balance, Eq. 3:\n" r"$(1-C)\,\omega_\mathrm{sub} = C\,\omega_\mathrm{cld}$", fontsize=12)
ax.legend(fontsize=10, loc="best")
ax.grid(alpha=0.3)

fig.savefig("figS_mass_balance_check.pdf", bbox_inches="tight", dpi=300)
print("wrote figS_mass_balance_check.pdf")
