"""
Minimal analysis script for the GRL cloud-fraction paper.

Design goals for this version:
  - Reproduces the four paper figures (CF/subsidence/rad-cooling/stability;
    dF/dT & S_inv; tau_i_sub/tau_i_cloud/CF; stepwise buffering bar chart)
  - Adds the CSC*dl "special case" comparison figure, in the same styled
    format as the paper figures (previously this lived only in an
    unstyled exploratory plot).
  - Adds a new Q_clr vs Q_dyn_clr balance-check figure, testing the WTG
    steady-state assumption that clear-sky radiative cooling is balanced
    by dynamical (adiabatic/advective) warming.
  - Computes the Q_T = 0 crossing temperature and its shift under warming,
    for comparison against the Seidel & Yang (2022) radiative-tropopause
    metric (~0.09-0.15 K/K for self-lofting ozone vs ~0.4 K/K for fixed-
    in-pressure ozone).

"""

import numpy as np
import os as _os
import matplotlib
# Batch-safe by default: never open interactive windows. Set SHOW_FIGS=1 in
# the environment to restore the old blocking plt.show() behavior.
SHOW_FIGS = bool(int(_os.environ.get("SHOW_FIGS", "0")))
if not SHOW_FIGS:
    matplotlib.use("Agg")
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# =====================================================
# CONFIG / CACHING
# =====================================================

run_new = False  # False: load from the cached .npz files below (control/plus4k_vars_minimal_full_record.npz), no netCDF access needed.

# Updated to local lustre1 copy (NERSC scratch currently down); adjust filenames if they differ.
# These netCDF files were generated via:
#   ncrcat 3hi_ne4pg2_anvil_diags.INSTANT.nhours_x3* anvil_diags
sim_control = '/p/lustre1/beydoun1/control_anvil_diags.nc'
sim_plus4k = '/p/lustre1/beydoun1/plus4k_anvil_diags.nc'

# Restrict analysis to a single full calendar year (Jan-Dec 2020), leaving ~5 months
# (Aug-Dec 2019) of spin-up rather than the paper's 1 month -- a check on whether the
# 18-month, non-integer-number-of-years averaging window was hiding a spin-up/seasonal-
# aliasing issue. Set to None to fall back to the full record (original behavior).
ANALYSIS_START = None
ANALYSIS_END = None

# Cache filenames are tagged with the analysis window so switching windows can never
# silently load a cache computed over a different period.
_window_tag = f"_{ANALYSIS_START}_{ANALYSIS_END}" if ANALYSIS_START else "_full_record"
CACHE_CONTROL = f'control_vars_minimal{_window_tag}.npz'
CACHE_PLUS4K = f'plus4k_vars_minimal{_window_tag}.npz'

dTs = 4.0          # Cess-Potter SST perturbation, K
peak_idx = 30      # index into T_l (anvil peak, ~228 K) -- matches original script

# Which "Q" (clear-sky radiative cooling) to use throughout the script:
#   'all_sky'       (default): the model's actual all-sky rrtmgp_T_mid_tend diagnostic,
#                    masked to clear-sky columns (qi+qc<=1e-5) -- what "Q" has always
#                    meant in this script so far.
#   'clearsky_flux': derived from the DEDICATED clear-sky flux diagnostic
#                    (SW/LW_clrsky_flux_up/dn), differenced per-column w.r.t. the real
#                    per-column pressure (p_mid), then masked the same way -- a
#                    genuinely different physical quantity (radiation-scheme's
#                    hypothetical "if there were no clouds anywhere" calculation,
#                    rather than the real all-sky tendency sampled at locally-clear
#                    columns). NOT yet verified against real data -- check that the
#                    resulting Q looks physically sensible (order 1e-5 K/s, same sign
#                    convention as the default) before trusting downstream numbers.
Q_SOURCE = 'all_sky'  # or 'clearsky_flux'

# Only pull the variables we actually need -- minimizes I/O vs. the full file.
NEEDED_VARS = [
    'T_mid', 'p_mid', 'count_where_qi_plus_qc_gt_0.00001',
    'SW_clrsky_flux_up', 'SW_clrsky_flux_dn',
    'LW_clrsky_flux_up', 'LW_clrsky_flux_dn',
    'omega_where_qi_plus_qc_gt_0.00001',
    'omega_where_qi_plus_qc_le_0.00001',
    'DryStaticEnergy_pvert_derivative_where_qi_le_0.00001',
    'rrtmgp_T_mid_tend_where_qi_plus_qc_le_0.00001',
    'homme_T_mid_tend_where_qi_plus_qc_le_0.00001',
    'omega_pvert_derivative_where_qi_plus_qc_gt_0.00001',
    'qi_where_qi_plus_qc_gt_0.00001', 'qc',
    'qv2qi_vapdep_where_qi_plus_qc_gt_0.00001',
    'shoc_cond_where_qi_plus_qc_gt_0.00001',
    # in-cloud native diagnostics for the other two microphysics terms, needed
    # to compute their clear-sky counterparts by subtraction (see get_vars()
    # below) rather than by masking the unconditional field directly:
    'shoc_evap_where_qi_plus_qc_gt_0.00001',
    'qi2qv_sublim_where_qi_plus_qc_gt_0.00001',
    'homme_qi_tend_where_qi_plus_qc_gt_0.00001', 'homme_qc_tend',
    # unconditioned (full-domain) versions, combined with (1-count_ice) below to
    # get the clear-sky microphysical/latent-heating tendency -- tests whether
    # subvisible cirrus (below the qi+qc>1e-5 cloud mask) sublimating/depositing
    # in nominally "clear" columns explains the Fig 6 cold-end residual.
    'shoc_cond', 'shoc_evap', 'qi2qv_sublim', 'qv2qi_vapdep',
    'lat', 'area',
]

# =====================================================
# GRID / AVERAGING HELPERS (unchanged from original)
# =====================================================

# `area` is set inside the `if run_new:` block below (only needed when actually
# reading netCDF); declared here so the functions below can close over it.
area = None


def tropical_average(var):
    var_mean = np.mean(var * area, (0, 1)) / np.mean(area, (0, 1))
    return var_mean


def in_cloud_average(var, count):
    var_copy = var.copy()
    var_mean = np.nanmean(var_copy * area * count, (0, 1)) / np.mean(area, (0, 1))
    return var_mean


def in_cloud_sum(var, count):
    var_copy = var.copy()
    var_copy[np.isnan(var_copy)] = 0.0
    var_mean = np.mean(np.sum(count * area * var_copy, 1) / np.sum(area, (0, 1)), 0)
    return var_mean


# ---------------------------------------------------------------------------
# OPTIONAL further speedup (not enabled by default): the three functions above
# still call .values inside get_vars() *before* any reduction happens, which
# forces dask to materialize the full (already tropics-sliced) time series in
# memory. Since the data is dask-backed (via chunks= above), the reductions
# can instead run lazily -- dask streams+reduces chunk by chunk and only the
# much smaller final result ever touches memory. To use this path: in
# get_vars(), replace every `ds['X'].values` with `ds['X']` (i.e. pass the
# raw xr.DataArray straight through, no .values), and swap in these lazy
# versions of the three helpers instead of the eager ones above:
#
#   def tropical_average_lazy(da):
#       area_da = xr.DataArray(area.squeeze(), dims=[col_dim])
#       return (da * area_da).sum(col_dim).mean('time') / area_da.sum(col_dim)
#
#   def in_cloud_average_lazy(da, count):
#       area_da = xr.DataArray(area.squeeze(), dims=[col_dim])
#       return (da * area_da * count).sum(col_dim).mean('time') / area_da.sum(col_dim)
#
#   def in_cloud_sum_lazy(da, count):
#       area_da = xr.DataArray(area.squeeze(), dims=[col_dim])
#       da_filled = da.fillna(0.0)
#       return ((count * area_da * da_filled).sum(col_dim) / area_da.sum(col_dim)).mean('time')
#
# Each of these returns a small dask-backed (lev,) DataArray; call .values (or
# .compute()) on the *result* of get_vars(), once, rather than on each raw
# field. This is the bigger win of the two changes here, but touches more
# lines, so it's left as an opt-in swap for you to test against known-good
# output from the eager version first, rather than baked in as the default.
# ---------------------------------------------------------------------------


# =====================================================
# MINIMAL get_vars: no mass_flux dependency
# =====================================================

def vertical_gradient(f, p):
    """d(f)/d(p) along the last (level) axis, supporting a coordinate array
    p with the SAME shape as f -- unlike np.gradient, which only supports a
    1D coordinate shared across the other dimensions. Needed here because
    pressure genuinely varies with time and column, not just vertical
    level, and we want the actual per-column pressure (p_mid) rather than
    the fixed reference profile (p_mean, from 'lev') used elsewhere in this
    script as a simpler approximation. Uses the same central-difference /
    one-sided-at-edges scheme as np.gradient's default algorithm."""
    df_dp = np.empty_like(f)
    df_dp[..., 1:-1] = (f[..., 2:] - f[..., :-2]) / (p[..., 2:] - p[..., :-2])
    df_dp[..., 0] = (f[..., 1] - f[..., 0]) / (p[..., 1] - p[..., 0])
    df_dp[..., -1] = (f[..., -1] - f[..., -2]) / (p[..., -1] - p[..., -2])
    return df_dp


def get_vars(ds):
    T = tropical_average(ds['T_mid'].values)
    p = ds['p_mid'].values
    count_ice = ds['count_where_qi_plus_qc_gt_0.00001'].values

    flux_sw_cs = ds['SW_clrsky_flux_up'].values - ds['SW_clrsky_flux_dn'].values
    flux_lw_cs = ds['LW_clrsky_flux_up'].values - ds['LW_clrsky_flux_dn'].values
    F_rad_clrsky = tropical_average(flux_sw_cs) + tropical_average(flux_lw_cs)

    # clear sky (kept exactly as in the original script's masking convention)
    omega_dn = in_cloud_sum(ds['omega_where_qi_plus_qc_le_0.00001'].values, 1 - count_ice)
    S = in_cloud_average(ds['DryStaticEnergy_pvert_derivative_where_qi_le_0.00001'].values, 1 - count_ice)

    if Q_SOURCE == 'clearsky_flux':
        # Q derived from the DEDICATED clear-sky flux diagnostic (SW/LW_clrsky_flux_up/dn),
        # differenced per-column w.r.t. the REAL per-column pressure (p_mid), THEN masked to
        # clear-sky columns -- a genuinely different quantity from the default 'all_sky'
        # option below (which masks the model's actual all-sky rrtmgp tendency to clear-sky
        # columns). Standard heating-rate identity: dT/dt = (g/cp)*d(F_net_up)/dp, with
        # F_net_up = (SW_up-SW_dn)+(LW_up-LW_dn) already computed above as flux_sw_cs+flux_lw_cs.
        _g, _cp = 9.81, 1004.0
        Q_clrsky_percol = (_g / _cp) * vertical_gradient(flux_sw_cs + flux_lw_cs, p)
        Q = in_cloud_average(Q_clrsky_percol, 1 - count_ice)
    else:
        Q = in_cloud_average(ds['rrtmgp_T_mid_tend_where_qi_plus_qc_le_0.00001'].values, 1 - count_ice)

    Q_dyn_clr = in_cloud_average(ds['homme_T_mid_tend_where_qi_plus_qc_le_0.00001'].values, 1 - count_ice)

    # Clear-sky (subvisible cirrus / below-cloud) microphysical process rates.
    # Computed by subtraction (true domain total minus the native in-cloud
    # diagnostic), divided by (1-CF) to give a true per-unit-clear-area
    # conditional average -- NOT by masking the unconditional field with
    # (1-count_ice) directly, which was the previous (incorrect) approach.
    # Mixing-ratio units (kg/kg/s); converted to temperature-tendency units
    # downstream via latent heats.
    _CF_for_clr = tropical_average(count_ice)

    _shoc_cond_incloud_native = ds['shoc_cond_where_qi_plus_qc_gt_0.00001'].values
    _shoc_evap_incloud_native = ds['shoc_evap_where_qi_plus_qc_gt_0.00001'].values
    _qi2qv_sublim_incloud_native = ds['qi2qv_sublim_where_qi_plus_qc_gt_0.00001'].values
    _qv2qi_vapdep_incloud_native = ds['qv2qi_vapdep_where_qi_plus_qc_gt_0.00001'].values

    shoc_cond_clr = (tropical_average(ds['shoc_cond'].values)
                      - in_cloud_sum(_shoc_cond_incloud_native, count_ice)) / (1 - _CF_for_clr)
    shoc_evap_clr = (tropical_average(ds['shoc_evap'].values)
                      - in_cloud_sum(_shoc_evap_incloud_native, count_ice)) / (1 - _CF_for_clr)
    qi2qv_sublim_clr = (tropical_average(ds['qi2qv_sublim'].values)
                         - in_cloud_sum(_qi2qv_sublim_incloud_native, count_ice)) / (1 - _CF_for_clr)
    qv2qi_vapdep_clr = (tropical_average(ds['qv2qi_vapdep'].values)
                         - in_cloud_sum(_qv2qi_vapdep_incloud_native, count_ice)) / (1 - _CF_for_clr)

    # True in-cloud conditional means of the same four terms, for a direct
    # clear-vs-cloudy magnitude comparison. in_cloud_sum gives the domain-
    # contribution style value (CF * true in-cloud mean); dividing by CF
    # converts it to the same true-conditional-mean footing as the *_clr
    # terms above, so the two are directly comparable.
    shoc_cond_incloud = in_cloud_sum(_shoc_cond_incloud_native, count_ice) / _CF_for_clr
    shoc_evap_incloud = in_cloud_sum(_shoc_evap_incloud_native, count_ice) / _CF_for_clr
    qi2qv_sublim_incloud = in_cloud_sum(_qi2qv_sublim_incloud_native, count_ice) / _CF_for_clr
    qv2qi_vapdep_incloud = in_cloud_sum(_qv2qi_vapdep_incloud_native, count_ice) / _CF_for_clr

    # cloudy sky
    div_ice = in_cloud_sum(ds['omega_pvert_derivative_where_qi_plus_qc_gt_0.00001'].values, count_ice)
    omega_up = in_cloud_sum(ds['omega_where_qi_plus_qc_gt_0.00001'].values, count_ice)

    qi_in_cloud = ds['qi_where_qi_plus_qc_gt_0.00001'].values
    qc_in_cloud = ds['qc'].values / count_ice
    qc_in_cloud[np.isinf(qc_in_cloud)] = np.nan

    vap_dep = ds['qv2qi_vapdep_where_qi_plus_qc_gt_0.00001'].values
    cond = ds['shoc_cond_where_qi_plus_qc_gt_0.00001'].values
    homme_qi_tend_in_cloud = ds['homme_qi_tend_where_qi_plus_qc_gt_0.00001'].values
    homme_qc_tend_in_cloud = ds['homme_qc_tend'].values / count_ice
    homme_qc_tend_in_cloud[np.isinf(homme_qc_tend_in_cloud)] = np.nan

    source_sum = (in_cloud_sum(cond, count_ice)
                  + in_cloud_sum(homme_qc_tend_in_cloud, count_ice)
                  + in_cloud_sum(homme_qi_tend_in_cloud, count_ice)
                  + in_cloud_sum(vap_dep, count_ice))

    CF = tropical_average(count_ice)
    # source-based in-cloud lifetime (tau_mp) -- no mass_flux/sinks needed.
    # NOTE: get_vars() previously also returned a `delta = CF/(tau*(omega_up/9.81))`
    # here, but that quantity is NOT the one used in the CSC comparison below --
    # it's dimensionally different (not a plain ratio) and was a leftover from an
    # earlier exploratory diagnostic. The CSC comparison needs the dimensionless
    # ratio delta = tau_cloud/tau_mp, which we build downstream from `tau` and
    # `tau_i_cloud` instead (see the CSC comparison section).
    tau = (in_cloud_sum(qi_in_cloud, count_ice) + in_cloud_sum(qc_in_cloud, count_ice)) / source_sum

    qa = in_cloud_average(qc_in_cloud, count_ice) + in_cloud_average(qi_in_cloud, count_ice)

    return (T, Q, S, omega_dn, omega_up, CF, tau, qa, div_ice, F_rad_clrsky, Q_dyn_clr,
            shoc_cond_clr, shoc_evap_clr, qi2qv_sublim_clr, qv2qi_vapdep_clr)


# =====================================================
# LOAD (with simple named caching -- no arr_N indexing)
# =====================================================

VARNAMES = ['T', 'Q', 'S', 'omega_dn', 'omega_up', 'CF', 'tau', 'qa', 'div', 'F_rad_clrsky', 'Q_dyn_clr',
            'shoc_cond_clr', 'shoc_evap_clr', 'qi2qv_sublim_clr', 'qv2qi_vapdep_clr']

if run_new:
    print("reading from netcdf files")

    # chunks + parallel=True let dask read/decode time-chunks in parallel instead
    # of one big single-threaded eager load; adjust the time chunk size to taste.
    _open_kwargs = dict(chunks={'time': 50}, parallel=True)
    ds_control_full = xr.open_mfdataset(sim_control, **_open_kwargs)
    ds_plus4k_full = xr.open_mfdataset(sim_plus4k, **_open_kwargs)

    p_mean = ds_control_full['lev'].values * 100
    lat = ds_control_full['lat'].values      # tiny (ncol,) array, cheap to pull fully
    I = (lat < 30) & (lat > -30)

    # Column dim name discovered from 'lat' rather than hardcoded (robust to
    # whatever the native grid dimension is actually called, e.g. 'ncol').
    col_dim = ds_control_full['lat'].dims[0]
    trop_idx = np.where(I)[0]

    # Slice to the tropical band *before* selecting variables / pulling .values
    # below -- this is the main I/O win: only ~half the columns (30S-30N) ever
    # get read off disk or decoded, instead of reading the full global field
    # and discarding the rest.
    ds_control = ds_control_full[NEEDED_VARS].isel({col_dim: trop_idx})
    ds_plus4k = ds_plus4k_full[NEEDED_VARS].isel({col_dim: trop_idx})

    if ANALYSIS_START is not None:
        ds_control = ds_control.sel(time=slice(ANALYSIS_START, ANALYSIS_END))
        ds_plus4k = ds_plus4k.sel(time=slice(ANALYSIS_START, ANALYSIS_END))
        n_ctrl, n_p4k = ds_control.sizes['time'], ds_plus4k.sizes['time']
        print(f"Restricted to {ANALYSIS_START} -- {ANALYSIS_END}: "
              f"{n_ctrl} control / {n_p4k} +4K time samples selected")
        if n_ctrl == 0 or n_p4k == 0:
            raise ValueError(
                "Time selection returned zero samples -- check that the file's time "
                "coordinate is actual calendar dates (not a relative/step index), and "
                "that ANALYSIS_START/END fall within the file's date range.")

    area = ds_control['area'].values.reshape((1, len(trop_idx), 1))

    print("control")
    control_vals = get_vars(ds_control)
    print("plus4k")
    plus4k_vals = get_vars(ds_plus4k)

    # =====================================================
    # Rain evaporation, from a SEPARATE netCDF file (mass_flux, not the main
    # anvil_diags files) -- self-contained read, does not touch the global
    # `area`/`lat`/`trop_idx`/col_dim set up above for the main files, to
    # avoid any risk of clobbering state get_vars() or anything downstream
    # relies on. qr2qv_evap here is domain-mean only (no clear/cloud split
    # available), so per your instruction we tropical-average it and divide
    # by (1-CF) -- i.e. assume ALL of the domain-mean signal is
    # attributable to clear-sky columns (the upper-bound end of the
    # assumption-free [0, domain_mean] bracket discussed earlier, not a
    # split estimate). Assumes qr2qv_evap sits on the same native 'lev' grid
    # as the main files (not independently verified here).
    # =====================================================
    # These netCDF files were generated via:
    #   ncrcat 3hi_ne4pg2_mass_flux_diags.INSTANT.nhours_x3* mass_flux_diags
    sim_control_mf = '/p/lustre1/beydoun1/control_mass_flux.nc'
    sim_plus4k_mf = '/p/lustre1/beydoun1/plus4k_mass_flux.nc'

    print("rain evaporation (separate mass_flux file)")
    _ds_mf_ctrl_full = xr.open_mfdataset(sim_control_mf, **_open_kwargs)
    _ds_mf_p4k_full = xr.open_mfdataset(sim_plus4k_mf, **_open_kwargs)

    _mf_lat = _ds_mf_ctrl_full['lat'].values
    _mf_I = (_mf_lat < 30) & (_mf_lat > -30)
    _mf_col_dim = _ds_mf_ctrl_full['lat'].dims[0]
    _mf_trop_idx = np.where(_mf_I)[0]

    _ds_mf_ctrl = _ds_mf_ctrl_full[['qr2qv_evap', 'area']].isel({_mf_col_dim: _mf_trop_idx})
    _ds_mf_p4k = _ds_mf_p4k_full[['qr2qv_evap', 'area']].isel({_mf_col_dim: _mf_trop_idx})
    if ANALYSIS_START is not None:
        _ds_mf_ctrl = _ds_mf_ctrl.sel(time=slice(ANALYSIS_START, ANALYSIS_END))
        _ds_mf_p4k = _ds_mf_p4k.sel(time=slice(ANALYSIS_START, ANALYSIS_END))

    _mf_area_ctrl = _ds_mf_ctrl['area'].values.reshape((1, len(_mf_trop_idx), 1))
    _mf_area_p4k = _ds_mf_p4k['area'].values.reshape((1, len(_mf_trop_idx), 1))

    # Local, self-contained tropical average -- deliberately NOT reusing the
    # module-level tropical_average()/`area` to avoid any dependence on or
    # interference with the main files' state.
    def _tropical_average_mf(var, area_local):
        return np.mean(var * area_local, (0, 1)) / np.mean(area_local, (0, 1))

    rain_evap_domain_mean = _tropical_average_mf(_ds_mf_ctrl['qr2qv_evap'].values, _mf_area_ctrl)
    rain_evap_domain_mean_plus4k = _tropical_average_mf(_ds_mf_p4k['qr2qv_evap'].values, _mf_area_p4k)

    # p_mean is cached alongside everything else so the else-branch below never
    # needs to touch netCDF at all.
    np.savez(CACHE_CONTROL, **dict(zip(VARNAMES, control_vals)), p_mean=p_mean,
             rain_evap_domain_mean=rain_evap_domain_mean)
    np.savez(CACHE_PLUS4K, **dict(zip(VARNAMES, plus4k_vals)),
             rain_evap_domain_mean=rain_evap_domain_mean_plus4k)
else:
    print("loading pre-saved variables (no netCDF access)")
    _cache_control = np.load(CACHE_CONTROL)
    control_vals = [_cache_control[v] for v in VARNAMES]
    plus4k_vals = [np.load(CACHE_PLUS4K)[v] for v in VARNAMES]
    p_mean = _cache_control['p_mean']
    rain_evap_domain_mean = _cache_control['rain_evap_domain_mean']
    rain_evap_domain_mean_plus4k = np.load(CACHE_PLUS4K)['rain_evap_domain_mean']

(T, Q, S, omega_dn, omega_up, CF, tau, qa, div, F_rad_clrsky, Q_dyn_clr,
 shoc_cond_clr, shoc_evap_clr, qi2qv_sublim_clr, qv2qi_vapdep_clr) = control_vals
(T_plus4k, Q_plus4k, S_plus4k, omega_dn_plus4k, omega_up_plus4k, CF_plus4k, tau_plus4k,
 qa_plus4k, div_plus4k, F_rad_clrsky_plus4k, Q_dyn_clr_plus4k,
 shoc_cond_clr_plus4k, shoc_evap_clr_plus4k, qi2qv_sublim_clr_plus4k, qv2qi_vapdep_clr_plus4k) = plus4k_vals

dl = np.log(qa) - np.log(1e-5)
dl_plus4k = np.log(qa_plus4k) - np.log(1e-5)

# =====================================================
# VERIFY EQ. 3 (MASS BALANCE): (1-C)*omega_sub = C*omega_cld
#
# Directly requested by the reviewer ("it would be nice for the authors to
# verify that their Eq. 3 holds in these simulations"). Checked on the
# native (pre-interpolation) diagnosed fields, both states, as a fractional
# residual: [(1-CF)*omega_dn - CF*omega_up] / (CF*omega_up). Should be
# small if the mass-balance identity holds to good approximation; a large,
# systematic residual would indicate the identity is not well satisfied
# and needs further explanation, not just a reported number.
# =====================================================

# =====================================================
# First: inspect the RAW sign and magnitude of omega_dn and omega_up
# directly, rather than assume a sign convention -- per the suggestion to
# check this empirically. Prints a few sample levels of each (raw, signed)
# so it's immediately visible whether they come out with the sign the
# established tau_i_sub/tau_i_cloud formulas' leading "-" signs would imply.
# =====================================================
_sample_idx = np.linspace(0, len(omega_dn) - 1, 8).astype(int)
print("\n[Raw sign check] omega_dn (clear-sky) and omega_up (in-cloud), a few native-level samples:")
print("  omega_dn (should be signed; compare against tau_i_sub's formula, which assumes omega_dn<0):")
print("   ", omega_dn[_sample_idx])
print("  omega_up (should be signed; compare against tau_i_cloud's formula, which assumes omega_up<0):")
print("   ", omega_up[_sample_idx])

# =====================================================
# Eq. 3 MASS-BALANCE CHECK, positive-definite (absolute value) version:
# (1-C)*|omega_dn| vs C*|omega_up|. This sidesteps the raw sign-convention
# question entirely (still not fully settled from the code alone -- see
# above) by comparing MAGNITUDES only. This is a real, meaningful check of
# whether the mass-balance identity holds in terms of relative SIZE, even
# though it is not a complete substitute for a fully signed verification.
# =====================================================
_lhs = (1 - CF) * np.abs(omega_dn)
_rhs = CF * np.abs(omega_up)
_massbal_resid = (_lhs - _rhs) / _rhs
_lhs_p4k = (1 - CF_plus4k) * np.abs(omega_dn_plus4k)
_rhs_p4k = CF_plus4k * np.abs(omega_up_plus4k)
_massbal_resid_p4k = (_lhs_p4k - _rhs_p4k) / _rhs_p4k

print(f"\n[Eq. 3 mass-balance check, MAGNITUDE-ONLY] (1-C)*|omega_sub| vs C*|omega_cld|, fractional residual:")
print(f"  Control: median={np.nanmedian(_massbal_resid)*100:.2f}%  "
      f"5th-95th pctile=[{np.nanpercentile(_massbal_resid,5)*100:.2f}%, {np.nanpercentile(_massbal_resid,95)*100:.2f}%]")
print(f"  +4K:     median={np.nanmedian(_massbal_resid_p4k)*100:.2f}%  "
      f"5th-95th pctile=[{np.nanpercentile(_massbal_resid_p4k,5)*100:.2f}%, {np.nanpercentile(_massbal_resid_p4k,95)*100:.2f}%]")
print("NOTE: if this magnitude-only check now looks reasonable (residual small, order 10s of %, not "
      "thousands), the remaining question is purely about SIGN convention, not about the omega_dn fix "
      "or the underlying physics -- worth confirming directly from the raw sign-check print above before "
      "trusting any signed version of this check.")

# =====================================================
# DIAGNOSTIC FIGURE: omega_dn and omega_up profiles, made positive-definite
# (absolute value) so they can be directly visually compared regardless of
# raw sign convention -- exactly as requested, to check these are behaving
# like genuinely distinct physical quantities (not the identical-array bug
# from before) and that their relative magnitudes look physically sensible.
# =====================================================
fig_diag, ax_diag = plt.subplots(figsize=(6, 6), constrained_layout=True)
_lev_idx = np.arange(len(omega_dn))
ax_diag.plot(np.abs(omega_dn), _lev_idx, lw=2, color="tab:blue", label=r"$|\omega_\mathrm{dn}|$ (clear-sky)")
ax_diag.plot(np.abs(omega_up), _lev_idx, lw=2, color="tab:red", label=r"$|\omega_\mathrm{up}|$ (in-cloud)")
ax_diag.set_xlabel(r"$|\omega|$ [Pa s$^{-1}$]")
ax_diag.set_ylabel("Native level index")
ax_diag.set_title("Diagnostic: omega_dn vs omega_up (both positive-definite)\ncontrol state, native levels")
ax_diag.legend(frameon=False)
ax_diag.grid(True, alpha=0.3)
ax_diag.invert_yaxis()
fig_diag.savefig("diagnostic_omega_dn_vs_up.pdf", bbox_inches="tight", dpi=300)
if SHOW_FIGS:
    plt.show()
print("[Diagnostic figure saved: diagnostic_omega_dn_vs_up.pdf] -- confirm these two curves now look "
      "like genuinely different physical quantities (they should generally differ noticeably in "
      "magnitude and vertical structure -- clear-sky subsidence is typically weaker but occurs over "
      "far more area than in-cloud ascent), not identical or near-identical as they would have been "
      "before the omega_dn fix.")

# =====================================================
# TEMPERATURE-COORDINATE INTERPOLATION
# =====================================================

T_l = np.linspace(197, 297, 100)


def interpolate_to_T(var, var_plus4k, ind, ind_plus4k):
    x_ctrl = T[ind:]
    x_p4k = T_plus4k[ind_plus4k:]
    f_ctrl = interp1d(x_ctrl, var[ind:], kind='linear')
    f_p4k = interp1d(x_p4k, var_plus4k[ind_plus4k:], kind='linear')
    # Clip query points to each array's own valid range. This absorbs tiny
    # (~1e-3 K) floating-point boundary mismatches that can arise from
    # different reduction orders (e.g. chunked/parallel vs. eager averaging)
    # without silently extrapolating past genuinely out-of-range values --
    # if you ever see a *large* clip here, that's worth investigating, but
    # sub-0.01K differences are just numerical noise from summation order.
    T_l_ctrl = np.clip(T_l, x_ctrl.min(), x_ctrl.max())
    T_l_p4k = np.clip(T_l, x_p4k.min(), x_p4k.max())
    return f_ctrl(T_l_ctrl), f_p4k(T_l_p4k)


p_l, p_l_plus4k = interpolate_to_T(p_mean, p_mean, 35, 27)
S_l, S_l_plus4k = interpolate_to_T(-S, -S_plus4k, 35, 27)
CF_l, CF_l_plus4k = interpolate_to_T(CF, CF_plus4k, 35, 27)
omega_up_l, omega_up_l_plus4k = interpolate_to_T(omega_up, omega_up_plus4k, 35, 27)
omega_dn_l, omega_dn_l_plus4k = interpolate_to_T(omega_dn, omega_dn_plus4k, 35, 27)
Q_l, Q_l_plus4k = interpolate_to_T(Q, Q_plus4k, 35, 27)
div_l, div_l_plus4k = interpolate_to_T(div, div_plus4k, 35, 27)
dl_l, dl_l_plus4k = interpolate_to_T(dl, dl_plus4k, 35, 27)
tau_l, tau_l_plus4k = interpolate_to_T(tau, tau_plus4k, 35, 27)  # source-based tau_mp, T-coordinates
Q_dyn_clr_l, Q_dyn_clr_l_plus4k = interpolate_to_T(Q_dyn_clr, Q_dyn_clr_plus4k, 35, 27)  # NEW
shoc_cond_clr_l, shoc_cond_clr_l_plus4k = interpolate_to_T(shoc_cond_clr, shoc_cond_clr_plus4k, 35, 27)
shoc_evap_clr_l, shoc_evap_clr_l_plus4k = interpolate_to_T(shoc_evap_clr, shoc_evap_clr_plus4k, 35, 27)
qi2qv_sublim_clr_l, qi2qv_sublim_clr_l_plus4k = interpolate_to_T(qi2qv_sublim_clr, qi2qv_sublim_clr_plus4k, 35, 27)
qv2qi_vapdep_clr_l, qv2qi_vapdep_clr_l_plus4k = interpolate_to_T(qv2qi_vapdep_clr, qv2qi_vapdep_clr_plus4k, 35, 27)
# Assumes rain_evap_domain_mean sits on the same native 'lev' grid/ordering
# as the main files -- not independently verified, since it comes from a
# separate netCDF file (see mass_flux read above).
rain_evap_domain_mean_l, rain_evap_domain_mean_l_plus4k = interpolate_to_T(
    rain_evap_domain_mean, rain_evap_domain_mean_plus4k, 35, 27)

# =====================================================
# DERIVED THERMODYNAMIC QUANTITIES (unchanged formulas)
# =====================================================

ro = p_l / (287 * T_l)
ro_plus4k = p_l_plus4k / (287 * T_l)

gam_d = 9.81 * 1e-3
gam = (1 - S_l * ro) * gam_d
gam_plus4k = (1 - S_l_plus4k * ro_plus4k) * gam_d

inv_S = ((1 / gam) - (1 / gam_d)) ** (-1)
inv_S_plus4k = ((1 / gam_plus4k) - (1 / gam_d)) ** (-1)

Q_T = Q_l * ro / gam
Q_T_plus4k = Q_l_plus4k * ro_plus4k / gam_plus4k

dZ = 1 / gam
dZ_plus4k = 1 / gam_plus4k
dp = dZ * 9.81 * ro
dp_plus4k = dZ_plus4k * 9.81 * ro_plus4k

tau_i_sub = np.abs((gam / np.gradient(T_l)) * omega_dn_l / ro / 9.81)
tau_i_sub_plus4k = np.abs((gam_plus4k / np.gradient(T_l)) * omega_dn_l_plus4k / ro_plus4k / 9.81)

tau_i_cloud = -(omega_up_l / CF_l) * (gam / np.gradient(T_l)) / ro / 9.81
tau_i_cloud_plus4k = -(omega_up_l_plus4k / CF_l_plus4k) * (gam_plus4k / np.gradient(T_l)) / ro_plus4k / 9.81

# Dimensionless delta = tau_cloud / tau_mp, matching the original script's
# `delta = (-1/tau_i_cloud)/tau_l` -- just using the source-based tau_l here
# (since we don't have the sink-based version without mass_flux). At steady
# state source ~= sink so these should track closely; worth cross-checking
# once mass_flux access is back.
delta_l = (-1 / tau_i_cloud) / tau_l
delta_l_plus4k = (-1 / tau_i_cloud_plus4k) / tau_l_plus4k

Q_clr = np.diff(F_rad_clrsky) / np.gradient(p_mean)
Q_clr_plus4k = np.diff(F_rad_clrsky_plus4k) / np.gradient(p_mean)
Q_clr_l, Q_clr_l_plus4k = interpolate_to_T(Q_clr, Q_clr_plus4k, 35, 27)

tau_i_sub_calc = -(Q_clr_l * 9.81 / 1000 * (ro / gam)) * inv_S * 1000 / dp
tau_i_sub_calc_plus4k = -(Q_clr_l_plus4k * 9.81 / 1000 * (ro_plus4k / gam_plus4k)) * inv_S_plus4k * 1000 / dp_plus4k

CF_calc = -(Q_clr_l * 9.81 / 1000 * (ro / gam)) * inv_S * 1000 / dp / tau_i_cloud
CF_calc_plus4k = -(Q_clr_l_plus4k * 9.81 / 1000 * (ro_plus4k / gam_plus4k)) * inv_S_plus4k * 1000 / dp_plus4k / tau_i_cloud_plus4k

# =====================================================
# QUANTITATIVE CHECK: how well does CF_calc (built from the Q/S-PREDICTED
# tau_i_sub, i.e. tau_i_sub_calc, combined with the diagnosed tau_i_cloud)
# reproduce the actually-diagnosed CF_l?
#
# This is the practical, end-to-end test of the Q/S approximation's
# validity for the paper's actual purpose (predicting cloud fraction) --
# NOT a trivial mass-balance identity check, since tau_i_sub_calc is a
# genuinely different quantity from the diagnosed tau_i_sub in Fig 3a
# (built from Q_clr_l/S_inv, not from omega_dn_l directly). Reported here
# as both a correlation (R^2) and a direct fractional-bias metric at the
# anvil peak, to give "the reconstruction works well" an actual number.
# =====================================================

from scipy.stats import linregress as _linregress_cfcheck

def _r2_between(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    if np.sum(valid) < 3:
        return np.nan, np.nan
    slope, intercept, rval, _, _ = _linregress_cfcheck(a[valid], b[valid])
    return slope, rval**2

slope_cf_ctrl, r2_cf_ctrl = _r2_between(CF_calc, CF_l)
slope_cf_p4k, r2_cf_p4k = _r2_between(CF_calc_plus4k, CF_l_plus4k)

frac_bias_ctrl = (CF_calc[peak_idx] - CF_l[peak_idx]) / CF_l[peak_idx] * 100
frac_bias_p4k = (CF_calc_plus4k[peak_idx] - CF_l_plus4k[peak_idx]) / CF_l_plus4k[peak_idx] * 100

print(f"\n[Fig 3c end-to-end Q/S check] CF_calc vs. diagnosed CF_l across the full T_l profile:")
print(f"  Control: slope={slope_cf_ctrl:.3f}  R^2={r2_cf_ctrl:.3f}")
print(f"  +4K:     slope={slope_cf_p4k:.3f}  R^2={r2_cf_p4k:.3f}")
print(f"[Fig 3c end-to-end Q/S check] Fractional bias at anvil peak (T~{T_l[peak_idx]:.0f}K): "
      f"control={frac_bias_ctrl:+.1f}%, +4K={frac_bias_p4k:+.1f}%")

# =====================================================
# GLOBAL STYLE
# =====================================================

c_ctrl = "tab:blue"
c_p4 = "tab:red"
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8       # one legend size for every panel of every figure
})

# Named text roles, so a size is never chosen ad hoc at the call site. The
# legends previously ran 9 / 7 / 9 pt within Fig 1 and 9 / 7 / 7.5 pt within
# Fig 3 -- three different sizes for the same role in one figure, because
# each was shrunk independently to escape a collision. The collisions are
# fixed by placement now, so the sizes no longer need to absorb them.
FS_LEGEND = 8.0    # every legend (set via rcParams above; do not override per call)
FS_ANNOT  = 7.5    # in-axes explanatory text
FS_VALUE  = 8.5    # numeric data labels

PANEL_W = 2.8
PANEL_H = 3.0

# AGU/GRL hard limits: 190 mm (7.48 in) max width, 240 mm (9.45 in) max
# height INCLUDING the caption.
#
# But the binding constraint is \linewidth, not the 190 mm ceiling. Measured
# off the compiled manuscript, agujournal2019 gives \linewidth = 421.5 pt =
# 5.85 in, and each figure is placed at width=0.95\linewidth (0.85 for
# Fig. 4). Anything WIDER than that gets scaled DOWN by LaTeX, which shrinks
# its type: at 7.32 in Fig. 2's 9 pt tick labels printed at 6.8 pt, and at
# 7.31 in the SI figure's printed at 6.4 pt, against 8.3 pt for the 6.02 in
# figures. Same source size, three different sizes on the page.
#
# So every figure is authored at exactly the width it will be printed at.
# Scale factor is then 1.0 everywhere and the pt sizes set below are the pt
# sizes that reach the reader. If you change a \includegraphics width= in
# the .tex, change the matching figsize here too.
# 5.50 in (396 pt). Measured by scaling: a figure authored 5.558 in wide
# came back with its 9 pt tick labels at 8.46 pt on the page, so the placed
# width was 0.94 x 5.558 = 5.224 in = 0.95 x linewidth. Do NOT measure this
# from body-text extents -- the manuscript sets line numbers in the left
# margin, which inflates the apparent text block by ~0.35 in.
LINEWIDTH_IN = 5.50              # \linewidth in agujournal2019
W_FULL  = 0.95 * LINEWIDTH_IN    # 5.225 in -- Figs 1, 2, 3, 6
W_INSET = 0.85 * LINEWIDTH_IN    # 4.675 in -- Fig 4

T_ANVIL = 228.0                # anvil-peak isotherm all headline numbers quote
T_TOP, T_BOT = 210.0, 270.0    # common vertical range for the profile figures

# One color per physical quantity, reused in EVERY figure so the reader can
# carry the mapping across panels (Fig 4 previously used matplotlib's default
# cycle, which contradicted the Fig 3 assignments).
C_SINV   = "#1B7837"   # dark green  -- stability, and S_inv
C_DP     = "#E08214"   # orange      -- radiative cooling, and Delta p
C_TAUSUB = "#762A83"   # dark purple -- the subsidence branch (omega_sub, tau_sub^-1)
C_TAUCLD = "#B15928"   # sienna      -- the in-cloud branch
C_CF     = "k"         # black       -- cloud fraction C
C_CFCALC = "0.35"      # grey        -- its Q_rad/S reconstruction


def mark_anvil(axs):
    """Faint marker at the anvil-peak isotherm on every profile panel. Every
    headline number in the paper is quoted at this level, so show it rather
    than leaving it to the caption."""
    for ax in np.ravel(axs):
        ax.axhline(T_ANVIL, color="0.45", lw=0.7, ls=(0, (4, 3)), zorder=0)


def style_temp_profile_axes(axs, T):
    for ax in np.ravel(axs):
        ax.grid(True, alpha=0.3)
        ax.set_ylim(np.max(T), np.min(T))
        ax.tick_params(direction="out")


def scale_ticks(ax, factor):
    ax.ticklabel_format(style='plain', axis='x')
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t/factor:g}" for t in ticks])


def frac_response(x_ctrl, x_p4k, mask_frac=0.05):
    """100*(ln(x_p4k)-ln(x_ctrl))/dTs, masked where either value is too
    close to zero (relative to the profile's typical magnitude) for the
    log-ratio to be meaningful, or where the two values have opposite
    signs (log-ratio undefined). Moved here (before Figure 1) so it's
    available for the new consolidated %/K panel (e) added to Fig 1,
    as well as every later figure that already used it."""
    scale = np.nanmedian(np.abs(x_ctrl))
    thresh = mask_frac * scale
    ok = (np.abs(x_ctrl) > thresh) & (np.abs(x_p4k) > thresh) & (np.sign(x_ctrl) == np.sign(x_p4k))
    out = np.full_like(x_ctrl, np.nan, dtype=float)
    out[ok] = 100 * (np.log(np.abs(x_p4k[ok])) - np.log(np.abs(x_ctrl[ok]))) / dTs
    return out


# =====================================================
# FIGURE 1
# =====================================================

omega_dn_clear = np.abs(omega_dn_l) / (1.0 - CF_l)
omega_dn_calc_pos = -(1000 * Q_l / S_l)  # Q/S diagnostic, matches original fig1b dashed line
Q_pos = -Q_l

omega_dn_clear_p4 = np.abs(omega_dn_l_plus4k) / (1.0 - CF_l_plus4k)
omega_dn_calc_pos_p4 = -(1000 * Q_l_plus4k / S_l_plus4k)
Q_pos_p4 = -Q_l_plus4k

# Estimated omega_sub using Q_rad + Q_micro_clr (clear-sky microphysical
# latent heating, now that shoc_cond_clr/shoc_evap_clr/qi2qv_sublim_clr/
# qv2qi_vapdep_clr are computed correctly via subtraction+(1-CF) rather
# than by masking the unconditional field -- see get_vars()). Tests
# whether adding this term closes the gap with the diagnosed omega_sub
# (Reviewer 2, Major Point 2). Local constants to avoid any dependence on
# the L_v/L_s/cp defined later (Fig 6b) -- same values, kept separate so
# this doesn't rely on execution order.
_L_v_fig1 = 2.501e6   # J/kg
_L_s_fig1 = 2.834e6   # J/kg
_cp_fig1 = 1004.0     # J/kg/K

Q_micro_clr_l_fig1 = ((_L_s_fig1/_cp_fig1)*(qv2qi_vapdep_clr_l - qi2qv_sublim_clr_l)
                      + (_L_v_fig1/_cp_fig1)*(shoc_cond_clr_l - shoc_evap_clr_l))
Q_micro_clr_l_fig1_plus4k = ((_L_s_fig1/_cp_fig1)*(qv2qi_vapdep_clr_l_plus4k - qi2qv_sublim_clr_l_plus4k)
                             + (_L_v_fig1/_cp_fig1)*(shoc_cond_clr_l_plus4k - shoc_evap_clr_l_plus4k))

# Q_micro_clr_l_fig1 is a true per-unit-clear-area conditional mean (the
# shoc_*_clr/qi2qv_sublim_clr/qv2qi_vapdep_clr inputs are now computed via
# subtraction + /(1-CF), per the get_vars() fix). Q_l is domain-contribution
# style (in_cloud_average normalizes by total domain area, not (1-CF)*area
# -- same convention S uses, so the Q/S ratio elsewhere is self-consistent
# by cancellation). Multiply by (1-CF_l) here to bring Q_micro down to Q_l's
# convention before adding them -- do NOT change Q_l/S_l themselves, since
# that ratio's cancellation is relied on throughout the rest of the script.
omega_dn_calc_with_micro = -(1000 * (Q_l + (1 - CF_l) * Q_micro_clr_l_fig1) / S_l)
omega_dn_calc_with_micro_p4 = -(1000 * (Q_l_plus4k + (1 - CF_l_plus4k) * Q_micro_clr_l_fig1_plus4k) / S_l_plus4k)

# Second, independent approach to estimating the clear-sky microphysical
# contribution: rather than reconstructing shoc+p3 process-by-process (as
# above, and limited to what SHOC's cond/evap/sublim/vapdep alone capture --
# no p3, no rain evaporation), infer (Q_shoc+Q_p3)_clr as the RESIDUAL
# needed to close the four-term clear-sky energy balance
# (homme + shoc + p3 + rrtmgp ~ 0 at steady state), using only the two
# terms we have natively in clear sky (Q_dyn_clr_l = homme, Q_l = rrtmgp):
#     (Q_shoc + Q_p3)_clr,residual = -(Q_dyn_clr_l + Q_l)
# Both Q_dyn_clr_l and Q_l are already domain-contribution style
# (in_cloud_average, same convention), so this residual is too -- no (1-CF)
# reconciliation needed here, unlike the shoc-only reconstruction above.
# Substituting back in (Q_rad + this residual) algebraically simplifies to
# -Q_dyn_clr_l alone, but kept unsimplified below so the physical steps stay
# visible. This assumes negligible storage/other imbalance beyond these
# four terms -- the same assumption the existing Fig 6/6b residual checks
# already probe, so any nonzero 2-term residual there is exactly this
# estimate's own uncertainty.
Q_shocp3_clr_residual = -(Q_dyn_clr_l + Q_l)
Q_shocp3_clr_residual_plus4k = -(Q_dyn_clr_l_plus4k + Q_l_plus4k)

omega_dn_calc_with_shocp3_residual = -(1000 * (Q_l + Q_shocp3_clr_residual) / S_l)
omega_dn_calc_with_shocp3_residual_p4 = -(1000 * (Q_l_plus4k + Q_shocp3_clr_residual_plus4k) / S_l_plus4k)

# Rain evaporation, from the separate mass_flux file (qr2qv_evap). Domain-
# mean only, so per your instruction: tropical-average, then /(1-CF) to get
# a true clear-sky conditional mean under the assumption that ALL of the
# domain-mean signal is attributable to clear-sky columns (the upper-bound
# end of the assumption-free bracket, not a measured split). Sign
# convention (does qr2qv_evap>0 mean evaporation, i.e. cooling?) not
# verified against the model's actual output -- flip the sign below if
# inverted. L_v since this is liquid (rain) evaporation.
Q_rainevap_clr_l = rain_evap_domain_mean_l / (1 - CF_l)
Q_rainevap_clr_l_plus4k = rain_evap_domain_mean_l_plus4k / (1 - CF_l_plus4k)

Q_rainevap_K_l = -(_L_v_fig1/_cp_fig1) * Q_rainevap_clr_l
Q_rainevap_K_l_plus4k = -(_L_v_fig1/_cp_fig1) * Q_rainevap_clr_l_plus4k

# Standalone test (Q_rad + Q_rainevap only, not stacked on Q_micro) -- same
# (1-CF) reconciliation as Q_micro, since Q_rainevap_clr_l is now a true
# conditional mean and Q_l is domain-contribution style.
omega_dn_calc_with_rainevap = -(1000 * (Q_l + (1 - CF_l) * Q_rainevap_K_l) / S_l)
omega_dn_calc_with_rainevap_p4 = -(1000 * (Q_l_plus4k + (1 - CF_l_plus4k) * Q_rainevap_K_l_plus4k) / S_l_plus4k)

# The omega'S'/S covariance-correction term, discovered from the residual
# between the full clear-sky dynamical tendency (Q_dyn_clr, homme) and the
# vertical-advection-only prediction from diagnosed omega_sub and S
# (fig_horizontal_advection_check.pdf). <omega'S'> = Q_dyn_clr - <omega><S>,
# a real term missing from the simple product-of-means picture; dividing by
# S converts it to omega-space (Pa/s), equivalent to adding <omega'S'>
# alongside Q_micro/Q_rainevap in Q-space before the final division by S.
# Both Q_dyn_clr_l and omega_dn_clear*S_l are already domain-contribution
# style (same convention as Q_l) -- no (1-CF) reconciliation needed here,
# same situation as the shoc+p3 residual term above.
Q_covariance_clr_l = Q_dyn_clr_l - (omega_dn_clear * S_l) / 1000
Q_covariance_clr_l_plus4k = Q_dyn_clr_l_plus4k - (omega_dn_clear_p4 * S_l_plus4k) / 1000

omega_dn_calc_with_covariance = -(1000 * (Q_l + Q_covariance_clr_l) / S_l)
omega_dn_calc_with_covariance_p4 = -(1000 * (Q_l_plus4k + Q_covariance_clr_l_plus4k) / S_l_plus4k)

# was (2*PANEL_W, 3*PANEL_H) = 5.6 x 9.0 in; 9.12 in of artwork plus a
# caption overruns AGU's 240 mm (9.45 in) height limit.
fig1 = plt.figure(figsize=(W_FULL, 8.1), constrained_layout=True)
_gs1 = fig1.add_gridspec(3, 4)
_ax_a = fig1.add_subplot(_gs1[0, 0:2])
_ax_b = fig1.add_subplot(_gs1[0, 2:4], sharey=_ax_a)
_ax_c = fig1.add_subplot(_gs1[1, 0:2], sharey=_ax_a)
_ax_d = fig1.add_subplot(_gs1[1, 2:4], sharey=_ax_a)
_ax_e = fig1.add_subplot(_gs1[2, 0:2], sharey=_ax_a)  # half-width, bottom row
_ax_leg = fig1.add_subplot(_gs1[2, 2:4])              # legend-only cell, to its right
_ax_leg.axis("off")
axs = [_ax_a, _ax_b, _ax_c, _ax_d, _ax_e]

axs[0].plot(CF_l, T_l, lw=2, color=c_ctrl, label="Control")
axs[0].plot(CF_l_plus4k, T_l, lw=2, color=c_p4, label="+4K")
axs[0].set_title("(a) Cloud fraction")   # was "(a) CF"; every other title is spelled out
axs[0].set_xlabel(r"Cloud fraction $C$")
axs[0].set_ylabel("Temperature [K]")
# was loc="upper right", where the swatches landed on the +4K curve's 220 K minimum
axs[0].legend(frameon=False, loc="lower right")
axs[0].set_xlim(0.0, 0.12)

axs[1].plot(omega_dn_clear, T_l, lw=2, color=c_ctrl, label="Diagnosed")
axs[1].plot(omega_dn_calc_pos, T_l, lw=2, ls="--", color=c_ctrl, label=r"$Q_\mathrm{rad}/S$ calc")
axs[1].plot(omega_dn_clear_p4, T_l, lw=2, color=c_p4)
axs[1].plot(omega_dn_calc_pos_p4, T_l, lw=2, ls="--", color=c_p4)
axs[1].set_title("(b) Subsidence")
axs[1].set_xlabel(r"$\omega_\mathrm{sub}$ [$10^{-2}$ Pa s$^{-1}$]")
axs[1].legend(frameon=False, loc="lower left")   # was upper right, over the curves
axs[1].set_xlim(0.0, 0.038)  # 0-3.8 in displayed 1e-2 Pa/s units
scale_ticks(axs[1], 1e-2)
axs[1].set_xlim(0.0, 0.038)  # re-assert: scale_ticks' set_xticks() can otherwise auto-expand the view

axs[2].plot(Q_pos, T_l, lw=2, color=c_ctrl)
axs[2].plot(Q_pos_p4, T_l, lw=2, color=c_p4)   # was ls="--": +4K is solid in
                                              # (a)/(b) and in the (a) legend
axs[2].set_title("(c) Radiative cooling")
axs[2].set_xlabel(r"$Q_\mathrm{rad}$ [$10^{-5}$ K s$^{-1}$]")  # match Eq. 2's symbol
axs[2].set_ylabel("Temperature [K]")
axs[2].set_xlim(0.0, 1.8e-5)   # was 2e-5; the data peak at ~1.7
scale_ticks(axs[2], 1e-5)
axs[2].set_xlim(0.0, 1.8e-5)  # re-assert: scale_ticks' set_xticks() can otherwise auto-expand the view

axs[3].plot(S_l, T_l, lw=2, color=c_ctrl)
axs[3].plot(S_l_plus4k, T_l, lw=2, color=c_p4)   # was ls="--", see (c)
axs[3].set_title("(d) Stability")
axs[3].set_xlabel(r"$S$ [K Pa$^{-1}$]")
axs[3].set_xlim(0.0, 2.1)   # was 2.0, which clipped the +4K curve exactly on
                            # the 210 K frame edge

# =====================================================
# PANEL (e): all four fractional (%/K) responses on ONE panel -- per
# Reviewer 2's Major Point 5 request for %/K analogues, consolidated
# rather than spread across separate replacement panels/figures, since
# dln(X)/dTs is unitless regardless of what X is and can share one axis.
# Restricted to the well-behaved anvil range (per the earlier Fig 8
# lesson: the log-derivative becomes ill-conditioned near the tropopause
# and near cloud base, where the underlying quantities pass through zero
# or a local extremum).
# =====================================================

dlnCF_e = frac_response(CF_l, CF_l_plus4k)
dlnOmegaDiag_e = frac_response(omega_dn_clear, omega_dn_clear_p4)
dlnOmegaCalc_e = frac_response(omega_dn_calc_pos, omega_dn_calc_pos_p4)  # matches panel (b): plain Q_rad/S, not Q_rad+Q_lat
dlnQ_e = frac_response(Q_l, Q_l_plus4k)
dlnS_e = frac_response(S_l, S_l_plus4k)

axs[4].plot(dlnCF_e, T_l, lw=2, color="k", label=r"$C$")
C_OMEGA_E, C_Q_E, C_S_E = C_TAUSUB, C_DP, C_SINV   # shared quantity palette
axs[4].plot(dlnOmegaDiag_e, T_l, lw=2, ls="--", color=C_OMEGA_E, label=r"$\omega_\mathrm{sub}$ (diag)")
axs[4].plot(dlnOmegaCalc_e, T_l, lw=2, ls=(0, (1, 1)), color=C_OMEGA_E, label=r"$\omega_\mathrm{sub}$ ($Q_\mathrm{rad}/S$ calc)")
axs[4].plot(dlnQ_e, T_l, lw=2, ls=":", color=C_Q_E, label=r"$Q_\mathrm{rad}$")
axs[4].plot(dlnS_e, T_l, lw=2, ls="-.", color=C_S_E, label=r"$S$")
axs[4].axvline(0, color="gray", lw=0.8)
axs[4].set_title("(e) Fractional response")
axs[4].set_xlabel(r"$\Delta\ln X/\Delta T_s$ [% K$^{-1}$]")
# was loc="lower right" INSIDE the axes, printed over the S and Q_rad curves.
# Moved below the panel: nothing to collide with, and the entries get bigger.
_ax_leg.legend(handles=axs[4].get_lines()[:5], loc="center left",
               frameon=False, handlelength=2.2, labelspacing=1.1,
               borderpad=0.0, handletextpad=0.7)
axs[4].set_xlim(-10, 10)
style_temp_profile_axes(axs, T_l)
mark_anvil(axs)
axs[4].set_ylim(270, 210)  # crop to the well-behaved anvil range; must come AFTER
                           # style_temp_profile_axes, which would otherwise reset
                           # this back to the full T_l range -- adjust once you see real output
fig1.savefig("fig1.pdf", dpi=300)   # no bbox_inches: figsize IS the printed size

# =====================================================
# FIGURE 2
# =====================================================

Q_T_pos = -Q_T
Q_T_pos_p4 = -Q_T_plus4k

# was (3*PANEL_W, PANEL_H) = 8.4 x 3.0 in -> 8.52 in on disk, over AGU's
# 190 mm (7.48 in) width limit; the journal would have scaled it down ~14%,
# dropping the 9 pt tick labels below 8 pt.
fig2, axs = plt.subplots(2, 2, figsize=(W_FULL, 5.8), sharey=True, constrained_layout=True)
axs = axs.ravel()   # 2x2 grid, matching Fig. 3's convention now that this figure
                     # also holds 4 panels -- keeps every figure at the same
                     # native width (W_FULL) so text scales identically once
                     # all are placed at the same \includegraphics width in
                     # the manuscript. The previous 1x4 layout widened the
                     # canvas instead, which left this figure's text visibly
                     # smaller than Figs. 1/3/6 at the same printed width,
                     # and further packed 4 panel titles too tight to read.

axs[0].plot(Q_T_pos, T_l, lw=2, color=c_ctrl, label="Control")
axs[0].plot(Q_T_pos_p4, T_l, lw=2, color=c_p4, label="+4K")
axs[0].set_title(r"(a) $Q_T$")
axs[0].set_xlabel(r"$Q_T$ [$10^{-3}$ W m$^{-3}$ K$^{-1}$]")
axs[0].set_ylabel("Temperature [K]")
axs[0].set_xlim(0.0, 1.75e-3)   # was 2e-3; ~20% of the axis held no data
axs[0].set_xticks([0.0, 0.5e-3, 1.0e-3, 1.5e-3])   # the auto locator drops to
                                # {0, 1} at this panel width
scale_ticks(axs[0], 1e-3)
axs[0].set_xlim(0.0, 1.75e-3)  # re-assert: scale_ticks' set_xticks() can otherwise auto-expand the view
axs[0].legend(frameon=False, loc="upper right")

axs[1].plot(inv_S, T_l, lw=2, color=c_ctrl)
axs[1].plot(inv_S_plus4k, T_l, lw=2, color=c_p4)
axs[1].set_title(r"(b) $S_\mathrm{inv}$")
axs[1].set_xlabel(r"$S_\mathrm{inv}$ [m K$^{-1}$]")
# Zero origin kept, to match panel (a) and so the eye can read the magnitude
# contraction between control and +4K against a true baseline -- S_inv is a
# ratio-scale quantity and this panel is about how much of it is lost, not
# just where the peak sits. Only the UPPER limit is refitted: the default ran
# to ~0.047 against a 0.0373 data maximum.
axs[1].set_xlim(0.0, 0.041)

# New panel (c): cloud fraction, same data/style as Fig. 1(a), for direct
# side-by-side comparison with S_inv in panel (b) -- S_inv's peaks and CF's
# peaks track each other, which is easier to see with both panels adjacent
# than by flipping back to Fig. 1.
axs[2].plot(CF_l, T_l, lw=2, color=c_ctrl, label="Control")
axs[2].plot(CF_l_plus4k, T_l, lw=2, color=c_p4, label="+4K")
axs[2].set_title("(c) Cloud fraction")
axs[2].set_xlabel(r"Cloud fraction $C$")
axs[2].set_ylabel("Temperature [K]")   # 2 x 2 layout: (c) is the bottom-left
                                       # panel, so it carries the second ylabel
axs[2].set_xlim(0.0, 0.12)

# Consolidated fractional-response panel, same rationale as Fig 1(e): dln(X)
# is unitless regardless of X, so dF/dT and S_inv's warming responses can
# share one axis rather than needing separate panels/figures.
dlnQT_e2 = frac_response(Q_T, Q_T_plus4k)
dlnSinv_e2 = frac_response(inv_S, inv_S_plus4k)
dlnCF_e2 = frac_response(CF_l, CF_l_plus4k)

axs[3].plot(dlnQT_e2, T_l, lw=2, ls="--", color=C_DP, label=r"$Q_T$")
axs[3].plot(dlnSinv_e2, T_l, lw=2, ls="-.", color=C_SINV, label=r"$S_\mathrm{inv}$")
axs[3].plot(dlnCF_e2, T_l, lw=2, color=C_CF, label=r"$C$")
axs[3].axvline(0, color="gray", lw=0.8)
axs[3].set_title("(d) Fractional response")
axs[3].set_xlabel(r"$\Delta\ln X/\Delta T_s$ [% K$^{-1}$]")
axs[3].legend(frameon=False, loc="upper right")
axs[3].set_xlim(-10, 10)

style_temp_profile_axes(axs, T_l)
mark_anvil(axs)
axs[3].set_ylim(270, 210)  # crop to the well-behaved anvil range; must come AFTER
                           # style_temp_profile_axes; adjust once you see real output
fig2.savefig("fig2.pdf", dpi=300)   # no bbox_inches: figsize IS the printed size

# =====================================================
# FIGURE 3
# =====================================================

# was 1 x 4 at (4*PANEL_W, PANEL_H) = 11.2 x 3.0 in -> 11.32 in on disk, 50%
# wider than AGU's 190 mm limit. Reflowed to 2 x 2, which fits the page at
# full size and matches Fig 1's grid.
fig3, axs = plt.subplots(2, 2, figsize=(W_FULL, 5.8), sharey=True, constrained_layout=True)
axs = axs.ravel()

axs[0].plot(tau_i_sub, T_l, lw=2, color=c_ctrl, label="Control")
axs[0].plot(tau_i_sub_plus4k, T_l, lw=2, color=c_p4, label="+4K")
axs[0].set_title(r"(a) $\tau_\mathrm{sub}^{-1}$")
axs[0].set_xlabel(r"$\tau_\mathrm{sub}^{-1}$ [$10^{-5}$ s$^{-1}$]")
axs[0].set_ylabel("Temperature [K]")
axs[0].set_xlim(0.0, 6e-5)
scale_ticks(axs[0], 1e-5)
axs[0].set_xlim(0.0, 6e-5)  # re-assert: scale_ticks' set_xticks() can otherwise auto-expand the view
axs[0].legend(frameon=False, loc="lower right")   # was upper right, on the control curve

axs[1].plot(tau_i_cloud, T_l, lw=2, color=c_ctrl)
axs[1].plot(tau_i_cloud_plus4k, T_l, lw=2, color=c_p4)
axs[1].set_title(r"(b) $\tau_\mathrm{cld}^{-1}$")
axs[1].set_xlabel(r"$\tau_\mathrm{cld}^{-1}$ [$10^{-4}$ s$^{-1}$]")
axs[1].set_xlim(0.0, 7e-4)
scale_ticks(axs[1], 1e-4)
axs[1].set_xlim(0.0, 7e-4)
axs[1].set_xlim(0.0, 6.6e-4)   # was 7e-4; the data peak at ~6.1  # re-assert: scale_ticks' set_xticks() can otherwise auto-expand the view

axs[2].plot(CF_l, T_l, lw=2, color=c_ctrl, label="Diagnosed")
axs[2].plot(CF_calc, T_l, lw=2, ls="--", color=c_ctrl, label=r"$Q_\mathrm{rad}/S$ calc")
axs[2].plot(CF_l_plus4k, T_l, lw=2, color=c_p4)
axs[2].plot(CF_calc_plus4k, T_l, lw=2, ls="--", color=c_p4)
axs[2].set_title("(c) Cloud fraction")
axs[2].set_xlabel(r"Cloud fraction $C$")
axs[2].set_ylabel("Temperature [K]")   # 2 x 2 layout: (c) is the bottom-left
                                       # panel, so it carries the second ylabel
axs[2].set_xlim(0.0, 0.12)
# The solid/dashed split in this panel was previously unexplained: the only
# legend in the figure was (a)'s, which covers color (experiment) alone.
axs[2].legend(frameon=False, loc="lower right")

# Consolidated fractional-response panel: diagnosed tau_i_sub, diagnosed
# tau_i_cloud, diagnosed CF, and CF_calc (the Q/S-predicted-tau_i_sub-based
# reconstruction, panel (c)'s dashed line) -- lets the reader directly see
# whether the end-to-end Q/S reconstruction (CF_calc) tracks the diagnosed
# CF's actual warming response, not just its absolute profile shape.
dlnTauSub_e3 = frac_response(tau_i_sub, tau_i_sub_plus4k)
dlnTauCloud_e3 = frac_response(tau_i_cloud, tau_i_cloud_plus4k)
dlnCF_diag_e3 = frac_response(CF_l, CF_l_plus4k)
dlnCF_calc_e3 = frac_response(CF_calc, CF_calc_plus4k)

axs[3].plot(dlnTauSub_e3, T_l, lw=2, ls="--", color=C_TAUSUB, label=r"$\tau_\mathrm{sub}^{-1}$")
axs[3].plot(dlnTauCloud_e3, T_l, lw=2, ls="-.", color=C_TAUCLD, label=r"$\tau_\mathrm{cld}^{-1}$")
axs[3].plot(dlnCF_diag_e3, T_l, lw=2, color=C_CF, label=r"$C$ (diag)")
axs[3].plot(dlnCF_calc_e3, T_l, lw=2, ls=(0, (1, 1)), color=C_CFCALC, label=r"$C$ (calc)")
axs[3].axvline(0, color="gray", lw=0.8)
axs[3].set_title("(d) Fractional response")
axs[3].set_xlabel(r"$\Delta\ln X/\Delta T_s$ [% K$^{-1}$]")
axs[3].legend(frameon=False, loc="lower right")  # was upper right, over CF (calc)
axs[3].set_xlim(-10, 10)

style_temp_profile_axes(axs, T_l)
mark_anvil(axs)
axs[3].set_ylim(270, 210)  # crop to the well-behaved anvil range; must come AFTER
                           # style_temp_profile_axes; adjust once you see real output
fig3.savefig("fig3.pdf", dpi=300)   # no bbox_inches: figsize IS the printed size

# =====================================================
# FIGURE 4: stepwise buffering bar chart
# =====================================================

dln_Sinv = 100 * (np.log(inv_S_plus4k[peak_idx]) - np.log(inv_S[peak_idx])) / dTs
dln_tsub = 100 * (np.log(tau_i_sub_calc_plus4k[peak_idx]) - np.log(tau_i_sub_calc[peak_idx])) / dTs
dln_tcld = 100 * (np.log(tau_i_cloud_plus4k[peak_idx]) - np.log(tau_i_cloud[peak_idx])) / dTs
dln_CF = 100 * (np.log(CF_calc_plus4k[peak_idx]) - np.log(CF_calc[peak_idx])) / dTs
dln_dp = dln_Sinv - dln_tsub

vals = [dln_Sinv, -dln_dp, dln_tsub, -dln_tcld, dln_CF]
labels = [r"$S_\mathrm{inv}$", r"$-\Delta p$", r"$\tau_\mathrm{sub}^{-1}$", r"$-\tau_\mathrm{cld}^{-1}$", r"$C$"]

# ---------------------------------------------------------------------------
# Redrawn as a WATERFALL rather than five independent bars from zero.
#
# The five numbers are not five parallel quantities: they are a cascade, and
# the identities that make it a cascade are the whole point of the figure --
#     dln(tau_sub) = dln(S_inv) + dln(1/Delta p)
#     dln(CF)      = dln(tau_sub) - dln(tau_cloud)
# The previous version drew every bar from y=0, so the reader had to do that
# arithmetic in their head to see the cancellation. Here the contribution
# bars (S_inv, -Delta p, -tau_cloud) float from the running total and the two
# derived quantities (tau_sub, CF) are drawn as filled totals from zero, so
# "-5.8 %/K of stability response is buffered down to -1.4 %/K of cloud
# response" is legible directly off the chart.
#
# Colors now come from the shared C_* map at the top of the style block, so
# tau_sub is the same blue and CF the same black as in Fig 3(d); the previous
# version used matplotlib's default cycle, which colored tau_sub green and
# CF purple.
# ---------------------------------------------------------------------------
_is_total = [False, False, True, False, True]
_colors = [C_SINV, C_DP, C_TAUSUB, C_TAUCLD, C_CF]

_bottoms, _running = [], 0.0
for _v, _tot in zip(vals, _is_total):
    if _tot:
        _bottoms.append(0.0)          # totals are drawn from zero
        _running = _v
    else:
        _bottoms.append(_running)
        _running = _running + _v

fig4, ax = plt.subplots(figsize=(W_INSET, 2.95), constrained_layout=True)
for xi, (v, b, col, tot) in enumerate(zip(vals, _bottoms, _colors, _is_total)):
    ax.bar(xi, v, bottom=b, width=0.66, color=col,
           alpha=1.0 if tot else 0.55,
           edgecolor="k", linewidth=1.1 if tot else 0.6,
           zorder=3)

# connectors: top of each step -> base of the next, so the chain is explicit
for xi in range(len(vals) - 1):
    _y = _bottoms[xi] + vals[xi] if not _is_total[xi] else vals[xi]
    ax.plot([xi + 0.33, xi + 1 - 0.33], [_y, _y], color="0.4", lw=0.9, ls=":", zorder=2)

ax.axhline(0, color="k", lw=1, zorder=4)
for xi, (v, b, tot) in enumerate(zip(vals, _bottoms, _is_total)):
    _top = b + v
    ax.text(xi, _top + (0.14 if v >= 0 else -0.14), f"{v:+.1f}".replace("-", "\u2212"),
            ha="center", va="bottom" if v >= 0 else "top",
            fontweight="bold" if tot else "normal", fontsize=FS_VALUE, zorder=5)

# Stage brackets, replacing the two unlabelled dashed vlines. Those were drawn
# with no color argument, so they picked up C0 -- the same blue as the first
# bar -- and read as data rather than as separators.
_tops = np.array(_bottoms) + np.array(vals)
_ymin = min(_tops.min(), min(_bottoms)) - 1.1
_ymax = max(_tops.max(), 0.0) + 1.5
for _x in (1.5, 3.5):
    ax.axvline(_x, color="0.75", lw=0.9, ls="-", zorder=1)
for _x, _txt in ((0.5, r"stability $\rightarrow$ subsidence"),
                 (2.5, r"subsidence $\rightarrow$ cloud"),
                 (4.0, "net")):
    ax.text(_x, _ymax - 0.15, _txt, ha="center", va="top", fontsize=FS_ANNOT, color="0.35")

ax.set_xticks(np.arange(len(vals)))
ax.set_xticklabels(labels)
ax.set_xlim(-0.65, len(vals) - 0.35)
ax.set_ylim(_ymin, _ymax)
ax.set_ylabel(r"$\Delta \ln X / \Delta T_s$ [% K$^{-1}$]")
# The isotherm was previously stated only in the caption, even though it is
# what makes the numbers quotable.
ax.set_title(f"Stepwise buffering at the anvil peak "
             f"($T = {T_l[peak_idx]:.1f}$ K)")
ax.grid(True, axis="y", alpha=0.3, zorder=0)
fig4.savefig("fig4.pdf", dpi=300)   # no bbox_inches: figsize IS the printed size

print(f"\n[Fig 4 numbers] dlnSinv={dln_Sinv:.2f}  dln(-dp)={-dln_dp:.2f}  "
      f"dln(tau_i_sub)={dln_tsub:.2f}  dln(-tau_i_cloud)={-dln_tcld:.2f}  dlnCF={dln_CF:.2f}  [%/K]")

# ---------------------------------------------------------------------------
# ROBUSTNESS CHECK: Fig 4's dln_tsub (and therefore dln_CF) is built from
# tau_i_sub_calc -- the Q/S-*predicted* subsidence timescale -- not from
# tau_i_sub, the fully *diagnosed* one (from the actual omega_dn_l field).
# Given the fractional-response profiles show diagnosed and Q/S-predicted
# omega_sub diverging by ~1-2%/K even in the well-behaved anvil range, this
# checks whether the headline -2.2%/K buffered-subsidence number is sensitive
# to that choice, i.e. whether it survives being computed the "diagnosed" way
# instead of the "Q/S-calculated" way Fig 4 currently uses.
# ---------------------------------------------------------------------------
dln_tsub_diagnosed = 100 * (np.log(tau_i_sub_plus4k[peak_idx]) - np.log(tau_i_sub[peak_idx])) / dTs
print(f"\n[Robustness check, anvil peak] dln(tau_i_sub), Q/S-calculated (Fig 4's current basis) = {dln_tsub:.2f} %/K")
print(f"[Robustness check, anvil peak] dln(tau_i_sub), fully diagnosed                           = {dln_tsub_diagnosed:.2f} %/K")
print(f"[Robustness check, anvil peak] difference = {dln_tsub_diagnosed - dln_tsub:.2f} %/K "
      f"-- if this is small relative to -2.2%/K, the headline number is robust to which omega_sub is used")


# =====================================================
# FIGURE 9 (NEW): two further robustness comparisons, as full profiles
#
# (a) tau_i_sub: fully diagnosed (from omega_dn_l) vs. Q/S-calculated
#     (tau_i_sub_calc, what Fig 4's headline numbers are actually built
#     from) -- the profile-wide version of the single-point check above.
#
# (b) dF/dT built two ways: Q_T (from Q_l, the in-cloud-averaged native
#     model tendency `rrtmgp_T_mid_tend`, used in Fig 1b/2a/tau_i_sub)
#     vs. a flux-divergence-based Q_T_clr (from Q_clr_l = d(F_rad_clrsky)/dp,
#     the same construction used for tau_i_sub_calc/CF_calc/Fig 4, and the
#     one more directly comparable to how dF/dT is typically computed in
#     the literature, e.g. Jeevanjee 2022 / Williams & Jeevanjee 2025).
# =====================================================

Q_T_clr = Q_clr_l * ro / gam
Q_T_clr_plus4k = Q_clr_l_plus4k * ro_plus4k / gam_plus4k

dln_tau_i_sub_diag_profile = frac_response(tau_i_sub, tau_i_sub_plus4k)
dln_tau_i_sub_calc_profile = frac_response(tau_i_sub_calc, tau_i_sub_calc_plus4k)
dln_QT_native_profile = frac_response(Q_T, Q_T_plus4k)          # same as dln_QT_profile in Fig 8, repeated here for side-by-side
dln_QT_clr_profile = frac_response(Q_T_clr, Q_T_clr_plus4k)

fig9, axs = plt.subplots(1, 2, figsize=(2*PANEL_W, PANEL_H), sharey=True, constrained_layout=True)

axs[0].plot(dln_tau_i_sub_diag_profile, T_l, lw=2, color='k', label="Diagnosed")
axs[0].plot(dln_tau_i_sub_calc_profile, T_l, lw=2, ls="--", color='k', label=r"$Q/S$ calculated")
axs[0].axvline(0, color='gray', lw=0.8)
axs[0].set_title(r"(a) $d\ln \tau_{i,\rm sub}/dT_s$")
axs[0].set_xlabel(r"[% K$^{-1}$]")
axs[0].set_ylabel("Temperature [K]")
axs[0].legend(frameon=False, loc="upper right", fontsize=8)
axs[0].set_xlim(-15, 5)  # tightened; adjust to taste once you see it

axs[1].plot(dln_QT_native_profile, T_l, lw=2, color='k', label=r"$Q$ (native tendency)")
axs[1].plot(dln_QT_clr_profile, T_l, lw=2, ls="--", color='k', label=r"$Q_\mathrm{clr}$ (flux div.)")
axs[1].axvline(0, color='gray', lw=0.8)
axs[1].set_title(r"(b) $d\ln(dF/dT)/dT_s$")
axs[1].set_xlabel(r"[% K$^{-1}$]")
axs[1].legend(frameon=False, loc="upper right", fontsize=8)
axs[1].set_xlim(-8, 5)  # tightened; adjust to taste once you see it

style_temp_profile_axes(axs, T_l)
for ax in np.ravel(axs):
    ax.set_ylim(270, 210)
fig9.savefig("fig9_alt_diagnostics_comparison.pdf", bbox_inches="tight", dpi=300)

print(f"\n[Fig 9, anvil peak T~{T_l[peak_idx]:.0f}K] "
      f"dln(tau_i_sub) diagnosed={dln_tau_i_sub_diag_profile[peak_idx]:.2f}  "
      f"Q/S-calc={dln_tau_i_sub_calc_profile[peak_idx]:.2f}  [%/K]")
print(f"[Fig 9, anvil peak T~{T_l[peak_idx]:.0f}K] "
      f"dln(dF/dT) native-Q={dln_QT_native_profile[peak_idx]:.2f}  "
      f"Q_clr(flux-div)={dln_QT_clr_profile[peak_idx]:.2f}  [%/K]")

# =====================================================
# FIGURE 8 (NEW): profile-wide fractional (%/K) response panels
#
# Reviewer 2 (Major Point 5) asked for %/K analogues of Figs 1a/b and
# 2a/b, rather than only the single anvil-peak numbers in Fig 4's bar
# chart. This gives the full T-coordinate profile of the fractional
# warming response for CF, omega_sub, dF/dT, and S_inv.
#
# Quantities that can approach zero or change sign (omega_sub near the
# tropopause, dF/dT at the T_RT crossing) are masked there using the
# same "small relative to typical magnitude" logic as the Fig 7 WTG
# check, to avoid log-derivative blowups at those points -- otherwise
# a single near-zero-crossing point can dominate the panel's y-scale.
# (frac_response() is defined above, before Fig 9's first use of it.)
# =====================================================

dlnCF_profile = frac_response(CF_l, CF_l_plus4k)
dln_omega_profile = frac_response(omega_dn_clear, omega_dn_clear_p4)
dln_omega_calc_profile = frac_response(omega_dn_calc_pos, omega_dn_calc_pos_p4)  # Q/S-predicted, for comparison
dln_QT_profile = frac_response(Q_T, Q_T_plus4k)          # Q_T (not Q_T_pos) -- frac_response handles sign via abs()
dln_Sinv_profile = frac_response(inv_S, inv_S_plus4k)

# =====================================================
# DOES SCREAM'S OWN GAMMA SCALE WITH TAU_I_SUB, AND IS S_INV'S RESPONSE
# DOMINATED BY THE PRESSURE (Delta p) TERM?
#
# The idealized-model check (entraining plume vs. non-dilute) found that
# in BOTH toy models, dln(Gamma_local)/dTs was small (~-0.15 to -0.34%/K)
# while dln(p)/dTs at fixed T did most of the work in setting dln(S_inv)/dTs
# (dlnSinv/dTs ~ dlnp/dTs - dlnGamma/dTs, with the Gamma term small). This
# checks whether the SAME pattern holds in SCREAM's own diagnosed fields --
# `gam` IS the paper's actual Gamma (it's what builds inv_S/S_inv already),
# so no new derivation is needed, just the fractional response of variables
# already computed above.
# =====================================================

dln_gamma_profile = frac_response(gam, gam_plus4k)
dln_p_profile = frac_response(p_l, p_l_plus4k)

# =====================================================
# DOES THE q*_v CORRECTION (W&J 2025 Eq. 7) EXPLAIN THE dln(p)-vs-dln(Sinv)
# GAP, OR IS THERE A REAL RESIDUAL -- INCLUDING IN THE MID-TROPOSPHERE?
#
# W&J's derivation gives dln(Sinv)/dTs ~ [1/(1+beta*qv_sat)] * dln(p)/dTs,
# an approximation that should hold best where qv_sat -> 0 (cold levels)
# and degrade monotonically toward warmer levels (larger qv_sat) -- NOT
# show deviation appearing already in the middle of the profile unrelated
# to that trend. This computes the actual q*_v-corrected prediction (not
# just the qv_sat->0 limit dln(p)/dTs used in Fig 11) from SCREAM's own
# T,p profile, and compares it to the diagnosed dln(Sinv)/dTs -- if a
# residual remains after this correction, that's a genuine departure from
# the theory, not just the expected asymptotic approximation error.
# =====================================================

L_v_wj = 2.5104e6
R_a_wj = 287.
R_v_wj = 461.
p_o_for_qv_sat_wj = 2.69e11  # matches the Clausius-Clapeyron prefactor used in the toy-model script


def qv_sat_screen(T, p):
    return 0.622 * (p_o_for_qv_sat_wj / p) * np.exp(-L_v_wj / (R_v_wj * T))


beta_wj = L_v_wj / (R_a_wj * T_l)  # beta = L/(Ra*T), W&J notation
qv_sat_ctrl = qv_sat_screen(T_l, p_l)  # use control p_l for the correction factor (evaluated at Ts=control per W&J's Fig 3b convention)

qv_correction = 1.0 / (1.0 + beta_wj * qv_sat_ctrl)
dln_Sinv_predicted_eq7 = qv_correction * dln_p_profile

print(f"\n[Eq.7 q*_v-corrected check] Comparing predicted (via q*_v correction) vs. diagnosed dln(Sinv)/dTs:")
for T_check in [220, 230, 240, 250, 260, 270]:
    idx_check = int(np.argmin(np.abs(T_l - T_check)))
    residual = dln_Sinv_profile[idx_check] - dln_Sinv_predicted_eq7[idx_check]
    print(f"  T~{T_l[idx_check]:.0f}K: diagnosed={dln_Sinv_profile[idx_check]:.2f}  "
          f"Eq.7-predicted={dln_Sinv_predicted_eq7[idx_check]:.2f}  "
          f"qv_sat={qv_sat_ctrl[idx_check]:.2e}  residual={residual:.2f} %/K")

# =====================================================
# THE CORE BUFFERING INEQUALITY: |dln(S_inv)/dTs| > |dln(Delta p)/dTs| ?
#
# The buffering argument does NOT require matching any idealized adiabat's
# magnitude for Gamma. It requires two structural facts:
#   (1) S_inv's response is an AMPLIFIED version of Gamma's response, via
#       the amplification factor S_inv/Gamma = Gamma_d/(Gamma_d-Gamma),
#       which is generically O(2-3) in the free troposphere since Gamma is
#       typically well below Gamma_d there -- a basic, near-universal fact
#       about moist tropical thermodynamics, not something that needs to
#       match any specific idealized calculation.
#   (2) Delta p's response is comparatively UN-amplified (dln(Delta p)/dTs
#       = dln(p)/dTs - dln(Gamma)/dTs =~ dln(p)/dTs once Gamma's own
#       response is small), and is dominated by the isotherm-rise/pressure
#       term, which both idealized models AND Williams & Jeevanjee's own
#       multi-model comparison find to be a robust ~3-5%/K feature.
#
# If (1) and (2) hold, |dln(S_inv)/dTs| > |dln(Delta p)/dTs| follows, and
# the DIFFERENCE (= dln(tau_sub)/dTs) preserves S_inv's sign but shrinks
# its magnitude -- the buffering -- without needing precise agreement with
# any idealized Gamma calculation. This is the one inequality the whole
# argument rests on; we test it directly here.
# =====================================================

amplification_factor = inv_S / gam  # = Gamma_d/(Gamma_d-Gamma); control-state, T_l-coordinate arrays

dln_deltap_profile = dln_p_profile - dln_gamma_profile  # dln(Delta p)/dTs = dln(rho)/dTs - dln(Gamma)/dTs = dln(p)/dTs - dln(Gamma)/dTs at fixed T

fig12, axs = plt.subplots(1, 2, figsize=(2*PANEL_W, PANEL_H), sharey=True, constrained_layout=True)

axs[0].plot(amplification_factor, T_l, lw=2, color='k')
axs[0].axhline(0, color='gray', lw=0.5)
axs[0].set_title(r"(a) Amplification factor $S_\mathrm{inv}/\Gamma$")
axs[0].set_xlabel(r"$S_\mathrm{inv}/\Gamma = \Gamma_d/(\Gamma_d-\Gamma)$ [-]")
axs[0].set_ylabel("Temperature [K]")

axs[1].plot(dln_Sinv_profile, T_l, lw=2, color='tab:green', label=r"$d\ln S_\mathrm{inv}/dT_s$")
axs[1].plot(dln_deltap_profile, T_l, lw=2, ls='--', color='tab:orange', label=r"$d\ln \Delta p/dT_s$")
axs[1].axvline(0, color='gray', lw=0.8)
axs[1].set_title(r"(b) Is $|d\ln S_\mathrm{inv}| > |d\ln \Delta p|$?")
axs[1].set_xlabel(r"[% K$^{-1}$]")
axs[1].legend(frameon=False, loc="upper right", fontsize=8)

style_temp_profile_axes(axs, T_l)
for ax in np.ravel(axs):
    ax.set_ylim(270, 210)
fig12.savefig("fig12_buffering_inequality_check.pdf", bbox_inches="tight", dpi=300)

print(f"\n[Core buffering inequality check]")
for T_check in [220, 230, 240, 250, 260, 270]:
    idx_check = int(np.argmin(np.abs(T_l - T_check)))
    print(f"  T~{T_l[idx_check]:.0f}K: amp.factor(Sinv/Gamma)={amplification_factor[idx_check]:.2f}  "
          f"|dlnSinv|={abs(dln_Sinv_profile[idx_check]):.2f}  "
          f"|dlnDeltap|={abs(dln_deltap_profile[idx_check]):.2f}  "
          f"inequality holds: {abs(dln_Sinv_profile[idx_check]) > abs(dln_deltap_profile[idx_check])}")

print(f"\n[SCREAM Gamma/p check, anvil peak T~{T_l[peak_idx]:.0f}K] "
      f"dln(Gamma)/dTs={dln_gamma_profile[peak_idx]:.2f}  "
      f"dln(tau_i_sub,diag)/dTs={dln_tau_i_sub_diag_profile[peak_idx]:.2f}  "
      f"dln(p)/dTs={dln_p_profile[peak_idx]:.2f}  "
      f"dln(Sinv)/dTs={dln_Sinv_profile[peak_idx]:.2f}  [%/K]")
print(f"[SCREAM Gamma/p check] dln(p)-dln(Gamma) = "
      f"{dln_p_profile[peak_idx] - dln_gamma_profile[peak_idx]:.2f} %/K "
      f"(cf. diagnosed dln(Sinv)={dln_Sinv_profile[peak_idx]:.2f} %/K -- "
      f"should match if the p-Gamma identity holds)")
print(f"[SCREAM Gamma/p check] for reference, idealized-model dln(Gamma)/dTs at same T: "
      f"entraining plume=-0.15%/K, non-dilute=-0.34%/K (much smaller than SCREAM's value above if so)")

fig11, axs = plt.subplots(1, 2, figsize=(2*PANEL_W, PANEL_H), sharey=True, constrained_layout=True)

axs[0].plot(dln_gamma_profile, T_l, lw=2, color='k', label=r"$d\ln\Gamma/dT_s$")
axs[0].plot(dln_tau_i_sub_diag_profile, T_l, lw=2, ls='--', color='tab:purple', label=r"$d\ln\tau_{i,\rm sub}/dT_s$ (diag.)")
axs[0].axvline(0, color='gray', lw=0.8)
axs[0].set_title(r"(a) Does $\Gamma$ scale with $\tau_{i,\rm sub}$?")
axs[0].set_xlabel(r"[% K$^{-1}$]")
axs[0].set_ylabel("Temperature [K]")
axs[0].legend(frameon=False, loc="upper right", fontsize=8)
axs[0].set_xlim(-7, 0)

axs[1].plot(dln_p_profile, T_l, lw=2, color='k', label=r"$d\ln p/dT_s$")
axs[1].plot(dln_Sinv_profile, T_l, lw=2, ls='--', color='tab:green', label=r"$d\ln S_\mathrm{inv}/dT_s$ (diagnosed)")
axs[1].plot(dln_Sinv_predicted_eq7, T_l, lw=2, ls=':', color='tab:red', label=r"$d\ln S_\mathrm{inv}/dT_s$ (Eq.7, $q^*_v$-corrected)")
axs[1].axvline(0, color='gray', lw=0.8)
axs[1].set_title(r"(b) Is $S_\mathrm{inv}$'s response $\approx$ the $p$ term?")
axs[1].set_xlabel(r"[% K$^{-1}$]")
axs[1].legend(frameon=False, loc="upper right", fontsize=7)
axs[1].set_xlim(-7, 0)

style_temp_profile_axes(axs, T_l)
for ax in np.ravel(axs):
    ax.set_ylim(270, 210)
fig11.savefig("fig11_gamma_p_scaling_check.pdf", bbox_inches="tight", dpi=300)

# =====================================================
# BETA DECOMPOSITION: omega_sub = beta * (Q/S)
#
# beta absorbs everything the pure WTG approximation misses -- conditional-
# sampling storage effects (cf. Beydoun et al. 2021 SI S4), ice sublimation/
# microphysics, and residual WTG error. Taking d/dTs of ln(omega_sub) then
# splits cleanly:
#
#   dln(omega_sub,diagnosed)/dTs = dln(beta)/dTs + dln(omega_sub,Q/S)/dTs
#
# i.e. dln(beta)/dTs is just the (already-computed) difference of the two
# profiles above. If this is small relative to the total response, that's
# a rigorous statement that Q/S captures the *warming sensitivity* even
# though beta != 1 means it has a real absolute-magnitude bias.
# =====================================================

beta = omega_dn_clear / omega_dn_calc_pos              # beta(T), control
beta_plus4k = omega_dn_clear_p4 / omega_dn_calc_pos_p4  # beta(T), +4K

dln_beta_profile = dln_omega_profile - dln_omega_calc_profile

fig10, axs = plt.subplots(1, 2, figsize=(2*PANEL_W, PANEL_H), sharey=True, constrained_layout=True)

axs[0].plot(beta, T_l, lw=2, color=c_ctrl, label="Control")
axs[0].plot(beta_plus4k, T_l, lw=2, color=c_p4, label="+4K")
axs[0].axvline(1, color='k', lw=1, ls=':')
axs[0].set_title(r"(a) $\beta = \omega_\mathrm{sub,diag}/\omega_\mathrm{sub,Q/S}$")
axs[0].set_xlabel(r"$\beta$ [-]")
axs[0].set_ylabel("Temperature [K]")
axs[0].legend(frameon=False, loc="upper right", fontsize=8)

axs[1].plot(dln_beta_profile, T_l, lw=2, color='k', label=r"$d\ln\beta/dT_s$")
axs[1].plot(dln_omega_profile, T_l, lw=2, ls='--', color='gray', label=r"$d\ln\omega_\mathrm{sub,diag}/dT_s$ (total)")
axs[1].axvline(0, color='gray', lw=0.8)
axs[1].set_title(r"(b) $d\ln\beta/dT_s$ vs. total response")
axs[1].set_xlabel(r"[% K$^{-1}$]")
axs[1].legend(frameon=False, loc="upper right", fontsize=8)

style_temp_profile_axes(axs, T_l)
for ax in np.ravel(axs):
    ax.set_ylim(270, 210)
fig10.savefig("fig10_beta_decomposition.pdf", bbox_inches="tight", dpi=300)

print(f"\n[Beta decomposition, anvil peak T~{T_l[peak_idx]:.0f}K] "
      f"beta_control={beta[peak_idx]:.3f}  beta_+4K={beta_plus4k[peak_idx]:.3f}")
print(f"[Beta decomposition, anvil peak] dln(beta)/dTs={dln_beta_profile[peak_idx]:.2f} %/K  "
      f"= dln(omega_diag)/dTs({dln_omega_profile[peak_idx]:.2f}) - "
      f"dln(omega_Q/S)/dTs({dln_omega_calc_profile[peak_idx]:.2f})  "
      f"-- beta's share of the total response: "
      f"{100*dln_beta_profile[peak_idx]/dln_omega_profile[peak_idx]:.0f}%")

fig8, axs = plt.subplots(2, 2, figsize=(2*PANEL_W, 2*PANEL_H), sharey=True, constrained_layout=True)

axs[0, 0].plot(dlnCF_profile, T_l, lw=2, color='k')
axs[0, 0].axvline(0, color='gray', lw=0.8)
axs[0, 0].set_title(r"(a) $d\ln\mathrm{CF}/dT_s$")
axs[0, 0].set_xlabel(r"[% K$^{-1}$]")
axs[0, 0].set_ylabel("Temperature [K]")
axs[0, 0].set_xlim(-15, 5)  # tightened for the 210-270K view; adjust to taste once you see it

axs[0, 1].plot(dln_omega_profile, T_l, lw=2, color='k', label="Diagnosed")
axs[0, 1].plot(dln_omega_calc_profile, T_l, lw=2, ls="--", color='k', label=r"$Q/S$ predicted")
axs[0, 1].axvline(0, color='gray', lw=0.8)
axs[0, 1].set_title(r"(b) $d\ln \omega_\mathrm{sub}/dT_s$")
axs[0, 1].set_xlabel(r"[% K$^{-1}$]")
axs[0, 1].legend(frameon=False, loc="upper right", fontsize=8)
axs[0, 1].set_xlim(-25, 10)

axs[1, 0].plot(dln_QT_profile, T_l, lw=2, color='k')
axs[1, 0].axvline(0, color='gray', lw=0.8)
axs[1, 0].set_title(r"(c) $d\ln(dF/dT)/dT_s$")
axs[1, 0].set_xlabel(r"[% K$^{-1}$]")
axs[1, 0].set_ylabel("Temperature [K]")
axs[1, 0].set_xlim(-8, 15)

axs[1, 1].plot(dln_Sinv_profile, T_l, lw=2, color='k')
axs[1, 1].axvline(0, color='gray', lw=0.8)
axs[1, 1].set_title(r"(d) $d\ln S_\mathrm{inv}/dT_s$")
axs[1, 1].set_xlabel(r"[% K$^{-1}$]")
axs[1, 1].set_xlim(-45, -20)

style_temp_profile_axes(axs, T_l)
for ax in np.ravel(axs):
    ax.set_ylim(270, 210)  # restrict to the well-behaved anvil range; avoids the
                           # near-tropopause (Q_T sign change) and near-cloud-base
                           # (omega_sub/CF local extrema) regions where the
                           # fractional diagnostic becomes ill-conditioned
fig8.savefig("fig8_fractional_response.pdf", bbox_inches="tight", dpi=300)

print(f"\n[Fig 8, anvil peak T~{T_l[peak_idx]:.0f}K] dlnCF/dTs={dlnCF_profile[peak_idx]:.2f}  "
      f"dln(omega_sub, diagnosed)/dTs={dln_omega_profile[peak_idx]:.2f}  "
      f"dln(omega_sub, Q/S predicted)/dTs={dln_omega_calc_profile[peak_idx]:.2f}  "
      f"dln(dF/dT)/dTs={dln_QT_profile[peak_idx]:.2f}  "
      f"dln(Sinv)/dTs={dln_Sinv_profile[peak_idx]:.2f}  [%/K]")

# =====================================================
# FIGURE 5 (NEW): CSC*dl special-case comparison
#
# Compares the new framework's diagnosed source term
# (delta * omega_sub / dp) against the Beydoun et al. (2021)
# closure (CSC * Delta_l = -div * dl), in the same styled format
# as Figs 1-4. Report the ratio of the two at the anvil peak
# (where they should roughly agree) vs. a midlevel point (where
# net convergence -> 0 and the CSC closure should break down)
# in the printed summary below the figure.
# =====================================================

source_new = delta_l * omega_up_l / dp
source_new_p4 = delta_l_plus4k * omega_up_l_plus4k / dp_plus4k
source_csc = -div_l * dl_l
source_csc_p4 = -div_l_plus4k * dl_l_plus4k

# ---------------------------------------------------------------------------
# Three changes here, all about making the anvil comparison actually visible:
#
# 1. The x-axis previously ran to 1.4e-4 because of a warm-cloud-base spike
#    near 290 K. That spike is nowhere near the anvil and is not what the
#    panel is arguing about, but it compressed the entire region of interest
#    (0 to ~0.4e-4) into the leftmost quarter of the frame. Now clipped.
# 2. style_temp_profile_axes() set the y-range to the full T_l grid
#    (197-297 K), so this figure alone used a different vertical range from
#    Figs 1-3 (210-270 K). Now on the shared range.
# 3. The 1e-4 exponent was left in matplotlib's corner offset box, where it
#    collided with the x-label and did not match the "[10^-5 K s^-1]" style
#    used in every other axis label. Folded into the label instead.
# ---------------------------------------------------------------------------
_SRC_SCALE = 1e-5

fig6, ax = plt.subplots(figsize=(W_FULL, 6.2), constrained_layout=True)
ax.plot(source_new / _SRC_SCALE, T_l, lw=2, color=c_ctrl, label="Control (this framework)")
ax.plot(source_new_p4 / _SRC_SCALE, T_l, lw=2, color=c_p4, label="+4K (this framework)")
ax.plot(source_csc / _SRC_SCALE, T_l, lw=2, ls="--", color=c_ctrl,
        label=r"Control (CSC $\times\ \Delta l$)")
ax.plot(source_csc_p4 / _SRC_SCALE, T_l, lw=2, ls="--", color=c_p4,
        label=r"+4K (CSC $\times\ \Delta l$)")
ax.set_title("Cloud source term: this framework vs. the\nBeydoun et al. (2021) CSC closure")
ax.set_xlabel(r"Source term [$10^{-5}$ s$^{-1}$]")
ax.set_ylabel("Temperature [K]")
ax.set_xlim(-0.3, 4.6)
fig6.legend(*ax.get_legend_handles_labels(), loc="outside lower center",
            ncol=2, frameon=False, columnspacing=2.4,
            handlelength=2.4)
style_temp_profile_axes(np.array([ax]), T_l)
ax.set_ylim(T_BOT, T_TOP)   # match Figs 1-3; must come AFTER
                            # style_temp_profile_axes, which resets it
mark_anvil(np.array([ax]))

# Annotate the closure ratio the text quotes, at the level it is quoted for,
# instead of leaving the reader to eyeball two curves against each other.
_ratio_peak = source_csc[peak_idx] / source_new[peak_idx]
ax.annotate(f"CSC closure recovers\n{_ratio_peak*100:.0f}% of the diagnosed\n"
            f"source at the anvil peak",
            xy=(source_csc[peak_idx] / _SRC_SCALE, T_l[peak_idx]),
            xytext=(0.12, 249), fontsize=FS_ANNOT, color="0.25", ha="left",
            arrowprops=dict(arrowstyle="->", color="0.45", lw=0.9,
                            connectionstyle="arc3,rad=0.2"))

# Named fig6.pdf directly: this is Figure 6 in the package and the README.
# It was previously written as fig5_csc_comparison.pdf and renamed by hand,
# which is how the code and the delivered PDF drifted apart. NOTE: whether
# this figure belongs in the main text (README) or the SI
# (response_to_reviewer1.tex, as figS1_csc_comparison.pdf) is still open.
fig6.savefig("fig6.pdf", dpi=300)   # no bbox_inches: figsize IS the printed size

midlevel_idx = 62  # ~T=260K; CHECK this actually sits near div_l~0 in your profile before quoting it
for label, idx in [("anvil peak (idx=%d)" % peak_idx, peak_idx), ("midlevel (idx=%d)" % midlevel_idx, midlevel_idx)]:
    ratio_ctrl = source_csc[idx] / source_new[idx]
    print(f"[CSC special-case check, {label}] CSC*dl / framework source (control) = {ratio_ctrl:.2f}  "
          f"(div_l={div_l[idx]:.3e})")

# =====================================================
# FIGURE 6 (NEW): Q_clr vs Q_dyn_clr steady-state balance check
#
# Tests the WTG assumption underlying Eq. (1)/(2): does clear-sky
# radiative cooling (Q_l) actually balance clear-sky dynamical/
# adiabatic warming (Q_dyn_clr_l) at steady state? Panel (a) overlays
# the two; panel (b) shows the residual (should be small if the
# balance holds).
# =====================================================

fig6, axs = plt.subplots(1, 2, figsize=(2*PANEL_W, PANEL_H), sharey=True, constrained_layout=True)

axs[0].plot(-Q_l, T_l, lw=2, color=c_ctrl, label=r"$-Q_\mathrm{clr}$ (Control)")
axs[0].plot(Q_dyn_clr_l, T_l, lw=2, ls="--", color=c_ctrl, label=r"$Q_\mathrm{dyn,clr}$ (Control)")
axs[0].plot(-Q_l_plus4k, T_l, lw=2, color=c_p4, label=r"+4K")
axs[0].plot(Q_dyn_clr_l_plus4k, T_l, lw=2, ls="--", color=c_p4)
axs[0].set_title(r"(a) $-Q_\mathrm{clr}$ vs $Q_\mathrm{dyn,clr}$")
axs[0].set_xlabel(r"[K s$^{-1}$]")
axs[0].set_ylabel("Temperature [K]")
axs[0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
axs[0].legend(frameon=False, loc="upper right", fontsize=7)

residual = Q_l + Q_dyn_clr_l           # should be ~0 if radiative cooling is balanced by dynamical warming
residual_p4 = Q_l_plus4k + Q_dyn_clr_l_plus4k
axs[1].plot(residual, T_l, lw=2, color=c_ctrl, label="Control")
axs[1].plot(residual_p4, T_l, lw=2, color=c_p4, label="+4K")
axs[1].axvline(0, color='k', lw=1)
axs[1].set_title(r"(b) Residual ($Q_\mathrm{clr}+Q_\mathrm{dyn,clr}$)")
axs[1].set_xlabel(r"[K s$^{-1}$]")
axs[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

style_temp_profile_axes(axs, T_l)
fig6.savefig("fig6_Q_balance_check.pdf", bbox_inches="tight", dpi=300)

# =====================================================
# FIGURE 6b (NEW): three-term balance including clear-sky microphysics
#
# Tests whether the Fig 6 residual is explained by latent heating from
# subvisible cirrus / below-cloud phase changes in nominally "clear"
# columns (below the qi+qc>1e-5 mask) -- i.e. Q_clr + Q_dyn_clr + Q_micro_clr
# should be closer to zero than the two-term version if this is the
# missing piece. Sign convention: assumes shoc_cond/qv2qi_vapdep are
# positive for the *forward* (mass-gaining, latent-heat-releasing) process
# and shoc_evap/qi2qv_sublim are positive for the *reverse* (mass-losing,
# latent-heat-absorbing) process -- verify this against your model's
# actual convention before trusting the sign of Q_micro_clr below; if
# inverted, the fix is just flipping the sign of the affected term(s).
# NOTE: this only tests the ice/liquid clear-sky phase-change terms --
# it does NOT include rain evaporation (needs mass_flux/qr2qv_evap, not
# yet available), so it's expected to help mainly at the cold end, not
# the warm end (~270-290K) where rain evaporation is the leading candidate.
# =====================================================

L_v = 2.501e6   # J/kg, latent heat of vaporization
L_s = 2.834e6   # J/kg, latent heat of sublimation (vapor <-> ice)
cp = 1004.0     # J/kg/K

# Split ice vs liquid rather than lumping into one term -- the ice terms
# (qi2qv_sublim, qv2qi_vapdep) are the ones actually relevant to the cold-
# anvil/thin-cirrus hypothesis; the liquid terms (shoc_cond, shoc_evap) show
# a large, warm-level (250-285K) signal that *worsens* an already-small
# residual there, which points to a units/double-counting/cloud-edge issue
# with those two variables specifically rather than genuine missing physics.
# Keep them separate until that's understood, rather than let the liquid
# term's problem contaminate the (apparently sensible) ice-term result.
Q_micro_ice_clr_l = (L_s/cp)*(qv2qi_vapdep_clr_l - qi2qv_sublim_clr_l)
Q_micro_ice_clr_l_plus4k = (L_s/cp)*(qv2qi_vapdep_clr_l_plus4k - qi2qv_sublim_clr_l_plus4k)

Q_micro_liq_clr_l = (L_v/cp)*(shoc_cond_clr_l - shoc_evap_clr_l)
Q_micro_liq_clr_l_plus4k = (L_v/cp)*(shoc_cond_clr_l_plus4k - shoc_evap_clr_l_plus4k)

Q_micro_clr_l = Q_micro_ice_clr_l + Q_micro_liq_clr_l              # kept for reference / full-sum comparison
Q_micro_clr_l_plus4k = Q_micro_ice_clr_l_plus4k + Q_micro_liq_clr_l_plus4k

residual = Q_l + Q_dyn_clr_l           # should be ~0 if radiative cooling is balanced by dynamical warming
residual_p4 = Q_l_plus4k + Q_dyn_clr_l_plus4k

# Ice-only 3-term residual -- the actual test of the cold-anvil hypothesis.
# Q_micro_ice_clr_l is a true per-unit-clear-area conditional mean (post
# get_vars() fix); Q_l/Q_dyn_clr_l are domain-contribution style. Multiply
# by (1-CF_l) here, at the point of combination, to match conventions --
# does not change Q_micro_ice_clr_l itself (still plotted standalone below
# as a true conditional mean, which is what you want to see there).
residual_3term_ice = Q_l + Q_dyn_clr_l + (1 - CF_l) * Q_micro_ice_clr_l
residual_3term_ice_p4 = Q_l_plus4k + Q_dyn_clr_l_plus4k + (1 - CF_l_plus4k) * Q_micro_ice_clr_l_plus4k

for T_check in [210, 220, 230, 250, 270, 285]:
    idx_check = int(np.argmin(np.abs(T_l - T_check)))
    print(f"[3-term check, ICE ONLY] T~{T_l[idx_check]:.0f}K: "
          f"2-term residual={residual[idx_check]:.2e}  "
          f"3-term (ice) residual={residual_3term_ice[idx_check]:.2e}  "
          f"Q_micro_ice={Q_micro_ice_clr_l[idx_check]:.2e}  "
          f"Q_micro_liq={Q_micro_liq_clr_l[idx_check]:.2e} (excluded, see note above)  [K/s]")


fig6b, axs = plt.subplots(1, 2, figsize=(2*PANEL_W, PANEL_H), sharey=True, constrained_layout=True)

axs[0].plot(residual, T_l, lw=2, ls="--", color=c_ctrl, label="Control, 2-term (Fig 6)")
axs[0].plot(residual_3term_ice, T_l, lw=2, color=c_ctrl, label="Control, 3-term (+ice micro. only)")
axs[0].plot(residual_p4, T_l, lw=2, ls="--", color=c_p4, label="+4K, 2-term")
axs[0].plot(residual_3term_ice_p4, T_l, lw=2, color=c_p4, label="+4K, 3-term")
axs[0].axvline(0, color='k', lw=1)
axs[0].set_title("(a) Residual: 2-term vs 3-term (ice micro. only)")
axs[0].set_xlabel(r"[K s$^{-1}$]")
axs[0].set_ylabel("Temperature [K]")
axs[0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
axs[0].legend(frameon=False, loc="upper right", fontsize=6)

axs[1].plot(Q_micro_ice_clr_l, T_l, lw=2, color=c_ctrl, label="Control, ice (sublim/vapdep)")
axs[1].plot(Q_micro_ice_clr_l_plus4k, T_l, lw=2, color=c_p4, label="+4K, ice")
axs[1].plot(Q_micro_liq_clr_l, T_l, lw=2, ls=":", color=c_ctrl, label="Control, liquid (SHOC cond/evap) -- unresolved")
axs[1].plot(Q_micro_liq_clr_l_plus4k, T_l, lw=2, ls=":", color=c_p4, label="+4K, liquid -- unresolved")
axs[1].axvline(0, color='k', lw=1)
axs[1].set_title(r"(b) $Q_\mathrm{micro,clr}$: ice vs liquid (dotted)")
axs[1].set_xlabel(r"[K s$^{-1}$]")
axs[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
axs[1].legend(frameon=False, loc="upper right", fontsize=6)

style_temp_profile_axes(axs, T_l)
fig6b.savefig("fig6b_three_term_balance.pdf", bbox_inches="tight", dpi=300)

# =====================================================
# FIGURE 6c (NEW): -Q_clr vs Q_dyn_clr as a scatter/regression, per state
#
# Beydoun et al. (2021) SI Text S4 makes exactly this point for the
# structurally identical problem (CSC vs CSC_Qr): the storage term in a
# conditionally-sampled clear-sky budget need not vanish, because the
# clear-sky fraction exchanges mass with the cloudy fraction at the
# instant a grid cell transitions between the two. Rather than expect a
# pointwise balance (-Q_clr = Q_dyn,clr exactly), that paper's SI Fig S2
# instead tests *proportionality*: <CSC> vs <CSC_Qr> across the SST
# hierarchy, finding r^2~0.95 with a non-unity (~1.35) slope, and
# concludes <CSC> ~ <CSC_Qr> rather than equality.
#
# Here we do the same test using vertical level (T_l) as the scatter
# dimension instead of SST, done separately for control and +4K to check
# whether the same proportional (not exact) relationship holds in both
# states -- restricted to the well-behaved 210-270K range to avoid the
# near-tropopause/near-surface points that are noisy for reasons already
# established (WTG/omega_sub ill-conditioning, rain-evap contamination).
# =====================================================

from scipy import stats

_scatter_mask = (T_l >= 210) & (T_l <= 270)

x_ctrl, y_ctrl = Q_dyn_clr_l[_scatter_mask], -Q_l[_scatter_mask]
x_p4k, y_p4k = Q_dyn_clr_l_plus4k[_scatter_mask], -Q_l_plus4k[_scatter_mask]

fit_ctrl = stats.linregress(x_ctrl, y_ctrl)
fit_p4k = stats.linregress(x_p4k, y_p4k)

fig6c, ax = plt.subplots(figsize=(2*PANEL_W, 2*PANEL_H), constrained_layout=True)

ax.scatter(x_ctrl, y_ctrl, color=c_ctrl, label=f"Control (r$^2$={fit_ctrl.rvalue**2:.3f}, slope={fit_ctrl.slope:.2f})")
ax.scatter(x_p4k, y_p4k, color=c_p4, label=f"+4K (r$^2$={fit_p4k.rvalue**2:.3f}, slope={fit_p4k.slope:.2f})")

_xfit = np.linspace(min(x_ctrl.min(), x_p4k.min()), max(x_ctrl.max(), x_p4k.max()), 50)
ax.plot(_xfit, fit_ctrl.intercept + fit_ctrl.slope*_xfit, color=c_ctrl, lw=2)
ax.plot(_xfit, fit_p4k.intercept + fit_p4k.slope*_xfit, color=c_p4, lw=2)
ax.plot(_xfit, _xfit, color='k', ls='--', lw=1, label="1:1 line")

ax.set_xlabel(r"$Q_\mathrm{dyn,clr}$ [K s$^{-1}$]")
ax.set_ylabel(r"$-Q_\mathrm{clr}$ [K s$^{-1}$]")
ax.set_title(r"$-Q_\mathrm{clr}$ vs $Q_\mathrm{dyn,clr}$ across levels (210-270K), cf. Beydoun et al. 2021 SI Fig. S2")
ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
ax.legend(frameon=False, loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)

fig6c.savefig("fig6c_Q_balance_scatter.pdf", bbox_inches="tight", dpi=300)

print(f"\n[Fig 6c, cf. 2021 SI S4] Control:  r^2={fit_ctrl.rvalue**2:.3f}  slope={fit_ctrl.slope:.3f}  "
      f"(2021 paper's CSC/CSC_Qr: r^2~0.95, slope~1.35)")
print(f"[Fig 6c, cf. 2021 SI S4] +4K:      r^2={fit_p4k.rvalue**2:.3f}  slope={fit_p4k.slope:.3f}")

# =====================================================
# FIGURE 7 (NEW): direct WTG check -- omega_sub * S vs Q
#
# Fig 6 tests Q_clr against Q_dyn_clr (homme_T_mid_tend), but that
# dycore tendency likely lumps horizontal advection, numerical
# diffusion, and sponge-layer damping together with the vertical
# advection term the WTG balance actually concerns -- so a residual
# there could reflect dycore bookkeeping, not a failure of the WTG
# assumption itself.
#
# This figure instead tests Eq. (1) directly and only: omega_sub = Q/S,
# i.e. omega_sub*S = Q, using the diagnosed (not dycore-tendency-based)
# quantities already computed for Fig 1b. This isolates the WTG
# assumption itself from dycore-tendency decomposition issues.
# =====================================================

# omega_dn_clear/omega_dn_calc_pos already carry the same sign convention
# used in Fig 1b (both positive-valued descent); Q_l/S_l are as computed
# above. omega_dn_calc_pos = -(1000*Q_l/S_l) IS the Q/S prediction, so:
wtg_residual = omega_dn_clear - omega_dn_calc_pos
wtg_residual_p4 = omega_dn_clear_p4 - omega_dn_calc_pos_p4

# The fractional residual (100*residual/omega_sub) blows up wherever omega_sub
# itself passes near zero (e.g. near the tropopause/model top) -- that's a
# denominator artifact, not a real failure of the balance, since the absolute
# residual there (panel a) stays small. Mask those points out rather than let
# them dominate the y-axis scale of panel (b). Threshold is relative to the
# typical |omega_sub| magnitude in the anvil region so it adapts to the
# actual data rather than a hardcoded absolute value.
_omega_scale = np.nanmedian(np.abs(omega_dn_clear))
_denom_thresh = 0.1 * _omega_scale  # mask where |omega_sub| < 10% of its typical anvil-region magnitude

wtg_frac_residual = np.where(np.abs(omega_dn_clear) > _denom_thresh,
                              100 * wtg_residual / omega_dn_clear, np.nan)
wtg_frac_residual_p4 = np.where(np.abs(omega_dn_clear_p4) > _denom_thresh,
                                 100 * wtg_residual_p4 / omega_dn_clear_p4, np.nan)

# Pointwise normalization (above) inflates the fractional residual wherever
# omega_sub itself is naturally small (e.g. approaching the T_RT crossing
# near the tropopause), even without hitting the near-zero mask -- this can
# look like "growing discrepancy toward cold temperatures" even if the
# absolute error (panel a) is flat. To separate a genuine growing error from
# this shrinking-denominator artifact, also compute the residual relative to
# a FIXED reference magnitude (peak |omega_sub| in the anvil region) rather
# than the local, pointwise value.
_omega_ref = np.abs(omega_dn_clear[peak_idx])
_omega_ref_p4 = np.abs(omega_dn_clear_p4[peak_idx])
wtg_frac_residual_fixedref = 100 * wtg_residual / _omega_ref
wtg_frac_residual_fixedref_p4 = 100 * wtg_residual_p4 / _omega_ref_p4

fig7, axs = plt.subplots(1, 3, figsize=(3*PANEL_W, PANEL_H), sharey=True, constrained_layout=True)

axs[0].plot(wtg_residual, T_l, lw=2, color=c_ctrl, label="Control")
axs[0].plot(wtg_residual_p4, T_l, lw=2, color=c_p4, label="+4K")
axs[0].axvline(0, color='k', lw=1)
axs[0].set_title(r"(a) $\omega_\mathrm{sub}-Q/S$ (diagnosed$-$WTG)")
axs[0].set_xlabel(r"[$10^{-2}$ Pa s$^{-1}$]")
axs[0].set_ylabel("Temperature [K]")
scale_ticks(axs[0], 1e-2)
axs[0].legend(frameon=False, loc="upper right", fontsize=8)

axs[1].plot(wtg_frac_residual, T_l, lw=2, color=c_ctrl)
axs[1].plot(wtg_frac_residual_p4, T_l, lw=2, color=c_p4)
axs[1].axvline(0, color='k', lw=1)
axs[1].set_title("(b) Fractional (pointwise-normalized)")
axs[1].set_xlabel("[%]")
axs[1].set_xlim(-100, 100)

axs[2].plot(wtg_frac_residual_fixedref, T_l, lw=2, color=c_ctrl)
axs[2].plot(wtg_frac_residual_fixedref_p4, T_l, lw=2, color=c_p4)
axs[2].axvline(0, color='k', lw=1)
axs[2].set_title("(c) Fractional (fixed anvil-peak ref.)")
axs[2].set_xlabel("[%]")

style_temp_profile_axes(axs, T_l)
fig7.savefig("fig7_WTG_direct_check.pdf", bbox_inches="tight", dpi=300)

# Print the fractional residual at a few reference temperatures, including
# the 220K point flagged as problematic in the Q_dyn_clr comparison. Report
# both normalizations -- if (b) grows toward cold T while (c) stays flat,
# that confirms it's the shrinking-omega_sub denominator artifact, not a
# genuinely growing WTG error.
for T_check in [220, 240, 260, 280]:
    idx_check = int(np.argmin(np.abs(T_l - T_check)))
    print(f"[WTG direct check] T~{T_l[idx_check]:.0f}K: "
          f"pointwise-frac={wtg_frac_residual[idx_check]:.1f}% (control), "
          f"fixedref-frac={wtg_frac_residual_fixedref[idx_check]:.1f}% (control)")

# =====================================================
# FIGURE 7b (NEW): diagnosed omega_sub vs Q/S-predicted omega_sub, as a
# scatter/regression -- the correct analog of Beydoun et al. (2021) SI
# Fig. S2 (<CSC> vs <CSC_Qr>) for THIS paper's actual Eq. (1) relationship.
#
# The 2021 paper's CSC/CSC_Qr comparison and this paper's omega_sub/Q-S-
# predicted-omega_sub comparison are both testing "does a directly
# diagnosed quantity track its radiatively/thermodynamically predicted
# counterpart" -- but the CSC pair there is a convergence (a p-derivative
# of omega_sub), whereas the pair relevant to this paper's core claim
# (Eq. 1: omega_sub = Q/S) is omega_sub itself against Q/S, not any
# derived convergence quantity. This is that direct test, done exactly
# the way the 2021 SI did it: scatter across levels, linear fit, r^2 and
# slope reported, for control and +4K separately.
# =====================================================

x_ctrl_om, y_ctrl_om = omega_dn_calc_pos[_scatter_mask], omega_dn_clear[_scatter_mask]
x_p4k_om, y_p4k_om = omega_dn_calc_pos_p4[_scatter_mask], omega_dn_clear_p4[_scatter_mask]

fit_ctrl_om = stats.linregress(x_ctrl_om, y_ctrl_om)
fit_p4k_om = stats.linregress(x_p4k_om, y_p4k_om)

fig7b, ax = plt.subplots(figsize=(2*PANEL_W, 2*PANEL_H), constrained_layout=True)

ax.scatter(x_ctrl_om, y_ctrl_om, color=c_ctrl,
           label=f"Control (r$^2$={fit_ctrl_om.rvalue**2:.3f}, slope={fit_ctrl_om.slope:.2f})")
ax.scatter(x_p4k_om, y_p4k_om, color=c_p4,
           label=f"+4K (r$^2$={fit_p4k_om.rvalue**2:.3f}, slope={fit_p4k_om.slope:.2f})")

_xfit_om = np.linspace(min(x_ctrl_om.min(), x_p4k_om.min()), max(x_ctrl_om.max(), x_p4k_om.max()), 50)
ax.plot(_xfit_om, fit_ctrl_om.intercept + fit_ctrl_om.slope*_xfit_om, color=c_ctrl, lw=2)
ax.plot(_xfit_om, fit_p4k_om.intercept + fit_p4k_om.slope*_xfit_om, color=c_p4, lw=2)
ax.plot(_xfit_om, _xfit_om, color='k', ls='--', lw=1, label="1:1 line")

ax.set_xlabel(r"$Q/S$-predicted $\omega_\mathrm{sub}$ [Pa s$^{-1}$]")
ax.set_ylabel(r"Diagnosed $\omega_\mathrm{sub}$ [Pa s$^{-1}$]")
ax.set_title(r"Diagnosed vs. $Q/S$-predicted $\omega_\mathrm{sub}$ (210-270K), cf. 2021 SI Fig. S2")
ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
ax.legend(frameon=False, loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)

fig7b.savefig("fig7b_omega_sub_scatter.pdf", bbox_inches="tight", dpi=300)

print(f"\n[Fig 7b, omega_sub vs Q/S, cf. 2021 SI S4] Control: r^2={fit_ctrl_om.rvalue**2:.3f}  "
      f"slope={fit_ctrl_om.slope:.3f}")
print(f"[Fig 7b, omega_sub vs Q/S, cf. 2021 SI S4] +4K:     r^2={fit_p4k_om.rvalue**2:.3f}  "
      f"slope={fit_p4k_om.slope:.3f}")

# =====================================================
# Q_T = 0 CROSSING (radiative tropopause temperature)
#
# Finds the temperature at which the temperature-coordinate clear-sky
# cooling Q_T crosses zero, in control and +4K, and reports the shift
# per K of surface warming -- directly comparable to Seidel & Yang
# (2022)'s radiative-tropopause-temperature metric (their Standard,
# fixed-in-pressure-ozone experiment gives ~0.4 K/K; self-lofting-ozone
# experiments give ~0.09-0.15 K/K).
#
# Search window restricted to 200-270K to avoid boundary-layer noise
# near the surface (cf. paper's Fig 2a note re: near-surface deviations).
# =====================================================

def find_zero_crossing(Q_T_arr, T_arr, Tmin=200.0, Tmax=270.0):
    mask = (T_arr > Tmin) & (T_arr < Tmax)
    Tm, Qm = T_arr[mask], Q_T_arr[mask]
    sign_changes = np.where(np.diff(np.sign(Qm)) != 0)[0]
    if len(sign_changes) == 0:
        return np.nan
    i = sign_changes[0]
    T1, T2 = Tm[i], Tm[i + 1]
    Q1, Q2 = Qm[i], Qm[i + 1]
    return T1 - Q1 * (T2 - T1) / (Q2 - Q1)


T_RT_ctrl = find_zero_crossing(Q_T, T_l)
T_RT_plus4k = find_zero_crossing(Q_T_plus4k, T_l)
dT_RT_per_K = (T_RT_plus4k - T_RT_ctrl) / dTs

print(f"\n[Radiative tropopause] T_RT control={T_RT_ctrl:.2f} K, +4K={T_RT_plus4k:.2f} K, "
      f"dT_RT/dTs={dT_RT_per_K:.3f} K/K")
print("  (compare: Seidel & Yang 2022 -- ~0.09-0.15 K/K self-lofting ozone, ~0.4 K/K fixed-in-pressure ozone)")

# =====================================================
# SHOW ALL
# =====================================================

if SHOW_FIGS:
    plt.show()
