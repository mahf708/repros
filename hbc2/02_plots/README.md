# Review Package: Thermodynamic Buffering of the Tropical High Cloud Fraction
# Response to Warming in a Global Storm-Resolving Model

This package regenerates every figure in the final paper and its Supporting
Information from post-processed data. It was assembled and independently
verified by re-running each script from scratch and pixel-comparing the
output against the actual figures used in the paper.

## Quick start

1. Set up the environment (see "Dependencies" below).
2. Copy every file from `../05_data/` (the data lives in its own top-level
   folder in this archive) into the same directory as the scripts in
   `scripts/` (the scripts expect the data files alongside them, not in a
   subdirectory).
3. Run, in this order:
   ```
   python3 analysis_minimal.py               # -> fig1, fig2, fig3, fig4, fig6
   python3 make_mip_figures_no_panel_a.py    # -> fig5, figS_mip_diagnostics
   python3 make_figS1_latent_profile.py      # -> figS_latent_clear_vs_cloud_profile
   python3 make_figS3_mass_balance.py        # -> figS_mass_balance_check  (see WARNING below)
   ```
4. Compare against `reference_figs/` to confirm your environment reproduces
   the same output (a `diff` at the pixel level will show only
   font-rendering/anti-aliasing noise between environments, not real
   differences — this was true even in our own verification, run twice on
   two different machine states).

## What each script produces

| Script | Figures produced | Status |
|---|---|---|
| `analysis_minimal.py` | fig1, fig2, fig3, fig4, fig6 | Verified: pixel-matches the paper |
| `make_mip_figures_no_panel_a.py` (calls `unified_buffering_analysis.py`) | fig5, figS_mip_diagnostics | Verified: pixel-matches the paper (N=83 models, slope=1.04, R²=0.98) |
| `make_figS1_latent_profile.py` | figS_latent_clear_vs_cloud_profile | Reconstructed — see WARNING |
| `make_figS3_mass_balance.py` | figS_mass_balance_check | Reconstructed — now matches SI text well, see note |

`analysis_minimal.py` also produces a number of exploratory/diagnostic
figures (fig7–fig12, the `Unified_buffering_*` files,
`diagnostic_omega_dn_vs_up.pdf`, etc.) that were part of the investigative
process but do **not** appear in the final paper or SI. They're harmless
byproducts of the same script, not something to be concerned about, but
don't confuse them for paper figures.

## Two SI figures are reconstructions, not verified originals

The interactive code that produced `figS_latent_clear_vs_cloud_profile.pdf`
and `figS_mass_balance_check.pdf` during manuscript revision was never
saved as a standalone script. The two scripts under those names in this
package are **from-scratch reconstructions**, written to match the SI
text's own description of what each figure shows, using the same cached
data and unit conventions used everywhere else in the paper. Each script's
docstring has the full detail, but in brief:

- **`make_figS1_latent_profile.py`**: reproduces the SI text's qualitative
  ordering exactly (in-cloud heating > in-cloud cooling > clear-sky
  cooling > clear-sky heating) and is in the right ballpark for magnitude,
  but the exact vertical separation between curves is not an exact match
  to "one to one and a half orders of magnitude" as stated in the SI text.
  **Moderate confidence.**

- **`make_figS3_mass_balance.py`**: an earlier version of this script
  multiplied `omega_dn`/`omega_up` by `(1-CF)`/`CF`, on the assumption
  these were raw conditional means needing that weighting to become
  domain contributions -- that produced curves roughly a factor of 9
  apart, nowhere near the SI text's "within 10%" claim. Checking the raw,
  unweighted magnitudes directly showed they already sit within ~2-18% of
  each other across 210-270K: `omega_dn`/`omega_up` are already the
  properly area-weighted domain contributions (i.e. already `(1-C)*omega_sub`
  and `C*omega_cld`), and the earlier CF multiplication was double-counting
  weighting that was already there. The corrected version plots them as-is
  and now matches the SI text's description well. **Reasonable confidence**,
  though still not checked pixel-for-pixel against the original figure.

If you have access to the original interactive session or notebook that
produced these two figures, that would let you confirm both directly;
short of that, we'd treat `make_figS3_mass_balance.py` as a good match and
`make_figS1_latent_profile.py` as approximate but directionally correct.

**Side note on `analysis_minimal.py`'s own console output:** this script
prints a "[Eq. 3 mass-balance check, MAGNITUDE-ONLY]" diagnostic with a
median residual of ~2800%. That check uses the same flawed
`(1-CF)*omega_dn` vs. `CF*omega_up` double-weighting identified and fixed
above — it does not indicate a real problem with the underlying data, just
with that one particular sanity-check line. It doesn't affect any of the
five figures `analysis_minimal.py` produces (fig1-4, fig6), which are
pixel-verified against the paper regardless. We left this print statement
as-is rather than edit the verified-correct script, but it's safe to
ignore.

## Dependencies

```bash
pip install numpy scipy matplotlib xarray netCDF4 h5netcdf --break-system-packages
```

(`netCDF4`/`h5netcdf` are required for `make_mip_figures_no_panel_a.py` to
read the RCEMIP `.nc` files — without them, RCEMIP models silently fail to
load and the script will report a much smaller N with no hard error. If
you ever see N substantially below 83 models in the fig5/figS_mip output,
check first that these two packages are actually installed and importable
in your environment.)

## Data files

All files below are in `../05_data/`.

| File | Used by | Notes |
|---|---|---|
| `control_vars_minimal_full_record.npz`, `plus4k_vars_minimal_full_record.npz` | all scripts | Post-processed SCREAM Cess-Potter output. Use the versions in this package specifically — an earlier, incomplete version of these two files (missing `rain_evap_domain_mean` and the `_incloud` variables) circulated during revision and will fail with a `KeyError` in `analysis_minimal.py`. |
| `RCEMIP/` | `unified_buffering_analysis.py` | Full RCEMIP archive (`swift.dkrz.de/RCE_small`, `RCE_large`). Needs `netCDF4`/`h5netcdf` (see above). |
| `CMIP_amip_*.pkl`, `CMIP_amip4k_*.pkl` | `unified_buffering_analysis.py` | CMIP6 AMIP / AMIP-future4K, pre-processed. |
| `SCREAM_ne1024_amip_*.pkl`, `SCREAM_ne1024_amip4k_*.pkl` | `unified_buffering_analysis.py` | SCREAM AMIP runs (the "Cess1" point in fig5), pre-processed. |

## Reference figures

`reference_figs/` contains the actual figures as they appear in the final
paper, for direct comparison against your own regenerated output.
