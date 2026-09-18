# Thermodynamic Buffering of the Tropical High Cloud Fraction
# Response to Warming in a Global Storm-Resolving Model

This code regenerates every figure in the final paper and its Supporting
Information from post-processed data.

## Quick start

1. Set up the environment (dependencies: numpy, scipy, matplotlib, xarray, netCDF4, and h5netcdf).
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
