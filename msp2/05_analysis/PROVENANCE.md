# 05_analysis provenance

Flat (history-stripped) snapshot of the analysis/plotting code for the warm-rain
perturbed-parameter ensemble.

Source repository:  https://github.com/mckenna-stanford-pnnl/ppe_e3sm_warm_rain
Branch / commit:    main @ 22ae509d98ffea6d7deafacf41ec9e9dbcc00d09

This is a snapshot of the repository contents at that commit with the git
history removed. The upstream `.ipynb_checkpoints/` directory (Jupyter editor
checkpoints, git-ignored upstream) was omitted; everything else is included
verbatim. See the upstream repository for full history and any updates.

Contents:
  *.ipynb                jupyter notebooks (priors, emulators, MCMC, posteriors, plots)
  emulators_production/  trained Gaussian-process emulators (.joblib) + scalers
  mcmc_chains/           posterior MCMC chains (.npy)
