"""Week 3 evaluation harness, built before any model (WR-1, Proposal §10).

metrics      window-level scores: AUROC, AUPRC, pooled AUROC from histograms, FPR
events       event-level scores: event recall, time-to-detect, range-based P/R/F1
thresholds   FPR-targeted thresholds from normal validation scores
calibration  isotonic calibration and reliability curves
stats        bootstrap CIs, paired comparisons, Wilcoxon signed-rank, Holm
sealed       the sealed test set and its access log (WR-2)
results      result artifacts (Implementation Plan §6.5)
checker      validation of result artifacts before any number is reported

Everything is implemented with numpy only, so the harness has no heavier
dependencies than the data pipeline and runs in CI.
"""
