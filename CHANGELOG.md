# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-08

### Added

- One JSON Lines predictions schema for every model, validated on load.
- Agreement metrics: QWK, Cohen's kappa, macro-F1, accuracy, mean absolute error,
  extreme error rate, confusion matrix and per-score precision, recall and F1.
- Question-cluster bootstrap confidence intervals.
- Paired comparison of two models with a bootstrap p-value.
- Calibration: ECE, Brier score, negative log-likelihood and a reliability diagram.
- Selective prediction: risk-coverage curve, AURC and agreement at fixed coverage.
- `asag-eval evaluate` and `asag-eval compare` with JSON, Markdown and PNG output.
- A synthetic example with reference results that CI regenerates on every push.

[0.1.0]: https://github.com/9ani/asag-eval/releases/tag/v0.1.0
