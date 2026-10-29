# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `asag-eval serve`: `POST /evaluate` returns the `metrics.json` of `asag-eval evaluate`;
  `GET /healthz` for probes. Standard library only.
- Multi-stage Dockerfile: a `dev` image that runs the checks and a non-root runtime image,
  both from the same wheel.
- Lock files with hashes for both images (#10).
- Jenkinsfile: tests, image build and an image smoke test on every commit; deployment to a
  kind staging cluster with rollback on `main`. The Jenkins controller and the cluster are
  defined as code in `ci/`.
- Kubernetes manifests for the staging deployment (`k8s/`).

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
