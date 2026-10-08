# Contributing

## Workflow

The project uses GitHub Flow: `main` is always releasable and every change arrives
through a pull request.

1. Open or pick an [issue](https://github.com/9ani/asag-eval/issues) that describes the change.
2. Create a short-lived branch from `main`: `feat/...`, `fix/...`, `docs/...` or `ci/...`.
3. Write the test first, then the code. Run the checks locally:

   ```bash
   ruff check . && ruff format --check .
   pytest --cov
   ```

4. Commit with a [Conventional Commits](https://www.conventionalcommits.org/) message,
   for example `feat: add temperature scaling`.
5. Open a pull request that references the issue (`Closes #12`). It is merged when the
   CI pipeline is green.

## What a change needs

- A test that fails without the change. New metrics are checked against an independent
  implementation or a hand-computed example.
- No new run-time dependency without a reason stated in the pull request.
- Deterministic output: no timestamps, no unseeded randomness.

If a change alters reported numbers on purpose, regenerate the reference results and
commit them together with the change, so the difference is visible in review:

```bash
asag-eval evaluate examples/sample_predictions.jsonl --out examples/expected
asag-eval compare examples/sample_predictions.jsonl examples/sample_baseline.jsonl --out examples/expected
```

## Releasing

1. Set the new version in `pyproject.toml` and `CITATION.cff`, regenerate
   `examples/expected` (the results record the tool version) and add a section to
   `CHANGELOG.md`.
2. Merge the pull request, then tag the merge commit: `git tag -a v0.2.0 -m "asag-eval 0.2.0"`
   and `git push origin v0.2.0`.
3. The release workflow runs the full pipeline and publishes the built packages.
