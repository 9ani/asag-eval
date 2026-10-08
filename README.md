# asag-eval

Reproducible evaluation of automated short-answer grading (ASAG) models.

One evaluator takes the predictions of any model in one schema and produces the same
set of tables: agreement metrics with question-cluster bootstrap confidence intervals,
calibration, and a risk-coverage analysis for human-in-the-loop grading.

Work in progress: see the [issues](https://github.com/9ani/asag-eval/issues) for the plan.

## Development

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; on Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest --cov
```

## License

[MIT](LICENSE)
