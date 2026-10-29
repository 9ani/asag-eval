"""Smoke test of a running asag-eval service.

    python3 ci/smoke_test.py http://host:port

Passes only if the service is healthy and returns, for the bundled example, exactly the
committed examples/expected/metrics.json: the deployed artifact must reproduce the
published numbers, not merely answer. Standard library only, so it runs on the CI host.
"""

import difflib
import json
import sys
import time
import urllib.request
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def wait_until_healthy(base: str, seconds: float = 60) -> dict:
    deadline = time.monotonic() + seconds
    while True:
        try:
            with urllib.request.urlopen(f"{base}/healthz", timeout=5) as reply:
                return json.load(reply)
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(1)


def main(base: str) -> None:
    print(f"healthy: {wait_until_healthy(base)}")
    body = (EXAMPLES / "sample_predictions.jsonl").read_bytes()
    request = urllib.request.Request(f"{base}/evaluate?name=sample_predictions.jsonl", data=body)
    with urllib.request.urlopen(request, timeout=120) as reply:
        served = reply.read().decode("utf-8")
    expected = (EXAMPLES / "expected" / "metrics.json").read_text(encoding="utf-8")
    if served != expected:
        diff = difflib.unified_diff(
            expected.splitlines(), served.splitlines(), "expected", "served", lineterm=""
        )
        print("\n".join(list(diff)[:40]))
        sys.exit("FAIL: the service does not reproduce examples/expected/metrics.json")
    qwk = json.loads(served)["metrics"]["qwk"]
    print(
        f"OK: QWK {qwk['value']} [{qwk['ci_low']}, {qwk['ci_high']}] "
        "and every other number equal the committed reference"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1].rstrip("/"))
