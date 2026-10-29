import http.client
import json
import os
import signal
import sys
import threading
import urllib.error
import urllib.request

import pytest

import asag_eval
from asag_eval import server
from asag_eval.cli import main


@pytest.fixture
def service():
    """Base URL of a server on a free port, stopped after the test."""
    httpd = server.make_server("127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def _request(url, body=None):
    """(status, decoded JSON) for GET, or POST when a body is given."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=body), timeout=30) as reply:
            return reply.status, json.loads(reply.read())
    except urllib.error.HTTPError as error:
        payload = json.loads(error.read())
        error.close()
        return error.code, payload


def test_health_reports_the_version(service):
    assert _request(f"{service}/healthz") == (
        200,
        {"status": "ok", "version": asag_eval.__version__},
    )


def test_evaluate_returns_the_committed_metrics(service, sample, expected, assert_matches):
    status, result = _request(
        f"{service}/evaluate?name=sample_predictions.jsonl", sample.read_bytes()
    )

    assert status == 200
    reference = json.loads((expected / "metrics.json").read_text(encoding="utf-8"))
    assert_matches(result, reference)


def test_parameters_reach_the_evaluation(service, sample):
    status, result = _request(f"{service}/evaluate?n_boot=50&seed=3&bins=5", sample.read_bytes())

    assert status == 200
    assert result["input"]["file"] == "request"
    assert {key: result["settings"][key] for key in ("n_boot", "seed", "n_bins")} == {
        "n_boot": 50,
        "seed": 3,
        "n_bins": 5,
    }


def test_bad_record_is_a_400_naming_the_line(service):
    body = b'{"id": "a1", "true_score": 7, "predicted_score": 1}\n'

    assert _request(f"{service}/evaluate", body) == (
        400,
        {"error": "request line 1: true_score must be an integer from 0 to 2, got 7"},
    )


@pytest.mark.parametrize(
    ("query", "message"),
    [
        ("n_boot=0", "n_boot must be from 1 to 10000, got 0"),
        ("n_boot=20000", "n_boot must be from 1 to 10000, got 20000"),
        ("seed=x", "seed must be an integer, got 'x'"),
        ("n_classes=1", "n_classes must be from 2 to 100, got 1"),
        ("nboot=5&color=red", "unknown parameter: color, nboot"),
    ],
)
def test_parameters_are_validated(service, sample, query, message):
    assert _request(f"{service}/evaluate?{query}", sample.read_bytes()) == (400, {"error": message})


@pytest.mark.parametrize(("method", "body"), [("GET", None), ("POST", b"{}")])
def test_unknown_paths_are_404(service, method, body):
    request = urllib.request.Request(f"{service}/metrics", data=body, method=method)
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request, timeout=30)
    assert error.value.code == 404
    error.value.close()


@pytest.mark.parametrize(("length", "status"), [(None, 411), ("-1", 400), ("ten", 400)])
def test_body_length_must_be_given_as_a_number(service, length, status):
    host, port = service.removeprefix("http://").split(":")
    connection = http.client.HTTPConnection(host, int(port), timeout=30)
    connection.putrequest("POST", "/evaluate")
    if length is not None:
        connection.putheader("Content-Length", length)
    connection.endheaders()
    reply = connection.getresponse()
    reply.read()
    connection.close()
    assert reply.status == status


def test_oversized_body_is_refused_unread(service, monkeypatch):
    monkeypatch.setattr(server, "MAX_BODY_BYTES", 10)
    assert _request(f"{service}/evaluate", b"x" * 11) == (413, {"error": "body exceeds 10 bytes"})


@pytest.mark.skipif(sys.platform == "win32", reason="Windows cannot deliver SIGTERM to itself")
def test_serve_stops_on_sigterm(capsys):
    previous = signal.getsignal(signal.SIGTERM)
    # Installed before the timer too, so an early signal cannot kill the test run.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    timer = threading.Timer(0.5, os.kill, (os.getpid(), signal.SIGTERM))
    timer.start()
    try:
        server.serve("127.0.0.1", 0)
    finally:
        timer.join()
        signal.signal(signal.SIGTERM, previous)
    assert "serving on http://127.0.0.1:" in capsys.readouterr().out


def test_serve_command_passes_host_and_port(monkeypatch):
    calls = []
    monkeypatch.setattr(server, "serve", lambda host, port: calls.append((host, port)))

    main(["serve", "--host", "0.0.0.0", "--port", "9000"])
    main(["serve"])

    assert calls == [("0.0.0.0", 9000), ("127.0.0.1", 8080)]


def test_address_in_use_exits_with_one_clear_line(capsys):
    taken = server.make_server("127.0.0.1", 0)
    try:
        with pytest.raises(SystemExit) as exit_info:
            main(["serve", "--port", str(taken.server_port)])
    finally:
        taken.server_close()
    assert exit_info.value.code == 2
    assert capsys.readouterr().err.startswith("asag-eval: error: ")
