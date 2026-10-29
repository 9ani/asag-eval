"""HTTP service: the evaluator behind one endpoint, for a grading backend or a cluster.

    GET  /healthz   -> {"status": "ok", "version": "..."}
    POST /evaluate?n_boot=2000&seed=0&n_classes=3&bins=10&name=preds.jsonl
         body: a predictions file in the schema of ``asag_eval.data``
         -> the metrics.json that ``asag-eval evaluate`` writes for the same file

Standard library only, so serving adds no dependency to the package or the image.
"""

import signal
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from asag_eval import __version__
from asag_eval.data import parse_predictions
from asag_eval.report import evaluate, to_json

MAX_BODY_BYTES = 20 * 2**20
# name: (default, minimum, maximum). The cap on n_boot keeps one request from holding a
# worker for minutes.
PARAMETERS = {
    "n_boot": (2000, 1, 10_000),
    "seed": (0, 0, 2**32 - 1),
    "n_classes": (3, 2, 100),
    "bins": (10, 1, 1000),
}


def _options(query: str) -> tuple[dict[str, int], str]:
    """Validated query parameters and the name that stands for the file in the result."""
    fields = parse_qs(query, keep_blank_values=True)
    unknown = sorted(set(fields) - set(PARAMETERS) - {"name"})
    if unknown:
        raise ValueError(f"unknown parameter: {', '.join(unknown)}")
    options = {}
    for key, (default, low, high) in PARAMETERS.items():
        text = fields.get(key, [str(default)])[-1]
        try:
            value = int(text)
        except ValueError:
            raise ValueError(f"{key} must be an integer, got {text!r}") from None
        if not low <= value <= high:
            raise ValueError(f"{key} must be from {low} to {high}, got {value}")
        options[key] = value
    return options, fields.get("name", ["request"])[-1]


class _Handler(BaseHTTPRequestHandler):
    server_version = f"asag-eval/{__version__}"

    def do_GET(self) -> None:
        if urlsplit(self.path).path == "/healthz":
            self._reply(HTTPStatus.OK, {"status": "ok", "version": __version__})
        else:
            self._reply(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        url = urlsplit(self.path)
        if url.path != "/evaluate":
            self._reply(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        length = self.headers.get("Content-Length")
        if length is None:
            self._reply(HTTPStatus.LENGTH_REQUIRED, {"error": "Content-Length is required"})
            return
        if not length.isdigit():
            self._reply(HTTPStatus.BAD_REQUEST, {"error": "Content-Length must be a number"})
            return
        if int(length) > MAX_BODY_BYTES:
            self._reply(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": f"body exceeds {MAX_BODY_BYTES} bytes"},
            )
            return
        try:
            options, name = _options(url.query)
            preds = parse_predictions(self.rfile.read(int(length)), name, options["n_classes"])
            result = evaluate(preds, options["n_boot"], options["seed"], options["bins"])
        except ValueError as exc:
            # Bad input is the client's to fix: the same one-line message the CLI prints.
            self._reply(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        self._reply(HTTPStatus.OK, result)

    def _reply(self, status: HTTPStatus, payload: dict) -> None:
        body = to_json(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def make_server(host: str, port: int) -> ThreadingHTTPServer:
    """A server that is bound but not yet serving; port 0 picks a free port."""
    return ThreadingHTTPServer((host, port), _Handler)


def serve(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Serve until SIGTERM or Ctrl+C."""
    server = make_server(host, port)
    # Docker and Kubernetes stop a container with SIGTERM; as PID 1 without a handler
    # Python would ignore it and be killed only after the grace period.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    print(f"asag-eval {__version__} serving on http://{host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
