# syntax=docker/dockerfile:1
#
# Two images from one file:
#   dev      locked tools, the wheel built once, and the checks as its command
#            docker build --target dev -t asag-eval-dev . && docker run --rm asag-eval-dev
#   runtime  the same wheel on top of the locked runtime dependencies, as a non-root service
#            docker build -t asag-eval . && docker run --rm -p 8080:8080 asag-eval
#
# The base image is pinned by digest and every dependency by version and hash, so a
# rebuild from the same commit gives the same environment.
ARG PYTHON_IMAGE=python:3.13-slim@sha256:2d44738ebf4ce5f63b9d9ea871e9c7cfdc0426f7b0c97cac915fd88ded1d86bb

FROM ${PYTHON_IMAGE} AS dev
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /src
# Dependencies before the sources: this layer stays cached until a lock file changes.
COPY requirements-dev.lock .
RUN pip install --require-hashes -r requirements-dev.lock
COPY . .
# Tests import the installed wheel, not src/: what is tested is what ships.
RUN python -m build --wheel --no-isolation --outdir /dist \
    && pip install --no-deps /dist/*.whl
CMD ["sh", "-c", "ruff check . && ruff format --check . && pytest --cov --cov-report=xml:/reports/coverage.xml --junitxml=/reports/junit.xml"]

FROM ${PYTHON_IMAGE} AS runtime
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib
COPY requirements.lock /tmp/requirements.lock
RUN pip install --require-hashes -r /tmp/requirements.lock && rm /tmp/requirements.lock
COPY --from=dev /dist/ /tmp/dist/
RUN pip install --no-deps /tmp/dist/*.whl && rm -r /tmp/dist
# nobody: numeric, so Kubernetes can verify runAsNonRoot.
USER 65534:65534
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2)"]
ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="asag-eval" \
      org.opencontainers.image.source="https://github.com/9ani/asag-eval" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.revision="${VCS_REF}"
ENTRYPOINT ["asag-eval"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
