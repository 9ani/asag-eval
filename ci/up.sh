#!/usr/bin/env sh
# Start the local CI/CD environment: the kind cluster 'asag' (staging) and Jenkins.
# Safe to run again. Stop with: docker compose -f ci/compose.yaml down && kind delete cluster --name asag
set -eu
cd "$(dirname "$0")"

kind get clusters | grep -qx asag || kind create cluster --config kind-cluster.yaml
# Jenkins runs in the Docker network 'kind', so it needs the internal API server address.
kind get kubeconfig --internal --name asag > .kubeconfig
[ -f .env ] || printf 'JENKINS_ADMIN_PASSWORD=%s\n' "$(openssl rand -hex 12)" > .env
docker compose up -d --build

echo "Jenkins:  http://localhost:8080  (admin / password in ci/.env)"
echo "Staging:  http://localhost:30080/healthz  (after the first build of main)"
