// CI/CD for asag-eval.
// Every commit on every branch: lint and tests in the locked dev image, then the runtime
// image and a smoke test of it. Commits on main are also deployed to the staging cluster
// (kind) and smoke-tested there; a failed deployment is rolled back.
// The Jenkins controller that runs this file is defined in ci/ (see README, "CI/CD").

def kube(String script) {
  withCredentials([file(credentialsId: 'kubeconfig-staging', variable: 'KUBECONFIG')]) {
    sh script
  }
}

pipeline {
  agent any
  options {
    timestamps()
    timeout(time: 30, unit: 'MINUTES')
    disableConcurrentBuilds()
    buildDiscarder(logRotator(numToKeepStr: '20'))
  }
  environment {
    KIND_CLUSTER = 'asag'
    NAMESPACE = 'asag-staging'
  }
  stages {
    stage('Prepare') {
      steps {
        script {
          env.TAG = env.GIT_COMMIT.take(12)
          env.IMAGE = "asag-eval:${env.TAG}"
          // Unique among builds that run at the same time, and valid as a container name.
          env.RUN = "${env.TAG}-e${env.EXECUTOR_NUMBER}"
        }
        sh 'git log -1 --format="commit %h by %an: %s"'
        sh 'docker version --format "Docker {{.Server.Version}}" && kubectl version --client'
      }
    }

    stage('Lint & unit tests') {
      steps {
        sh 'docker build --target dev -t "asag-eval-dev:$TAG" .'
        // ruff, then pytest with the 90% coverage gate; a failure fails the build here.
        sh 'docker run --name "asag-test-$RUN" "asag-eval-dev:$TAG"'
      }
      post {
        always {
          sh 'rm -rf reports && docker cp "asag-test-$RUN:/reports" reports || true'
          sh 'docker rm -f "asag-test-$RUN" || true'
          junit allowEmptyResults: true, testResults: 'reports/junit.xml'
          recordCoverage tools: [[parser: 'COBERTURA', pattern: 'reports/coverage.xml']],
                         sourceCodeRetention: 'NEVER'
        }
      }
    }

    stage('Build image') {
      steps {
        // Reuses the cached dev stage: the wheel that was tested is the wheel that ships.
        sh 'docker build --build-arg VCS_REF="$GIT_COMMIT" -t "$IMAGE" .'
        sh 'docker image inspect "$IMAGE" --format "{{.Id}} {{.Size}} bytes"'
      }
    }

    stage('Smoke test image') {
      steps {
        sh 'docker run -d --name "asag-smoke-$RUN" --network kind "$IMAGE"'
        sh 'python3 ci/smoke_test.py "http://asag-smoke-$RUN:8080"'
      }
      post {
        always {
          sh 'docker rm -f "asag-smoke-$RUN" || true'
        }
      }
    }

    stage('Push to Docker Hub') {
      when {
        branch 'main'
        // Off until DOCKERHUB_REPO and the 'dockerhub' credential exist (ci/jenkins/casc.yaml).
        expression { env.DOCKERHUB_REPO?.trim() }
      }
      steps {
        withCredentials([usernamePassword(credentialsId: 'dockerhub',
                                          usernameVariable: 'DOCKER_USER',
                                          passwordVariable: 'DOCKER_TOKEN')]) {
          sh '''
            # A throwaway config: the token never lands in the Jenkins home.
            export DOCKER_CONFIG="$(mktemp -d)"
            trap 'rm -rf "$DOCKER_CONFIG"' EXIT
            echo "$DOCKER_TOKEN" | docker login -u "$DOCKER_USER" --password-stdin "${DOCKERHUB_REPO%%/*}"
            for tag in "$TAG" latest; do
              docker tag "$IMAGE" "$DOCKERHUB_REPO:$tag"
              docker push "$DOCKERHUB_REPO:$tag"
            done
          '''
        }
      }
    }

    stage('Deploy to staging') {
      when { branch 'main' }
      steps {
        sh 'kind load docker-image "$IMAGE" --name "$KIND_CLUSTER"'
        kube '''
          kubectl apply -f k8s/namespace.yaml -f k8s/service.yaml
          sed -e 's|${IMAGE}|'"$IMAGE"'|' -e 's|${CHANGE_CAUSE}|'"$TAG, build $BUILD_NUMBER"'|' \
            k8s/deployment.yaml | kubectl apply -f -
          if ! kubectl -n "$NAMESPACE" rollout status deployment/asag-eval --timeout=180s; then
            kubectl -n "$NAMESPACE" rollout undo deployment/asag-eval
            exit 1
          fi
          kubectl -n "$NAMESPACE" get deployment,pods,service -o wide
        '''
      }
    }

    stage('Smoke test staging') {
      when { branch 'main' }
      steps {
        sh 'python3 ci/smoke_test.py "http://$KIND_CLUSTER-control-plane:30080"'
        kube 'kubectl -n "$NAMESPACE" rollout history deployment/asag-eval'
      }
      post {
        failure {
          kube 'kubectl -n "$NAMESPACE" rollout undo deployment/asag-eval'
        }
      }
    }
  }

  post {
    always {
      // Keep the runtime images (rollback), drop the dev image tag of this commit.
      sh 'docker image rm "asag-eval-dev:$TAG" || true'
    }
  }
}
