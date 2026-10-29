# Jenkins controller for this repository, configured as code: plugins from plugins.txt,
# users, credentials and the pipeline job from casc.yaml. No setup wizard, no clicks.
FROM jenkins/jenkins:2.580.1-lts-jdk21

USER root
# Docker CLI and buildx: builds run on the host's Docker daemon through the mounted socket.
COPY --from=docker:29.8.2-cli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=docker:29.8.2-cli /usr/local/libexec/docker/cli-plugins/docker-buildx /usr/local/libexec/docker/cli-plugins/docker-buildx
# kubectl matches the kind node (v1.37); kind loads built images into the cluster.
# python3 runs ci/smoke_test.py.
ARG TARGETARCH
ARG KUBECTL_VERSION=v1.37.0
ARG KIND_VERSION=v0.33.0
RUN curl -fsSLo /usr/local/bin/kubectl "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${TARGETARCH}/kubectl" \
    && curl -fsSLo /usr/local/bin/kind "https://kind.sigs.k8s.io/dl/${KIND_VERSION}/kind-linux-${TARGETARCH}" \
    && chmod +x /usr/local/bin/kubectl /usr/local/bin/kind \
    && apt-get update \
    && apt-get install -y --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/*

USER jenkins
COPY plugins.txt /usr/share/jenkins/ref/plugins.txt
RUN jenkins-plugin-cli --plugin-file /usr/share/jenkins/ref/plugins.txt
COPY casc.yaml /usr/share/jenkins/casc.yaml
ENV JAVA_OPTS="-Djenkins.install.runSetupWizard=false" \
    CASC_JENKINS_CONFIG=/usr/share/jenkins/casc.yaml
