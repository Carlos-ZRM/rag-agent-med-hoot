#!/bin/bash
name=quiz-db-app
tag=0.1
quay_repo=quay.io/medi-hoot/quiz-db-app
push=true

podman build --platform linux/amd64 -t quiz-db-app:0.1-amd64 .
podman build --platform linux/arm64 -t quiz-db-app:0.1-arm64 .

if [ "$push" = true ]; then
  podman tag quiz-db-app:0.1-amd64 $quay_repo:0.1-amd64
  podman tag quiz-db-app:0.1-arm64 $quay_repo:0.1-arm64
  podman push $quay_repo:0.1-amd64
  podman push $quay_repo:0.1-arm64
fi
