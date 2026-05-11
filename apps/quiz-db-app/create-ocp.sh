#!/bin/bash

appns=xpk  # Set your desired namespace here
name=quiz-db-app   # Set your build config name here

# 1. Create namespace if it does not exist
if ! oc get namespace "$appns" >/dev/null 2>&1; then
  echo "Namespace $appns does not exist. Creating..."
  oc create namespace "$appns"
else
  echo "Namespace $appns already exists."
fi

# Set the current project to the namespace
oc project "$appns"

# 2. Verify if BuildConfig exists, if not, create it
if ! oc get bc "$name" >/dev/null 2>&1; then
  echo "BuildConfig $name does not exist. Creating..."
  oc new-build --name "$name" --binary --strategy docker
else
  echo "BuildConfig $name already exists."
fi

# 3. Start the build
oc start-build "$name" --from-dir . --follow



