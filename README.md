# OpenShift to OpenShift AI demo

This repo contains a minimal, fully runnable demo that shows a small path from a simple “current state” architecture to a deployable OpenShift 4.19 demo.
It includes:

* A **Rust queue broker** (REST API + enhanced UI) that: accepts requests, writes a job row to Postgres, publishes a message to RabbitMQ, and exposes job status endpoints and a `/ui` page for quick interaction.
* A **Python worker** that consumes RabbitMQ messages, downloads two small **pickled** Hugging Face models ([iris classifier](https://huggingface.co/skops-tests/iris-sklearn-1.0-logistic_regression-without-config) + [diabetes regressor](https://huggingface.co/skops-tests/tabularregression-sklearn-latest-hist_gradient_boosting_regressor-with-config-pickle)), runs inference, and writes results back to Postgres.
* Kubernetes resources (OpenShift Template) that build images (BuildConfigs), create ImageStreams, and deploy as **Kubernetes Deployments** (OpenShift 4.19 friendly).
* Helpful deployment, build, and debug steps for common platform issues (disk pressure, build failures due to Rust transitive crates, build resource hints, tolerations for demo).

---

## Quick surface (what you get)

* UI: `GET /ui` — submit Iris or Diabetes requests, view recent jobs (polling UI).
* API:

  * `POST /predict/iris` → body `{"features":[sepal_len,sepal_wid,petal_len,petal_wid]}`
  * `POST /predict/diabetes` → body `{"features":{age,sex,bmi,bp,s1,s2,s3,s4,s5,s6}}`
  * `GET /jobs/{uuid}` → job status + result JSON
  * `GET /jobs?limit=20` → recent jobs (used by UI)
* OpenShift target: **4.19** (uses Deployments + ImageStream triggers + lookupPolicy.local)

---

## Repo layout (what to look at)

* `rust-broker/` — rust broker source, Cargo.toml, Dockerfile
* `py-worker/` — python worker source (app.py, requirements.txt, Dockerfile)
* `k8s/openshift-template.yaml` — OpenShift Template for building & deploying (pass `GIT_URI` when processing)

---

## Deploy (step-by-step)

1. Create / switch project:

```bash
oc new-project ai-demo || oc project ai-demo
```

2. (Optional) If you plan to use `registry.redhat.io` Postgres image, create a pull secret and link it:

```bash
oc create secret docker-registry redhat-registry \
  --docker-server=registry.redhat.io \
  --docker-username="$RH_REG_USER" \
  --docker-password="$RH_REG_PASS" \
  --docker-email=unused@example.com
oc secrets link default redhat-registry --for=pull
```

3. Process & apply the template (point `GIT_URI` to your repo):

```bash
oc process -f k8s/openshift-template.yaml \
  -p GIT_URI=https://github.com/YOURORG/demo-v1.git \
  -p GIT_REF=main \
| oc apply -f -
```

4. Start builds (if BuildConfigs did not auto-run):

```bash
oc start-build broker --follow
oc start-build worker --follow
```

5. Watch imagestreams & deployments:

```bash
oc get builds
oc get is
oc get deploy -w
oc get pods -w
```

6. Open UI when route is created:

```bash
echo "https://$(oc get route broker -o jsonpath='{.spec.host}')/ui"
```

---

## Test examples (curl)

* **Submit Iris Classification** (small flower, likely setosa):

```bash
BROKER=$(oc get route broker -o jsonpath='{.spec.host}')
curl -s -X POST "https://${BROKER}/predict/iris" \
  -H "content-type: application/json" \
  -d '{"features":[5.1,3.5,1.4,0.2]}'
# Expected response: {"type":"classification","label_name":"setosa","label_id":0,"confidence":0.95}
```

* **Submit Diabetes Regression** (standardized physiological measurements):

```bash
curl -s -X POST "https://${BROKER}/predict/diabetes" \
  -H "content-type: application/json" \
  -d '{"features":{"age":0.03,"sex":-0.0446,"bmi":0.02,"bp":0.01,"s1":0.1,"s2":0.08,"s3":0.02,"s4":0.03,"s5":0.04,"s6":-0.01}}'
# Expected response: {"type":"regression","prediction":152.3}
```

* Poll job:

```bash
curl -s "https://${BROKER}/jobs/<JOB_UUID>"
```

---

## Why the template uses Deployments & ImageStream triggers

* `DeploymentConfig` is deprecated and `Deployment` is the recommended standard on newer OpenShift versions (4.19+).
* The template uses `ImageStream` + `image.openshift.io/triggers` annotation + `lookupPolicy.local: true` so `image: broker:latest` resolves internally and Deployments roll automatically when a Build produces a new ImageStreamTag.

---

## Rust build issue & resolution (edition2024 / base64ct)

**Symptom:** Build fails with:

```
feature `edition2024` is required
error: failed to parse manifest ... base64ct-1.8.0/Cargo.toml
```

That happens when a transitive crate requires Rust `edition2024` and your builder uses stable Cargo (1.79). There are two safe options applied here:

1. **Pin the transitive crate** to a pre-1.8 (edition-2021) release via `[patch.crates-io] base64ct = "1.7.1"` in `Cargo.toml`. This forces dependency resolution to stay on 1.7.x and avoids edition2024 transitive crates.
2. Use a slightly newer Rust builder image (we recommend `rust:1.81`) in the Dockerfile to avoid tooling mismatches.

The repo includes both fixes:

* `Cargo.toml` has a `[patch.crates-io]` entry to pin `base64ct` to 1.7.x.
* `rust-broker/Dockerfile` uses `rust:1.81` in the build stage and installs `pkg-config libssl-dev` to compile native TLS / openssl-sys.

If you must use a cluster that restricts egress, see “Offline / Proxy” section below.

---

## Build / ephemeral-storage problems and how to handle them

**Symptom:** Build pods fail with ephemeral-storage errors or node has `DiskPressure` taint. You’ll see messages like:

* `The node was low on resource: ephemeral-storage`
* `node.kubernetes.io/disk-pressure: True` → pods not schedulable

**Actions:**

A — **Fix the node** (recommended, permanent):

* Identify the tainted node:

  ```bash
  oc get nodes
  oc describe node <NODE>
  ```
* Debug the node and free space:

  ```bash
  oc debug node/<NODE>
  # then inside the debug shell:
  crictl -r unix:///host/var/run/crio/crio.sock images
  crictl -r unix:///host/var/run/crio/crio.sock ps -a
  # remove unused images:
  crictl -r unix:///host/var/run/crio/crio.sock images -q | xargs -r crictl -r unix:///host/var/run/crio/crio.sock rmi
  # trim journal logs:
  chroot /host bash -lc 'journalctl --vacuum-time=3d'
  ```
* Re-check that `DiskPressure` flips to `False`.

B — **(Temporary) Tolerate the taint on demo deployments** (demo-only — not recommended for production):

```bash
oc patch deploy/broker -p '{"spec":{"template":{"spec":{"tolerations":[{"key":"node.kubernetes.io/disk-pressure","operator":"Exists","effect":"NoSchedule"}]}}}}'
oc patch deploy/worker -p '{"spec":{"template":{"spec":{"tolerations":[{"key":"node.kubernetes.io/disk-pressure","operator":"Exists","effect":"NoSchedule"}]}}}}'
```

Note: the kubelet may still evict pods if pressure remains.

C — **Add BuildConfig resource hints (so scheduler can make better decisions)**

```bash
oc patch bc/broker -p '{"spec":{"resources":{"requests":{"cpu":"200m","memory":"1Gi","ephemeral-storage":"1Gi"},"limits":{"cpu":"2","memory":"2Gi","ephemeral-storage":"6Gi"}}}}'
oc patch bc/worker -p '{"spec":{"resources":{"requests":{"cpu":"100m","memory":"512Mi","ephemeral-storage":"512Mi"},"limits":{"cpu":"1","memory":"1Gi","ephemeral-storage":"3Gi"}}}}'
```

These help the scheduler but do not negate the need to free up node disk space.

---

## Small resource guidance for demo components

These are tuned to be light for a single-node demo environment:

* Broker: `requests=50m CPU, 128Mi RAM, 200Mi ephemeral`, `limits=200m,256Mi,1Gi`
* Worker: `requests=50m CPU, 256Mi RAM, 200Mi ephemeral`, `limits=500m,512Mi,1Gi`
* Builds: allocate greater ephemeral-storage until node is healthy (see BuildConfig patch above)

Apply resource limits to Deployments using `oc set resources`.

---

## UI details

* UI lives at `GET /ui` on the broker route. It is a single page HTML/JS form that:

  * Submits Iris & Diabetes requests,
  * Shows the most recent 20 jobs (polls every 2s),
  * Displays raw response for quick feedback.
* If you want push updates instead of polling, we can add Server-Sent Events (SSE) to broadcast job updates.

---

## Machine Learning Models

This demo uses two pre-trained scikit-learn models from Hugging Face:

### 1. Iris Classification Model
* **HuggingFace Repo**: [`skops-tests/iris-sklearn-1.0-logistic_regression-without-config`](https://huggingface.co/skops-tests/iris-sklearn-1.0-logistic_regression-without-config)
* **Model Type**: Logistic Regression classifier
* **Dataset**: [Iris flower dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) - classic ML dataset
* **Purpose**: Predicts iris species (setosa, versicolor, virginica) from flower measurements

**Input Format**: 
```json
{"features": [sepal_length, sepal_width, petal_length, petal_width]}
```
- All values in centimeters (cm)
- Example: `[5.1, 3.5, 1.4, 0.2]` (likely setosa)
- Typical ranges: sepal_length (4.3-7.9), sepal_width (2.0-4.4), petal_length (1.0-6.9), petal_width (0.1-2.5)

**Output Format**:
```json
{
  "type": "classification",
  "label_id": 0,
  "label_name": "setosa", 
  "confidence": 0.95
}
```

### 2. Diabetes Regression Model  
* **HuggingFace Repo**: [`skops-tests/tabularregression-sklearn-latest-hist_gradient_boosting_regressor-with-config-pickle`](https://huggingface.co/skops-tests/tabularregression-sklearn-latest-hist_gradient_boosting_regressor-with-config-pickle)
* **Model Type**: Histogram-based Gradient Boosting regressor
* **Dataset**: [Diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) from scikit-learn
* **Purpose**: Predicts diabetes progression (disease severity score) from physiological measurements

**Input Format**:
```json
{"features": {"age": 0.03, "sex": -0.0446, "bmi": 0.02, "bp": 0.01, "s1": 0.1, "s2": 0.08, "s3": 0.02, "s4": 0.03, "s5": 0.04, "s6": -0.01}}
```
- All values are **standardized** (mean=0, std=1)
- Fields: age, sex, bmi (body mass index), bp (blood pressure), s1-s6 (serum measurements)
- Typical range: approximately -2.0 to +2.0 for each feature

**Output Format**:
```json
{
  "type": "regression", 
  "prediction": 152.3
}
```
- Prediction range: approximately 25-346 (diabetes progression score)

### Models & Security Notes

* Models are **pickled scikit-learn artifacts** fetched from public Hugging Face repos. Pickle files execute code at load-time. **DO NOT** load untrusted pickles in production. This demo assumes you trust the given repos. Review and (re)host vetted artifacts for any production rollout.
* The worker downloads the pickles from Hugging Face at startup and caches them under `/tmp/hf-cache` inside the container. That cache is ephemeral and will be redownloaded each time the pod is recreated.
* For production use, consider:
  - Converting to ONNX format for safer inference
  - Using OpenShift AI Model Serving instead of custom workers
  - Hosting models in your own secure registry

---

## Offline / proxy cluster notes

If your cluster **blocks egress** (crates.io, PyPI, apt, HF Hub), you have several options:

1. **Provide HTTP(S) proxy** settings in build pods (set `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` for BuildConfigs).
2. **Mirror images & dependencies** into an internal registry (UBI/mirrored base images, mirrored PyPI wheel index, vendored crates).
3. **Pre-build images** outside cluster and push to your internal registry, then skip BuildConfigs and reference images directly in the template.

Tell me which restrictions exist and I’ll produce a build strategy and exact BuildConfig envs for a proxy or a mirroring plan.

---

## Troubleshooting checklist

1. `oc get builds` — check build status & logs (`oc logs -f bc/<name>` or `oc start-build <name> --follow`).
2. `oc get is` — ImageStreams should show `broker:latest` and `worker:latest` tags created by builds.
3. `oc get deploy` & `oc rollout status deploy/<deployment>` — confirm rollouts succeed after image updates.
4. `oc get pods -o wide` — if pods pending, `oc describe pod <pod>` and look at events for `Taint`, `Insufficient`, `DiskPressure`.
5. If Rust build still fails with edition errors, ensure the updated `Cargo.toml` and Dockerfile are committed to the `rust-broker/` directory and the BuildConfig `contextDir` is `rust-broker`. Then re-run `oc start-build broker --follow`.

---

## Next steps you might want

* Replace ephemeral RabbitMQ/Postgres with Operators + PVCs (production-like).
* Replace the Python worker with an OpenShift AI model server (v2 planned).
* Add SSE or WebSocket to the broker to push job updates to the UI.
* Add metrics + tracing (Prometheus + Jaeger) and a small sample Grafana dashboard.
* Add simple auth (service account JWT or OAuth proxy) if exposing the UI externally.

---

## Quick command crib (common commands)

```bash
# reapply template (point GIT_URI to your repo)
oc process -f k8s/openshift-template.yaml -p GIT_URI=https://github.com/YOURORG/demo-v1.git | oc apply -f -

# kick builds
oc start-build broker --follow
oc start-build worker --follow

# check build logs
oc logs -f bc/broker
oc logs -f bc/worker

# view image streams and deployments
oc get is
oc get deploy
oc get pods -w

# open UI
echo "https://$(oc get route broker -o jsonpath='{.spec.host}')/ui"
```