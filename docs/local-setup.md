# Local development setup

Setting up `ru-jailbreak-guard` for local development on macOS (M-series).

## Prerequisites

- macOS 14+ on Apple Silicon (M1/M2/M3/M4)
- Homebrew
- 32 GB+ system RAM (16 GB allocated to Docker Desktop)
- ~50 GB free disk

## Step 1 — Docker Desktop

Install Docker Desktop and bump resource allocation:

1. `brew install --cask docker` (or download from docker.com)
2. Open Docker Desktop → Settings → Resources → Advanced
3. Set:
   - **Memory: 16 GB**
   - **CPUs: 8**
   - Swap: 2 GB
4. Click *Apply & Restart*
5. Settings → Kubernetes → check *Enable Kubernetes* → Apply (waits ~2 min)

Verify:

```bash
docker info | grep -E "(CPUs|Total Memory)"
kubectl config current-context  # should be: docker-desktop
kubectl get nodes               # should show one Ready node
```

## Step 2 — Tooling

```bash
# Python package manager
brew install uv

# Kubectl & ArgoCD CLI
brew install kubectl argocd

# (Optional) GitHub CLI for PRs
brew install gh
```

## Step 3 — Clone and install

```bash
git clone https://github.com/Meffazm/ru-jailbreak-guard.git
cd ru-jailbreak-guard
uv python install 3.13
make setup
make all
```

`make all` runs lint + type-check + tests. All should pass.

## Step 4 — ArgoCD

ArgoCD is the GitOps engine for this project. Install it once into your local cluster:

```bash
kubectl create namespace argocd
ARGOCD_VERSION="v3.3.8"  # or run: gh release list --repo argoproj/argo-cd --limit 1
kubectl apply -n argocd --server-side \
  -f "https://raw.githubusercontent.com/argoproj/argo-cd/${ARGOCD_VERSION}/manifests/install.yaml"
kubectl wait --for=condition=Available deployment --all -n argocd --timeout=300s
```

**Why `--server-side`:** ArgoCD's `applicationsets.argoproj.io` CRD now exceeds 262 KB
(annotation size limit for client-side apply). Without `--server-side`, you'll see
`metadata.annotations: Too long`.

Apply the project's AppProject + ApplicationSet:

```bash
kubectl apply --server-side -f gitops/project.yaml
kubectl apply --server-side -f gitops/applicationset.yaml
```

Verify:

```bash
kubectl get appproject,applicationset -n argocd
# Expected: ru-jailbreak-guard listed for both
```

Get the initial admin password:

```bash
kubectl -n argocd get secret argocd-initial-admin-secret \
  -o jsonpath='{.data.password}' | base64 -d ; echo
```

Open the UI:

```bash
make argocd-ui
# browse https://localhost:8080  → login admin / <password>
```

## Step 5 — Tear-down

ArgoCD remains installed (small footprint when idle). To stop:

- Quit Docker Desktop (cleanest), OR
- `kubectl delete namespace argocd` to remove ArgoCD only

## Troubleshooting

### `uv sync --frozen` reports lockfile drift

Lockfile and `pyproject.toml` are out of sync. Either:

- Update lockfile: `uv sync` (without `--frozen`), then commit the new `uv.lock`
- Verify your local uv version matches `0.11.7` (CI version): `uv --version`

### `ty` install fails

`ty` is in early alpha (Astral's new type checker). If install fails on a transient
PyPI issue, retry `uv sync` — it usually resolves. As a longer-term fallback, swap
`ty>=0.0.1a0` for `mypy>=1.13` in `pyproject.toml` dev deps and update the Makefile
`type-check` target to `uv run mypy src tests`.

### Docker Desktop refuses 16 GB allocation

Confirm system RAM is 32 GB+ and no other RAM-heavy app is running. Restart Docker
Desktop. If still refused, try 12 GB — the project will run, just with less headroom.

### `kubectl` shows no nodes

Docker Desktop's Kubernetes is disabled. Settings → Kubernetes → Enable.

### ArgoCD pods stuck in Pending or ImagePullBackOff

If pods can't pull images from `public.ecr.aws/docker/library/redis:*`, you may be
hitting the same CDN-edge TLS issue we saw during initial bootstrap. Workaround:

```bash
docker pull mirror.gcr.io/library/redis:8.2-alpine
docker tag mirror.gcr.io/library/redis:8.2-alpine \
  public.ecr.aws/docker/library/redis:8.2.3-alpine
docker save public.ecr.aws/docker/library/redis:8.2.3-alpine | \
  docker exec -i desktop-control-plane ctr -n=k8s.io images import -
PATCH='[{"op":"replace",
  "path":"/spec/template/spec/containers/0/imagePullPolicy",
  "value":"IfNotPresent"}]'
kubectl -n argocd patch deployment argocd-redis --type=json -p="${PATCH}"
kubectl -n argocd rollout restart deployment argocd-redis
```

This is a one-time workaround; if the redis pod is ever recreated and the network
issue resolves, the patch becomes redundant. Re-apply if it bites again.
