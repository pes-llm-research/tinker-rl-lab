# syntax=docker/dockerfile:1.6
# =============================================================================
# TinkerRL-Bench — Reproducible Environment
# =============================================================================
# Build:         docker build -t tinker-rl-lab:latest .
# Pinned build:  docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) \
#                             --build-arg BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
#                             -t tinker-rl-lab:$(git rev-parse --short HEAD) .
# Run (GPU):     docker run --gpus all --rm -it \
#                  -e TINKER_API_KEY -e WANDB_API_KEY -e HF_TOKEN \
#                  -v $(pwd)/results:/workspace/tinker-rl-lab/results \
#                  tinker-rl-lab:latest bash
# Smoke test:    docker run --gpus all --rm -e TINKER_API_KEY tinker-rl-lab:latest \
#                  bash scripts/smoke_test.sh
# =============================================================================

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base

# Build-time provenance args (populated from git at build time)
ARG GIT_COMMIT=unknown
ARG GIT_REF=unknown
ARG BUILD_DATE=unknown

# OCI image labels — embed provenance in the image itself
LABEL org.opencontainers.image.title="TinkerRL-Bench" \
      org.opencontainers.image.description="A Unified Benchmark for RL Post-Training of Language Models" \
      org.opencontainers.image.source="https://github.com/arvindcr4/tinker-rl-lab" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.ref.name="${GIT_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.authors="PES LLM Research Team <arvindcr4@gmail.com>"

# Non-interactive + deterministic hashing + unbuffered stdio
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=42 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# --- System deps --------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        git git-lfs curl wget ca-certificates \
        build-essential pkg-config \
        jq tini \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --system

RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# --- Workdir ------------------------------------------------------------------
WORKDIR /workspace/tinker-rl-lab

# --- Python deps (cached layer) ----------------------------------------------
COPY requirements.txt pyproject.toml ./
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# --- Project sources ----------------------------------------------------------
COPY . .

# Record build provenance inside the image so reviewers can verify
RUN printf 'git_commit=%s\ngit_ref=%s\nbuild_date=%s\n' \
    "${GIT_COMMIT}" "${GIT_REF}" "${BUILD_DATE}" > /workspace/tinker-rl-lab/.build_info && \
    cat /workspace/tinker-rl-lab/.build_info

# Optional editable installs (best-effort — do not fail the build)
RUN pip install -e . 2>/dev/null || true && \
    pip install -e atropos/ 2>/dev/null || true

# --- Runtime defaults ---------------------------------------------------------
ENV SEED=42 \
    WANDB_MODE=offline \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=8

# --- Self-check: verify the environment imports the core stack ---------------
RUN python - <<'PY'
import importlib, sys
mods = ["torch", "transformers", "datasets", "accelerate", "peft", "trl", "numpy", "scipy"]
for m in mods:
    v = importlib.import_module(m).__version__
    print(f"{m:15s} {v}")
import torch
print(f"CUDA available: {torch.cuda.is_available()} | device_count={torch.cuda.device_count()}")
PY

# --- Lightweight healthcheck: make sure python + torch still import ----------
HEALTHCHECK --interval=5m --timeout=30s --start-period=30s --retries=3 \
    CMD python -c "import torch,transformers,trl; print('ok')" || exit 1

# Use tini as PID 1 for clean signal handling in interactive/Kubernetes runs
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
