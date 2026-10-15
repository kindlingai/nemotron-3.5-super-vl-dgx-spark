#!/bin/bash
# Nemotron 3.5 Super VL 120B-A12B NVFP4, one DGX Spark, model card recipe. Usage: ./serve.sh [spec|nospec] [extra vllm args...]
MODE=${1:-spec}; shift
MODELS=${MODELS:-$HOME/models/super}; PORT=${PORT:-8000}
IMAGE=vllm/vllm-openai:nightly-3f1bcfd82bb2e93fcd364fc7a5491718453a77e2
SPEC=()
[ "$MODE" = spec ] && SPEC=(--speculative_config.model /models/drafter --speculative_config.num_speculative_tokens 7 --speculative_config.method dflash)
docker rm -f super 2>/dev/null
docker run -d --name super --gpus all --ipc=host -p $PORT:8000 -v "$MODELS":/models:ro -v super-cache:/root/.cache \
  $IMAGE /models/model \
  --served-model-name nemotron-3.5-super-vl --max-num-seqs 1 --trust-remote-code \
  --gpu-memory-utilization 0.75 --max-num-batched-tokens 8192 \
  "${SPEC[@]}" \
  --mamba-backend flashinfer --mamba-ssm-cache-dtype float16 \
  --enable-mamba-cache-stochastic-rounding --mamba-cache-philox-rounds 5 \
  --enable-mamba-shared-prefix-checkpoint \
  --video-pruning-rate 0.5 --media-io-kwargs '{"video": {"fps": 2, "num_frames": 256}}' \
  --block-size 64 --prefix-match-unit 64 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 "$@"
