#!/bin/bash
# Nemotron 3.5 Super VL 120B-A12B NVFP4, tensor parallel across 2 DGX Sparks.
# Usage (rank 0 first, then rank 1): HEAD=<rank0 IP> IFACE=<nic> ./serve-tp2.sh RANK [spec|nospec] [extra vllm args]
RANK=$1; MODE=${2:-spec}; shift 2
HEAD=${HEAD:?set HEAD to rank 0 IP on the fast link}; IFACE=${IFACE:?set IFACE to the fast-link interface}
MYIP=$(ip -4 -o addr show $IFACE | awk '{print $4}' | cut -d/ -f1)
IMAGE=vllm/vllm-openai:nightly-3f1bcfd82bb2e93fcd364fc7a5491718453a77e2
SPEC=(); [ "$MODE" = spec ] && SPEC=(--speculative_config.model /models/drafter --speculative_config.num_speculative_tokens 7 --speculative_config.method dflash)
ROLE=(); [ "$RANK" != 0 ] && ROLE=(--headless)
docker rm -f super 2>/dev/null
docker run -d --name super --gpus all --ipc=host --network host --privileged --ulimit memlock=-1 \
  -v "${MODELS:-$HOME/models/super}":/models:ro -v super-cache:/root/.cache \
  -e NCCL_SOCKET_IFNAME=$IFACE -e GLOO_SOCKET_IFNAME=$IFACE -e VLLM_HOST_IP=$MYIP \
  $IMAGE /models/model --port 8000 \
  --tensor-parallel-size 2 --nnodes 2 --node-rank $RANK --master-addr $HEAD --master-port 29511 "${ROLE[@]}" \
  --served-model-name nemotron-3.5-super-vl --max-num-seqs 1 --trust-remote-code \
  --gpu-memory-utilization 0.75 --max-num-batched-tokens 8192 \
  "${SPEC[@]}" \
  --mamba-backend flashinfer --mamba-ssm-cache-dtype float16 \
  --enable-mamba-cache-stochastic-rounding --mamba-cache-philox-rounds 5 \
  --enable-mamba-shared-prefix-checkpoint \
  --video-pruning-rate 0.5 --media-io-kwargs '{"video": {"fps": 2, "num_frames": 256}}' \
  --block-size 64 --prefix-match-unit 64 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 "$@"
