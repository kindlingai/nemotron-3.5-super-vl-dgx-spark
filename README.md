# Nemotron 3.5 Super VL 120B-A12B NVFP4 on DGX Spark (1 or 2 boxes)

Working vLLM recipes for [nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4)
on one DGX Spark (GB10), and tensor-parallel across two. The flags are the model card's DGX Spark recipe;
the two-box version adds only vLLM's built-in multi-node options (no Ray).

## Download (on every box)

```bash
export MODELS=$HOME/models/super
hf download nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4          --local-dir $MODELS/model
hf download nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr --local-dir $MODELS/drafter
```

Image: `vllm/vllm-openai:nightly-3f1bcfd82bb2e93fcd364fc7a5491718453a77e2` (from the model card; the `v0.32.0` tag
named on the card was not on Docker Hub when we tried). API model name: `nemotron-3.5-super-vl`, port 8000.

## One Spark

```bash
./serve-1spark.sh spec      # DFlash LiLiCorr drafter, 7 tokens, 1 sequence (card recipe)
./serve-1spark.sh nospec --max-num-seqs 4
```

## Two Sparks (TP2)

Both boxes need the model in `$MODELS` and a direct high-speed link (we used the RoCE ports; NCCL picks them up).
Start rank 0 first, then rank 1. Rank 0 serves the API.

```bash
# box A (rank 0)
HEAD=<box A IP on the fast link> IFACE=<fast-link NIC> ./serve-tp2.sh 0 spec
# box B (rank 1)
HEAD=<box A IP on the fast link> IFACE=<fast-link NIC> ./serve-tp2.sh 1 spec
```

Use the same mode (`spec` or `nospec`) and extra flags on both ranks.

## Results

512-token decode, warm runs. "Streams fired at once" = all requests sent at the same moment; total = all tokens / wall time.
Context 1M (`max-model-len` default from the model). With TP2 each box holds half the weights and half of each token's KV cache (the model has 2 KV heads), so far more KV memory is free per box.

| setup | GPU clock | context booted | prefill 8K | prefill 32K | 1 stream | 4 streams fired at once |
|---|---|---:|---:|---:|---:|---|
| 1 Spark, drafter | stock | 1M | 3,058 tok/s | 2,870 tok/s | 25-26 tok/s | - |
| 1 Spark, no drafter | stock | 1M | - | - | 18.7 tok/s | 47 tok/s total, 11.9 per stream |
| 2 Sparks TP2, drafter | capped 2000 MHz | 1M | 4,216 tok/s | 3,968 tok/s | 26-27 tok/s | - |
| 2 Sparks TP2, no drafter | capped 2000 MHz | 1M | 4,046 tok/s | 3,946 tok/s | 17.9 tok/s | 53 tok/s total, 13.5 per stream |

Long context, 1 Spark, no drafter: a needle at 60% depth of a 250K-token prompt was found (first token after 137 s).
At 900K tokens the model did not find it (1 try).

## TP2 decode tuning (update)

Two extra TP2 profiles (see the header of `serve-tp2.sh`). Both Sparks GPU clock capped at 2000 MHz.
512-token decode, thinking off. "prose" = a story prompt, "code" = a Python module with tests.
Code decode varies between runs (sampling at temperature 1.0 changes how many drafted tokens are accepted).

**Which one:** for one user, use **balanced** (`b12x`): same single-stream speed as `marlin` and ~14% faster prefill.
Use **decode** (`marlin`) for the highest total throughput with 4 parallel chats.

| TP2 profile | prefill 8K | prefill 32K | 1 stream prose | 1 stream code | 4 streams at once (total) | 250K needle |
|---|---:|---:|---:|---:|---|---|
| prefill (card default, drafter 7 tokens) | 4,216 tok/s | 3,968 tok/s | 26-27 tok/s | - | - | - |
| balanced: `b12x` MoE + linear, drafter 4 tokens | 3,314-3,372 tok/s | 3,300-3,314 tok/s | 43-44 tok/s | 59-74 tok/s | 82 prose / 140 code tok/s | found, 1st token 102 s |
| decode: `marlin` MoE, drafter 4 tokens | 2,798-3,014 tok/s | 2,892-2,900 tok/s | 44-46 tok/s | 71-79 tok/s | 95 prose / 155 code tok/s | found, 1st token 113 s |

The 4-stream rows used `--max-num-seqs 4`; single-stream speed is the same with that setting
(re-checked: `marlin` 45.4-45.6 tok/s prose with 4 sequences, two fresh boots).

Draft-token count (`marlin`, 1 stream, prose): 2, 3 and 4 tokens all give 43-46 tok/s; 7 tokens (the card value)
gives 26-28. Measured per verify step: 2 tokens = 44.8 ms and 2.0 tokens accepted per step; 4 tokens = 51.8 ms and
2.3 tokens per step. On our prompts only ~1-1.3 drafted tokens are accepted per step, so longer drafts mostly add
verify cost. (An early 2-token boot measured 30 tok/s; two fresh boots did not reproduce it.)

## Notes

- With the card's flags, TP2 helps prefill (+35%) and multi-stream decode but not single-stream decode. The `marlin` + 4-token drafter profile above fixes that.
- The first boot takes about 8 minutes (kernel compile and autotune); later boots about 4-5 minutes.
- Unified memory: anything else in RAM (for example files in `/dev/shm`) counts against the model. We had one boot killed by the OOM killer for this reason.
- The card's `--gpu-memory-utilization 0.75` is kept on purpose: it leaves room for the OS and the vision encoder.

## Files

- `serve-1spark.sh`, `serve-tp2.sh` - launchers
- `bench.py` - smoke test, prefill, decode
- `bench_streams.py N [prose|code]` - N streams fired at the same moment
- `needle.py WORDS` - long-context needle test
