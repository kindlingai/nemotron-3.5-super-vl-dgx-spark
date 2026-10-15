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
1 stream, 512 tokens, thinking off. "prose" = a story prompt, "code" = a Python module with tests.

| TP2 profile | prefill 8K | prefill 32K | 1 stream, prose | 1 stream, code |
|---|---:|---:|---:|---:|
| prefill (card default, drafter 7 tokens) | 4,216 tok/s | 3,968 tok/s | 26-27 tok/s | - |
| balanced: `b12x` MoE + linear, drafter 4 tokens | 3,372 tok/s | 3,300 tok/s | 43 tok/s | 59-67 tok/s |
| decode: `marlin` MoE, drafter 4 tokens | 2,798-3,014 tok/s | 2,892-2,900 tok/s | 44-46 tok/s | 71-79 tok/s |

Decode profile, more results:
- 250K-token needle at 60% depth: found (first token after 113 s).
- With `--max-num-seqs 4`: 4 streams fired at once gave 95 tok/s total (prose) and 155 tok/s total (code). In that
  configuration single-stream decode was lower (31 prose / 50 code). We have not found why yet; use the default
  `--max-num-seqs 1` for single-user speed.

Draft-token count matters: with `marlin`, 2 tokens gave 30/38 tok/s (prose/code), 3 gave 44-46/69-73,
4 gave 44-46/71-79, 7 gave 26-28 (prose). Our reading of 7 (not proven): each verify step checks 8 tokens; every token routes to
different experts, so a step reads far more expert weights while only ~2-3 drafts are accepted.

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
