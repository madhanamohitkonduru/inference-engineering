# PyTorch profiler runs (vLLM 0.32.0, Qwen3-0.6B, RTX 4090, driver 595.91.07)

Server (profiler on, capped at 300 iterations, no Python stacks, frontend ignored so the cap applies):
```bash
bash scripts/serve.sh --profiler-config '{"profiler":"torch","torch_profiler_dir":"/root/prof/torch","torch_profiler_with_stack":false,"ignore_frontend":true,"max_iterations":300}'
```
Warm-up (not recorded), then recorded runs (random prompts, 512 in / 128 out):
```bash
vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --model Qwen/Qwen3-0.6B \
  --dataset-name random --random-input-len 512 --random-output-len 128 --num-prompts 30 --max-concurrency 8
# c=1:  --num-prompts 4  --max-concurrency 1  --profile
# c=32: --num-prompts 64 --max-concurrency 32 --profile
python3 scripts/trace_top_kernels.py /root/prof/c1/*.pt.trace.json.gz
```
Raw traces (~5 MB each) are not committed. Summaries below were produced by `scripts/trace_top_kernels.py`.
