# Day 6: prefix caching

Server: vLLM 0.32.0, Qwen3-0.6B, RTX 4090, `scripts/serve.sh` defaults (prefix caching is **on by default**; block size 16, SHA-256 block hashes).
Experiment script: [`prefix_test.py`](prefix_test.py). Raw output: [`results/raw_output.txt`](results/raw_output.txt).

## 1. Self-check answers (my own words)
1. **Why can't two prompts that differ at token 5 share a cache?** Every token's saved key/value numbers depend on all the tokens before it, so one different early token changes everything after it. The block fingerprints chain the same way: each block's hash includes the previous block's hash.
2. **Why does a 100-token shared prefix save only 96 tokens?** The cache stores blocks of 16 tokens and only full blocks are saved and shared. 100 = 6 full blocks (96 tokens) + 4 leftover tokens that do not fill a block, so those 4 are recomputed.
3. **Why won't caching change TPOT?** Caching skips prefill, not decode. By the time decode starts the prompt's KV cache is identical either way, so every decode step reads the same bytes.

## 2. Predictions vs measurements
Prefill arithmetic, Qwen3-0.6B, 2,000-token prompt (prefill FLOPs ~ 2 x parameters x tokens):
- 0.6B parameters: 2.4 x 10^12 FLOPs. Layers only (about 440M parameters; the embedding table is a lookup and the lm_head runs for one token): 1.76 x 10^12 FLOPs.
- RTX 4090 dense BF16 tensor peak used: 165.2 TFLOPS. Ideal prefill: 14.5 ms (0.6B) or 10.65 ms (layers only).
- For a 15-token question: 0.08 ms of arithmetic, far below the ~1.2 ms weight-read time, so a tiny prefill is memory-bound like decode.

| Prediction | My prediction | Reference prediction (Claude's) | Measured (mean over 2 runs where noted) | Gap vs mine / reference |
|---|---|---|---|---|
| Cold TTFT, caching on | 14.5 ms (arithmetic only, no overhead) | ~30 ms | 28.4 ms (28.6, 28.1) | +96% / -5% |
| TTFT requests 2-10, caching on | "time to compute the short question" (no number) | ~8 ms | 16.0 ms (run1) / 15.9 ms (run2) | n/a / +99% |
| TTFT every request, caching off | 14.5 ms | ~30 ms | 27.1 ms (mean of 10) | +87% / -10% |
| Speedup, request 1 vs 2-10 | "depends on the amount to compute" (no number) | ~4x | 1.8x (28.4 / 16.0) | n/a / -55% |

Why the gaps: my own predictions counted GPU arithmetic only and left out the fixed per-request overhead; the reference predictions guessed that overhead at 5-10 ms, but the fastest warm requests took ~13 ms.

## 3. Was the benchmark hitting the cache? (Day 4 question)
`bench_quick.sh` (100 prompts x 2 concurrency levels, 512-token random prompts, same prompts each run):

| | Hit rate | Hit tokens |
|---|---|---|
| First pass | 49.2% | 51,200 of 104,000 (= exactly 100 x 512: one of the two runs found all its prompts stored) |
| Second pass | 98.5% | 102,400 of 104,000 |

| | Caching on (98.5% hits) | Caching off | Change when off |
|---|---|---|---|
| c=1 mean TTFT | 19.1 ms | 25.7 ms | +34% |
| c=32 mean TTFT | 95.6 ms | 131.3 ms | +37% |
| c=1 TPOT | 2.50 ms | 2.53 ms | +1% |
| c=32 TPOT | 5.47 ms | 5.76 ms | +5% |
| c=1 / c=32 throughput | 380 / 4,514 tok/s | 368 / 4,215 tok/s | -3% / -7% |

- **Finding:** the benchmark's repeated prompts were cached heavily (49% on a first pass, 98.5% on repeats). Turning caching off raised TTFT by about a third and left c=1 decode speed unchanged.
- **Hypothesis:** the Day 1-3 throughput numbers were affected only slightly, since a 512-token prefill is ~1% of a request's time (~3-7 ms vs ~320 ms of decode). Not re-measured on the Day 1 data.
- **Hypothesis:** the 5% TPOT rise at c=32 with caching off comes from prefill work sharing steps with other requests' decoding.

## 4. Shared-prefix experiment (2,000-token prefix, 10 requests one at a time)
Mean TTFT of requests 2-10 (request 1 is cold):

| Case | Hit tokens (total, 9 requests) | Request 1 | Requests 2-10 | Rule demonstrated |
|---|---|---|---|---|
| Caching off | 0 | 28.9 ms | 27.1 ms | - |
| Caching on (2 runs) | 18,000 (= 9 x 2000, 125 full blocks) | 28.6 / 28.1 ms | 16.0 / 15.9 ms | - |
| One token changed at position 1,000 | 8,928 (= 9 x 992 = 62 blocks before the changed token) | 26.8 ms | 20.0 ms | Rule 2: exact prefix |
| Shared text after the unique question | 0 | 30.9 ms | 30.3 ms | Rule 2: where the difference sits |
| 100-token prefix | 864 (= 9 x 96) | 18.7 ms | 12.4 ms | Rule 1: only full blocks |
| TPOT unchanged by caching (section 3) | - | - | - | Rule 3 |

- **Finding:** hit counts match the block arithmetic exactly in all three edge cases.
- **Finding:** put static text (system prompt, document) first and changing text (the question) last.
- **Finding:** at 2,000 tokens caching saved ~11 ms per request, about equal to the 10.65 ms ideal arithmetic.
- **Hypothesis:** at 512 tokens it saved 6.6 ms against 2.7 ms ideal, so larger prefills may run closer to peak compute. Different request paths (chat vs completions) and a few ms of noise; not established.
- **Hypothesis:** the fixed non-GPU floor is ~11-13 ms (HTTP, scheduling, first step, streaming), and it grows a little with request size. Not broken down.
- **Hypothesis:** a cache lookup that finds nothing costs a few ms (case "after" 30.3 ms vs caching off 27.1 ms). Single runs, small gap.

## 5. Reading the code (`vllm/v1/core/`, v0.32.0)
- Block hash: `kv_cache_utils.py:695 hash_block_tokens` hashes (parent block hash, the block's token ids, extra keys); the first block uses a fixed starting value; blocks are assumed full.
- Lookup: `kv_cache_manager.py:264 get_computed_blocks` returns early if caching is disabled, otherwise calls `find_longest_cache_hit` with the request's block hashes and `max_cache_hit_length = num_tokens - 1`.
- Release: `kv_cache_manager.py:620 free`, then `block_pool.py` (`free_blocks`, ~774-803) decrements `ref_cnt` and puts blocks with count 0 into `free_block_queue`; a hit on such a block (~762-766) takes it back out of the queue.

How a new request finds its cached blocks:
1. When a request arrives, vLLM splits the prompt into 16-token blocks and computes a hash for each full block; each hash depends on the previous block's hash.
2. `get_computed_blocks` asks for the longest run of blocks, from the start, whose hashes are already stored.
3. It stops at the first miss, and the tokens in the matched blocks are skipped.
4. It never takes the very last prompt token from the cache (the model must recompute it to produce the first output word), which can mean redoing the whole last block.
5. A finished request's blocks go to the free queue and stay there, still findable by hash, until the memory is needed and they are evicted.

**Finished requests' blocks: kept until evicted, not deleted immediately** (`free_blocks` puts them in the free queue; `get_new_blocks`, line 681, pops from the front of that queue when new memory is needed).

## Caveats
- Single short runs; differences of a few ms or a few percent may be noise.
- This pod's KV pool was 155,872 tokens (Day 1-4 pods: 167,728) and its TPOT was ~10% higher (2.49-2.53 ms vs ~2.25 ms on Day 1 and Day 4). Reason not investigated.
- `prefix_test.py` sends token IDs directly to `/v1/completions`, so tokenizing cost is not in TTFT.

## Next
- Chunked prefill (Day 7): how a long prefill shares a step with ongoing decodes.
- Rerun the Day 1 sweep with caching off to measure its effect on those numbers.
