# Day 5: continuous batching and the vLLM scheduler

Server: vLLM 0.32.0, Qwen3-0.6B, RTX 4090, `scripts/serve.sh` (`--max-model-len 4096 --gpu-memory-utilization 0.85`) plus `--max-num-seqs N`.

## 1. Orca paper (my answers)
1. **Why short requests wait in old-style batching:** a batch is returned only when every request in it is done, so short requests wait for the longest one, and new requests cannot join until the batch ends.
2. **What iteration-level scheduling changes:** the scheduler and execution engine interact after every iteration (decode step), so finished requests leave immediately and waiting requests take the free seats at the next step.
3. **What the two flags limit:** `max_num_seqs` limits how many sequences are in flight at once (seats); `max_num_batched_tokens` limits the tokens processed in a single step (work per step; matters most for prefill, where a prompt costs its full length).

## 2. Experiment: 1 long (400 tokens) + 3 short (20 tokens) requests, sent 0.05 s apart
Script: [`batching_test.py`](batching_test.py) (uses `ignore_eos` so lengths are exact). Raw output: [`results/raw_output.txt`](results/raw_output.txt). Two runs per setting.

Finish time in seconds (run 1 / run 2):

| Request | `--max-num-seqs 1` | `--max-num-seqs 4` |
|---|---|---|
| long (400 tok) | 0.92 / 0.90 | 0.96 / 0.95 |
| short1 (20 tok) | 0.97 / 0.95 | 0.13 / 0.13 |
| short2 (20 tok) | 1.04 / 1.01 | 0.18 / 0.19 |
| short3 (20 tok) | 1.09 / 1.06 | 0.22 / 0.22 |

- **Finding:** with 1 seat the short requests queue behind the long one (each needs ~0.05 s of work but finishes at ~1 s). With 4 seats they join the running batch and finish 5-8x sooner; the long request slows by ~5%.
- The 0.05 s spacing between short finish times at 4 seats comes from the script's 0.05 s send stagger.
- **Hypothesis (not measured):** the small slowdown of the long request is the cost of sharing each step with 3 others.
- `--max-num-seqs 16` was skipped: with only 4 requests it cannot differ from 4. Only 4 requests and 2 runs per setting, so treat the figures as indicative.

## 3. Reading the scheduler (`vllm/v1/core/sched/scheduler.py`, v0.32.0)
Line numbers refer to the installed 0.32.0 file.
- Waiting queue created at line 223, running list at 231; `schedule()` starts at 591.
- Seat limit: `--max-num-seqs` -> `max_num_running_reqs` (129) -> `max_num_active_reqs` (132), checked at 923.
- Token budget: `--max-num-batched-tokens` -> `max_num_scheduled_tokens` (137) -> `token_budget` (611).

One scheduling step, in five sentences:
1. At the start of a step, `schedule()` sets a token budget equal to `max_num_batched_tokens` (0 if the server is paused).
2. It first goes through the running list and schedules the next token for each running request, spending budget on each.
3. If budget is left, it looks at the waiting queue and tries to admit requests.
4. It stops admitting when the running count reaches the seat limit or the budget runs out.
5. An admitted request is popped from the waiting queue, appended to the running list, and its prompt is processed in the same step (budget spent at line 1366). Removal of finished requests was not in the lines read.

Also seen but not studied: lines ~798-834 handle preemption (removing a running request when memory runs out).
