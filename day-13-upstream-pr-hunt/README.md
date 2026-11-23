# Day 13: Upstream PR hunt

Goal: find one upstream contribution target I can finish, check that nobody else already has it, and open an issue with evidence. No PR today, by design.
*Finding* = checked or measured; *hypothesis* = not verified. Everything was checked between 2026-11-20 and 2026-11-22; GitHub state changes quickly, so every row carries the date it was read.

## Result in one paragraph
Every issue I examined that looked like a "first contribution" already had a PR, a claimant, or a maintainer saying no. The target I chose is therefore not an existing issue but my own Day 12 finding: **vLLM 0.32.0 accepts `--hf-overrides '{"rope_theta": N}'` on Qwen3 without any message and applies nothing.** I searched for an existing report and found none, then opened **https://github.com/vllm-project/vllm/issues/69279** (2026-11-22) with evidence and an offer to send a small PR only if a maintainer agrees. **Outcome (read 2026-11-23): a vLLM maintainer (hmellor) closed the issue as "user error".** His reason: since Transformers v5, `rope_theta` belongs inside `rope_parameters`, so the nested form is the correct way to override it. Another contributor had already opened PR #69323 to make the flat key work (it says it fixes #69279); the same maintainer closed that PR too ("This is not usage we want to encourage"). Nobody replied to the idea of a warning. So the claim ended without a PR, which is a valid result of the plan: no PR was written, so no work was thrown away.

## Chosen target
- Issue: https://github.com/vllm-project/vllm/issues/69279 (text in `issue_69279_text.md`).
- Evidence (*finding*, measured on Day 12, Qwen3-0.6B BF16, vLLM 0.32.0, RTX 4090, 50 passages x 512 tokens; raw gate output in `gate_output_flat_vs_nested.txt`):

| Run (reference `new_bf16_c1`) | Flip rate | Mean KL | Perplexity change |
|---|---|---|---|
| flat `{"rope_theta": 500000}` | 0.00000% | 0.00000 | 0.00000% |
| nested `{"rope_parameters": {"rope_type": "default", "rope_theta": 500000}}` | 5.06% | 0.00831 | +0.64% |

- Why the flat key does nothing (*finding* for the Day 12 run: vLLM's config object still showed `rope_parameters.rope_theta` = 1,000,000; *code reading, not executed on main*: `vllm/transformers_utils/config.py` applies the override with `config.update(...)`, logged only at debug level, then `patch_rope_parameters` runs and leaves `rope_parameters` unchanged).
- What "done" looked like: a maintainer chooses between (1) a warning when a flat override key does not change the resolved config, or (2) a docs note on the nested form; then a small PR with a test.
- Needs a GPU: no for the change itself (config-level); the original measurement needed one.
- **What happened:** issue #69279 closed as user error (the nested form is the supported one); PR #69323, which applied the flat key instead of warning, closed with "not a usage we want to encourage". The warning option was not taken up or discussed.
- *Finding (independent confirmation):* PR #69323's description measured the same behaviour on another machine (RTX 5090 laptop, transformers 5.23.0): with the flat key on the base commit, top-1 changed at 0 of 459 positions (identical to no override); with the nested spelling, 17 of 459. It also explains the mechanism more exactly than my code reading did: `patch_rope_parameters` calls `standardize_rope_params`, which does `rope_parameters.setdefault("rope_theta", ...)`, so a value already in `rope_parameters` wins and the flat one is dropped.
- *Hypothesis (untested):* a startup warning for override keys that change nothing may still be wanted by other maintainers, but the one who answered made clear that the flat form is not supported, and I did not push back.

## Shortlist (checked 2026-11-20 to 2026-11-22)
| # | Candidate | Assignee / linked PR | Last activity | GPU? | What "done" would be | Verdict |
|---|---|---|---|---|---|---|
| 1 | vllm #69279 (my own report, flat `rope_theta` ignored) | none; competing PR #69323 (closed unmerged) | opened 2026-11-22, closed by a maintainer as user error by 2026-11-23 | no | warning or docs note, chosen by a maintainer | **Chosen; closed by maintainer** |
| 2 | vllm #61231 EAGLE draft silently uses RoPE base 10000 | none assigned; **3 open PRs** (#61320, #61321, #61334) | commit referencing it 2 days before 2026-11-20 | no | log + warn on startup | Disqualified: three people already built fixes |
| 3 | vllm #58675 `hf-overrides` dict drops `rope_theta` | none assigned; fix PR #68807 opened about 3 days before 2026-11-22 | PR created ~2026-11-19, no review yet | no | keep checkpoint's `rope_theta` | Disqualified: has a fix PR. Related, not the same bug as #1 |
| 4 | vllm #28498 port conflict (RL) | none | label `stale`, opened Nov 2025 | likely (multi-process RL) | unclear | Disqualified: stale and hard to reproduce |
| 5 | vllm #25750 Whisper tracking | none | opened Sep 2025 | yes | a list of features, no single "done" | Disqualified: umbrella issue |
| 6 | vllm #23957 Phi4Flash in V1 | none; PR #44963 linked | opened Aug 2025 | yes | port a whole model | Disqualified: large, has a PR |
| 7 | guidellm #1592 timings use `time.time()` | none | maintainer replied 4 days before 2026-11-20 | no | switch to monotonic clock | Disqualified: maintainer said no to an external PR for this ("very disruptive") |
| 8 | guidellm #1565 ITL biased low (EOS token) | none; PR #1566 | opened 2 weeks earlier | no | align ITL/TPOT token counts | Disqualified: has a PR |
| 9 | guidellm #1202 streaming `JSONDecodeError` (U+2028) | none; PR #1203 open since Sep 29 | labelled `community contribution` Oct 9 | no | spec-compliant SSE line splitter | Disqualified: PR already waiting |
| 10 | guidellm #1522 empty metrics shown as 0.0 | none; PR #1523 | 3 weeks old | no | show "--" | Disqualified: has a PR |
| 11 | guidellm #1379 warmup by completion order | none; claimed in a comment; PRs #1532, #1570 | active | no | maintainer decision first | Disqualified: claimed, PRs exist, design undecided |
| 12 | guidellm #1529 console tables printed twice | none; PRs #1530 and #1574 | 3 weeks old | no | print once | Disqualified: two competing PRs |
| 13 | Lead: async scheduling disabled by n-gram speculation | n/a | n/a | n/a | n/a | Not a target: I found no such logic in main's `vllm/config/vllm.py` (*finding*, code reading). Whether it was fixed after 0.32.0 is *hypothesis*; I did not read release notes |
| 14 | Lead: `prompt_logprobs` vs prefix caching | n/a | n/a | n/a | n/a | Not a target: handled on main by `Request.get_prefix_cache_hit_limit` and a docs page (*finding*, code reading) |

Counts (one-time, 2026-11-20): vLLM had 14 open `good first issue` issues, 3 without assignee (rows 4 to 6, all disqualified). guidellm has the label but no open issue carries it; it has 59 open issues, 45 without assignee, and a `community contribution` label (2 open issues).

*Finding:* "no assignee" does not mean "free". In both projects nobody uses the assignee field; people claim by opening a PR within days. The signals that matter are the linked PRs (the Development box), the comments, and whether a maintainer has already answered.

## vLLM `check_logprobs_close` (Step 3)
File: `tests/models/utils.py`, function `check_logprobs_close` (read on main at b492a09, 2026-11-20).

**What does it compare?** For each prompt it takes two lists of generated outputs (for example vLLM versus Hugging Face, or two vLLM configurations). Each output has generated token ids and the top-N logprobs at each generated position. It walks the positions in order. While both sides generated the same token, nothing is asserted (unless `always_check_logprobs=True`). At the first position where the sampled tokens differ, it asserts that each side's chosen token appears in the other side's top-N logprobs, and then stops checking that prompt (`break`), because after a divergence the two sequences are no longer comparable. If prompt logprobs are given for both sides, the top-k token-id sets must match at every prompt position. Text that differs only produces a warning.

**How does it decide "close enough", and how does that differ from my gate?**
- It decides with yes/no membership checks: is my token inside your top-N list? A `near_tie_tol` option allows a token that is within a tolerance of the lowest included logprob. There is no size of difference anywhere: a token at rank 2 with probability 0.45 and a token at rank 19 with probability 0.001 both pass.
- My gate is teacher-forced (both models score the same fixed text, so one early difference does not change what follows), looks at the whole distribution (KL), perplexity and the top-1 flip rate, and takes its thresholds from a measured noise floor with bootstrap intervals.
- Theirs is a correctness test that must be fast and stable on many models, so a coarse rule makes sense there. Mine answers a different question: how big is the change, and is it larger than the noise?
- *Hypothesis* (not tested, and would need a maintainer's design decision): a magnitude-aware helper next to `check_logprobs_close` could catch drift that membership checks miss.

## Self-check answers
1. **Why check for an existing issue or open PR before starting?** So I do not spend days on something already taken or already refused. Evidence from this day: #61231 had three competing PRs within days, and only one can be merged; guidellm #1379 was claimed with two PRs; #1592 was refused by a maintainer. Checking is also polite, because maintainers' review time is limited.
2. **Why is a small PR more likely to be merged?** It is quicker to review and easier to test, it has less to go wrong, and a reviewer can say yes without a design discussion. Size is not the only factor, though: PR #69323 was tiny and was closed within a day, because the maintainer disagreed with the idea, not the code. That is why the plan says to ask before writing a PR. vLLM's own rules ask for an RFC issue above about 500 lines, and a maintainer on guidellm called a wide change "very disruptive" for an outside contributor.
3. **What do I do if CI fails?** Read the failing job's log first and decide whether my change caused it. If yes, fix it and push. If the same tests fail on the base commit (vLLM PR #68807 reported hub-dependent tests failing both before and after its change), say so in a comment. If it is unrelated or flaky, use `/ci retry` (vLLM allows it once the PR is approved or labelled ready) or ask a maintainer. Do not ignore a red check and do not force-push over reviewers' comments.

## Predictions versus measurement
| My prediction (before the search) | What I found | Verdict |
|---|---|---|
| Many `good first issue` items get assigned quickly because someone claims them | In vLLM the three unassigned ones are old or hard; in guidellm nobody uses assignees and the label has no open issues. Claiming happens through PRs and comments | Wrong in mechanism (*hypothesis* for why: the easy ones were already fixed) |
| guidellm is the most approachable repo | Many small, GPU-free bugs, but every one I opened had a PR, a claimant, or a refusal | Partly right on size and GPU needs, wrong on availability |
| My own finding would be about support across different GPUs | The finding was a silently ignored override key (from Day 12), not GPU-related | Wrong |
| (not predicted) How a maintainer would respond to my own finding | Closed within about a day as "user error", with a competing PR opened and closed in between | Lesson: a real, reproducible observation can still be "working as intended" 

## What I did not do
- No pull request, and no fork (a fork is only needed once a maintainer agrees to a PR).
- No config-level reproduction on vLLM 0.32.0 or main without a GPU (the GPU pod had been terminated), and no `collect_env` output. The "where it happens" paragraph is code reading of main, not executed.
- I read #68807's description but not its diff, so "it does not change flat-key behaviour" is *hypothesis*, based on its description and test table.
- I did not read the other related issues I found (#37435, #56311, #60165, #54094).
- guidellm: only the newest page of unassigned issues (about 25 of 45); #1616, #1588, #1582, #1531, #1456, #1390, #1378 and #1337 were not opened. SGLang and vLLM's recipes repository were not checked at all.
- The capture `../day-12-correctness-gate/captures/rope_flat_ignored_c1.json.gz` carries the internal label `bad_rope_mild_c1` and a note describing `rope_theta=500000`; it was saved under that name and renamed afterwards. The Day 12 README and `server_flags_check.txt` record that it is the flat run. I did not attach it to the issue.
- I did not reply to the maintainer after the close, and did not argue for the warning. One maintainer's answer is a signal, not a project-wide policy (*hypothesis*), but a second comment would not have changed the closure.
- Before posting I did not check Transformers v5's migration notes, which would have shown that flat `rope_theta` is deprecated there. I checked the vLLM side only (code, issues, PRs).

## What I learned
- A bug report can be accurate and still be closed: "it does nothing" was true, but the maintainer's answer was "it is not supposed to do anything; use the nested form". Next time, before posting, read the upstream dependency's migration notes (here Transformers v5) and ask whether the behaviour is deprecated rather than broken.
- Posting an issue first and writing no PR protected me: another contributor wrote a PR for the same idea and had it closed with no discussion.
- Most of the value here was in the checking: the shortlist, the existing-PR search, the contribution rules, and the independent confirmation of my measurement.

## Next steps
Nothing to do on #69279. For the next contribution attempt: pick something a maintainer has marked as wanted (for example a `help wanted` or `community contribution` item with no PR), read the dependency's changelog first, and comment to claim before writing code.
