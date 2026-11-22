# Text of the issue as posted

Posted 2026-11-22 as https://github.com/vllm-project/vllm/issues/69279 (title: `[Bug]: --hf-overrides '{"rope_theta": N}' is accepted silently but has no effect on Qwen3 (the value lives under rope_parameters)`).

Sections: environment (by hand; `collect_env` was not available because the GPU pod no longer existed), bug description with the command, observed (flip rate 0%, KL 0, perplexity change 0 versus no override; vLLM's config object still showed `rope_parameters.rope_theta` = 1,000,000), expected (apply or warn), control (nested spelling: flip 5.06%, KL 0.0083), where it seems to happen (code reading of main at b492a09), transformers counterpart, why it matters, related issues (#58675 / #68807, #61231), search terms used, two possible directions (warn, or document the nested form), an offer to send a small PR only if a maintainer agrees, and a one-line AI-assistance disclosure.

The numbers come from `../day-12-correctness-gate/` (see `gate_output_flat_vs_nested.txt` here, produced by running that folder's `gate.py` on its captures).
