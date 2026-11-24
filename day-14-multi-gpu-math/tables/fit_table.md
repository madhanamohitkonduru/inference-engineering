Fit table (8,192-token sequences, gpu_memory_utilization 0.9, activation 2.0 GiB assumed). no = doesn't fit; tight = fits but fewer than 4 sequences; FITS = at least 4 sequences. Cell = mark (sequences of 8,192).

| Model | GPU | TP=1 | TP=2 | TP=4 | TP=8 |
|---|---|---|---|---|---|
| qwen3-8b | rtx4090 | tight (3) | FITS (21) | FITS (55) | FITS (125) |
| qwen3-8b | a100-80 | FITS (48) | FITS (110) | FITS (235) | FITS (483) |
| qwen3-8b | h100-80 | FITS (48) | FITS (110) | FITS (235) | FITS (483) |
| qwen3-8b | h200 | FITS (97) | FITS (208) | FITS (430) | FITS (874) |
| qwen3-32b | rtx4090 | no | no | FITS (8) | FITS (47) |
| qwen3-32b | a100-80 | FITS (4) | FITS (39) | FITS (109) | FITS (249) |
| qwen3-32b | h100-80 | FITS (4) | FITS (39) | FITS (109) | FITS (249) |
| qwen3-32b | h200 | FITS (31) | FITS (94) | FITS (219) | FITS (468) |
| qwen2.5-72b | rtx4090 | no | no | no | FITS (8) |
| qwen2.5-72b | a100-80 | no | tight (1) | FITS (57) | FITS (169) |
| qwen2.5-72b | h100-80 | no | tight (1) | FITS (57) | FITS (169) |
| qwen2.5-72b | h200 | no | FITS (45) | FITS (145) | FITS (345) |
