# c=1 (one request at a time), 4 requests, first 300 iterations

- GPU events: 107,717; summed kernel time: 674.5 ms; observed window: 707.3 ms
- GPU busy (union of intervals): 674.1 ms = 95.3% of window; idle: 33.2 ms
- Gaps between kernels: 107,092; median 0.1 us; p99 1.8 us; gaps > 20 us: 165 totalling 18.9 ms
- 293 decode iterations (`execute_context_0(0)_generation_1(1)`, avg 1.873 ms CUDA time each) and 3 prompt iterations

| Rank | Kernel | Class | Calls | Total ms | % of kernel time |
|---|---|---|---|---|---|
| 1 | `internal::gemvx::kernel<int, int,...` | GEMM | 33,112 | 436.09 | 64.7% |
| 2 | `flash::flash_fwd_splitkv_kernel<...` | attention | 8,204 | 59.30 | 8.8% |
| 3 | `flash::flash_fwd_splitkv_combine_kernel<...` | attention | 8,204 | 55.35 | 8.2% |
| 4 | `triton_red_fused_3` | other | 7,992 | 23.29 | 3.5% |
| 5 | `cunn_SoftMaxForward<4, float, ...` | other | 296 | 14.88 | 2.2% |

| Class | Total ms | % of kernel time |
|---|---|---|
| attention | 126.99 | 18.8% |
| GEMM | 440.25 | 65.3% |
| other | 107.23 | 15.9% |

The `aten::mm` operator (the lm_head matmul) took 97.0 ms over 296 calls (327.7 us each).
