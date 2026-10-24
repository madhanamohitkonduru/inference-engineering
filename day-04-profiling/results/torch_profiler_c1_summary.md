Output of `scripts/trace_top_kernels.py` on the c=1 PyTorch-profiler trace (copied from the terminal; raw trace not committed, 5 MB).

- GPU events: 107,717; summed kernel time: 674.5 ms; observed window: 707.3 ms
- GPU busy (union of intervals): 674.1 ms = 95.3% of window; idle: 33.2 ms
- Gaps between kernels: 107,092; median 0.1 us; p99 1.8 us; gaps > 20 us: 165 totalling 18.9 ms

| Rank | Kernel | Class | Calls | Total ms | % of kernel time |
|---|---|---|---|---|---|
| 1 | `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int,...` | GEMM | 33,112 | 436.09 | 64.7% |
| 2 | `void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, ...` | attention | 8,204 | 59.30 | 8.8% |
| 3 | `void flash::flash_fwd_splitkv_combine_kernel<Flash_fwd_kernel_traits<1...` | attention | 8,204 | 55.35 | 8.2% |
| 4 | `triton_red_fused_3` | other | 7,992 | 23.29 | 3.5% |
| 5 | `void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, ...` | other | 296 | 14.88 | 2.2% |

| Class | Total ms | % of kernel time |
|---|---|---|
| attention | 126.99 | 18.8% |
| GEMM | 440.25 | 65.3% |
| other | 107.23 | 15.9% |

From PyTorch's own summary table (`profiler_out.txt`): `execute_context_0(0)_generation_1(1)` = 293 decode iterations, 1.873 ms CUDA time each;
`execute_context_1(8)_generation_0(0)` = 3 iterations (4.75 ms each); `aten::mm` (LM head) = 296 calls, 327.7 us average.
