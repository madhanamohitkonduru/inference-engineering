Output of `scripts/trace_top_kernels.py` on the c=32 PyTorch-profiler trace (copied from the terminal; raw trace not committed).

- GPU events: 131,891; summed kernel time: 1,368.5 ms; observed window: 1,433.8 ms
- GPU busy (union of intervals): 1,367.9 ms = 95.4% of window; idle: 65.9 ms
- Gaps between kernels: 126,718; median 0.1 us; p99 2.2 us; gaps > 20 us: 186 totalling 15.6 ms

| Rank | Kernel | Class | Calls | Total ms | % of kernel time |
|---|---|---|---|---|---|
| 1 | `void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, ...` | attention | 6,832 | 603.79 | 44.1% |
| 2 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_3...` | GEMM | 13,832 | 196.24 | 14.3% |
| 3 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64...` | GEMM | 704 | 106.39 | 7.8% |
| 4 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_1...` | GEMM | 13,832 | 105.27 | 7.7% |
| 5 | `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_tn` | GEMM | 868 | 84.73 | 6.2% |

| Class | Total ms | % of kernel time |
|---|---|---|
| attention | 677.80 | 49.5% |
| GEMM | 529.50 | 38.7% |
| other | 161.23 | 11.8% |
