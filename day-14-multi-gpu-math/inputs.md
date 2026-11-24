# Day 14 inputs (read 2026-11-23)

## Model shapes and checkpoint sizes
Raw files in `raw/` (downloaded from `https://huggingface.co/<repo>/raw/main/config.json` and `model.safetensors.index.json`). Qwen3-0.6B has one weights file and no index; its size is the file's `x-linked-size` header (1,503,300,328 bytes), which matches Day 3.

| Model | layers | hidden | attn heads | KV heads | head_dim | checkpoint bytes (`total_size`) | tied embeddings |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 28 | 1024 | 16 | 8 | 128 | 1,503,300,328 (file size) | yes |
| Qwen3-8B | 36 | 4096 | 32 | 8 | 128 | 16,381,470,720 | no |
| Qwen3-32B | 64 | 5120 | 64 | 8 | 128 | 65,524,246,528 | no |
| Qwen2.5-72B-Instruct | 80 | 8192 | 64 | 8 | not in config (hidden / heads = 128) | 145,412,407,296 | no |

Other fields: vocab 151,936 (Qwen3) and 152,064 (Qwen2.5-72B); dtype bfloat16.

## GPU specs (NVIDIA pages; what each page says, not what I assume)
| GPU / variant | Memory | Memory bandwidth | Interconnect as listed | Source |
|---|---|---|---|---|
| GeForce RTX 4090 | 24 GB GDDR6X, 384-bit, 21 Gbps | 1008 GB/s | PCI Express Gen 4; NVLink: No | nvidia.com RTX 4090 page; Ada architecture whitepaper (images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf, Appendix A) |
| A100 80GB | 80 GB | 2,039 GB/s (SXM; see caveat) | NVLink 600 GB/s; PCIe Gen4 64 GB/s | A100 80GB datasheet (nvidia.com .../a100-80gb-datasheet-update-nvidia-us-1521051-r2-web.pdf) |
| H100 SXM | 80 GB | 3.35 TB/s | NVLink 900 GB/s; PCIe Gen5 128 GB/s | nvidia.com/en-us/data-center/h100 |
| H200 SXM | 141 GB | 4.8 TB/s | NVLink 900 GB/s; PCIe Gen5 128 GB/s | nvidia.com/en-us/data-center/h200 |

Caveats:
- A100: the datasheet table came out garbled when I read it (three value columns, 40GB and 80GB mixed). 2,039 GB/s is the figure the text ties to the 80GB model; I did not confirm which of SXM or PCIe it belongs to. A100 80GB PCIe may have a different bandwidth. To verify before using it for an exact claim.
- NVIDIA's pages do not say whether the NVLink or PCIe figures are per direction or both directions combined. My own arithmetic for PCIe 4.0 x16: 16 GT/s x 16 lanes x 128/130 / 8 = about 31.5 GB/s per direction, so the listed 64 GB/s is both directions added (*finding*, arithmetic from the PCIe 4.0 rate; the 16 GT/s figure is from PCI-SIG coverage). For NVLink, 900 GB/s on H100/H200 is most likely both directions combined (about 450 GB/s each way); that is a *hypothesis* until I read the Hopper architecture whitepaper.

Update (2026-11-24): Runpod's H100 guide (not NVIDIA) says the H100 SXM NVLink 4.0 figure is "900 GB/s bidirectional bandwidth per GPU", which supports the hypothesis above that it is about 450 GB/s each way. See `pod_options.md`.
