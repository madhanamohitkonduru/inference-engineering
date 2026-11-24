# Day 14 Step 7: 2-GPU pods on Runpod (public pages read 2026-11-24; live availability NOT checked)

Sources: https://www.runpod.io/pricing (page says "Updated November 12, 2026"), https://www.runpod.io/gpu-models/rtx-4090, https://www.runpod.io/articles/guides/nvidia-h100, https://docs.runpod.io/api-reference-v2/pods/create-a-pod, https://docs.runpod.io/references/gpu-types.

## Prices (per GPU per hour, on demand; "from" = starting price on the page)
| GPU | VRAM | Secure Cloud | Community Cloud | Two GPUs, Secure (my arithmetic) |
|---|---|---|---|---|
| RTX 4090 | 24 GB | $0.89 | from $0.34 (from the 4090 page; the pricing page lists $0.89 only) | $1.78/hr |
| A100 SXM | 80 GB | $2.19 | from $1.39 | $4.38/hr |
| H100 PCIe | 80 GB | $2.89 | from $1.99 | $5.78/hr |
| H100 SXM | 80 GB | $4.39 | from $2.69 | $8.78/hr |
| H200 (SXM) | 141 GB | $5.29 | from $3.59 | $10.58/hr |

Pods can have more than one GPU: the create-pod API has a GPU `count` field (default 1, maximum 1024), and the RTX 4090 page says "Runpod supports multi-GPU pod configurations". The pricing page does not list pod sizes, so whether a 2x pod of a given type is in stock has to be checked in the console when deploying.

## Interconnect
- RTX 4090: no NVLink (NVIDIA's own page says "NVIDIA NVLink (SLI-Ready): No"; Runpod's 4090 page agrees). A 2x 4090 pod can only talk over PCIe.
- Discrepancy: Runpod's 4090 page lists "PCIe Gen5 x16, 63 GB/s", but NVIDIA's page lists PCI Express Gen 4 for the 4090. I trust NVIDIA (about 31.5 GB/s each way for Gen 4 x16); the real speed also depends on the host's slots, which a rental page does not show.
- H100 SXM: Runpod's H100 guide says it "supports NVLink 4.0 with 900 GB/s bidirectional bandwidth per GPU"; H100 PCIe: "Not supported (PCIe only)". That page (not NVIDIA) is the first source that states 900 GB/s is bidirectional, which supports the 450 GB/s-per-direction hypothesis; still not confirmed from NVIDIA. Whether the two SXM GPUs in one 2x pod are NVLinked to each other is not stated on any page I read; to check with `nvidia-smi topo -m` on the pod.

## What this means for Day 15 (my suggestion)
- PCIe case: 2x RTX 4090 (about $1.78/hr Secure). Qwen3-8B at TP=2 fits (fit table: 21 sequences of 8,192).
- NVLink case: 2x H100 SXM (about $8.78/hr Secure) for a short session; this is the only way to test the NVLink prediction. Check `nvidia-smi topo -m` shows NV links, not PIX/PHB.
- Cost estimate (arithmetic, not a quote): 1.5 hours of 2x 4090 is about $2.67, and 1 hour of 2x H100 SXM is about $8.78, before storage.
