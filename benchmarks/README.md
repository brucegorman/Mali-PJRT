# Mali benchmarks on Android

These benchmarks run on a vivo Y36m with a Mali-G57 MC2 GPU and Android 13. The phone compiles and executes the JAX training graph. Qwen decode offloads eligible Q4 projections through Vulkan and runs the rest of the model on the CPU. [Qwen build and data](data/qwen-y36m/SOURCE.json) · [PJRT build and data](data/valhall-y36m/results.json)

## PJRT training and matrix multiplication

The 96-step training run uses a `32 → 1024 → 1024 → 1` network and a batch size of 4,096. Every run uses the same data, initial weights, and sample order. Timing starts after one warm-up and includes execution and device synchronization. In mixed precision, the forward and backward matrix multiplications take FP16 inputs and request FP32 accumulation and output. The model parameters, optimizer state, and loss use FP32.

| Configuration | CPU, 96 steps | Mali, 96 steps |
|---|---:|---:|
| FP32 control | 62.560 s | 105.837 s |
| Mixed precision, default tiles | 63.645 s | 83.340 s |
| Mixed precision, `reuse-gradients` | — | **64.441 s** |
| Mixed precision, `reuse` in forward and backward | — | 74.838 s |

Changing only the gradient tiles cut mixed-precision Mali training time by **22.7%**. All runs reduced held-out cross-entropy from `0.9799` to `0.3545` and reached `0.9831` NDCG@5. [Training measurements](data/valhall-y36m/results.json) · [FP32 and GEMM controls](data/valhall-y36m/compute-controls.json)

The standalone GEMM runs 512 multiplications of `[4096, 1024] × [1024, 1024]` with resident FP16 inputs and FP32 output. Mali reached **63.88 GFLOP/s** (68.844 s) and the CPU reached **57.01 GFLOP/s** (77.139 s). Both outputs passed a NumPy check. [Per-block measurements](data/valhall-y36m/compute-controls.json)

### Valhall GEMM and activation storage

The probe ran the default and four tile/workgroup candidates twice in forward and reverse order. Each process reported the median of seven calls after warm-up. Both forward GEMMs use FP16 inputs. The two forward columns vary only the storage type for the ReLU result between layers. [Probe](../pjrt/smoke.py) · [Tuning candidate generator](mali_tuning.py)

| Mali configuration | Forward, FP32 activations | Forward, FP16 activations | Two backward GEMMs |
|---|---:|---:|---:|
| Default | 224.2 ms | 221.9 ms | 791.1 ms |
| `light` | 388.1 ms | 383.8 ms | 1,028.1 ms |
| `balanced` | 235.3 ms | 257.5 ms | 458.4 ms |
| `reuse` | 376.3 ms | 377.9 ms | 459.3 ms |
| `shared` | 701.3 ms | 787.6 ms | 780.3 ms |
| `reuse-gradients` | 224.3 ms | 221.9 ms | **456.7 ms** |

- **Activation storage.** In the probe, writing the ReLU result in FP16 reduced the intermediate buffer from 16 MiB to 8 MiB and cut default-tile forward time by about 1%. In the compiled training graph, the two hidden GEMMs already fuse bias and ReLU and emit FP16 activations for the next layer. Each also emits an FP32 pre-activation for backpropagation.
- **Valhall tiling.** `MALI_IREE_TUNING_SPEC` enables the selected tuning spec. It changes two matching mixed-precision gradients from `8 × 2` workgroups and `4 × 64 × 8` tiles to `8 × 4` workgroups and `16 × 64 × 16` tiles. The two backward GEMMs take about 42% less time in the probe. All four forward candidates were slower than the default. The FP32 control and standalone GEMM use the compiler's default tiles. [Compiler entry point](../pjrt/executable.c) · [Gradient tuning spec](data/valhall-y36m/reuse-gradients.mlir)
- **Dispatch count.** The on-device IREE compiler emits 21 compute dispatches and 12 buffer copies per optimizer step in both the default and gradient-tuned graphs. The count comes from the generated HAL loop.

### Standalone GEMM profiling

A separate Streamline capture repeated the 512-call Mali GEMM in **58.096 s** at **75.70 GFLOP/s**. Across 57 steady one-second bins, FMA pipe utilization was **65.25%**, and **99.994%** of warps used the full register allocation. Shader-side load/store read beats were nonzero. The L2 lookup and aggregate external-bus counters returned zero under load, so this capture leaves the cache hit rate and DRAM bandwidth unresolved. [Counter totals and sample selection](data/valhall-y36m/streamline-gemm.json)

Mali Offline Compiler 2026.5.0 reported **63 work registers**, **50% estimated occupancy**, **zero spill bytes**, and **0% 16-bit arithmetic** for the standalone GEMM shader. Its longest-path estimate identified the load/store pipe as the limiting shader pipeline. That estimate covers the shader's load/store work, which can include accesses served by cache. [Offline shader analysis](data/valhall-y36m/streamline-gemm.json)

## Qwen Q4 decode

The model uses Q4_0 weights. The Mali projections use Q8_0 inputs, integer products, and FP32 scaling and accumulation. With shader profiling disabled, the 96-token run measured a steady-state p50 of **56.38 ms/token** on the CPU and **259.19 ms/token** on Mali hybrid. All 14,585,856 teacher-forced logits matched bit for bit. [96-token measurements](data/qwen-y36m/shape-profile/long-aligned/compare.json)

| Figure | Measurement |
|---|---|
| [Per-token latency and cold start](figures/qwen-decode.png) | Matched 96-token runs showing time to first token, the first decode interval, and steady-state latency. |
| [Weight and activation layouts](figures/qwen-layouts.png) | Two ABBA comparisons that isolate Q4 weight alignment and Q8 activation alignment. |
| [Time by projection shape](figures/qwen-projections.png) | Shader, submission, and host call time in the profiled 24-token run. |
| [Kernels with real weights](figures/qwen-kernels.png) | GPU time for separate FP32 dequantization and fused GEMV with shared-memory and subgroup reductions. |

Q4 weight alignment reduced steady-state p50 from `353.54` to `315.36 ms/token`. In a separate ABBA comparison, Q8 activation alignment reduced it from `314.36` to `260.88 ms/token`. Both prompt-level comparisons and CPU fallback with the shader unavailable passed. [Q4 weight results](data/qwen-y36m/weight-layout) · [Q8 activation results](data/qwen-y36m/activation-layout) · [Correctness checks](data/qwen-y36m/verified)

## Reproduce the benchmarks

Prepare the Android NDK, Zig, a Python environment with NumPy, the PJRT workspace on the phone, and a sibling `mali-q4-decode` checkout with its pinned GGUF. Plotting requires Matplotlib. [Phone setup](../pjrt/README.md)

```powershell
.venv/Scripts/python.exe benchmarks/run_y36m.py build/y36m-rerun
python benchmarks/plot_qwen.py --qwen-results build/y36m-rerun/qwen --out build/y36m-rerun/figures
```

Use `--suite qwen` or `--suite valhall` to run one suite. Valhall's `--resume` reuses successful runs when the device, bundle, and tuning specs match. To rebuild the four figures from the checked-in data, run `python benchmarks/plot_qwen.py`. [Benchmark runner](run_y36m.py)

Use `--suite power` or the standalone collector for power measurements. Whole-phone energy at the battery requires a valid discharge window. An external meter can record power separately. [Power collector](capture_power.py) · [Energy plots](plot_energy.py)
