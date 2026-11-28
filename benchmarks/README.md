# Mali 基准

## Valhall 矩阵乘和激活存储

`--valhall` 比较两层连续的混合精度矩阵乘以及输入梯度、权重梯度两个乘法。
矩阵乘使用 FP16 输入和 FP32 结果。第一层的 ReLU 可以把激活写成 FP16，第二层直接读取该 buffer。
`4096 × 1024` 的跨层激活从 16 MiB 降到 8 MiB。两种路径在第二次矩阵乘前都使用 FP16，适合隔离跨层存储和转换的影响。
生成的 FP16 路径把转换融合进生产者的 ReLU dispatch，消费端省掉了独立的转换 dispatch。
编译和读回校验位于计时之外。 [设备探针](../pjrt/smoke.py)

固定版本的 IREE G57 默认配置随形状变化。对本探针的三个 FP16 矩阵乘，生成的默认 workgroup 为 `8 × 2`，tile 为 `4 × 64 × 8`。
下面四个候选用同一套精确的 dtype、M/N/K 与 indexing map 选择三个 dispatch，覆盖前向和两种转置梯度。

| 配置 | Workgroup X × Y | Tile M × N × K | 调整 |
|---|---|---|---|
| `light` | `8 × 2` | `4 × 32 × 8` | 每线程输出列从 8 降到 4，减少活跃累加值 |
| `balanced` | `8 × 4` | `16 × 32 × 8` | 每组 32 个线程，每线程输出 `4 × 4` |
| `reuse` | `8 × 4` | `16 × 64 × 16` | 更大的输出 tile 与 K tile，测试访存复用 |
| `shared` | `8 × 4` | `16 × 64 × 16` | 使用 IREE 的 Workgroup 共享内存 promotion 流水线 |

生成候选后，用已构建的 PJRT bundle 在手机上逐一复跑。`MALI_IREE_TUNING_SPEC` 作用于新编译，`--valhall` 通过独立进程在手机上编译。

```powershell
python benchmarks/mali_tuning.py build/valhall-tuning
python pjrt/build.py --zig build/pjrt-deps/zig-x86_64-windows-0.15.2/zig.exe --output build/pjrt-valhall
$deviceDir = '/data/local/tmp/jax-mali-train.qCUUWZ'
adb push build/pjrt-valhall "$deviceDir/"
adb push build/valhall-tuning/light.mlir build/valhall-tuning/balanced.mlir build/valhall-tuning/reuse.mlir build/valhall-tuning/shared.mlir "$deviceDir/"
adb shell -T sh "$deviceDir/pjrt-valhall/run_on_phone.sh" --valhall
foreach ($profile in @('light', 'balanced', 'reuse', 'shared')) {
    adb shell -T "MALI_IREE_TUNING_SPEC=$deviceDir/$profile.mlir sh $deviceDir/pjrt-valhall/run_on_phone.sh --valhall"
}
```

在电脑上可以先用固定版本的 IREE 编译生成的 StableHLO fixture，核对目标配置和存储类型。

```powershell
$out = 'build/valhall-tuning'
.venv/Scripts/iree-compile.exe "$out/activation-f16.mlir" --iree-input-type=stablehlo `
    --iree-hal-target-device=vulkan --iree-vulkan-target=mali-g57 `
    "--iree-codegen-transform-dialect-library=$out/shared.mlir" `
    "--iree-hal-dump-executable-configurations-to=$out/config-shared" -o "$out/shared.vmfb"
rg -n 'workgroup_size|lowering_config|writeonly:tensor' "$out/config-shared"
```

固定版本的 Windows Vulkan runtime 已通过默认配置和四套候选的数值抽查，覆盖两种激活存储。
四套配置保留为可选项。每线程寄存器占用和缓存命中率需要硬件计数器才能确认。

