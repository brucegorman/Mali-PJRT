"""Generate G57 tuning candidates and an FP16 activation-storage comparison."""

import argparse
from pathlib import Path


# Workgroup X/Y, per-thread M/N, and reduction K. Every candidate keeps at
# least one 16-lane Valhall warp and tiles the selected dimensions exactly.
TILES = {
    "light": (8, 2, 2, 4, 8),
    "balanced": (8, 4, 4, 4, 8),
    "reuse": (8, 4, 4, 8, 16),
}

# Exact row-major affine maps and logical M/N/K for the three dense products.
PRODUCTS = (
    ("forward", "d0, d2", "d2, d1", 4096, 1024, 1024),
    ("input_gradient", "d0, d2", "d1, d2", 4096, 1024, 1024),
    ("weight_gradient", "d2, d0", "d2, d1", 1024, 1024, 4096),
)


def tuning_spec(x, y, tm, tn, tk, *, shared=False):
    if shared:
        tiles = f"[[{tm * y}, {tn * x}, {tk}]]"
        pipeline = (f"SPIRVMatmulPromoteVectorize workgroup_size = [{x}, {y}, 1], "
                    "{pipeline_depth = 0 : i64, store_stage = 1 : i64}")
    else:
        tiles = f"[[{tm * y}, {tn * x}], [{tm}, {tn}], [0, 0, {tk}]]"
        pipeline = f"SPIRVBaseVectorize workgroup_size = [{x}, {y}, 1]"
    matchers = []
    for name, lhs, rhs, m, n, k in PRODUCTS:
        matchers.append(f"""  transform.named_sequence @match_{name}(%op: !transform.any_op {{transform.readonly}})
      -> (!transform.any_op, !transform.any_param) {{
    transform.iree.match.has_no_lowering_config %op : !transform.any_op
    %batch, %m, %n, %k = transform.iree.match.contraction %op,
      lhs_type = f16, rhs_type = f16, output_type = f32,
      indexing_maps = [affine_map<(d0, d1, d2) -> ({lhs})>,
                       affine_map<(d0, d1, d2) -> ({rhs})>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>] : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %m, [{m}] : !transform.param<i64>
    transform.iree.match.dims_equal %n, [{n}] : !transform.param<i64>
    transform.iree.match.dims_equal %k, [{k}] : !transform.param<i64>
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = {tiles}>,
      translation_info = #iree_codegen.translation_info<pipeline = {pipeline}>> -> !transform.any_param
    transform.yield %op, %config : !transform.any_op, !transform.any_param
  }}
""")
    rules = ",\n      ".join(f"@match_{name} -> @apply_config" for name, *_ in PRODUCTS)
    return f"""module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @apply_config(%op: !transform.any_op {{transform.readonly}},
                                         %config: !transform.any_param {{transform.readonly}}) {{
    transform.annotate %op "compilation_info" = %config : !transform.any_op, !transform.any_param
    transform.yield
  }}
{''.join(matchers)}
  transform.named_sequence @__kernel_config(%variant: !transform.any_op {{transform.consumed}})
      -> !transform.any_op attributes {{iree_codegen.tuning_spec_entrypoint}} {{
    %result = transform.foreach_match in %variant
      {rules}
      : (!transform.any_op) -> !transform.any_op
    transform.yield %result : !transform.any_op
  }}
}}
"""


def dot(result, left, right):
    return f"""    %{result} = stablehlo.dot_general %{left}, %{right},
      contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      : (tensor<4096x1024xf16>, tensor<1024x1024xf16>) -> tensor<4096x1024xf32>"""


def workload(storage):
    # Split producer and consumer at the activation boundary. The output is
    # a real device buffer retained by the caller between the two invocations.
    activation_type = f"tensor<4096x1024x{storage}>"
    if storage == "f16":
        producer = f"    %stored = stablehlo.convert %relu : (tensor<4096x1024xf32>) -> {activation_type}\n"
        stored = "stored"
        consumer = ""
        input_name = "activation"
    else:
        producer = ""
        stored = "relu"
        consumer = "    %half = stablehlo.convert %activation : (tensor<4096x1024xf32>) -> tensor<4096x1024xf16>\n"
        input_name = "half"
    return f"""module {{
  func.func @gemm(%x: tensor<4096x1024xf16>, %w: tensor<1024x1024xf16>) -> tensor<4096x1024xf32> {{
{dot('product', 'x', 'w')}
    return %product : tensor<4096x1024xf32>
  }}
  func.func @produce(%x: tensor<4096x1024xf16>, %w: tensor<1024x1024xf16>) -> {activation_type} {{
{dot('product', 'x', 'w')}
    %zero = stablehlo.constant dense<0.0> : tensor<4096x1024xf32>
    %relu = stablehlo.maximum %product, %zero : tensor<4096x1024xf32>
{producer}    return %{stored} : {activation_type}
  }}
  func.func @consume(%activation: {activation_type}, %w: tensor<1024x1024xf16>) -> tensor<4096x1024xf32> {{
{consumer}{dot('product', input_name, 'w')}
    return %product : tensor<4096x1024xf32>
  }}
  func.func @input_gradient(%dy: tensor<4096x1024xf16>, %w: tensor<1024x1024xf16>) -> tensor<4096x1024xf32> {{
    %dx = stablehlo.dot_general %dy, %w, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT]
      : (tensor<4096x1024xf16>, tensor<1024x1024xf16>) -> tensor<4096x1024xf32>
    return %dx : tensor<4096x1024xf32>
  }}
  func.func @weight_gradient(%x: tensor<4096x1024xf16>, %dy: tensor<4096x1024xf16>) -> tensor<1024x1024xf32> {{
    %dw = stablehlo.dot_general %x, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT]
      : (tensor<4096x1024xf16>, tensor<4096x1024xf16>) -> tensor<1024x1024xf32>
    return %dw : tensor<1024x1024xf32>
  }}
}}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, tile in TILES.items():
        (args.output / f"{name}.mlir").write_text(tuning_spec(*tile), encoding="utf-8")
    (args.output / "shared.mlir").write_text(tuning_spec(*TILES["reuse"], shared=True), encoding="utf-8")
    for storage in ("f32", "f16"):
        (args.output / f"activation-{storage}.mlir").write_text(workload(storage), encoding="utf-8")


if __name__ == "__main__":
    main()
