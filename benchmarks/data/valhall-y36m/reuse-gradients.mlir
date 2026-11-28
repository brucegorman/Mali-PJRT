module attributes {transform.with_named_sequence} {
  transform.named_sequence @apply_config(%op: !transform.any_op {transform.readonly},
                                         %config: !transform.any_param {transform.readonly}) {
    transform.annotate %op "compilation_info" = %config : !transform.any_op, !transform.any_param
    transform.yield
  }
  transform.named_sequence @match_input_gradient(%op: !transform.any_op {transform.readonly})
      -> (!transform.any_op, !transform.any_param) {
    transform.iree.match.has_no_lowering_config %op : !transform.any_op
    %batch, %m, %n, %k = transform.iree.match.contraction %op,
      lhs_type = f16, rhs_type = f16, output_type = f32,
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>] : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %m, [4096] : !transform.param<i64>
    transform.iree.match.dims_equal %n, [1024] : !transform.param<i64>
    transform.iree.match.dims_equal %k, [1024] : !transform.param<i64>
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[16, 64], [4, 8], [0, 0, 16]]>,
      translation_info = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [8, 4, 1]>> -> !transform.any_param
    transform.yield %op, %config : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @match_weight_gradient(%op: !transform.any_op {transform.readonly})
      -> (!transform.any_op, !transform.any_param) {
    transform.iree.match.has_no_lowering_config %op : !transform.any_op
    %batch, %m, %n, %k = transform.iree.match.contraction %op,
      lhs_type = f16, rhs_type = f16, output_type = f32,
      indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>] : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %m, [1024] : !transform.param<i64>
    transform.iree.match.dims_equal %n, [1024] : !transform.param<i64>
    transform.iree.match.dims_equal %k, [4096] : !transform.param<i64>
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[16, 64], [4, 8], [0, 0, 16]]>,
      translation_info = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [8, 4, 1]>> -> !transform.any_param
    transform.yield %op, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @__kernel_config(%variant: !transform.any_op {transform.consumed})
      -> !transform.any_op attributes {iree_codegen.tuning_spec_entrypoint} {
    %result = transform.foreach_match in %variant
      @match_input_gradient -> @apply_config,
      @match_weight_gradient -> @apply_config
      : (!transform.any_op) -> !transform.any_op
    transform.yield %result : !transform.any_op
  }
}
