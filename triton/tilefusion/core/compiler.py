from triton._C.libtriton import ir, passes

import tempfile


def fuse_kernels_in_ttir(
    mlir_file_path: str,
    producer_kernel_name: str,
    consumer_kernel_name: str,
    producer_output_arg_idx: list,
    consumer_input_arg_idx: list,
    enable_debug: bool = True,
) -> str:
    """
    对 TTIR 模块执行 kernel 融合 pass，并返回融合后的 TTIR 文本。
    
    Args:
        mlir_file_path: 输入的 .mlir 文件路径（包含两个 kernel）
        gemm_kernel_name: GEMM kernel 的函数名
        softmax_kernel_name: Softmax kernel 的函数名
        gemm_output_arg_idx: GEMM kernel 中输出张量的参数索引（如 C_ptr 是 arg#2）
        softmax_input_arg_idx: Softmax kernel 中输入张量的参数索引（如 x_ptr 是 arg#0）
        enable_debug: 是否启用 pass manager debug 输出
    
    Returns:
        str: 融合后的 TTIR 模块文本
    """
    # 创建 IR 上下文
    ctx = ir.context()
    ir.load_dialects(ctx)
    
    # 解析 MLIR 模块
    mod = ir.parse_mlir_module(mlir_file_path, ctx)
    
    # 构建 Pass Manager
    pm = ir.pass_manager(ctx)
    if enable_debug:
        pm.enable_debug()
    
    # 添加融合 pass：格式为 "gemm_arg_idx->softmax_arg_idx"
    pairs = [f"{p}->{c}" for p, c in zip(producer_output_arg_idx, consumer_input_arg_idx)]
    mapping = ",".join(pairs)
    passes.ttir.add_fuse_kernels(pm, producer_kernel_name, consumer_kernel_name, mapping)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)  
    passes.ttir.add_optimize_fused_loops(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    # 执行 pass pipeline
    pm.run(mod, "fuse_ttir")
    
    # 返回结果
    fused_ttir_text = str(mod)
    with tempfile.NamedTemporaryFile("w", suffix=".ttir", delete=False) as f2:
        f2.write(fused_ttir_text)
        fused_path = f2.name
    # print(f"Fused module written to: {fused_path}")
    return fused_path, fused_ttir_text



def opt_kernels_in_ttir(
    mlir_file_path: str,
    enable_debug: bool = True,
    direction: str = "0",
    targetaxis: int = 0,
    mask_arg_index: int = 17,
    enable_rle: bool = False,
) -> str:

    # 创建 IR 上下文
    ctx = ir.context()
    ir.load_dialects(ctx)
    
    # 解析 MLIR 模块
    mod = ir.parse_mlir_module(mlir_file_path, ctx)
    
    # 构建 Pass Manager
    pm = ir.pass_manager(ctx)
    if enable_debug:
        pm.enable_debug()
    passes.ttir.add_pointer_opt(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.ttir.add_pointer_opt(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    # passes.ttir.add_triton_fuse_softmax_gemm(pm)
    # passes.common.add_canonicalizer(pm)
    # passes.common.add_cse(pm)
    passes.ttir.add_triton_flash_attention_fusion(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.ttir.add_pointer_opt(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.ttir.add_pointer_opt(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.ttir.add_triton_dead_store_elimination(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    if enable_rle:
        passes.ttir.add_triton_redundant_load_elimination(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
    passes.ttir.add_triton_hoist_invariant_loads(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    # 预缩放 Q：将 mulf(dot(Q,K), scale) 改写为 dot(Q*scale, K)
    passes.ttir.add_triton_pre_scale_query(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.ttir.add_triton_softmax_micro_opts(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    # 由调用方传入 direction（例如 "0" | "1" | "2"），用于控制 OptimizeCausalMask 行为
    if direction != "2":     
        passes.ttir.add_triton_optimize_causal_mask(pm, direction, mask_arg_index, targetaxis)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
    passes.ttir.add_triton_early_load_v(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.ttir.add_triton_fuse_mask_into_dot(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    # Drop mask addition before exp chain (semantics-changing, for perf A/B)
    passes.ttir.add_triton_eliminate_cast_roundtrip(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.ttir.add_triton_add_divisibility_attr(pm)
    passes.ttir.add_triton_combine_scalar_mults(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.common.add_licm(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_combine(pm)
    passes.ttir.add_reorder_broadcast(pm)
    passes.common.add_cse(pm)
    passes.common.add_symbol_dce(pm)
    passes.ttir.add_loop_unroll(pm)
    passes.ttir.add_triton_licm(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_combine(pm)
    passes.ttir.add_reorder_broadcast(pm)
    passes.common.add_cse(pm)
    passes.common.add_symbol_dce(pm)
    pm.run(mod, "opt_ttir")
    
    # 返回结果
    fused_ttir_text = str(mod)
    with tempfile.NamedTemporaryFile("w", suffix=".ttir", delete=False) as f2:
        f2.write(fused_ttir_text)
        fused_path = f2.name
    # print(f"Fused module written to: {fused_path}")
    return fused_path, fused_ttir_text



def opt_kernels_in_ttir_test(
    mlir_file_path: str,
    enable_debug: bool = True,
) -> str:

    # 创建 IR 上下文
    ctx = ir.context()
    ir.load_dialects(ctx)
    
    # 解析 MLIR 模块
    mod = ir.parse_mlir_module(mlir_file_path, ctx)
    
    # 构建 Pass Manager
    pm = ir.pass_manager(ctx)
    if enable_debug:
        pm.enable_debug()
    passes.ttir.add_pointer_opt(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.ttir.add_pointer_opt(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    # passes.ttir.add_triton_fuse_softmax_gemm(pm)
    # passes.common.add_canonicalizer(pm)
    # passes.common.add_cse(pm)
    passes.ttir.add_triton_flash_attention_fusion(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.ttir.add_pointer_opt(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.ttir.add_pointer_opt(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.ttir.add_triton_dead_store_elimination(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.ttir.add_triton_hoist_invariant_loads(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    # # # 预缩放 Q：将 mulf(dot(Q,K), scale) 改写为 dot(Q*scale, K)
    passes.ttir.add_triton_pre_scale_query(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.ttir.add_triton_softmax_micro_opts(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.ttir.add_triton_optimize_causal_mask(pm, '0', 19, 1,)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.ttir.add_triton_early_load_v(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.ttir.add_triton_fuse_mask_into_dot(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    # Drop mask addition before exp chain (semantics-changing, for perf A/B)
    passes.ttir.add_triton_eliminate_cast_roundtrip(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.ttir.add_triton_add_divisibility_attr(pm)
    passes.ttir.add_triton_combine_scalar_mults(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.common.add_licm(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_combine(pm)
    passes.ttir.add_reorder_broadcast(pm)
    passes.common.add_cse(pm)
    passes.common.add_symbol_dce(pm)
    passes.ttir.add_loop_unroll(pm)
    passes.ttir.add_triton_licm(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_combine(pm)
    passes.ttir.add_reorder_broadcast(pm)
    passes.common.add_cse(pm)
    passes.common.add_symbol_dce(pm)
    # 执行 pass pipeline
    pm.run(mod, "opt_ttir")
    
    # 返回结果
    fused_ttir_text = str(mod)
    with tempfile.NamedTemporaryFile("w", suffix=".ttir", delete=False) as f2:
        f2.write(fused_ttir_text)
        fused_path = f2.name
    # print(f"Fused module written to: {fused_path}")
    return fused_path, fused_ttir_text

