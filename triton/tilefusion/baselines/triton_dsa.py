from typing import Optional

import torch
import triton
import triton.language as tl
try:
    from tilelang_dsa import sparse_mla_fwd_interface
    HAS_TILELANG = True
except ImportError:
    HAS_TILELANG = False
import numpy as np

spar_mla_fwd_configs = [
    triton.Config({'num_stages': 4, 'num_warps': 8}),
    # triton.Config({'num_stages': 2, 'num_warps': 4}),
]

@triton.autotune( # Decorate the kernel
    configs=spar_mla_fwd_configs,
    key=['K', 'is_causal'],
)

@triton.jit
def triton_sparse_mla_fwd(
    q,
    kv,
    indices,
    sm_scale: tl.constexpr,
    output,
    lse,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kvb, stride_kvg, stride_kvn, stride_kvd,
    stride_tb, stride_tg, stride_tm, stride_tt, # topk，对应indices
    stride_ob, stride_oh, stride_om, stride_od,
    stride_lb, stride_lh, stride_lm,
    B: tl.constexpr,
    SQ: tl.constexpr, # seqlen
    SKV: tl.constexpr,
    K: tl.constexpr, # topk
    D: tl.constexpr, # QKV dim
    TD: tl.constexpr, # tail dim
    DP: tl.constexpr,
    TDP: tl.constexpr,
    H: tl.constexpr, # q_head_dim
    G: tl.constexpr, # group_size
    VG: tl.constexpr, # H/G KV groups
    BK: tl.constexpr, 
    BH: tl.constexpr,
    # BD: tl.constexpr, # block of output dim
    is_causal: tl.constexpr
):
    i_b, i_sq, i_gbh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_g, i_bh = i_gbh // G, i_gbh % G
    q_base = q + i_b*stride_qb + i_sq*stride_qm + i_gbh*(BH*stride_qh) # 留两个维度，后面逐块载入
    tq_base = q_base + D*stride_qd
    kv_base = kv + i_b*stride_kvb + i_g*stride_kvg
    tkv_base = kv_base + D*stride_kvd
    t_base = indices + i_b*stride_tb + i_sq*stride_tm + i_g*stride_tg
    o_base = output + i_b*stride_ob + i_sq*stride_om + i_gbh*(BH*stride_oh)
    l_base = lse + i_b*stride_lb + i_sq*stride_lm + i_gbh*(BH*stride_lh)

    offs_h = tl.arange(0, BH)
    offs_d = tl.arange(0, DP)
    offs_td = tl.arange(0, TDP)
    offs_od = tl.arange(0, DP)
    offs_t = tl.arange(0, BK)
    mask_h = i_bh * BH + offs_h < G
    mask_d = offs_d < D
    mask_td = offs_td < TD
    mask_od = mask_d

    q_ptr = q_base + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q_msk = mask_h[:, None] & mask_d[None, :]
    q_blk = tl.load(q_ptr, q_msk, other=0.0).to(tl.float16)

    tq_ptr = tq_base + offs_h[:, None] * stride_qh + offs_td[None, :] * stride_qd
    tq_msk = mask_h[:, None] & mask_td[None, :]
    tq_blk = tl.load(tq_ptr, tq_msk, other=0.0).to(tl.float16)
    
    max_log = tl.full([BH], float('-inf'), dtype=tl.float16)
    sum_exp = tl.full([BH], 1.0, dtype=tl.float16)
    acc = tl.zeros([BH, DP], dtype=tl.float16)
    qk = tl.zeros([BH, BK], dtype=tl.float16)

    log_scale: tl.constexpr = sm_scale * 1.44269504

    # max_col = max(0, i_sq + SKV - SQ) if is_causal else SKV-1
    max_col = i_sq if is_causal else SQ-1

    NK = tl.cdiv(K, BK)
    for ck in range(NK):
        t_ptr = (BK * ck + offs_t) * stride_tt
        t_msk = t_ptr < K
        t_ptr += t_base
        kv_ids = tl.load(t_ptr, t_msk, other=-1)
        mask_ids = (kv_ids <= max_col) & (kv_ids >= 0)

        # if mask_ids.max(0) > 0:
        if ck * BK <= max_col:
            kv_ptr = kv_base + offs_d[:, None]*stride_kvd + kv_ids[None, :]*stride_kvn
            kv_msk = mask_d[:, None] & mask_ids[None, :]
            kv_blk = tl.load(kv_ptr, kv_msk, other=0.0).to(tl.float16) #[DP, BK]
            # kv_blk = tl.full([DP, BK], 0.001, dtype=tl.float16)

            tkv_ptr = tkv_base + offs_td[:, None]*stride_kvd + kv_ids[None, :]*stride_kvn
            tkv_msk = mask_td[:, None] & mask_ids[None, :]
            tkv_blk = tl.load(tkv_ptr, tkv_msk, other=0.0).to(tl.float16) #[TDP, BK]
            # tkv_blk = tl.full([TDP, BK], 0.001, dtype=tl.float16)

            qk = tl.dot(q_blk, kv_blk, out_dtype=tl.float16)
            qk = tl.dot(tq_blk, tkv_blk, qk, out_dtype=tl.float16) * log_scale
            # qk = tl.dot(tq_blk, tkv_blk, qk, out_dtype=tl.float16) * sm_scale

            qk = tl.where(mask_ids[None, :], qk, float('-inf')) #[BH, BK]

            new_max = tl.maximum(max_log, tl.max(qk, axis=1))
            exp_qk = tl.math.exp2(qk - new_max[:, None]).to(tl.float16)
            # exp_qk = tl.math.exp(qk - new_max[:, None]).to(tl.float16)
            sum_qk = tl.sum(exp_qk, axis=1)
            alpha = tl.math.exp2(max_log - new_max).to(tl.float16)
            # alpha = tl.math.exp(max_log - new_max).to(tl.float16)
            sum_exp = sum_exp*alpha + sum_qk
            acc = acc*alpha[:, None]
            acc = tl.dot(exp_qk, kv_blk.trans(), acc, out_dtype=tl.float16) #[BH, BK] @ [BK, DP] = [BH, DP]
            
            max_log = new_max.to(tl.float16)

    out_vals = acc / sum_exp[:, None]
    o_ptr = o_base + offs_h[:, None] * stride_oh + offs_od[None, :] * stride_od
    o_msk = mask_h[:, None] & mask_od[None, :]
    # o_msk &= tl.zeros_like(o_msk)
    tl.store(o_ptr, out_vals.to(q_blk.dtype), o_msk)

    fin_log = max_log + tl.math.log2(sum_exp.to(tl.float32)) # 返回 lse / ln2
    # fin_log *= 0.69314718
    # fin_log = max_log + tl.math.log(sum_exp.to(tl.float32))
    # fin_log *= 1.44269504 # 返回 lse / ln2
    l_ptr = l_base + offs_h * stride_lh
    l_msk = mask_h
    tl.store(l_ptr, fin_log.to(q_blk.dtype), l_msk)
    
    # if (i_b == 0) & (i_gbh == (H//BH)-1) & (i_sq == SQ-1):
    #     print("kv_ids", kv_ids)


@triton.jit
def triton_sparse_mla_fwd_mlir(
    q, kv, indices, output,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kvb, stride_kvh, stride_kvn, stride_kvd,
    stride_tb, stride_th, stride_tm, stride_tt,
    stride_ob, stride_oh, stride_om, stride_od,
    H, K,
    sm_scale: tl.constexpr,
):
    i_b = tl.program_id(0)
    i_sq = tl.program_id(1)
    i_h_block = tl.program_id(2)

    offs_h = i_h_block * 64 + tl.arange(0, 64)
    mask_h = offs_h < H

    q_base = q + i_b * stride_qb + i_sq * stride_qm
    offs_d_512 = tl.arange(0, 512)
    offs_d_64 = 512 + tl.arange(0, 64)
    
    q_ptr_1 = q_base + offs_h[:, None] * stride_qh + offs_d_512[None, :] * stride_qd
    q_ptr_2 = q_base + offs_h[:, None] * stride_qh + offs_d_64[None, :] * stride_qd
    
    q_blk_1 = tl.load(q_ptr_1, )
    q_blk_2 = tl.load(q_ptr_2, )

    t_base = indices + i_b * stride_tb + i_sq * stride_tm
    kv_base = kv + i_b * stride_kvb
    
    sum_exp = tl.zeros([64, 1], dtype=tl.float32)
    acc = tl.zeros([64, 512], dtype=tl.float32)

    loop_end = tl.minimum(K, (i_sq // 64 + 1) * 64)

    for ck in range(0, loop_end, 64):
        offs_k = ck + tl.arange(0, 64)
        t_ptr = t_base + offs_k * stride_tt
        
        # Match IR: No mask for kv_ids load
        kv_ids = tl.load(t_ptr)
        
        # Load KV - Match MLIR: No masks
        kv_ptr_1 = kv_base + offs_d_512[:, None] * stride_kvd + kv_ids[None, :] * stride_kvn
        kv_part1 = tl.load(kv_ptr_1)
        
        kv_ptr_2 = kv_base + offs_d_64[:, None] * stride_kvd + kv_ids[None, :] * stride_kvn
        kv_part2 = tl.load(kv_ptr_2)
        
        # Dot products
        qk = tl.dot(q_blk_1, kv_part1)
        qk = tl.dot(q_blk_2, kv_part2, qk)
        
        # 1. Truncate to F16 (Crucial for MLIR match)
        qk = qk.to(tl.float16)
        
        # 2. Scale in F16
        qk = qk * sm_scale
        
        # 3. Masking in F16 (Addition style)
        # 修正: 使用 float("-inf") 而非 0xFC00
        mask = i_sq < offs_k
        mask_val = tl.where(mask[None, :], float("-inf"), 0.0).to(tl.float16)
        qk = qk + mask_val
        
        # 4. Ext back to F32 for Exp2
        p = tl.math.exp2(qk.to(tl.float32) * 1.44269502)
        
        # Update
        sum_exp += tl.sum(p, axis=1)[:, None]
        acc = tl.dot(p.to(tl.float16), tl.trans(kv_part1).to(tl.float16), acc)

    acc = acc / sum_exp
    
    o_base = output + i_b * stride_ob + i_sq * stride_om + i_h_block * 64 * stride_oh
    o_ptr = o_base + tl.arange(0, 64)[:, None] * stride_oh + tl.arange(0, 512)[None, :] * stride_od
    tl.store(o_ptr, acc.to(tl.float16), )

def triton_sparse_mla_fwd_interface(q, kv, indices, sm_scale=None, return_p_sum: bool = False, d_v=512):
    is_causal = True
    assert return_p_sum == False, "This kernel file is for fwd only"
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    B, SQ, H, DT = q.shape
    _, S, VG, _ = kv.shape

    # assert DT == 576, "you should assign dim otherwise"
    D = d_v

    assert kv.shape[-1] == DT
    TD = DT - D
    DP = triton.next_power_of_2(D)
    TDP = triton.next_power_of_2(TD)
    assert kv.shape[0] == B
    _, _, _, K = indices.shape
    assert indices.shape == (B, SQ, VG, K)
    G = H//VG
    if sm_scale is None:
        sm_scale = DT ** -0.5
    BH = 64
    NH = G // BH
    BK = 32
    # BD = min(256, max(triton.next_power_of_2(DT), 16))
    # ND = triton.cdiv(DT, BD)
    output = torch.zeros((B, SQ, H, D), device=q.device, dtype=q.dtype)
    lse = torch.full((B, SQ, H), float('-inf'), device=q.device, dtype=q.dtype)
    grid = (B, SQ, VG*NH) # (SQ//BQ, B*H)
    # kernel = sparse_mla_fwd[grid](heads, dim, tail_dim, topk, kv_group, sm_scale, is_causal)
    triton_sparse_mla_fwd[grid](
        q, kv, indices, sm_scale,
        output, lse,
        q.stride(0), q.stride(2), q.stride(1), q.stride(3), # [B, H, SQ, DT]
        kv.stride(0), kv.stride(2), kv.stride(1), kv.stride(3), # [B, VG, SKV, DT]
        indices.stride(0), indices.stride(2), indices.stride(1), indices.stride(3), # [B, VG, SQ, K]
        output.stride(0), output.stride(2), output.stride(1), output.stride(3), # [B, H, SQ, D]
        lse.stride(0), lse.stride(2), lse.stride(1), # [B, H, SQ]
        B, SQ, S, K, D, TD, DP, TDP, H, G, VG,
        BK,
        BH,
        # BD,
        is_causal
    )
    
    # sparse_mla_fwd[grid](q, kv, indices, output)
    return output, lse


def triton_sparse_mla_fwd_mlir_interface(q, kv, indices, sm_scale=None, d_v=512):
    # q: [B, H, SQ, DT]
    # kv: [B, SKV, HKV, DT]
    # indices: [B, HKV, SQ, K]
    B, H, SQ, DT = q.shape
    _, SKV, HKV, _ = kv.shape
    _, _, _, K = indices.shape

    if sm_scale is None:
        sm_scale = DT**-0.5

    output = torch.zeros((B, H, SQ, d_v), device=q.device, dtype=q.dtype)

    # Grid: (Batch, Seq, Head_Blocks)
    grid = (B, SQ, triton.cdiv(H, 64))

    triton_sparse_mla_fwd_mlir[grid](
        q, kv, indices, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        kv.stride(0), kv.stride(1), kv.stride(2), kv.stride(3),
        indices.stride(0), indices.stride(1), indices.stride(2), indices.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        H, K, sm_scale,
        num_warps=8, num_stages=4
    )
    return output


def ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale=None, is_casual=True, d_v=512):
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)
    b, sq, h, dim_q = q.shape
    b, sk, g, _ = kv.shape

    dim = d_v
    # assert kv.shape[-1] == 576, "you should assign dim otherwise"
    # dim = 512
    k = kv
    v = kv[..., :dim]

    b, _, _, dim_v = v.shape
    g_index = g
    h_index = h // g
    compressed_casual_mask = torch.arange(
        0, sq, dtype=torch.int32, device="cuda").view(-1, 1) >= torch.arange(
            1 - 1, sk * 1, 1, dtype=torch.int32, device="cuda").view(1, -1)

    mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
    mask = mask[..., :-1]
    mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
    mask[:, :, :1 - 1, 0] = True
    mask = mask.view(b, g_index, 1, sq, sk)

    q = q.view(b, sq, g, -1, dim_q)
    score = torch.einsum("bmghd,bngd->bghmn", q, k)
    sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
    p = score.softmax(dim=-1)
    p = p.view(b, g_index, h_index, -1, sq, sk)
    p = p.view(b, g, -1, sq, sk)
    o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
    o = o.reshape(b, sq, h, dim_v)
    return o.to(torch.float16)


def test_sparse_mla_fwd(B=1,
                        S=4096,
                        SKV=4096,
                        H=128,
                        HKV=1,
                        DQK=576,
                        DV=512,
                        topk=2048,
                        dtype=torch.float16):
    torch.random.manual_seed(0)
    q = torch.randn((B, S, H, DQK), dtype=dtype, device="cuda").requires_grad_(True)
    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device="cuda").requires_grad_(True)

    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[b, t, h, :len(i_i)] = i_i

    # print("indices", indices)
    from copy import deepcopy
    q2 = deepcopy(q)
    kv2 = deepcopy(kv)
    indices2 = deepcopy(indices)

    q3 = q.transpose(1, 2).contiguous()  # [B, H, S, DQK]
    kv3 = kv.transpose(1, 2).contiguous()  # [B, HKV, S, DQK]
    indices3 = indices.transpose(1, 2).contiguous()  # [B, HKV, S, topk]

    triton_out, triton_lse = triton_sparse_mla_fwd_interface(q, kv, indices, d_v=DV)
    print("triton done \n triton lse tensor: \n", triton_lse)
    print()
    mlir_out = triton_sparse_mla_fwd_mlir_interface(q3, kv3, indices3, d_v=DV)
    print("mlir triton done")
    print()
    
    if HAS_TILELANG:
        tilelang_out, tilelang_lse = sparse_mla_fwd_interface(q2, kv2, indices2, d_v=DV)
        print("tilelang done \n tilelang lse tensor: \n", tilelang_lse)
        print()

        diff_out = torch.mean(torch.abs(triton_out - tilelang_out))
        diff_lse = torch.mean(torch.abs(triton_lse - tilelang_lse))
        print("std_out: ", diff_out)
        print("std_lse: ", diff_lse)
        print()

        # Compare MLIR out with Tilelang out
        mlir_out_trans = mlir_out.transpose(1, 2)
        diff_mlir = torch.mean(torch.abs(mlir_out_trans - tilelang_out))
        print("mlir_std_out: ", diff_mlir)
        print()
    else:
        print("Tilelang not available, skipping comparison.")
        # Compare MLIR out with Triton out
        mlir_out_trans = mlir_out.transpose(1, 2)
        diff_mlir = torch.mean(torch.abs(mlir_out_trans - triton_out))
        print("mlir_vs_triton_std_out: ", diff_mlir)
        print()

    def fn():
        return triton_sparse_mla_fwd_interface(q, kv, indices, d_v = DV)

    from tilelang.profiler import do_bench

    ms = do_bench(
        fn,
        rep=100,
        warmup=250,
    )
    print(f"Triton kernel Average time: {ms:.3f} ms")
    print("Triton kernel fwd io bandwidth = ", (B * S * DQK * topk * 2) / (ms * 1e-3) / 1e12)
    print("Triton kernel fwd tflops = ", (B * S * (DQK + DV) * topk * 2 * H) / (ms * 1e-3) / 1e12)
    print()

    if HAS_TILELANG:
        def fn2():
            return sparse_mla_fwd_interface(q, kv, indices)

        ms = do_bench(
            fn2,
            rep=100,
            warmup=250,
        )
        print(f"Tilelang kernel Average time: {ms:.3f} ms")
        print("Tilelang kernel fwd io bandwidth = ", (B * S * DQK * topk * 2) / (ms * 1e-3) / 1e12)
        print("Tilelang kernel fwd tflops = ", (B * S * (DQK + DV) * topk * 2 * H) / (ms * 1e-3) / 1e12)
        print()


    def fn3():
        return triton_sparse_mla_fwd_mlir_interface(q3, kv3, indices3, d_v=DV)

    ms = do_bench(
        fn3,
        rep=100,
        warmup=250,
    )
    print(f"MLIR Triton kernel Average time: {ms:.3f} ms")
    print("MLIR Triton kernel fwd io bandwidth = ", (B * S * DQK * topk * 2) / (ms * 1e-3) / 1e12)
    print("MLIR Triton kernel fwd tflops = ", (B * S * (DQK + DV) * topk * 2 * H) / (ms * 1e-3) / 1e12)
    print()

if __name__ == "__main__":
    # test_sparse_mla_fwd(
    #     B=1, S=128, SKV=1024, H=32, HKV=1, DQK=256+32, DV=256, topk=64, dtype=torch.float32)
    test_sparse_mla_fwd(
        B=1, S=4096, SKV=4096, H=128, HKV=1, DQK=576, DV=512, topk=2048, dtype=torch.float16)
