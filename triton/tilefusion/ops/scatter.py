import torch
import triton
import triton.language as tl
from tilefusion.utils.utils import DEVICE


@triton.jit
def scatter4d_kernel_scalar(
    out_ptr, index_ptr,
    H, M, N, K,
    stride_ob, stride_oh, stride_om, stride_on,
    stride_ib, stride_ih, stride_im, stride_ik,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Each program handles a tile of rows (BLOCK_M) for one (b, h), and writes K positions along dim=-1.
    We perform sequential masked stores across K lanes to emulate torch.scatter last-wins semantics.
    Shapes:
        out:   [B, H, M, N]
        index: [B, H, M, K]
    Grid: (ceil_div(M, BLOCK_M), B*H)
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = (pid_bh // H).to(tl.int32)
    h = (pid_bh % H).to(tl.int32)

    m0 = pid_m * BLOCK_M
    offs_m = m0 + tl.arange(0, BLOCK_M)
    mmask = offs_m < M

    offs_k = tl.arange(0, BLOCK_K)
    kmask = offs_k < K

    # Base pointers for (b,h) and (m,k)
    base_out = b * stride_ob + h * stride_oh + offs_m * stride_om  # [BLOCK_M]
    base_idx = b * stride_ib + h * stride_ih + offs_m * stride_im  # [BLOCK_M]

    # Load indices [BLOCK_M, BLOCK_K]
    idx_ptrs = index_ptr + base_idx[:, None] + offs_k[None, :] * stride_ik
    mask_mk = mmask[:, None] & kmask[None, :]
    idxs = tl.load(idx_ptrs, mask=mask_mk, other=-1)

    # Compute output pointers [BLOCK_M, BLOCK_K]
    out_ptrs = out_ptr + base_out[:, None] + idxs * stride_on
    valid = mask_mk & (idxs >= 0) & (idxs < N)

    # Broadcast scalar value to matrix [BLOCK_M, BLOCK_K]
    val_mat = tl.full([BLOCK_M, BLOCK_K], 0, tl.float32)

    # Sequential per-lane store along K to ensure last-wins for duplicates within a row
    for i in tl.static_range(BLOCK_K):
        lane_mask = valid & (offs_k[None, :] == i)
        tl.store(out_ptrs, val_mat, mask=lane_mask)



def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def scatter_4d_triton(
    input_tensor: torch.Tensor,
    indices: torch.Tensor,
    src,
    block_k: int | None = None,
    block_m: int | None = 32,
) -> torch.Tensor:
    """Scatter on dim=-1 for 4D inputs.

    Args:
      input_tensor: [B, H, M, N] (CUDA)
      indices: [B, H, M, K] or [B, M, K] (CUDA, int)
      src: scalar or tensor with shape [B, H, M, K] or [B, M, K]
      block_k: optional power-of-two block for K unrolling; defaults to next_pow2(K) capped at 256

    Returns:
      output: [B, H, M, N]
    """
    assert input_tensor.is_cuda and indices.is_cuda, "input and indices must be CUDA tensors"
    assert input_tensor.dim() == 4, "input must be [B,H,M,N]"

    B, H, M, N = input_tensor.shape

    K = indices.shape[-1]

    # Decide BLOCK_K
    if block_k is None:
        block_k = min(256, _next_pow2(int(K)))
    # Decide BLOCK_M
    if block_m is None:
        block_m = 32

    # Prepare output
    out = input_tensor

    # 2D grid over (M tiles, B*H)
    def _cdiv(a, b):
        return (a + b - 1) // b
    grid = (_cdiv(M, int(block_m)), B * H)
    scatter4d_kernel_scalar[grid](
        out, indices,
        H, M, N, K,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        indices.stride(0), indices.stride(1), indices.stride(2), indices.stride(3),
        BLOCK_M=int(block_m), BLOCK_K=int(block_k),
    )
    
    return out


def _ttir_of_scatter(M: int, N: int, K: int, BLOCK_M: int, BLOCK_N: int, B: int, H: int):
    """Generate TTIR IR for scatter kernels.

    When use_scalar=True, returns TTIR of the scalar variant; otherwise tensor variant.
    """
    # Allocate example tensors
    out = torch.empty((B, H, M, N), device=DEVICE, dtype=torch.float16)
    indices = torch.randint(0, N, (B, H, M, K), device=DEVICE, dtype=torch.int32)

    # Match BLOCK_K policy with runtime
    def _next_pow2_py(x: int) -> int:
        return 1 if x <= 1 else 1 << (x - 1).bit_length()

    grid = ((M + BLOCK_M - 1) // BLOCK_M, B * H)

    triton_kernel = scatter4d_kernel_scalar[grid](
        out, indices,
        H, M, N, K,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        indices.stride(0), indices.stride(1), indices.stride(2), indices.stride(3),
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_N,
    )

    return triton_kernel.asm['ttir']

