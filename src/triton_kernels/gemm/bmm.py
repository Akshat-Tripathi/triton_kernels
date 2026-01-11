import torch
import triton
import triton.language as tl


@triton.jit
def bmm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    stride_a_b,
    stride_a_m,
    stride_a_k,
    stride_b_b,
    stride_b_k,
    stride_b_n,
    stride_c_b,
    stride_c_m,
    stride_c_n,
    is_fp16: bool,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_blocks = tl.num_programs(1)
    n_blocks = tl.num_programs(2)

    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, m_blocks, n_blocks, GROUP_SIZE)

    if is_fp16:
        a_ptr = a_ptr.to(tl.pointer_type(tl.float16))
        b_ptr = b_ptr.to(tl.pointer_type(tl.float16))
        c_ptr = c_ptr.to(tl.pointer_type(tl.float16))

    offs_m = tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
    offs_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, K, BLOCK_K):
        offs_k = tl.arange(0, BLOCK_K) + k

        mask_k = offs_k < K

        offs_a = (
            (pid_b * stride_a_b)
            + (offs_m[:, None] * stride_a_m)
            + (offs_k[None, :] * stride_a_k)
        )
        offs_b = (
            (pid_b * stride_b_b)
            + (offs_k[:, None] * stride_b_k)
            + (offs_n[None, :] * stride_b_n)
        )

        mask_a = mask_m[:, None] & mask_k[None, :]
        mask_b = mask_k[:, None] & mask_n[None, :]

        a = tl.load(a_ptr + offs_a, mask_a)
        b = tl.load(b_ptr + offs_b, mask_b)

        acc = tl.dot(a, b, acc)

    offs_c = (
        (pid_b * stride_c_b)
        + (offs_m[:, None] * stride_c_m)
        + (offs_n[None, :] * stride_c_n)
    )

    mask_c = mask_m[:, None] & mask_n[None, :]

    if is_fp16:
        tl.store(c_ptr + offs_c, acc.to(tl.float16), mask_c)
    else:
        tl.store(c_ptr + offs_c, acc, mask_c)


# a, b, c are tensors on the GPU
def bmm(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    BATCH: int,
    M: int,
    N: int,
    K: int,
):
    # a: (B, M, K), b: (B, K, N), c: (B, M, N)
    # TODO: Autotune
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    GROUP_SIZE = 4

    stride_a_b, stride_a_m, stride_a_k = a.stride()
    stride_b_b, stride_b_k, stride_b_n = b.stride()
    stride_c_b, stride_c_m, stride_c_n = c.stride()

    grid = (BATCH, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    bmm_kernel[grid](
        a,
        b,
        c,
        stride_a_b,
        stride_a_m,
        stride_a_k,
        stride_b_b,
        stride_b_k,
        stride_b_n,
        stride_c_b,
        stride_c_m,
        stride_c_n,
        a.dtype == torch.float16,
        M,
        N,
        K,
        BLOCK_M,  # ty:ignore[invalid-argument-type]
        BLOCK_N,  # ty:ignore[invalid-argument-type]
        BLOCK_K,  # ty:ignore[invalid-argument-type]
        GROUP_SIZE,  # ty:ignore[invalid-argument-type]
    )
    return c


if __name__ == "__main__":
    B = 32
    M = 256
    N = 256
    K = 256

    a = torch.randn(
        (B, M, K),
        dtype=torch.float32,
        device="cuda",
    )
    b = torch.randn(
        (B, K, N),
        dtype=torch.float32,
        device="cuda",
    )

    ref_c = torch.einsum("bmk,bkn->bmn", a, b)
    c = torch.zeros_like(ref_c)

    bmm(a, b, c, B, M, N, K)

    print(torch.allclose(c, ref_c, atol=1e-2, rtol=1e-2))
    print(c.sum(dim=2), ref_c.sum(dim=2), sep="\n")
    breakpoint()