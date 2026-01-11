import torch
import triton
import triton.language as tl


@triton.jit
def ms_reduction_kernel(
    x_ptr,
    ms_ptr,
    B: int,
    N: int,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_b = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
    offs_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N

    mask_b = offs_b < B
    mask_n = offs_n < N

    stride_b = N

    offs_x = offs_b[:, None] * stride_b + offs_n[None, :]
    mask_x = mask_b[:, None] & mask_n[None, :]

    x = tl.load(x_ptr + offs_x, mask_x, 0)

    ms = tl.sum(x * x, axis=1) / N

    tl.atomic_add(ms_ptr + offs_b, ms, mask_b, sem="relaxed")


@triton.jit
def rms_norm_kernel(
    x_ptr,
    ms_ptr,
    out_ptr,
    gamma: float,
    beta: float,
    eps: float,
    B: int,
    N: int,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_b = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
    offs_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N

    mask_b = offs_b < B
    mask_n = offs_n < N

    stride_b = N

    offs_x = offs_b[:, None] * stride_b + offs_n[None, :]
    mask_x = mask_b[:, None] & mask_n[None, :]

    x = tl.load(x_ptr + offs_x, mask_x, 0)
    ms = tl.load(ms_ptr + offs_b, mask_b)

    out = x * tl.rsqrt(ms + eps) * gamma + beta

    tl.store(out_ptr + offs_x, out, mask_x)


def rmsnorm(
    x: torch.Tensor,
    out: torch.Tensor,
    gamma: float,
    beta: float,
    eps: float,
    B: int,
    N: int,
):
    BLOCK_B = 1
    BLOCK_N = 256

    grid = (triton.cdiv(B, BLOCK_B), triton.cdiv(N, BLOCK_N))

    ms = torch.zeros((B,), dtype=x.dtype, device=x.device)

    ms_reduction_kernel[grid](x, ms, B, N, BLOCK_B, BLOCK_N)  # ty:ignore[invalid-argument-type]
    rms_norm_kernel[grid](x, ms, out, gamma, beta, eps, B, N, BLOCK_B, BLOCK_N)  # ty:ignore[invalid-argument-type]

    return out


if __name__ == "__main__":
    B, N = 64, 1024

    x = torch.randn((B, N), device="cuda")
    gamma = 4.2
    beta = 3.42
    eps = 1e-5
    ref_output = gamma * x * ((x * x).mean(dim=1) + eps).rsqrt()[:, None] + beta

    output = torch.zeros_like(ref_output)
    rmsnorm(x, output, gamma, beta, eps, B, N)

    print(torch.allclose(output, ref_output, rtol=1e-3, atol=1e-3))
