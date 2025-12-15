import triton
import torch
import triton.language as tl


def _get_autotune_configs():
    return [
        triton.Config(
            kwargs={
                "BLOCK_B": 1,
                "BLOCK_S": bs,
                "BLOCK_H": bh,
                "BLOCK_D": bd,
                "NUM_D_STAGES": nds,
                "NUM_H_STAGES": nhs,
            },
            num_warps=nw,
        )
        for nds in [1, 2, 4]
        for nhs in [1, 2, 4]
        for nw in [2, 4, 8, 16, 32]
        for bs in [4, 8, 16]
        for bh in [16, 32, 64, 128]
        for bd in [16, 32, 64, 128]
    ]


@triton.autotune(
    configs=_get_autotune_configs(),
    key=["H", "D"],
)
@triton.jit
def rope_forward_kernel(
    x_ptr,
    z_ptr,
    cos_ptr,
    sin_ptr,
    B,
    S,
    H,
    D,
    BLOCK_B: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_D_STAGES: tl.constexpr,
    NUM_H_STAGES: tl.constexpr,
    WARP_SPECIALISE: tl.constexpr = False,
):
    """
    Apply rotary positional embeddings to input

    :param x_ptr: (B, S, H, D), bf16/fp16/fp32
    :param z_ptr: (B, S, H, D), bf16/fp16/fp32
    :param cos_ptr: (S, D), bf16/fp16/fp32
    :param sin_ptr: (S, D), bf16/fp16/fp32
    """

    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)

    mask_b = offs_b < B
    mask_s = offs_s < S

    for d in tl.range(
        0, D, BLOCK_D, num_stages=NUM_D_STAGES, warp_specialize=WARP_SPECIALISE
    ):
        offs_d = d + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        offs_df = d + tl.arange(0, BLOCK_D // 2) * 2
        mask_df = offs_df < D

        offs_freqs = offs_s[:, None] * D + offs_df[None, :]
        mask_freqs = mask_s[:, None] & mask_df[None, :]
        cos = tl.load(cos_ptr + offs_freqs, mask_freqs)
        sin = tl.load(sin_ptr + offs_freqs + 1, mask_freqs)

        for h in tl.range(
            0, H, BLOCK_H, num_stages=NUM_H_STAGES, warp_specialize=WARP_SPECIALISE
        ):
            offs_h = h + tl.arange(0, BLOCK_H)
            mask_h = offs_h < H

            offs_block = (
                offs_b[:, None, None, None] * S * H * D
                + offs_s[None, :, None, None] * H * D
                + offs_h[None, None, :, None] * D
                + offs_d[None, None, None, :]
            )
            mask_block = (
                mask_b[:, None, None, None]
                & mask_s[None, :, None, None]
                & mask_h[None, None, :, None]
                & mask_d[None, None, None, :]
            )

            x = tl.load(x_ptr + offs_block, mask_block)
            x = tl.reshape(x, (BLOCK_B, BLOCK_S, BLOCK_H, BLOCK_D // 2, 2))
            x1, x2 = tl.split(x)

            o1 = x1 * cos[None, :, None, :] - x2 * sin[None, :, None, :]
            o2 = x1 * sin[None, :, None, :] + x2 * cos[None, :, None, :]

            o = tl.interleave(o1, o2)
            tl.store(z_ptr + offs_block, o, mask_block)


def rope_forward(hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    B, S, H, D = hidden_states.shape

    assert cos.shape == sin.shape
    assert cos.shape == (S, D)

    def grid(META):
        return (
            triton.cdiv(B, META["BLOCK_B"]),
            triton.cdiv(S, META["BLOCK_S"]),
        )

    z = torch.zeros_like(hidden_states)
    rope_forward_kernel[grid](hidden_states, z, cos, sin, B, S, H, D)

    return z


@triton.autotune(
    configs=_get_autotune_configs(),
    key=["H", "D"],
)
@triton.jit
def rope_forward_inplace_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    B,
    S,
    H,
    D,
    BLOCK_B: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_D_STAGES: tl.constexpr,
    NUM_H_STAGES: tl.constexpr,
    WARP_SPECIALISE: tl.constexpr = False,
):
    """
    Apply rotary positional embeddings to input

    :param x_ptr: (B, S, H, D), bf16/fp16/fp32
    :param z_ptr: (B, S, H, D), bf16/fp16/fp32
    :param cos_ptr: (S, D), bf16/fp16/fp32
    :param sin_ptr: (S, D), bf16/fp16/fp32
    """

    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)

    mask_b = offs_b < B
    mask_s = offs_s < S

    for d in tl.range(
        0, D, BLOCK_D, num_stages=NUM_D_STAGES, warp_specialize=WARP_SPECIALISE
    ):
        offs_d = d + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        offs_df = d + tl.arange(0, BLOCK_D // 2) * 2
        mask_df = offs_df < D

        offs_freqs = offs_s[:, None] * D + offs_df[None, :]
        mask_freqs = mask_s[:, None] & mask_df[None, :]
        cos = tl.load(cos_ptr + offs_freqs, mask_freqs)
        sin = tl.load(sin_ptr + offs_freqs + 1, mask_freqs)

        for h in tl.range(
            0, H, BLOCK_H, num_stages=NUM_H_STAGES, warp_specialize=WARP_SPECIALISE
        ):
            offs_h = h + tl.arange(0, BLOCK_H)
            mask_h = offs_h < H

            offs_block = (
                offs_b[:, None, None, None] * S * H * D
                + offs_s[None, :, None, None] * H * D
                + offs_h[None, None, :, None] * D
                + offs_d[None, None, None, :]
            )
            mask_block = (
                mask_b[:, None, None, None]
                & mask_s[None, :, None, None]
                & mask_h[None, None, :, None]
                & mask_d[None, None, None, :]
            )

            x = tl.load(x_ptr + offs_block, mask_block)
            x = tl.reshape(x, (BLOCK_B, BLOCK_S, BLOCK_H, BLOCK_D // 2, 2))
            x1, x2 = tl.split(x)

            o1 = x1 * cos[None, :, None, :] - x2 * sin[None, :, None, :]
            o2 = x1 * sin[None, :, None, :] + x2 * cos[None, :, None, :]

            o = tl.interleave(o1, o2)
            tl.store(x_ptr + offs_block, o, mask_block)


def rope_forward_inplace(
    hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
):
    B, S, H, D = hidden_states.shape

    assert cos.shape == sin.shape
    assert cos.shape == (S, D)

    def grid(META):
        return (
            triton.cdiv(B, META["BLOCK_B"]),
            triton.cdiv(S, META["BLOCK_S"]),
        )

    rope_forward_inplace_kernel[grid](hidden_states, cos, sin, B, S, H, D)
    return hidden_states


def apply_rotary_emb(
    hidden_states: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
):
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


if __name__ == "__main__":
    B, S, H, D = 4, 2048, 64, 128

    cos = torch.randn((S, D), device="cuda")
    sin = torch.randn((S, D), device="cuda")

    h = torch.randn((B, S, H, D), device="cuda")

    ref = apply_rotary_emb(h, cos[None, :, None, :], sin[None, :, None, :])

    z = rope_forward_inplace(h.clone(), cos, sin)
    assert torch.allclose(z, ref, 0.001, 0.001)
