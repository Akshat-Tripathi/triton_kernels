import pytest
import torch

from triton_kernels.rope import rope_forward


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for RoPE test")
@pytest.mark.parametrize(
    "B,S,H,D",
    [
        (1, 128, 8, 64),
        (2, 512, 16, 64),
        (4, 2048, 64, 128),
    ],
)
def test_rope_forward_matches_reference(B, S, H, D):
    torch.manual_seed(0)

    cos = torch.randn((S, D), device="cuda")
    sin = torch.randn((S, D), device="cuda")

    h = torch.randn((B, S, H, D), device="cuda")

    ref = apply_rotary_emb(h, cos[None, :, None, :], sin[None, :, None, :])

    z = rope_forward(h, cos, sin)

    assert z.shape == h.shape
    assert torch.allclose(z, ref, rtol=1e-3, atol=1e-3)
