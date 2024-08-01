import os
os.environ['ENABLE_EXPERIMENTAL_FLAGS'] = '1'
os.environ['VISUALIZATION_MODE'] = '0'

# os.environ['LOG_LEVEL_ALL'] = '0'
# os.environ['HABANA_LOGS'] = './habana_logs'

import torch
from torch import nn
import habana_frameworks.torch.hpu as ht
from habana_frameworks.torch.hpex.kernels import RotaryPosEmbeddingHelperV2 as FusedRoPE


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device='hpu'):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = max_position_embeddings

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, seq_len=None, device='hpu'):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len is not None and seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=device)

        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q: torch.Tensor, cos, sin, position_ids, unsqueeze_dim=1):
    # b, h, s, d = q.shape
    # q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed


B = 64
M = 32
QT = 1
KVT = 1024
D = 128
max_position_embeddings = 2048

rotary_emb = RotaryEmbedding(D, max_position_embeddings=max_position_embeddings).to(torch.bfloat16)
cos, sin = rotary_emb.cos_cached, rotary_emb.sin_cached

q = torch.randn([B, M, QT, D]).to(torch.bfloat16).to('hpu')
# k = torch.randn([B, M, KVT, D]).to(torch.bfloat16).to('hpu')

q_pos = torch.randint(0, max_position_embeddings-1, (B, QT), dtype=torch.long, device='hpu')

ht.synchronize()

with torch.no_grad():
    q_fused_rope = FusedRoPE.apply(q, cos.unsqueeze(0).unsqueeze(0).clone(), sin.unsqueeze(0).unsqueeze(0).clone(), q_pos)
    q_default_rope = apply_rotary_pos_emb(q, cos, sin, q_pos)

# print(q_fused_rope)
# print(q_default_rope)
print(torch.allclose(q_fused_rope, q_default_rope, rtol=0.3, atol=0.3))
assert torch.allclose(q_fused_rope, q_default_rope, rtol=0.3, atol=0.3)

ht.synchronize()