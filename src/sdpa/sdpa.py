import os
os.environ['ENABLE_EXPERIMENTAL_FLAGS'] = '1'
os.environ['VISUALIZATION_MODE'] = '0'

import torch
import torch.nn as nn
import habana_frameworks.torch.hpu as ht
from habana_frameworks.torch.hpex.kernels import FusedSDPA as sdpa

B = 64
QT = 1
KVT = 1024
M = 32
D = 128


q = torch.randn([B, M, QT, D]).to(torch.bfloat16).to('hpu')
k = torch.randn([B, M, KVT, D]).to(torch.bfloat16).to('hpu')
v = torch.randn([B, M, KVT, D]).to(torch.bfloat16).to('hpu')
m = torch.ones(B, M, KVT, KVT).to(torch.bool).to('hpu')

ht.synchronize()

with torch.no_grad():
    with ht.sdp_kernel(enable_recompute=True):
        attn_output = sdpa.apply(q, k, v, None, 0.0, False, None, "fast")


ht.synchronize()

# import torch
# import habana_frameworks.torch.core as htcore
# from habana_frameworks.torch.hpex.normalization import FusedRMSNorm as fn
# from habana_frameworks.torch.hpex.kernels import FusedSDPA as sdpa
# q = torch.randn(1, 32, 8, 128).to(torch.bfloat16).to('hpu')
# k = torch.randn(1, 32, 8, 128).to(torch.bfloat16).to('hpu')
# v = torch.randn(1, 32, 8, 128).to(torch.bfloat16).to('hpu')
# #m = torch.randn(2, 1, 8, 8).to(torch.bool).to('hpu')
# max_p = 8
# m = torch.tril(torch.ones((max_p, max_p), dtype=torch.bool)).view(1, 1, 8, 8)
# m = (1 - m) * - 1000

# o = sdpa.apply(q, k, v, m, 0.0, False, None)

# o1 = sdpa.apply(q, k, v, None, 0.0, True, None)

# print((o - o1).abs().max())
