import os
os.environ['ENABLE_EXPERIMENTAL_FLAGS'] = '1'
os.environ['VISUALIZATION_MODE'] = '0'

import torch
import habana_frameworks.torch.hpu as ht
from habana_frameworks.torch.hpex.normalization import FusedRMSNorm

B = 64
T = 2
H = 4096

x = torch.randn([B, T, H]).to(torch.bfloat16).to('hpu')
w = torch.ones([H]).to(torch.bfloat16).to('hpu')

ht.synchronize()

with torch.no_grad():
    hidden_states, _ = torch.ops.hpu.rms_norm(x, w, 1e-5)

ht.synchronize()

x = torch.randn([B, T, H]).to(torch.float32).to('hpu')
w = torch.ones([H]).to(torch.float32).to('hpu')
with torch.no_grad():
    hidden_states, _ = torch.ops.hpu.rms_norm(x, w, 1e-5)

ht.synchronize()
