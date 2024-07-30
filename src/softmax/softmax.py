import os
os.environ['ENABLE_EXPERIMENTAL_FLAGS'] = '1'
os.environ['VISUALIZATION_MODE'] = '0'

import torch
import torch.nn as nn
import habana_frameworks.torch.hpu as ht

B = 64
T = 2
H = 4096


x = torch.randn([B, T, H]).to(torch.float32).to('hpu')
ht.synchronize()

with torch.no_grad():
    hidden_states = nn.functional.softmax(x, dim=-1, dtype=torch.float32)

ht.synchronize()
x = torch.randn([B, T, H]).to(torch.bfloat16).to('hpu')
ht.synchronize()

with torch.no_grad():
    hidden_states = nn.functional.softmax(x, dim=-1, dtype=torch.bfloat16)

ht.synchronize()
