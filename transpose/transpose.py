import os
os.environ['ENABLE_EXPERIMENTAL_FLAGS'] = '1'
os.environ['VISUALIZATION_MODE'] = '0'

import torch
import torch.nn as nn
import habana_frameworks.torch.hpu as ht

B = 64
T = 2
M = 32
D = 128


x = torch.randn([B, M, T, D]).to(torch.float32).to('hpu')
ht.synchronize()

with torch.no_grad():
    y = x.permute([0,2,1,3])

ht.synchronize()

x = torch.randn([B, M, T, D]).to(torch.bfloat16).to('hpu')

ht.synchronize()

with torch.no_grad():
    y = x.permute([0,2,1,3])

ht.synchronize()
