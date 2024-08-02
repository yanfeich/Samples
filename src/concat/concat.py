import os
os.environ['ENABLE_EXPERIMENTAL_FLAGS'] = '1'
os.environ['VISUALIZATION_MODE'] = '0'
os.environ['GRAPH_VISUALIZATION'] = '1'
os.environ['HABANA_LOGS'] = 'logs'
os.environ['LOG_LEVEL_ALL'] = '0'

import torch
import torch.nn as nn
import habana_frameworks.torch.hpu as ht

L = 32
T = 16
M = 20
N = 10
B = 64

states = torch.randn([M, L]).to(torch.bfloat16).to('hpu')
append = torch.randn([N, L]).to(torch.bfloat16).to('hpu')

with torch.no_grad():
    outputs = torch.cat((states,append), 0)

ht.synchronize()


states = torch.randn([M, L]).to(torch.float16).to('hpu')
append = torch.randn([M, T]).to(torch.float16).to('hpu')

with torch.no_grad():
    outputs = torch.concat((states, append), 1)

ht.synchronize()

list = []
list.append(torch.randn([B, M, L]).to(torch.float32).to('hpu'))
list.append(torch.randn([B, M, L]).to(torch.float32).to('hpu'))
list.append(torch.randn([B, M, L]).to(torch.float32).to('hpu'))
list.append(torch.randn([B, M, L]).to(torch.float32).to('hpu'))
list.append(torch.randn([B, M, L]).to(torch.float32).to('hpu'))

with torch.no_grad():
    outputs = torch.concat(list, -1)

ht.synchronize()
