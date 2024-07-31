import os
os.environ['ENABLE_EXPERIMENTAL_FLAGS'] = '1'
os.environ['VISUALIZATION_MODE'] = '0'
os.environ['GRAPH_VISUALIZATION'] = '1'
os.environ['HABANA_LOGS'] = 'logs'
os.environ['LOG_LEVEL_ALL'] = '0'

import torch
import torch.nn as nn
import habana_frameworks.torch.hpu as ht

B = 32
S = 32000

scores = torch.randn([B, S]).to(torch.bfloat16).to('hpu')

with torch.no_grad():
    outputs = torch.argmax(scores, dim = -1)

ht.synchronize()


scores = torch.randn([B, S]).to(torch.float32).to('hpu')

with torch.no_grad():
    outputs = torch.argmax(scores, dim = 0)

ht.synchronize()
