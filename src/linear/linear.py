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
IF = 11008
OF = 4096

seq_length = 4

proj = nn.Linear(in_features = IF, out_features = OF, dtype=torch.bfloat16).to('hpu')

states = torch.randn([B, IF]).to(torch.bfloat16).to('hpu')

with torch.no_grad():
    outputs = proj(states)

ht.synchronize()

