# Intel Gaudi Samples
========================================

The tool reproduces networks written in Synapse API (C\C++) or serialized graphs and runs them on Gaudi.

## Samples

- [gather](./gather/README.md)
- [sdpa](./sdpa/README.md)
- [rope](./rope/README.md)
- [unary](./unary/README.md)
- [gemmTest](./gemmTest/README.md)
- [reduction](./reduction/README.md)
- [argmax](./argmax/README.md)
- [bandwidthTest](./bandwidthTest/README.md)
- [softmax](./softmax/README.md)
- [cast](./cast/README.md)
- [reshape](./reshape/README.md)
- [index_copy](./index_copy/README.md)
- [binary](./binary/README.md)
- [normalization](./normalization/README.md)
- [transpose](./transpose/README.md)
- [batch_gemm](./batch_gemm/README.md)
- [reluTest](./reluTest/README.md)
- [slice](./slice/README.md)
- [linear](./linear/README.md)

## header files

located in `/usr/include/habanalabs`

```bash
.
|-- hccl.h
|-- hccl_types.h
|-- hlml.h
|-- hlthunk.h
|-- hlthunk_err_inject.h
|-- hlthunk_tests.h
|-- perf_lib_layer_params.h
|-- shim_default_plugins.hpp
|-- syn_sl_api.h
|-- synapse_api.h
|-- synapse_api_types.h
|-- synapse_common_types.h
|-- synapse_common_types.hpp
|-- synapse_types.h

```

## Build Graph

![synapse](imgs/synapse.png)
