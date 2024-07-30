# Synapse API

## guid

```bash
index_copy_fwd_bf16
index_copy_fwd_f16
index_copy_fwd_f32
index_copy_fwd_f8
```

## params

located in `/usr/include/habanalabs/perf_lib_layer_params.h`

```cpp
/*
 * Kernel name : index_copy_fwd_f32, index_copy_fwd_f16, index_copy_fwd_bf16,
                 index_copy_fwd_i32, index_copy_fwd_u16,
                 index_copy_fwd_i16, index_copy_fwd_u8, index_copy_fwd_i8
 */
namespace ns_IndexCopy
{
    struct Params
    {
        int axis;   //  Axis is in python order
    };
} // namespace ns_IndexCopy

```

## build

```bash
make
```

## run

```bash
./index_copy
```
