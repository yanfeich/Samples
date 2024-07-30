# Synapse API

## guid

```bash
batch_gemm
```

## params

located in `/usr/include/habanalabs/synapse_common_types.hpp`

```cpp
struct synGEMMParams
{
    synGEMMParams(bool transpose_a = false, bool transpose_b = false): transpose_a(transpose_a), transpose_b(transpose_b) {}

    bool transpose_a; // transpose A or not
    bool transpose_b; // transpose B or not
};
```

## build

```bash
make
```

## run

```bash
./batch_gemm
```
