# Synapse API

## guid

```bash
# RMS Norm
rms_norm_ex_fwd_bf16
rms_norm_ex_fwd_f16
rms_norm_ex_fwd_f32

# LayerNorm
layer_norm_fwd_bf16
layer_norm_fwd_f16
layer_norm_fwd_f32
```

## params

```cpp
/*
 * Kernel name: layer_norm_f32,
 *              layer_norm_fwd_f32, layer_norm_fwd_bf16
 */
namespace ns_LayerNormKernel
{
    struct Params
    {
        bool epsValid;
        float eps;
    };
    struct ParamsNorm : Params
    {
        int NormAxisBmp;  // A bit-map for CWHN. Set res bit for the dim to be normalized
        int ParamAxisBmp; // A bit-map for CWHN. Set res bit for the dim to be normalized
    };
    ASSERT_SIZE_IS_GREATER(ParamsNorm, Params);
    // It should derive after Params as ParamsNorm contents are meaningless in ParamsPt case
    // but only linear params hierarchy is currently supported
    struct ParamsPt : ParamsNorm
    {
        unsigned normalizedShapeDims;
    };
    ASSERT_SIZE_IS_GREATER(ParamsPt, ParamsNorm);
    struct ParamsRmsNorm : public ParamsPt
    {
        bool fastMath;
    };
    ASSERT_SIZE_IS_GREATER(ParamsRmsNorm, ParamsPt);
} // namespace ns_LayerNormKernel
```

## build

```bash
make
```

## run

```bash
./rms
./layernorm
```
