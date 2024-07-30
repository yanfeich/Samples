# Synapse API

## guid

```bash
sdpa_fwd_bf16
sdpa_recomp_fwd_bf16
```

## params

```cpp
namespace ns_Sdpa
{
    struct Params
    {
        float scale;    // Softmax scale, typ. 1.0/sqrt(head dim)
        bool is_causal; // is attention mask a lower triangular matrix of 1s
        ns_DropoutKernel::ParamsOptionalMaskOut dropout;
        bool is_inference;
    };
    struct ParamsV2 : public Params
    {
        SdpaSoftmaxMode_t softmax_mode;
    };
    struct ParamsV3 : public ParamsV2
    {
        unsigned int flags; // Flags to convey different operating
                            // modes like fp8 measurement
    };
} // namespace ns_Sdpa
```

## build

```bash
make
```

## run

```bash
./sdpa
./sdpa_recompute
```
