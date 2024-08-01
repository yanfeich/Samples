# Synapse API

## guid

```bash
rope_st2_fwd_bf16
rope_st2_fwd_f32
```

## params

```cpp
namespace ns_RoPESt2
{
    struct Params
    {
        unsigned int offset;
    };
    struct ParamsV2 : public Params
    {
        RotaryPosEmbeddingMode_t mode;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_RoPESt2
```

## build

```bash
make
```

## run

```bash
./rope
```