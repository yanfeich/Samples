# Synapse API

## guid

```bash
slice
```

## params

```cpp
struct synSliceParamsV2
{
    unsigned axes[HABANA_DIM_MAX];
    unsigned _structPadding = 0;
    TSize    starts[HABANA_DIM_MAX];
    TSize    ends[HABANA_DIM_MAX];
    TSize    steps[HABANA_DIM_MAX];
};
```

## build

```bash
make
```

## run

```bash
./slice
```
