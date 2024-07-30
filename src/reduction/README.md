# Synapse API

## guid

``` bash
reduce_max_bf16
reduce_max_f16
reduce_max_f32
reduce_mean_bf16
reduce_mean_f16
reduce_mean_f32
reduce_min_bf16
reduce_min_f16
reduce_min_f32
reduce_prod_bf16
reduce_prod_f16
reduce_prod_f32
reduce_sum_bf16
reduce_sum_f16
reduce_sum_f32
```

## params

```cpp
/*
 * Kernel name: reduce_sum_f32,
 *              reduce_sum_i8,
 *              reduce_sum_i16,
 *              reduce_prod_f32,
 *              reduce_L1_f32,
 *              reduce_L2_f32,
 *              reduce_log_sum_f32,
 *              reduce_log_sum_exp_f32,
 *              reduce_sum_square_f32,
 *              reduce_max_f32,
 *              reduce_min_f32,
 *              reduce_mean_f32,
 *              argmin_i8,
 *              argmin_i16,
 *              argmin_f32,
 *              argmax_i8,
 *              argmax_i16,
 *              argmax_f32,
 *              hardmax_f32
 */
namespace ns_Reduction
{
    struct Params
    {
        unsigned int reductionDimension;
    };

    struct ParamsV2
    {
        unsigned int reductionDimensionMask; // Bit mask representing dimensions(axes) to be reduced.
                                             // A '1' at Bit_n means dimension 'n' should be reduced
                                             // (with Bit_0 being LSB).
                                             // Eg: mask with binary value 1011 means
                                             // dimensions 3, 1 and 0 are to be reduced(TPC order).
        bool keepDim;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************

} // namespace ns_Reduction
```

## build

```bash
make
```

## run

```bash
./reduction
```
