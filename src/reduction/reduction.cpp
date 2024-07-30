#include <time.h>
#include <unistd.h>
#include <string.h>
#include <algorithm>
#include <synapse_api.h>
#include <perf_lib_layer_params.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <map>

int main(int argc, char *argv[])
{
    unsigned B = 64;
    unsigned T = 128;
    unsigned H = 4096;

    std::map<synDataType, unsigned> dtype2size{
        {syn_type_bf16, 2}};

    synStatus status = synInitialize();
    assert(status == synSuccess && "Failed to call  synInitialize()");

    uint32_t deviceId = 0;
    status = synDeviceAcquire(&deviceId, nullptr);
    assert(status == synSuccess && "Failed to call  synDeviceAcquire()");

    synDeviceInfo deviceInfo;
    status = synDeviceGetInfo(deviceId, &deviceInfo);
    assert(status == synSuccess && "Failed to call  synDeviceGetInfo()");

    synDeviceType device_type = deviceInfo.deviceType;

    //************************************************ create rms node
    std::vector<std::string> reduction_guids = {"reduce_max", "reduce_min", "reduce_mean", "reduce_sum", "reduce_prod"};
    std::vector<std::string> dtypes = {"f32", "bf16", "f16"};

    for (size_t i = 0; i < reduction_guids.size(); i++)
    {
        for (size_t j = 0; j < dtypes.size(); j++)
        {
            synGraphHandle graphHandle = nullptr;
            status = synGraphCreate(&graphHandle, device_type);
            assert(status == synSuccess && "Failed to call synGraphCreate()");

            std::vector<synTensor> outputs;
            std::vector<synTensor> inputs;

            std::string guid = reduction_guids[i] + "_" + dtypes[j];

            unsigned int offset = 0;
            unsigned X_shape[] = {H, T, B};
            unsigned X_size = H * T * B * 2;
            synSectionHandle X_SectionHandle = nullptr;
            status = synSectionCreate(&X_SectionHandle, 0, graphHandle);
            assert(status == synSuccess && "Failed to call synSectionCreate()");

            synTensorDescriptor X_desc;
            X_desc.m_dataType = syn_type_bf16;
            X_desc.m_dims = 3UL;
            X_desc.m_name = "X";
            memset(X_desc.m_strides, 0, sizeof(X_desc.m_strides));
            memset(X_desc.m_sizes, 0, sizeof(X_desc.m_sizes));
            memcpy(X_desc.m_sizes, X_shape, 3 * sizeof(unsigned));

            synTensor syn_X = nullptr;
            status = synTensorCreate(&syn_X, &X_desc, X_SectionHandle, offset);
            assert(status == synSuccess && "Failed to call synTensorCreate()");
            inputs.push_back(syn_X);

            unsigned Y_shape[] = {1, T, B};
            unsigned Y_size = T * B * 2;
            synSectionHandle Y_SectionHandle = nullptr;
            status = synSectionCreate(&Y_SectionHandle, 0, graphHandle);
            assert(status == synSuccess && "Failed to call synSectionCreate()");

            synTensorDescriptor Y_desc;
            Y_desc.m_dataType = syn_type_bf16;
            Y_desc.m_dims = 3UL;
            Y_desc.m_name = "Y";
            memset(Y_desc.m_strides, 0, sizeof(Y_desc.m_strides));
            memset(Y_desc.m_sizes, 0, sizeof(Y_desc.m_sizes));
            memcpy(Y_desc.m_sizes, Y_shape, 3 * sizeof(unsigned));
            synTensor syn_Y = nullptr;

            status = synTensorCreate(&syn_Y, &Y_desc, Y_SectionHandle, offset);
            assert(status == synSuccess && "Failed to call synTensorCreate()");
            outputs.push_back(syn_Y);

            std::string node_name = guid;
            ns_Reduction::Params params;
            params.reductionDimension = 0;
            status = synNodeCreate(graphHandle,
                                   inputs.data(),
                                   outputs.data(),
                                   1,
                                   1,
                                   &params,
                                   sizeof(params),
                                   guid.c_str(),
                                   node_name.c_str(),
                                   nullptr,
                                   nullptr);
            assert(status == synSuccess && "Failed to call synNodeCreate()");
            synRecipeHandle recipeHandle = nullptr;

            status = synGraphCompile(&recipeHandle,
                                     graphHandle,
                                     guid.c_str(),
                                     nullptr);
            assert(status == synSuccess && "Failed to call synGraphCompile()");

            status = synGraphDestroy(graphHandle);
            assert(status == synSuccess && "Failed to call synGraphDestroy()");

            status = synRecipeDestroy(recipeHandle);
            assert(status == synSuccess && "Failed to call synRecipeDestroy()");
        }
    }

    status = synDeviceRelease(deviceId);
    assert(status == synSuccess && "Failed to call synDeviceRelease()");

    status = synDestroy();
    assert(status == synSuccess && "Failed to call synDestroy()");
    return 0;
}
