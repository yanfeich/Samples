#include <time.h>
#include <unistd.h>
#include <string.h>
#include <algorithm>
#include <synapse_api.h>
#include <synapse_common_types.hpp>
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
    std::vector<std::string> dtypes = {"bf16", "f16", "f32", "f8"};

    for (size_t i = 0; i < dtypes.size(); i++)
    {
        synGraphHandle graphHandle = nullptr;
        status = synGraphCreate(&graphHandle, device_type);
        assert(status == synSuccess && "Failed to call synGraphCreate()");

        std::vector<synTensor> inputs;
        synDataType tensor_type = syn_type_fp16;
        unsigned X_shape[] = {H, T, B};
        unsigned Y_shape[] = {H, T, B};
        if (dtypes[i] == "bf16")
        {
            tensor_type = syn_type_bf16;
        }
        else if (dtypes[i] == "f32")
        {
            tensor_type = syn_type_single;
        }
        else if (dtypes[i] == "f8")
        {
            tensor_type = syn_type_fp8_143;
        }

        // ************************************************ create X tensor
        unsigned int offset = 0;
        synSectionHandle X_SectionHandle = nullptr;
        status = synSectionCreate(&X_SectionHandle, 0, graphHandle);
        assert(status == synSuccess && "Failed to call synSectionCreate()");

        synTensorDescriptor X_desc;
        X_desc.m_dataType = tensor_type;
        X_desc.m_dims = 3UL;
        X_desc.m_name = "X";
        memset(X_desc.m_strides, 0, sizeof(X_desc.m_strides));
        memset(X_desc.m_sizes, 0, sizeof(X_desc.m_sizes));
        memcpy(X_desc.m_sizes, X_shape, 3 * sizeof(unsigned));

        synTensor syn_X = nullptr;
        status = synTensorCreate(&syn_X, &X_desc, X_SectionHandle, offset);
        assert(status == synSuccess && "Failed to call synTensorCreate()");
        inputs.push_back(syn_X);

        // I tensor
        unsigned num_index = 1;
        unsigned I_shape[] = {num_index};
        unsigned I_size = num_index * 4;
        synSectionHandle I_SectionHandle = nullptr;
        status = synSectionCreate(&I_SectionHandle, 0, graphHandle);
        assert(status == synSuccess && "Failed to call synSectionCreate()");

        synTensorDescriptor I_desc;
        I_desc.m_dataType = syn_type_int32;
        I_desc.m_dims = 1UL;
        I_desc.m_name = "I";
        memset(I_desc.m_strides, 0, sizeof(I_desc.m_strides));
        memset(I_desc.m_sizes, 0, sizeof(I_desc.m_sizes));
        memcpy(I_desc.m_sizes, I_shape, 1 * sizeof(unsigned));

        synTensor syn_I = nullptr;
        status = synTensorCreate(&syn_I, &I_desc, I_SectionHandle, offset);
        assert(status == synSuccess && "Failed to call synTensorCreate()");
        inputs.push_back(syn_I);

        // Source tensor
        unsigned S_shape[] = {H, num_index, B};
        unsigned S_size = H * num_index * B;
        synSectionHandle S_SectionHandle = nullptr;
        status = synSectionCreate(&S_SectionHandle, 0, graphHandle);
        assert(status == synSuccess && "Failed to call synSectionCreate()");

        synTensorDescriptor S_desc;
        S_desc.m_dataType = tensor_type;
        S_desc.m_dims = 3UL;
        S_desc.m_name = "S";
        memset(S_desc.m_strides, 0, sizeof(S_desc.m_strides));
        memset(S_desc.m_sizes, 0, sizeof(S_desc.m_sizes));
        memcpy(S_desc.m_sizes, S_shape, 3 * sizeof(unsigned));

        synTensor syn_S = nullptr;
        status = synTensorCreate(&syn_S, &S_desc, S_SectionHandle, offset);
        assert(status == synSuccess && "Failed to call synTensorCreate()");
        inputs.push_back(syn_S);

        // output tensor
        std::vector<synTensor> outputs;

        synSectionHandle Y_SectionHandle = nullptr;
        status = synSectionCreate(&Y_SectionHandle, 0, graphHandle);
        assert(status == synSuccess && "Failed to call synSectionCreate()");

        synTensorDescriptor Y_desc;
        Y_desc.m_dataType = tensor_type;
        Y_desc.m_dims = 3UL;
        Y_desc.m_name = "Y";
        memset(Y_desc.m_strides, 0, sizeof(Y_desc.m_strides));
        memset(Y_desc.m_sizes, 0, sizeof(Y_desc.m_sizes));
        memcpy(Y_desc.m_sizes, Y_shape, 3 * sizeof(unsigned));
        synTensor syn_Y = nullptr;

        status = synTensorCreate(&syn_Y, &Y_desc, Y_SectionHandle, offset);
        assert(status == synSuccess && "Failed to call synTensorCreate()");
        outputs.push_back(syn_Y);

        std::string guid = "index_copy_fwd_" + dtypes[i];
        std::string node_name = "Index_Copy";

        ns_IndexCopy::Params params{};
        params.axis = 1;
        status = synNodeCreate(graphHandle,
                               inputs.data(),
                               outputs.data(),
                               inputs.size(),
                               outputs.size(),
                               &params,
                               sizeof(params),
                               guid.c_str(),
                               node_name.c_str(),
                               nullptr,
                               nullptr);
        assert(status == synSuccess && "Failed to call synNodeCreate()");

        synRecipeHandle recipeHandle = nullptr;
        std::string name = guid + ".recipe";
        status = synGraphCompile(&recipeHandle,
                                 graphHandle,
                                 name.c_str(),
                                 nullptr);
        assert(status == synSuccess && "Failed to call synGraphCompile()");
        status = synGraphDestroy(graphHandle);
        assert(status == synSuccess && "Failed to call synGraphDestroy()");

        status = synRecipeDestroy(recipeHandle);
        assert(status == synSuccess && "Failed to call synRecipeDestroy()");
    }

    status = synDeviceRelease(deviceId);
    assert(status == synSuccess && "Failed to call synDeviceRelease()");

    status = synDestroy();
    assert(status == synSuccess && "Failed to call synDestroy()");
    return 0;
}
