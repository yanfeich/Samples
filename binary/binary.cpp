#include <time.h>
#include <unistd.h>
#include <string.h>
#include <algorithm>
#include <synapse_api.h>
#include <synapse_common_types.hpp>
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
    std::vector<std::string> unary_guids = {"add", "mult", "sub", "pow", "div", "mod",
                                            "min", "max"};
    std::vector<std::string> dtypes = {"bf16", "f32", "f16"};

    for (size_t i = 0; i < unary_guids.size(); i++)
    {
        for (size_t j = 0; j < dtypes.size(); j++)
        {
            if (dtypes[j] == "f16" || dtypes[j] == "bf16")
            {
                if (unary_guids[i] == "mod" ||
                    unary_guids[i] == "div")
                {
                    continue;
                }
            }

            // ************************************************ compile
            synGraphHandle graphHandle = nullptr;
            status = synGraphCreate(&graphHandle, device_type);
            assert(status == synSuccess && "Failed to call synGraphCreate()");

            std::vector<synTensor> inputs;
            unsigned X_shape[] = {H, T, B};
            unsigned X_size = H * T * B;

            unsigned Y_shape[] = {H, T, B};
            unsigned Y_size = H * T * B;

            synDataType tensor_type = syn_type_fp16;

            if (dtypes[j] == "bf16")
            {
                X_size = X_size * 2;
                Y_size = Y_size * 2;
            }
            else if (dtypes[j] == "bf16")
            {
                X_size = X_size * 2;
                Y_size = Y_size * 2;
                tensor_type = syn_type_bf16;
            }
            else if (dtypes[j] == "f32")
            {
                X_size = X_size * 4;
                Y_size = Y_size * 4;
                tensor_type = syn_type_single;
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
            inputs.push_back(syn_X);

            std::vector<synTensor> outputs;
            std::string guid = unary_guids[i] + "_fwd_" + dtypes[j];

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

            std::string node_name = guid + "_" + std::to_string(i) + "_" + std::to_string(j);
            status = synNodeCreate(graphHandle,
                                   inputs.data(),
                                   outputs.data(),
                                   2,
                                   1,
                                   nullptr,
                                   0,
                                   guid.c_str(),
                                   node_name.c_str(),
                                   nullptr,
                                   nullptr);
            assert(status == synSuccess && "Failed to call synNodeCreate()");
            //************************************************ compile graph
            synRecipeHandle recipeHandle = nullptr;
            std::string name = guid + ".recipe";
            status = synGraphCompile(&recipeHandle,
                                     graphHandle,
                                     name.c_str(),
                                     nullptr);
            assert(status == synSuccess && "Failed to call synGraphCompile()");

            // clean
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
