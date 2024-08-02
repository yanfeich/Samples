#include <time.h>
#include <unistd.h>
#include <string.h>
#include <algorithm>
#include <synapse_api.h>
#include <synapse_common_types.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <map>

int main(int argc, char *argv[])
{
    unsigned L = 40;
    unsigned M = 32;
    unsigned N1 = 10;
    unsigned N2 = 20;
    unsigned N3 = 30;

    synStatus status = synInitialize();
    assert(status == synSuccess && "Failed to call  synInitialize()");

    uint32_t deviceId = 0;
    status = synDeviceAcquire(&deviceId, nullptr);
    assert(status == synSuccess && "Failed to call  synDeviceAcquire()");

    synDeviceInfo deviceInfo;
    status = synDeviceGetInfo(deviceId, &deviceInfo);
    assert(status == synSuccess && "Failed to call  synDeviceGetInfo()");

    synDeviceType device_type = deviceInfo.deviceType;

    std::vector<std::string> names = {"bf16", "f16", "f32"};
    std::vector<unsigned int> dsizes = {2, 2, 4};
    std::vector<synDataType> dtypes = {syn_type_bf16, syn_type_fp16, syn_type_single};

    for (size_t j = 0; j < dtypes.size(); j++)
    {
        synGraphHandle graphHandle = nullptr;
        status = synGraphCreate(&graphHandle, device_type);
        assert(status == synSuccess && "Failed to call synGraphCreate()");

        std::vector<synTensor> inputs;
        std::vector<synTensor> outputs;

        // ************************************************ create tensors
        unsigned int offset = 0;
        unsigned int N_dims = 2;

        unsigned S1_shape[] = {M, N1};
        unsigned S1_size = M * N1 * dsizes[j];
        unsigned S2_shape[] = {M, N2};
        unsigned S2_size = M * N2 * dsizes[j];
        unsigned S3_shape[] = {M, N3};
        unsigned S3_size = M * N3 * dsizes[j];
        unsigned O_shape[] = {M, N1+N2+N3};
        unsigned O_size = M * (N1+N2+N3) * dsizes[j];
        
        synDataType tensor_type = dtypes[j];

        // ************************************************ create input tensor
        synSectionHandle S1_SectionHandle = nullptr;
        status = synSectionCreate(&S1_SectionHandle, 0, graphHandle);
        assert(status == synSuccess && "Failed to call synSectionCreate()");

        synTensorDescriptor S1_desc;
        S1_desc.m_dataType = tensor_type;
        S1_desc.m_dims = N_dims;
        S1_desc.m_name = "S1";
        memset(S1_desc.m_strides, 0, sizeof(S1_desc.m_strides));
        memset(S1_desc.m_sizes, 0, sizeof(S1_desc.m_sizes));
        memcpy(S1_desc.m_sizes, S1_shape, N_dims * sizeof(unsigned));

        synTensor syn_S1 = nullptr;
        status = synTensorCreate(&syn_S1, &S1_desc, S1_SectionHandle, offset);
        assert(status == synSuccess && "Failed to call synTensorCreate()");
        inputs.push_back(syn_S1);

        synSectionHandle S2_SectionHandle = nullptr;
        status = synSectionCreate(&S2_SectionHandle, 0, graphHandle);
        assert(status == synSuccess && "Failed to call synSectionCreate()");

        synTensorDescriptor S2_desc;
        S2_desc.m_dataType = tensor_type;
        S2_desc.m_dims = N_dims;
        S2_desc.m_name = "S2";
        memset(S2_desc.m_strides, 0, sizeof(S2_desc.m_strides));
        memset(S2_desc.m_sizes, 0, sizeof(S2_desc.m_sizes));
        memcpy(S2_desc.m_sizes, S2_shape, N_dims * sizeof(unsigned));

        synTensor syn_S2 = nullptr;
        status = synTensorCreate(&syn_S2, &S2_desc, S2_SectionHandle, offset);
        assert(status == synSuccess && "Failed to call synTensorCreate()");
        inputs.push_back(syn_S2);
        
        synSectionHandle S3_SectionHandle = nullptr;
        status = synSectionCreate(&S3_SectionHandle, 0, graphHandle);
        assert(status == synSuccess && "Failed to call synSectionCreate()");

        synTensorDescriptor S3_desc;
        S3_desc.m_dataType = tensor_type;
        S3_desc.m_dims = N_dims;
        S3_desc.m_name = "S3";
        memset(S3_desc.m_strides, 0, sizeof(S3_desc.m_strides));
        memset(S3_desc.m_sizes, 0, sizeof(S3_desc.m_sizes));
        memcpy(S3_desc.m_sizes, S3_shape, N_dims * sizeof(unsigned));

        synTensor syn_S3 = nullptr;
        status = synTensorCreate(&syn_S3, &S3_desc, S3_SectionHandle, offset);
        assert(status == synSuccess && "Failed to call synTensorCreate()");
        inputs.push_back(syn_S3);

        //************************************************ create output tensor
        synSectionHandle O_SectionHandle = nullptr;
        status = synSectionCreate(&O_SectionHandle, 0, graphHandle);
        assert(status == synSuccess && "Failed to call synSectionCreate()");

        synTensorDescriptor O_desc;
        O_desc.m_dataType = tensor_type;
        O_desc.m_dims = N_dims;
        O_desc.m_name = "O";
        memset(O_desc.m_strides, 0, sizeof(O_desc.m_strides));
        memset(O_desc.m_sizes, 0, sizeof(O_desc.m_sizes));
        memcpy(O_desc.m_sizes, O_shape, N_dims * sizeof(unsigned));
        synTensor syn_O = nullptr;

        status = synTensorCreate(&syn_O, &O_desc, O_SectionHandle, offset);
        assert(status == synSuccess && "Failed to call synTensorCreate()");
        outputs.push_back(syn_O);
        
        std::string guid = "concat";
        std::string node_name = "Concat";
        synConcatenateParams concat_params;
        concat_params.axis = 1;

        status = synNodeCreate(graphHandle,
                               inputs.data(),
                               outputs.data(),
                               3,
                               1,
                               &concat_params,
                               sizeof(concat_params),
                               guid.c_str(),
                               node_name.c_str(),
                               nullptr,
                               nullptr);
        assert(status == synSuccess && "Failed to call synNodeCreate()");

        //************************************************ compile graph
        synRecipeHandle recipeHandle = nullptr;
        std::string name = "concat_" + names[j];
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
