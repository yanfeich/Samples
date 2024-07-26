#include <time.h>
#include <unistd.h>
#include <string.h>
#include <algorithm>
#include <synapse_api.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <map>

int main(int argc, char *argv[])
{
    unsigned B = 64;
    unsigned M = 32;
    unsigned QT = 1;
    unsigned KVT = 1024;
    unsigned D = 128;

    std::map<synDataType, unsigned> dtype2size{
        {syn_type_fp8_143, 1},
        {syn_type_fp8_152, 1},
        {syn_type_bf16, 2},
        {syn_type_fp16, 2},
        {syn_type_uint32, 4},
        {syn_type_int32, 4},
        {syn_type_single, 4},
    };

    synStatus status = synInitialize();
    assert(status == synSuccess && "Failed to call  synInitialize()");

    uint32_t deviceId = 0;
    status = synDeviceAcquire(&deviceId, nullptr);
    assert(status == synSuccess && "Failed to call  synDeviceAcquire()");

    synDeviceInfo deviceInfo;
    status = synDeviceGetInfo(deviceId, &deviceInfo);
    assert(status == synSuccess && "Failed to call  synDeviceGetInfo()");

    synDeviceType device_type = deviceInfo.deviceType;

    std::vector<std::string> dtypes = {"bf16", "f16", "fp8_143"};

    for (size_t j = 0; j < dtypes.size(); j++)
    {
        synGraphHandle graphHandle = nullptr;
        status = synGraphCreate(&graphHandle, device_type);
        assert(status == synSuccess && "Failed to call synGraphCreate()");

        std::vector<synTensor> inputs;
        std::vector<synTensor> outputs;

        // ************************************************ create A tensor
        unsigned int offset = 0;
        size_t max_dim = 4;
        unsigned A_shape[] = {D, QT, M, B};
        unsigned A_size = D * QT * M * B;

        unsigned B_shape[] = {D, KVT, M, B};
        unsigned B_size = D * KVT * M * B;

        unsigned C_shape[] = {KVT, QT, M, B};
        unsigned C_size = KVT * QT * M * B;

        synDataType tensor_type = syn_type_fp16;

        if (dtypes[j] == "bf16" || dtypes[j] == "f16")
        {
            A_size = A_size * 2;
            B_size = B_size * 2;
            C_size = C_size * 2;
            tensor_type = syn_type_fp16;
        }
        else if (dtypes[j] == "fp8_143")
        {
            A_size = A_size * 1;
            B_size = B_size * 1;
            C_size = C_size * 1;
            tensor_type = syn_type_fp8_143;
        }

        synSectionHandle A_SectionHandle = nullptr;
        status = synSectionCreate(&A_SectionHandle, 0, graphHandle);
        assert(status == synSuccess && "Failed to call synSectionCreate()");

        synTensorDescriptor A_desc;
        A_desc.m_dataType = tensor_type;
        A_desc.m_dims = max_dim;
        A_desc.m_name = "A";
        memset(A_desc.m_strides, 0, sizeof(A_desc.m_strides));
        memset(A_desc.m_sizes, 0, sizeof(A_desc.m_sizes));
        memcpy(A_desc.m_sizes, A_shape, max_dim * sizeof(unsigned));

        synTensor syn_A = nullptr;
        status = synTensorCreate(&syn_A, &A_desc, A_SectionHandle, offset);
        assert(status == synSuccess && "Failed to call synTensorCreate()");
        inputs.push_back(syn_A);

        //************************************************ create B tensor

        synSectionHandle B_SectionHandle = nullptr;
        status = synSectionCreate(&B_SectionHandle, 0, graphHandle);
        assert(status == synSuccess && "Failed to call synSectionCreate()");

        synTensorDescriptor B_desc;
        B_desc.m_dataType = tensor_type;
        B_desc.m_dims = max_dim;
        B_desc.m_name = "B";
        memset(B_desc.m_strides, 0, sizeof(B_desc.m_strides));
        memset(B_desc.m_sizes, 0, sizeof(B_desc.m_sizes));
        memcpy(B_desc.m_sizes, B_shape, max_dim * sizeof(unsigned));
        synTensor syn_B = nullptr;
        status = synTensorCreate(&syn_B, &B_desc, B_SectionHandle, offset);
        assert(status == synSuccess && "Failed to call synTensorCreate()");
        inputs.push_back(syn_B);

        //************************************************ create C tensor
        // NCHW => WHCN

        synSectionHandle C_SectionHandle = nullptr;
        status = synSectionCreate(&C_SectionHandle, 0, graphHandle);
        assert(status == synSuccess && "Failed to call synSectionCreate()");

        synTensorDescriptor C_desc;
        C_desc.m_dataType = tensor_type;
        C_desc.m_dims = max_dim;
        C_desc.m_name = "C";
        memset(C_desc.m_strides, 0, sizeof(C_desc.m_strides));
        memset(C_desc.m_sizes, 0, sizeof(C_desc.m_sizes));
        memcpy(C_desc.m_sizes, C_shape, max_dim * sizeof(unsigned));
        synTensor syn_C = nullptr;

        status = synTensorCreate(&syn_C, &C_desc, C_SectionHandle, offset);
        assert(status == synSuccess && "Failed to call synTensorCreate()");
        outputs.push_back(syn_C);

        std::string guid = "batch_gemm";
        std::string node_name = "BatchGemm";
        synGEMMParams gemm_params;
        bool transpose_a = false;
        bool transpose_b = true;
        gemm_params.transpose_a = transpose_a;
        gemm_params.transpose_b = transpose_b;

        status = synNodeCreate(graphHandle,
                               inputs.data(),
                               outputs.data(),
                               2,
                               1,
                               &gemm_params,
                               sizeof(gemm_params),
                               guid.c_str(),
                               node_name.c_str(),
                               nullptr,
                               nullptr);
        assert(status == synSuccess && "Failed to call synNodeCreate()");

        //************************************************ compile graph
        synRecipeHandle recipeHandle = nullptr;
        std::string name = guid + "_" + dtypes[j];
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
