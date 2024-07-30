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
    unsigned M = 32;
    unsigned T = 128;
    unsigned D = 128;

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

    // ************************************************ compile
    synGraphHandle graphHandle = nullptr;
    status = synGraphCreate(&graphHandle, device_type);
    assert(status == synSuccess && "Failed to call synGraphCreate()");

    std::vector<synTensor> inputs;
    std::vector<synTensor> outputs;

    // ************************************************ create Q tensor
    unsigned int offset = 0;
    unsigned X_shape[] = {D, T, M, B};
    unsigned X_size = D * T * M * B * 2;
    synSectionHandle X_SectionHandle = nullptr;
    status = synSectionCreate(&X_SectionHandle, 0, graphHandle);
    assert(status == synSuccess && "Failed to call synSectionCreate()");

    synTensorDescriptor X_desc;
    X_desc.m_dataType = syn_type_bf16;
    X_desc.m_dims = 4UL;
    X_desc.m_name = "Q";
    memset(X_desc.m_strides, 0, sizeof(X_desc.m_strides));
    memset(X_desc.m_sizes, 0, sizeof(X_desc.m_sizes));
    memcpy(X_desc.m_sizes, X_shape, 4 * sizeof(unsigned));

    synTensor syn_X = nullptr;
    status = synTensorCreate(&syn_X, &X_desc, X_SectionHandle, offset);
    assert(status == synSuccess && "Failed to call synTensorCreate()");
    inputs.push_back(syn_X);
    
    // K
    unsigned K_shape[] = {D, T, M, B};
    unsigned K_size = D * T * M * B * 2;
    synSectionHandle K_SectionHandle = nullptr;
    status = synSectionCreate(&K_SectionHandle, 0, graphHandle);
    assert(status == synSuccess && "Failed to call synSectionCreate()");

    synTensorDescriptor K_desc;
    K_desc.m_dataType = syn_type_bf16;
    K_desc.m_dims = 4UL;
    K_desc.m_name = "K";
    memset(K_desc.m_strides, 0, sizeof(K_desc.m_strides));
    memset(K_desc.m_sizes, 0, sizeof(K_desc.m_sizes));
    memcpy(K_desc.m_sizes, K_shape, 4 * sizeof(unsigned));

    synTensor syn_K = nullptr;
    status = synTensorCreate(&syn_K, &K_desc, K_SectionHandle, offset);
    assert(status == synSuccess && "Failed to call synTensorCreate()");
    inputs.push_back(syn_K);

    // V
    unsigned V_shape[] = {D, T, M, B};
    unsigned V_size = D * T * M * B * 2;
    synSectionHandle V_SectionHandle = nullptr;
    status = synSectionCreate(&V_SectionHandle, 0, graphHandle);
    assert(status == synSuccess && "Failed to call synSectionCreate()");

    synTensorDescriptor V_desc;
    V_desc.m_dataType = syn_type_bf16;
    V_desc.m_dims = 4UL;
    V_desc.m_name = "V";
    memset(V_desc.m_strides, 0, sizeof(V_desc.m_strides));
    memset(V_desc.m_sizes, 0, sizeof(V_desc.m_sizes));
    memcpy(V_desc.m_sizes, V_shape, 4 * sizeof(unsigned));

    synTensor syn_V = nullptr;
    status = synTensorCreate(&syn_V, &V_desc, V_SectionHandle, offset);
    assert(status == synSuccess && "Failed to call synTensorCreate()");
    inputs.push_back(syn_V);

    //************************************************ create C tensor
    unsigned Y_shape[] = {D, T, M, B};
    unsigned Y_size = D * T * M * B * 2;
    synSectionHandle Y_SectionHandle = nullptr;
    status = synSectionCreate(&Y_SectionHandle, 0, graphHandle);
    assert(status == synSuccess && "Failed to call synSectionCreate()");

    synTensorDescriptor Y_desc;
    Y_desc.m_dataType = syn_type_bf16;
    Y_desc.m_dims = 4UL;
    Y_desc.m_name = "Y";
    memset(Y_desc.m_strides, 0, sizeof(Y_desc.m_strides));
    memset(Y_desc.m_sizes, 0, sizeof(Y_desc.m_sizes));
    memcpy(Y_desc.m_sizes, Y_shape, 4 * sizeof(unsigned));
    synTensor syn_Y = nullptr;

    status = synTensorCreate(&syn_Y, &Y_desc, Y_SectionHandle, offset);
    assert(status == synSuccess && "Failed to call synTensorCreate()");
    outputs.push_back(syn_Y);

    unsigned Z_shape[] = {T, T, M, B};
    unsigned Z_size = T * T * M * B * 2;
    synSectionHandle Z_SectionHandle = nullptr;
    status = synSectionCreate(&Z_SectionHandle, 0, graphHandle);
    assert(status == synSuccess && "Failed to call synSectionCreate()");

    synTensorDescriptor Z_desc;
    Z_desc.m_dataType = syn_type_bf16;
    Z_desc.m_dims = 4UL;
    Z_desc.m_name = "Z";
    memset(Z_desc.m_strides, 0, sizeof(Z_desc.m_strides));
    memset(Z_desc.m_sizes, 0, sizeof(Z_desc.m_sizes));
    memcpy(Z_desc.m_sizes, Z_shape, 4 * sizeof(unsigned));
    synTensor syn_Z = nullptr;

    status = synTensorCreate(&syn_Z, &Z_desc, Z_SectionHandle, offset);
    assert(status == synSuccess && "Failed to call synTensorCreate()");
    outputs.push_back(syn_Z);

    //************************************************ create sdpa node
    std::string guid = "sdpa_fwd_bf16";
    std::string node_name = "SDPA";
    ns_Sdpa::ParamsV3 params;

    params.is_inference = true;
    params.scale = 0.08838834764;
    params.is_causal = true;
    params.softmax_mode = SDPA_FAST_SOFTMAX;

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

    //************************************************ compile graph
    synRecipeHandle recipeHandle = nullptr;
    std::string name = guid;
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

    status = synDeviceRelease(deviceId);
    assert(status == synSuccess && "Failed to call synDeviceRelease()");

    status = synDestroy();
    assert(status == synSuccess && "Failed to call synDestroy()");
    return 0;
}
