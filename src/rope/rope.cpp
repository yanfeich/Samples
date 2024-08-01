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

int main(int argc, char* argv[])
{
    unsigned B = 64;
    unsigned MX = 32;
    unsigned MCosSin = 1;
    unsigned T = 1;
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

    unsigned int offset = 0;

    // ************************************************ create input tensor
    unsigned x_shape[] = {D, T, MX, B};
    unsigned x_size = D * T * MX * B * 2;
    synSectionHandle X_SectionHandle = nullptr;
    status = synSectionCreate(&X_SectionHandle, 0, graphHandle);
    assert(status == synSuccess && "Failed to call synSectionCreate()");

    synTensorDescriptor x_desc;
    x_desc.m_dataType = syn_type_bf16;
    x_desc.m_dims = 4UL;
    x_desc.m_name = "X";
    memset(x_desc.m_strides, 0, sizeof(x_desc.m_strides));
    memset(x_desc.m_sizes, 0, sizeof(x_desc.m_sizes));
    memcpy(x_desc.m_sizes, x_shape, 4 * sizeof(unsigned));

    synTensor syn_x = nullptr;
    status = synTensorCreate(&syn_x, &x_desc, X_SectionHandle, offset);
    assert(status == synSuccess && "Failed to call synTensorCreate()");
    inputs.push_back(syn_x);
    
    // ************************************************ create cos tensor
    unsigned cos_shape[] = {D, T, MCosSin, B};
    unsigned cos_size = D * T * MCosSin * B * 2;
    synSectionHandle Cos_SectionHandle = nullptr;
    status = synSectionCreate(&Cos_SectionHandle, 0, graphHandle);
    assert(status == synSuccess && "Failed to call synSectionCreate()");

    synTensorDescriptor cos_desc;
    cos_desc.m_dataType = syn_type_bf16;
    cos_desc.m_dims = 4UL;
    cos_desc.m_name = "Cos";
    memset(cos_desc.m_strides, 0, sizeof(cos_desc.m_strides));
    memset(cos_desc.m_sizes, 0, sizeof(cos_desc.m_sizes));
    memcpy(cos_desc.m_sizes, cos_shape, 4 * sizeof(unsigned));

    synTensor syn_cos = nullptr;
    status = synTensorCreate(&syn_cos, &cos_desc, Cos_SectionHandle, offset);
    assert(status == synSuccess && "Failed to call synTensorCreate()");
    inputs.push_back(syn_cos);
    
    // ************************************************ create sin tensor
    unsigned sin_shape[] = {D, T, MCosSin, B};
    unsigned sin_size = D * T * MCosSin * B * 2;
    synSectionHandle Sin_SectionHandle = nullptr;
    status = synSectionCreate(&Sin_SectionHandle, 0, graphHandle);
    assert(status == synSuccess && "Failed to call synSectionCreate()");

    synTensorDescriptor sin_desc;
    sin_desc.m_dataType = syn_type_bf16;
    sin_desc.m_dims = 4UL;
    sin_desc.m_name = "Sin";
    memset(sin_desc.m_strides, 0, sizeof(sin_desc.m_strides));
    memset(sin_desc.m_sizes, 0, sizeof(sin_desc.m_sizes));
    memcpy(sin_desc.m_sizes, sin_shape, 4 * sizeof(unsigned));

    synTensor syn_sin = nullptr;
    status = synTensorCreate(&syn_sin, &sin_desc, Sin_SectionHandle, offset);
    assert(status == synSuccess && "Failed to call synTensorCreate()");
    inputs.push_back(syn_sin);

    // ************************************************ create sin tensor
    synSectionHandle Y_SectionHandle = nullptr;
    status = synSectionCreate(&Y_SectionHandle, 0, graphHandle);
    assert(status == synSuccess && "Failed to call synSectionCreate()");

    synTensorDescriptor y_desc;
    y_desc.m_dataType = syn_type_bf16;
    y_desc.m_dims = 4UL;
    y_desc.m_name = "Y";
    memset(y_desc.m_strides, 0, sizeof(y_desc.m_strides));
    memset(y_desc.m_sizes, 0, sizeof(y_desc.m_sizes));
    memcpy(y_desc.m_sizes, x_shape, 4 * sizeof(unsigned));
    synTensor syn_y = nullptr;

    status = synTensorCreate(&syn_y, &y_desc, Y_SectionHandle, offset);
    assert(status == synSuccess && "Failed to call synTensorCreate()");
    outputs.push_back(syn_y);

    // ************************************************ create rope node
    std::string guid = "rope_st2_fwd_bf16";
    std::string node_name = "RoPE";
    ns_Sdpa::ParamsV3 params;

    status = synNodeCreate(graphHandle,
                           inputs.data(),
                           outputs.data(),
                           3,
                           1,
                           &params,
                           sizeof(params),
                           guid.c_str(),
                           node_name.c_str(),
                           nullptr,
                           nullptr);
    assert(status == synSuccess && "Failed to call synNodeCreate()");

    // ************************************************ compile

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